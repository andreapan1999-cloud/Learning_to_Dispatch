import argparse
import os
import random
import shutil
import time
import warnings
from typing import List, Tuple, Optional

import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_CREATE_SHM", "0")
import torch
from stable_baselines3 import PPO
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
except Exception:
    MaskablePPO = None
    ActionMasker = None

try:
    from train.joint_maskable_ppo import JointMaskablePPO
except Exception:
    JointMaskablePPO = None
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env_dispatch_ppo import DispatchPPOEnv
import json
import csv
from pathlib import Path
import gymnasium as gym
import yaml
from data_gen.module2_network import build_network_from_config, save_network_csv
from data_gen.module3_travel_time import generate_travel_times, save_travel_times_csv, load_edges
from data_gen.module4_orders import generate_orders, save_orders_csv, load_nodes
from data_gen.module5_riders import generate_riders, save_riders_csv

warnings.filterwarnings(
    "ignore",
    message=r".*env\.action_masks.*deprecated.*",
    category=UserWarning,
)

PROFILE_HOTSPOTS = False
_PROFILE_STATS: dict[str, dict[str, float]] = {}


def record_hotspot(name: str, elapsed_s: float) -> None:
    if not PROFILE_HOTSPOTS:
        return
    stat = _PROFILE_STATS.setdefault(
        name,
        {"calls": 0.0, "total_s": 0.0, "max_ms": 0.0},
    )
    stat["calls"] += 1.0
    stat["total_s"] += float(elapsed_s)
    stat["max_ms"] = max(float(stat["max_ms"]), float(elapsed_s) * 1e3)
    calls = int(stat["calls"])
    if calls <= 5 or (calls % 100 == 0):
        avg_ms = 1e3 * float(stat["total_s"]) / max(1, calls)
        print(
            "[profile]",
            f"fn={name}",
            f"calls={calls}",
            f"avg_ms={avg_ms:.3f}",
            f"max_ms={float(stat['max_ms']):.3f}",
            flush=True,
        )


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_config_sections(cfg: dict | None) -> tuple[dict, dict, dict, dict, dict, dict]:
    cfg = cfg or {}
    sim_cfg = cfg.get("simulation", {}) if isinstance(cfg, dict) else {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}
    travel_time_cfg = cfg.get("travel_time", {}) if isinstance(cfg, dict) else {}
    orders_cfg = cfg.get("orders", {}) if isinstance(cfg, dict) else {}
    riders_cfg = cfg.get("riders", {}) if isinstance(cfg, dict) else {}
    network_cfg = cfg.get("network", {}) if isinstance(cfg, dict) else {}
    return sim_cfg, out_cfg, travel_time_cfg, orders_cfg, riders_cfg, network_cfg


def _resolve_eval_config(cfg: dict | None) -> tuple[int, int, str]:
    cfg = cfg or {}
    sim_cfg, out_cfg, _travel_time_cfg, _orders_cfg, _riders_cfg, _network_cfg = _get_config_sections(cfg)
    duration = sim_cfg.get("duration", cfg.get("duration"))
    seed = sim_cfg.get("seed", cfg.get("seed"))
    data_dir = out_cfg.get("dir", cfg.get("output_dir", cfg.get("data_dir", "outputs")))
    if duration is None:
        raise ValueError("Missing simulation.duration in config")
    if seed is None:
        raise ValueError("Missing simulation.seed in config")
    return int(duration), int(seed), str(data_dir)


def _resolve_eval_files(cfg: dict | None, config_path: str | None = None) -> dict:
    cfg = cfg or {}
    sim_cfg, out_cfg, travel_time_cfg, orders_cfg, riders_cfg, network_cfg = _get_config_sections(cfg)
    duration = sim_cfg.get("duration", cfg.get("duration"))
    seed = sim_cfg.get("seed", cfg.get("seed"))
    if duration is None:
        raise ValueError("Missing simulation.duration in config")
    if seed is None:
        raise ValueError("Missing simulation.seed in config")

    base_dir = Path(out_cfg.get("dir", cfg.get("output_dir", cfg.get("data_dir", "outputs"))))
    config_stem = Path(config_path).stem if config_path else "default"
    data_dir = base_dir / config_stem

    orders_file = str(orders_cfg.get("filename", "orders.csv"))
    riders_file = str(riders_cfg.get("filename", "riders.csv"))
    nodes_file = str(network_cfg.get("nodes_filename", "nodes.csv"))
    travel_times_file = str(travel_time_cfg.get("filename", "travel_times.csv"))
    edges_file = str(network_cfg.get("edges_filename", "edges.csv"))

    synthetic_cfg = network_cfg.get("synthetic", {}) if isinstance(network_cfg, dict) else {}
    return {
        "data_dir": data_dir,
        "duration": int(duration),
        "seed": int(seed),
        "orders_file": orders_file,
        "riders_file": riders_file,
        "nodes_file": nodes_file,
        "travel_times_file": travel_times_file,
        "edges_file": edges_file,
        "n_orders": int(orders_cfg.get("n_orders", 200)),
        "n_riders": int(riders_cfg.get("n_riders", 30)),
        "network_width": synthetic_cfg.get("width"),
        "network_height": synthetic_cfg.get("height"),
    }


def _prepare_eval_dataset(cfg: dict | None, config_path: str | None = None) -> dict:
    files = _resolve_eval_files(cfg, config_path=config_path)
    data_dir = Path(files["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    net = build_network_from_config(cfg or {})
    save_network_csv(net, data_dir)
    if str(files["nodes_file"]) != "nodes.csv":
        shutil.copyfile(data_dir / "nodes.csv", data_dir / str(files["nodes_file"]))
    if str(files["edges_file"]) != "edges.csv":
        shutil.copyfile(data_dir / "edges.csv", data_dir / str(files["edges_file"]))

    edges = load_edges(data_dir / str(files["edges_file"]))
    rows = generate_travel_times(edges, int(files["duration"]), int((cfg or {}).get("travel_time", {}).get("time_step_min", 15)), cfg or {})
    save_travel_times_csv(rows, data_dir / str(files["travel_times_file"]))

    nodes = load_nodes(data_dir / str(files["nodes_file"]))
    orders = generate_orders(cfg or {}, nodes)
    save_orders_csv(orders, data_dir / str(files["orders_file"]))

    riders = generate_riders(cfg or {}, nodes)
    save_riders_csv(riders, data_dir / str(files["riders_file"]))

    files["nodes_path"] = str((data_dir / str(files["nodes_file"])).resolve())
    files["edges_path"] = str((data_dir / str(files["edges_file"])).resolve())
    files["travel_times_path"] = str((data_dir / str(files["travel_times_file"])).resolve())
    files["orders_path"] = str((data_dir / str(files["orders_file"])).resolve())
    files["riders_path"] = str((data_dir / str(files["riders_file"])).resolve())
    files["data_dir"] = str(data_dir.resolve())
    files["n_orders"] = int(len(orders))
    files["n_riders"] = int(len(riders))
    return files


def make_env(
    cfg: dict | None = None,
    env_files: dict | None = None,
    flat_compat: bool = True,
    node_feat_dim: int | None = None,
    debug: bool = False,
    progress_debug: bool = False,
    greedy_top_k_couriers: int | None = None,
    greedy_top_k_positions: int | None = None,
    use_action_masker: bool = False,
    alpha_eta_mu: float = 0.01,
    beta_eta_sigma: float = 0.01,
    data_dir: str | None = None,
    duration: int | None = None,
    profile_hotspots: bool = False,
):
    cfg = cfg or {}
    sim_cfg, out_cfg, _travel_time_cfg, _orders_cfg, _riders_cfg, _network_cfg = _get_config_sections(cfg)
    env_files = env_files or {}
    if data_dir is None:
        data_dir = env_files.get("data_dir", out_cfg.get("dir", cfg.get("output_dir", cfg.get("data_dir", "outputs"))))
    if duration is None:
        duration = env_files.get("duration", sim_cfg.get("duration", cfg.get("duration")))
    if duration is None:
        raise ValueError("Missing simulation.duration in config")
    orders_file = str(env_files.get("orders_file", "orders.csv"))
    riders_file = str(env_files.get("riders_file", "riders.csv"))
    nodes_file = str(env_files.get("nodes_file", "nodes.csv"))
    travel_times_file = str(env_files.get("travel_times_file", "travel_times.csv"))
    edges_file = str(env_files.get("edges_file", "edges.csv"))

    env = DispatchPPOEnv(
        data_dir=str(data_dir),
        duration=int(duration),
        orders_file=orders_file,
        riders_file=riders_file,
        nodes_file=nodes_file,
        travel_times_file=travel_times_file,
        edges_file=edges_file,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
        R=4,
        O=4,
        invalid_action_penalty=-0.02,
        wait_penalty=-0.01,
        debug=bool(debug),
        progress_debug=bool(progress_debug),
        greedy_top_k_couriers=greedy_top_k_couriers,
        greedy_top_k_positions=greedy_top_k_positions,
        profile_hotspots=bool(profile_hotspots),
        node_feat_dim=node_feat_dim,
        alpha_eta_mu=float(alpha_eta_mu),
        beta_eta_sigma=float(beta_eta_sigma),
    )
    print(
        "[eval][env-config]",
        f"duration={int(duration)}",
        f"orders_file={(Path(str(data_dir)) / orders_file).resolve()}",
        f"riders_file={(Path(str(data_dir)) / riders_file).resolve()}",
        f"nodes_file={(Path(str(data_dir)) / nodes_file).resolve()}",
        f"travel_times_file={(Path(str(data_dir)) / travel_times_file).resolve()}",
        f"n_orders={env_files.get('n_orders', 'NA')}",
        f"n_riders={env_files.get('n_riders', 'NA')}",
        f"network_width={env_files.get('network_width', 'NA')}",
        f"network_height={env_files.get('network_height', 'NA')}",
    )

    env = GymnasiumAPIWrapper(env)
    env = LastDictWrapper(env)

    if hasattr(env, "_eta_supervise_every"):
        env._eta_supervise_every = 1
        env._eta_supervise_step = 0

    if flat_compat:
        env = DictToFlatObs(env, key="flat")

    if hasattr(env, "_debug_force_no_fallback"):
        env._debug_force_no_fallback = True

    if use_action_masker:
        if ActionMasker is None:
            raise RuntimeError(
                "sb3-contrib is required to evaluate maskable models. "
                "Install it with: pip install sb3-contrib"
            )
        env = ActionMasker(env, lambda e: e.action_masks())

    return env

def to_int_action(a) -> int:
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    return int(a.item()) if a.size == 1 else int(a.reshape(-1)[0])


def to_env_action(a, action_space):
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)

    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        if a.ndim >= 2:
            a = a[0]
        a = a.reshape(-1).astype(np.int64, copy=False)
        if a.size < nvec.size:
            pad = np.zeros((nvec.size - a.size,), dtype=np.int64)
            a = np.concatenate([a, pad], axis=0)
        a = a[: nvec.size]
        a = np.clip(a, 0, nvec - 1)
        return a

    return to_int_action(a)

def load_any_model(path: str):
    maskable_policy_mismatch = None

    if MaskablePPO is not None:
        try:
            m = MaskablePPO.load(path)
            print("[eval] Loaded algo: MaskablePPO")
            return m, "maskable"
        except ValueError as e:
            msg = str(e)
            if "MaskableActorCriticPolicy" in msg:
                maskable_policy_mismatch = e
            else:
                raise
        except Exception:
            pass

    if JointMaskablePPO is not None:
        try:
            m = JointMaskablePPO.load(path)
            print("[eval] Loaded algo: JointMaskablePPO")
            return m, "maskable"
        except Exception:
            pass

    m = PPO.load(path)
    if maskable_policy_mismatch is not None:
        print("[eval] MaskablePPO rejected plain policy, fallback to PPO.load")
    print("[eval] Loaded algo: PPO")
    return m, "plain"
def as_eta_mat(x, R: int = 4, O: int = 4):
    """
    Accepts:
      - np array / torch tensor of shape [R,O], [1,R,O], [B,R,O], [R*O], [1,R*O], etc.
    Returns:
      - np.ndarray shape [R,O]
    """
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    # drop batch dim if exists
    if x.ndim == 3:
        x = x[0]
    if x.ndim == 2:
        return x

    # flatten to [R*O]
    x = x.reshape(-1)
    if x.size == R * O:
        return x.reshape(R, O)

    # last resort: can't parse
    return None

class GymnasiumAPIWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) >= 2:
            obs, info = out[0], out[1] if isinstance(out[1], dict) else {}
            return obs, info
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            return out
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
            return obs, reward, terminated, truncated, info
        raise RuntimeError(f"Unexpected step() return: {type(out)} {out}")


class LastDictWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_dict = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dict = obs if isinstance(obs, dict) else None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_dict = obs if isinstance(obs, dict) else None
        return obs, reward, terminated, truncated, info
      
class DictToFlatObs(gym.Wrapper):
    def __init__(self, env: gym.Env, key: str = "flat"):
        super().__init__(env)
        self.key = str(key)
        self.last_dict = None

        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert self.key in env.observation_space.spaces
        self.observation_space = env.observation_space.spaces[self.key]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dict = obs
        return obs[self.key], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_dict = obs
        return obs[self.key], reward, terminated, truncated, info

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _safe_nan_stats_impl(arr):
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x)
    if int(valid.sum()) == 0:
        return float("nan"), float("nan")
    y = x[valid]
    return float(y.mean()), float(y.std())


def to_numpy_float32(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)

def compute_eta_metrics(eta_pred, eta_sigma, eta_tgt, eta_mask) -> Tuple[float, float, float, bool]:
    if eta_pred is None or eta_tgt is None or eta_mask is None:
        return float("nan"), float("nan"), float("nan"), False

    yp = to_numpy_float32(eta_pred)
    ys = None if eta_sigma is None else to_numpy_float32(eta_sigma)
    yt = to_numpy_float32(eta_tgt)
    m = to_numpy_float32(eta_mask)
    if yp.ndim == 3:
        yp = yp[0]
    if ys is not None and ys.ndim == 3:
        ys = ys[0]
    if yt.ndim == 3:
        yt = yt[0]
    if m.ndim == 3:
        m = m[0]
    if ys is None:
        ys = np.ones_like(yp, dtype=np.float32)
    if yp.shape != yt.shape or m.shape != yt.shape or ys.shape != yt.shape:
        return float("nan"), float("nan"), float("nan"), True

    ys = np.maximum(ys, 1e-6)
    valid = (m > 0.5) & np.isfinite(yp) & np.isfinite(yt) & np.isfinite(ys)
    if not bool(np.any(valid)):
        return float("nan"), float("nan"), float("nan"), True

    diff = yp[valid] - yt[valid]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    var = ys[valid] * ys[valid]
    nll = 0.5 * ((diff * diff) / var + np.log(var))
    return mae, rmse, float(np.mean(nll)), True


def _to_numpy_if_tensor(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def force_eta_prediction_from_obs(model, obs_input):
    if model is None or obs_input is None:
        return None, None

    policy = getattr(model, "policy", None)
    fe = getattr(policy, "features_extractor", None)
    if policy is None or fe is None:
        return None, None

    try:
        with torch.no_grad():
            obs_tensor, _ = policy.obs_to_tensor(obs_input)
            policy.extract_features(obs_tensor)
        eta_pred = getattr(fe, "last_eta", None)
        eta_sigma = getattr(fe, "last_eta_sigma", None)
        print(
            "[force_eta_prediction_from_obs]",
            f"last_eta_exists={bool(eta_pred is not None)}",
            f"last_eta_shape={None if eta_pred is None else tuple(int(x) for x in eta_pred.shape)}",
            f"last_eta_sigma_exists={bool(eta_sigma is not None)}",
            f"last_eta_sigma_shape={None if eta_sigma is None else tuple(int(x) for x in eta_sigma.shape)}",
        )
        if eta_sigma is None:
            eta_log_var = getattr(fe, "last_eta_log_var", None)
            if eta_log_var is not None:
                eta_sigma = torch.exp(0.5 * eta_log_var).clamp(min=1e-6)
        return eta_pred, eta_sigma
    except Exception as e:
        print("[force_eta_prediction_from_obs][EXCEPTION]", repr(e))
        raise


def predict_eta_from_model(model, model_kind: str, obs, action_masks, deterministic: bool, full_obs=None, force_eta: bool = False):
    if model is None:
        return None, None, None

    action_pred = None
    if model_kind == "maskable":
        try:
            action_pred, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
            )
        except Exception:
            action_pred = None
    else:
        try:
            action_pred, _ = model.predict(obs, deterministic=deterministic)
        except Exception:
            action_pred = None

    fe = getattr(model.policy, "features_extractor", None)
    eta_pred = getattr(fe, "last_eta", None)
    eta_sigma = getattr(fe, "last_eta_sigma", None)
    if eta_sigma is None:
        eta_log_var = getattr(fe, "last_eta_log_var", None)
        if eta_log_var is not None:
            eta_sigma = torch.exp(0.5 * eta_log_var).clamp(min=1e-6)

    # Force a fresh ETA forward pass when requested, or if policy extractor didn't populate it.
    if force_eta or eta_pred is None:
        obs_space = getattr(model.policy, "observation_space", None)
        if isinstance(obs_space, gym.spaces.Dict):
            obs_input = full_obs if isinstance(full_obs, dict) else obs
        else:
            obs_input = obs

        eta_pred_forced, eta_sigma_forced = force_eta_prediction_from_obs(model, obs_input)
        if eta_pred_forced is not None:
            eta_pred = eta_pred_forced
        if eta_sigma is None and eta_sigma_forced is not None:
            eta_sigma = eta_sigma_forced
    return action_pred, eta_pred, eta_sigma


def reset_vecenv(venv, seed: int):
    # Compatible with older SB3 where VecEnv.reset() has no seed kwarg
    try:
        venv.seed(seed)
    except Exception:
        pass
    return venv.reset()
    
def inject_eta_pred_to_env(model, venv, eta_hint=None, eta_sigma_hint=None) -> None:
    fe = getattr(model.policy, "features_extractor", None)
    eta = eta_hint if eta_hint is not None else getattr(fe, "last_eta", None)
    eta_sigma = eta_sigma_hint if eta_sigma_hint is not None else getattr(fe, "last_eta_sigma", None)
    eta_log_var = getattr(fe, "last_eta_log_var", None)
    if eta_sigma is None:
        if eta_log_var is not None:
            eta_sigma = torch.exp(0.5 * eta_log_var).clamp(min=1e-6)
    if eta is None and eta_sigma is None:
        return

    eta_np = None if eta is None else eta.detach().cpu().numpy()
    eta_sigma_np = None if eta_sigma is None else eta_sigma.detach().cpu().numpy()
    if eta_sigma_np is not None:
        eta_sigma_np = np.maximum(np.nan_to_num(eta_sigma_np, nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6)

    base = venv
    for _ in range(10):
        if hasattr(base, "envs"):
            break
        if hasattr(base, "venv"):
            base = base.venv
        else:
            return

    for i, w in enumerate(base.envs):
        e = getattr(w, "unwrapped", w)
        if hasattr(e, "set_eta_pred"):
            if eta_np is not None:
                if eta_np.ndim == 3:
                    e.set_eta_pred(eta_np[i])
                else:
                    e.set_eta_pred(eta_np)
        if hasattr(e, "set_eta_sigma"):
            if eta_sigma_np is not None:
                if eta_sigma_np.ndim == 3:
                    e.set_eta_sigma(eta_sigma_np[i])
                else:
                    e.set_eta_sigma(eta_sigma_np)

def get_masks_if_available(venv):
    t0 = time.perf_counter() if PROFILE_HOTSPOTS else 0.0
    e = unwrap_env_for_attr(venv, "envs")
    if e is not None and len(e.envs) > 0:
        w = e.envs[0]
        if hasattr(w, "get_wrapper_attr"):
            try:
                fn = w.get_wrapper_attr("action_masks")
                if callable(fn):
                    out = np.asarray(fn(), dtype=bool)
                    record_hotspot("env.action_masks", time.perf_counter() - t0)
                    return out
            except Exception:
                pass
    try:
        from sb3_contrib.common.maskable.utils import get_action_masks
        out = get_action_masks(venv)
        record_hotspot("env.action_masks", time.perf_counter() - t0)
        return out
    except Exception:
        return None

def _make_wait_action(action_space, eta_shape: tuple[int, int] | None = None):
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        a = np.zeros((nvec.size,), dtype=np.int64)
        if nvec.size >= 1:
            wait_ri = int(nvec[0] - 1)
            if eta_shape is not None:
                wait_ri = min(wait_ri, int(eta_shape[0]))
            a[0] = max(0, wait_ri)
        if nvec.size >= 2:
            wait_oi = int(nvec[1] - 1)
            if eta_shape is not None:
                wait_oi = min(wait_oi, int(eta_shape[1]))
            a[1] = max(0, wait_oi)
        if nvec.size >= 3:
            a[2] = 0
        if nvec.size >= 4:
            a[3] = 1 if int(nvec[3]) > 1 else 0
        return a
    return 0


def greedy_action_from_eta(eta_mat: np.ndarray, action_masks: np.ndarray, O: int, action_space=None):
    """
    eta_mat: [R,O] or [1,R,O]
    action_masks: [n_actions] or [1,n_actions]
    """
    eta = to_numpy_float32(eta_mat)
    if eta.ndim == 3:
        eta = eta[0]  # -> [R,O]

    m = np.asarray(action_masks).astype(bool)
    if m.ndim == 2:
        m = m[0]

    if eta.ndim != 2:
        return _make_wait_action(action_space, None)
    rr, oo = int(eta.shape[0]), int(eta.shape[1])

    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        if nvec.size >= 2 and m.size >= int(np.sum(nvec)):
            n0 = int(nvec[0])
            n1 = int(nvec[1])
            s0 = 0
            s1 = s0 + n0
            rider_valid = [ri for ri in range(0, n0 - 1) if s0 + ri < m.size and bool(m[s0 + ri])]
            order_valid = [oi for oi in range(0, n1 - 1) if s1 + oi < m.size and bool(m[s1 + oi])]

            best = None
            best_eta = float("inf")
            for ri in rider_valid:
                for oi in order_valid:
                    if ri >= rr or oi >= oo:
                        continue
                    e = float(eta[ri, oi].item())
                    if e < best_eta:
                        best_eta = e
                        best = (ri, oi)

            if best is None:
                return _make_wait_action(action_space, (rr, oo))

            out = _make_wait_action(action_space, (rr, oo))
            out[0] = int(best[0])
            out[1] = int(best[1])
            return out

    valid = np.where(m)[0]
    valid_pairs = valid[valid > 0]
    if valid_pairs.size == 0:
        return _make_wait_action(action_space, (rr, oo))

    best_a = None
    best_eta = float("inf")
    col_o = max(1, int(O))
    for a in valid_pairs:
        idx = int(a) - 1
        ri = idx // col_o
        oi = idx % col_o
        if ri >= rr or oi >= oo:
            continue
        e = float(eta[ri, oi].item())
        if e < best_eta:
            best_eta = e
            best_a = int(a)

    if best_a is None:
        return _make_wait_action(action_space, (rr, oo))
    return best_a


def enumerate_valid_actions(action_space, action_masks):
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        per_dim = decode_mask(action_masks, nvec)
        if len(per_dim) != len(nvec):
            return []
        out = []
        for ri in per_dim[0]:
            for oi in per_dim[1]:
                for pi in per_dim[2]:
                    for di in per_dim[3]:
                        act = np.asarray([ri, oi, pi, di], dtype=np.int64)
                        if _action_is_valid_by_mask(action_space, action_masks, act):
                            out.append(act)
        return out

    m = _mask_1d(action_masks)
    if m is None:
        return []
    return [int(i) for i in np.where(m)[0]]


def sample_random_valid_action(action_space, action_masks):
    candidates = enumerate_valid_actions(action_space, action_masks)
    if not candidates:
        return None
    idx = int(np.random.randint(0, len(candidates)))
    act = candidates[idx]
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        return np.asarray(act, dtype=np.int64)
    return int(act)


def choose_risk_aware_action(
    venv,
    action_space,
    action_masks,
    eta_pred,
    eta_sigma,
    alpha_eta_mu: float,
    beta_eta_sigma: float,
):
    t0 = time.perf_counter() if PROFILE_HOTSPOTS else 0.0
    candidates = enumerate_valid_actions(action_space, action_masks)
    if not candidates:
        record_hotspot("choose_risk_aware_action", time.perf_counter() - t0)
        return None, {"reason": "no_valid_actions", "num_candidates": 0}

    eta_mu_mat = as_eta_mat(eta_pred)
    eta_sigma_mat = as_eta_mat(eta_sigma)
    if eta_sigma_mat is not None:
        eta_sigma_mat = np.maximum(to_numpy_float32(eta_sigma_mat), 1e-6)

    best_action = None
    best_score = -float("inf")
    best_diag = {
        "reason": "no_dispatch_candidate",
        "num_candidates": int(len(candidates)),
        "num_dispatch_candidates": 0,
    }
    dispatch_candidates = 0

    for act in candidates:
        breakdown = get_action_breakdown_from_env(venv, act)
        if not isinstance(breakdown, dict):
            continue
        if not bool(breakdown.get("dispatched", 0.0) > 0.5):
            continue
        if bool(breakdown.get("invalid", 0.0) > 0.5):
            continue

        a = np.asarray(act, dtype=np.int64).reshape(-1)
        ri = int(a[0])
        oi = int(a[1])
        if eta_mu_mat is None or eta_sigma_mat is None:
            continue
        if (
            eta_mu_mat.ndim != 2
            or eta_sigma_mat.ndim != 2
            or ri < 0
            or oi < 0
            or ri >= eta_mu_mat.shape[0]
            or oi >= eta_mu_mat.shape[1]
            or ri >= eta_sigma_mat.shape[0]
            or oi >= eta_sigma_mat.shape[1]
        ):
            continue

        eta_mu_selected = float(eta_mu_mat[ri, oi])
        eta_sigma_selected = float(max(1e-6, eta_sigma_mat[ri, oi]))
        if not (np.isfinite(eta_mu_selected) and np.isfinite(eta_sigma_selected)):
            continue

        dispatch_candidates += 1
        dispatch_score = (
            float(breakdown.get("dispatch", 0.0))
            + float(breakdown.get("route_delta", 0.0))
            + float(breakdown.get("dispatch_extra", 0.0))
            + float(breakdown.get("repeat", 0.0))
            + float(breakdown.get("invalid_pen", 0.0))
            + float(breakdown.get("wait_action_pen", 0.0))
            + float(breakdown.get("backlog_pen", 0.0))
        )
        score = (
            dispatch_score
            - float(alpha_eta_mu) * eta_mu_selected
            - float(beta_eta_sigma) * eta_sigma_selected
        )
        if score > best_score:
            best_score = float(score)
            best_action = np.asarray(act, dtype=np.int64)
            best_diag = {
                "reason": "ok",
                "num_candidates": int(len(candidates)),
                "dispatch_score": float(dispatch_score),
                "score": float(score),
                "ri": int(ri),
                "oi": int(oi),
                "pickup_idx": int(a[2]) if a.size > 2 else None,
                "dropoff_idx": int(a[3]) if a.size > 3 else None,
                "eta_mu_selected": float(eta_mu_selected),
                "eta_sigma_selected": float(eta_sigma_selected),
            }

    if best_action is None:
        best_diag["num_dispatch_candidates"] = int(dispatch_candidates)
        record_hotspot("choose_risk_aware_action", time.perf_counter() - t0)
        return None, best_diag
    best_diag["num_dispatch_candidates"] = int(dispatch_candidates)
    record_hotspot("choose_risk_aware_action", time.perf_counter() - t0)
    return best_action, best_diag

def _get_env0(venv):
    base = getattr(venv, "venv", venv)
    return base.envs[0]

def unwrap_env_for_attr(venv, attr: str):
    e = venv
    for _ in range(20):
        if hasattr(e, attr):
            return e
        if hasattr(e, "venv"):
            e = e.venv
            continue
        break
    return None



def get_full_obs(venv):
    e = unwrap_env_for_attr(venv, "envs")
    if e is None:
        return None
    w = e.envs[0]
    if hasattr(w, "get_wrapper_attr"):
        try:
            return w.get_wrapper_attr("last_dict")
        except Exception:
            pass
    return getattr(w, "last_dict", None)


def get_greedy_action_from_env(venv):
    e = unwrap_env_for_attr(venv, "envs")
    if e is None:
        raise RuntimeError("Cannot access base envs for greedy action.")
    w = e.envs[0]
    if hasattr(w, "get_wrapper_attr"):
        try:
            fn = w.get_wrapper_attr("greedy_insertion_action")
            if callable(fn):
                return fn()
        except Exception:
            pass
    env0 = getattr(w, "unwrapped", w)
    if not hasattr(env0, "greedy_insertion_action"):
        raise RuntimeError("Env does not implement greedy_insertion_action().")
    return env0.greedy_insertion_action()

def get_action_breakdown_from_env(venv, action):
    e = unwrap_env_for_attr(venv, "envs")
    if e is None:
        return None
    w = e.envs[0]
    if hasattr(w, "get_wrapper_attr"):
        try:
            fn = w.get_wrapper_attr("action_reward_breakdown")
            if callable(fn):
                return fn(action)
        except Exception:
            pass
    env0 = getattr(w, "unwrapped", w)
    fn = getattr(env0, "action_reward_breakdown", None)
    if callable(fn):
        try:
            return fn(action)
        except Exception:
            return None
    return None

def get_wait_reason_from_env(venv) -> str:
    e = unwrap_env_for_attr(venv, "envs")
    if e is None:
        return "UNKNOWN"
    w = e.envs[0]
    if hasattr(w, "get_wrapper_attr"):
        try:
            fn = w.get_wrapper_attr("get_wait_allowed_reason")
            if callable(fn):
                return str(fn())
        except Exception:
            pass
    env0 = getattr(w, "unwrapped", w)
    fn = getattr(env0, "get_wait_allowed_reason", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            return "UNKNOWN"
    return "UNKNOWN"


def _mask_1d(action_masks):
    if action_masks is None:
        return None
    m = np.asarray(action_masks).astype(bool)
    if m.ndim == 2:
        return m[0]
    return m.reshape(-1)

def decode_mask(mask: np.ndarray, nvec: np.ndarray) -> List[np.ndarray]:
    m = _mask_1d(mask)
    if m is None:
        return []
    out = []
    off = 0
    for n in np.asarray(nvec, dtype=np.int64):
        seg = m[off:off + int(n)]
        out.append(np.where(seg)[0].astype(np.int64))
        off += int(n)
    return out


def _action_is_valid_by_mask(action_space, action_masks, action) -> bool:
    m = _mask_1d(action_masks)
    if m is None:
        return True
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size < nvec.size:
            pad = np.zeros((nvec.size - a.size,), dtype=np.int64)
            a = np.concatenate([a, pad], axis=0)
        a = a[: nvec.size]
        if np.any(a < 0) or np.any(a >= nvec):
            return False
        offset = 0
        for i, n in enumerate(nvec):
            if offset + int(a[i]) >= m.size:
                return False
            if not bool(m[offset + int(a[i])]):
                return False
            offset += int(n)
        return True
    ai = int(np.asarray(action).reshape(-1)[0])
    return bool(0 <= ai < m.size and m[ai])


def _has_dispatch_decision(action_space, action_masks) -> bool:
    m = _mask_1d(action_masks)
    if m is None:
        return True
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        if nvec.size < 2:
            return False
        n0 = int(nvec[0])
        n1 = int(nvec[1])
        s0 = 0
        s1 = s0 + n0
        rider_ok = bool(np.any(m[s0:s0 + max(0, n0 - 1)]))
        order_ok = bool(np.any(m[s1:s1 + max(0, n1 - 1)]))
        return rider_ok and order_ok
    if m.size <= 1:
        return False
    return bool(np.any(m[1:]))


def _decode_action_for_trace(venv, action):
    e = unwrap_env_for_attr(venv, "envs")
    if e is None or len(e.envs) == 0:
        return {"raw": np.asarray(action).tolist()}
    w = e.envs[0]
    env0 = getattr(w, "unwrapped", w)
    if hasattr(env0, "decode_action"):
        try:
            return env0.decode_action(action)
        except Exception:
            pass
    return {"raw": np.asarray(action).tolist()}


def _inspect_action_for_trace(venv, action, require_eta: bool = True):
    e = unwrap_env_for_attr(venv, "envs")
    if e is None or len(e.envs) == 0:
        return {"action": np.asarray(action).tolist()}
    w = e.envs[0]
    env0 = getattr(w, "unwrapped", w)
    if hasattr(env0, "inspect_action"):
        try:
            return env0.inspect_action(action, require_eta=require_eta)
        except Exception:
            pass
    decoded = _decode_action_for_trace(venv, action)
    return {
        "action": np.asarray(action).tolist(),
        "dispatch_valid": False,
        "action_in_range": False,
        "fallback_missing_env_helper": True,
        "decoded": decoded,
    }

def _is_wait_from_decoded(decoded) -> bool:
    if isinstance(decoded, dict) and ("is_wait" in decoded):
        return bool(decoded.get("is_wait", False))
    return False


def _trace_env_state(venv):
    e = unwrap_env_for_attr(venv, "envs")
    if e is None or len(e.envs) == 0:
        return None, None
    w = e.envs[0]
    env0 = getattr(w, "unwrapped", w)
    base = getattr(env0, "base", None)
    if base is not None:
        return int(getattr(base, "t", 0)), int(len(getattr(base, "pending_orders", [])))
    return int(getattr(env0, "t", 0)), int(len(getattr(env0, "pending_orders", [])))


def print_trace_action_space_info(venv):
    base = getattr(venv, "venv", venv)
    action_space = base.action_space
    n = getattr(action_space, "n", None)
    print(f"[trace] action_space={action_space} n={n}")
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(action_space.nvec, dtype=np.int64)
        print(f"[trace] action_space.nvec={nvec.tolist()} total={int(np.prod(nvec))}")
        e = unwrap_env_for_attr(venv, "envs")
        if e is not None and len(e.envs) > 0:
            w = e.envs[0]
            env0 = getattr(w, "unwrapped", w)
            if hasattr(env0, "decode_action"):
                ex0 = np.asarray([0, 0, 0, 1], dtype=np.int64)
                ex1 = np.asarray([int(nvec[0] - 1), int(nvec[1] - 1), 0, 1], dtype=np.int64)
                ex2 = np.asarray([min(1, max(0, int(nvec[0] - 2))), min(1, max(0, int(nvec[1] - 2))), 0, 1], dtype=np.int64)
                print(f"[trace] decode ex0={env0.decode_action(ex0)}")
                print(f"[trace] decode ex1={env0.decode_action(ex1)}")
                print(f"[trace] decode ex2={env0.decode_action(ex2)}")


def resolve_eta_pred(venv, model, full_obs, eta_pred_hint=None):
    if eta_pred_hint is not None:
        return eta_pred_hint

    e = unwrap_env_for_attr(venv, "envs")
    if e is not None and len(e.envs) > 0:
        w = e.envs[0]
        env0 = getattr(w, "unwrapped", w)

        if hasattr(env0, "get_eta_pred"):
            try:
                v = env0.get_eta_pred()
                if v is not None:
                    return to_numpy_float32(v)
            except Exception:
                pass

        for attr in ("_last_eta_pred", "_last_eta"):
            v = getattr(env0, attr, None)
            has_pair = bool(getattr(env0, "_last_has_pair", True))
            if v is not None and has_pair:
                return to_numpy_float32(v)

    return None

def resolve_eta_sigma(venv, model, full_obs, eta_sigma_hint=None):
    if eta_sigma_hint is not None:
        return eta_sigma_hint

    if model is not None:
        fe = getattr(model.policy, "features_extractor", None)
        s = getattr(fe, "last_eta_sigma", None)
        if s is not None:
            return to_numpy_float32(s)

        lv = getattr(fe, "last_eta_log_var", None)
        if lv is not None:
            return np.exp(0.5 * to_numpy_float32(lv))

    e = unwrap_env_for_attr(venv, "envs")
    if e is not None and len(e.envs) > 0:
        w = e.envs[0]
        env0 = getattr(w, "unwrapped", w)
        sigma_cache = getattr(env0, "_eta_sigma_cache", None)
        if sigma_cache is not None:
            return to_numpy_float32(sigma_cache)

    if isinstance(full_obs, dict):
        eta_tgt = full_obs.get("eta_tgt")
        if eta_tgt is not None:
            return np.ones_like(to_numpy_float32(eta_tgt), dtype=np.float32)
    return None


def eval_eta_for_assignment(venv, model, full_obs, eta_pred_hint=None, eta_sigma_hint=None):
    if eta_pred_hint is None and model is None:
        eta_eval_pred = None
        eta_eval_sigma = None
    else:
        eta_eval_pred = resolve_eta_pred(venv, model, full_obs, eta_pred_hint=eta_pred_hint)
        eta_eval_sigma = resolve_eta_sigma(venv, model, full_obs, eta_sigma_hint=eta_sigma_hint)

    if not isinstance(full_obs, dict):
        return float("nan"), float("nan"), float("nan"), False, {}

    eta_tgt = full_obs.get("eta_tgt")
    eta_mask = full_obs.get("eta_mask")
    mae_i, rmse_i, nll_i, enabled_i = compute_eta_metrics(
        eta_eval_pred, eta_eval_sigma, eta_tgt, eta_mask
    )

    diag = {}
    try:
        yp = None if eta_eval_pred is None else to_numpy_float32(eta_eval_pred).reshape(-1)
        ys = None if eta_eval_sigma is None else to_numpy_float32(eta_eval_sigma).reshape(-1)
        yt = None if eta_tgt is None else to_numpy_float32(eta_tgt).reshape(-1)
        m = None if eta_mask is None else to_numpy_float32(eta_mask).reshape(-1)
        if yp is not None and yt is not None and m is not None:
            n = min(yp.size, yt.size, m.size)
            valid = (m[:n] > 0.5) & np.isfinite(yp[:n]) & np.isfinite(yt[:n])
            idx = np.where(valid)[0]
            if idx.size > 0:
                take = idx[:3]
                diag["pred_sample"] = [float(x) for x in yp[take]]
                diag["gt_sample"] = [float(x) for x in yt[take]]
                if ys is not None and ys.size >= n:
                    diag["sigma_sample"] = [float(x) for x in ys[take]]
                else:
                    diag["sigma_sample"] = []
                diag["valid_count"] = int(idx.size)
    except Exception:
        diag = {}

    return mae_i, rmse_i, nll_i, enabled_i, diag


def resolve_greedy_eta_targets(venv, full_obs):
    if isinstance(full_obs, dict):
        eta_tgt = full_obs.get("eta_tgt")
        eta_mask = full_obs.get("eta_mask")
        if eta_tgt is not None and eta_mask is not None:
            return eta_tgt, eta_mask, "full_obs['eta_tgt']/full_obs['eta_mask']"

    e = unwrap_env_for_attr(venv, "envs")
    if e is not None and len(e.envs) > 0:
        w = e.envs[0]
        env0 = getattr(w, "unwrapped", w)
        if hasattr(env0, "_eta_targets"):
            try:
                eta_tgt, eta_mask = env0._eta_targets()
                if eta_tgt is not None and eta_mask is not None:
                    return eta_tgt, eta_mask, "env._eta_targets()"
            except Exception:
                pass

    return None, None, "missing"


def eval_eta_for_greedy_action(
    venv,
    model,
    action_space,
    env_action,
    chosen_decoded,
    full_obs,
    eta_pred_hint=None,
    eta_sigma_hint=None,
):
    pred_source = "model.policy.features_extractor.last_eta"
    sigma_source = "model.policy.features_extractor.last_eta_sigma/last_eta_log_var"
    gt_source = "full_obs['eta_tgt']"

    if not isinstance(full_obs, dict):
        return float("nan"), float("nan"), float("nan"), False, {}

    if not isinstance(action_space, gym.spaces.MultiDiscrete):
        return float("nan"), float("nan"), float("nan"), False, {
            "pred_source": pred_source,
            "sigma_source": sigma_source,
            "gt_source": gt_source,
            "reason": "unsupported_action_space",
        }

    a = np.asarray(env_action, dtype=np.int64).reshape(-1)
    if a.size < 2:
        return float("nan"), float("nan"), float("nan"), False, {
            "pred_source": pred_source,
            "sigma_source": sigma_source,
            "gt_source": gt_source,
            "reason": "bad_action_shape",
        }

    action_ri = int(a[0])
    action_oi = int(a[1])
    nvec = np.asarray(action_space.nvec, dtype=np.int64)
    wait_ri = int(nvec[0] - 1)
    wait_oi = int(nvec[1] - 1)
    if action_ri == wait_ri or action_oi == wait_oi:
        return float("nan"), float("nan"), float("nan"), False, {
            "pred_source": pred_source,
            "sigma_source": sigma_source,
            "gt_source": gt_source,
            "reason": "wait_action",
        }

    eta_pred = eta_pred_hint
    if eta_pred is None and model is not None:
        fe = getattr(model.policy, "features_extractor", None)
        eta_pred = getattr(fe, "last_eta", None)
    eta_sigma = resolve_eta_sigma(venv, model, full_obs, eta_sigma_hint=eta_sigma_hint)
    eta_tgt, eta_mask, gt_source = resolve_greedy_eta_targets(venv, full_obs)

    if eta_pred is None or eta_sigma is None or eta_tgt is None or eta_mask is None:
        return float("nan"), float("nan"), float("nan"), False, {
            "pred_source": pred_source,
            "sigma_source": sigma_source,
            "gt_source": gt_source,
            "reason": "missing_eta_inputs",
            "pred_present": bool(eta_pred is not None),
            "sigma_present": bool(eta_sigma is not None),
            "gt_present": bool(eta_tgt is not None),
            "mask_present": bool(eta_mask is not None),
        }

    ri = int(action_ri)
    oi = int(action_oi)
    coord_source = "local_action_indices"
    if ri < 0 or oi < 0:
        return float("nan"), float("nan"), float("nan"), False, {
            "pred_source": pred_source,
            "sigma_source": sigma_source,
            "gt_source": gt_source,
            "coord_source": coord_source,
            "action_ri": action_ri,
            "action_oi": action_oi,
            "reason": "missing_eta_coords",
        }

    yp = to_numpy_float32(eta_pred)
    ys = to_numpy_float32(eta_sigma)
    yt = to_numpy_float32(eta_tgt)
    m = to_numpy_float32(eta_mask)
    if yp.ndim == 3:
        yp = yp[0]
    if ys.ndim == 3:
        ys = ys[0]
    if yt.ndim == 3:
        yt = yt[0]
    if m.ndim == 3:
        m = m[0]

    diag = {
        "pred_source": pred_source,
        "sigma_source": sigma_source,
        "gt_source": gt_source,
        "action_ri": action_ri,
        "action_oi": action_oi,
        "selected_ri": ri,
        "selected_oi": oi,
        "selected_rider_id": None if not isinstance(chosen_decoded, dict) else chosen_decoded.get("rider_id"),
        "selected_order_id": None if not isinstance(chosen_decoded, dict) else chosen_decoded.get("order_id"),
        "coord_source": coord_source,
        "pred_shape": tuple(int(x) for x in yp.shape),
        "sigma_shape": tuple(int(x) for x in ys.shape),
        "gt_shape": tuple(int(x) for x in yt.shape),
        "mask_shape": tuple(int(x) for x in m.shape),
    }

    if (
        yp.ndim != 2 or ys.ndim != 2 or yt.ndim != 2 or m.ndim != 2
        or ri < 0 or oi < 0
        or ri >= yp.shape[0] or oi >= yp.shape[1]
        or ri >= ys.shape[0] or oi >= ys.shape[1]
        or ri >= yt.shape[0] or oi >= yt.shape[1]
        or ri >= m.shape[0] or oi >= m.shape[1]
    ):
        diag["reason"] = "shape_or_index_mismatch"
        diag["pred_shape"] = tuple(int(x) for x in yp.shape)
        diag["sigma_shape"] = tuple(int(x) for x in ys.shape)
        diag["gt_shape"] = tuple(int(x) for x in yt.shape)
        diag["mask_shape"] = tuple(int(x) for x in m.shape)
        return float("nan"), float("nan"), float("nan"), False, diag

    if not bool(m[ri, oi] > 0.5):
        diag["reason"] = "masked_selected_pair"
        diag["mask_value"] = float(m[ri, oi])
        return float("nan"), float("nan"), float("nan"), False, diag

    pred = float(yp[ri, oi])
    sigma = float(ys[ri, oi])
    gt = float(yt[ri, oi])
    diag["mask_value"] = float(m[ri, oi])

    diag["pred_sample"] = [pred]
    diag["gt_sample"] = [gt]
    diag["sigma_sample"] = [sigma]
    diag["valid_count"] = 1
    diag["pred_eq_gt"] = bool(np.allclose([pred], [gt], rtol=1e-6, atol=1e-6))
    diag["collected"] = True

    if not (np.isfinite(pred) and np.isfinite(gt) and np.isfinite(sigma)) or sigma <= 0.0:
        diag["reason"] = "non_finite_or_non_positive_sigma"
        return float("nan"), float("nan"), float("nan"), False, diag

    diff = pred - gt
    mae = float(abs(diff))
    rmse = float(abs(diff))
    var = sigma * sigma
    nll = float(0.5 * ((diff * diff) / var + np.log(var)))
    return mae, rmse, nll, True, diag

def run_one(
    model,
    model_kind: str,
    venv,
    seed: int,
    deterministic: bool,
    no_inject: bool,
    force_policy: bool,
    alpha_eta_mu: float = 0.01,
    beta_eta_sigma: float = 0.01,
    progress_debug: bool = False,
    max_steps: int = 5000,
    max_sim_time: int | None = None,
    greedy_only: bool = False,
    random_policy: bool = False,
    trace: bool = False,
) -> Tuple[float, float, float, float, bool]:
    obs = reset_vecenv(venv, seed)
    debug_eval = False

    done = [False]
    ep_reward = 0.0

    eta_maes, eta_rmses, eta_nlls = [], [], []
    eta_metrics_enabled = False
    eta_zero_diag = None
    greedy_pred_eq_gt_flags = []
    greedy_pred_eq_gt_diag = None
    greedy_eta_debug_rows = []
    step_cnt = 0
    trace_lines = 0
    R, O = 4, 4
    last_progress_line = None

    if progress_debug:
        env0 = _get_env0(venv)
        envu = getattr(env0, "unwrapped", env0)
        base0 = getattr(envu, "base", None)
        num_couriers = int(len(getattr(base0, "riders", []))) if base0 is not None else -1
        num_orders = int(len(getattr(base0, "orders", []))) if base0 is not None else -1
        print(
            f"[progress] episode_start seed={int(seed)} couriers={num_couriers} generated_orders={num_orders}",
            flush=True,
        )

    while not done[0]:
        action_masks = get_masks_if_available(venv) if (model_kind == "maskable" or random_policy) else None
        base = getattr(venv, "venv", venv)
        action_space = base.action_space
        decision_step = _has_dispatch_decision(action_space, action_masks)
        risk_diag = None

        obs_flat = obs
        full_obs = get_full_obs(venv)

        action_pi = None
        eta_pred = None
        eta_sigma = None
        if model is not None:
            action_pi, eta_pred, eta_sigma = predict_eta_from_model(
                model=model,
                model_kind=model_kind,
                obs=obs_flat,
                action_masks=action_masks,
                deterministic=deterministic,
                full_obs=full_obs,
                force_eta=bool(greedy_only or random_policy),
            )
        if debug_eval and step_cnt < 5:
            if isinstance(full_obs, dict):
                print("[dbg full_obs keys]", sorted(list(full_obs.keys()))[:50])
                em = full_obs.get("eta_mask", None)
                et = full_obs.get("eta_tgt", None)
                print("[dbg eta_tgt/mask is None?]", et is None, em is None)
                if em is not None:
                    em_np = np.asarray(em)
                    print("[dbg eta_mask sum]", float(em_np.sum()), "shape", em_np.shape, "min/max", float(em_np.min()), float(em_np.max()))
                if et is not None:
                    et_np = np.asarray(et)
                    print("[dbg eta_tgt shape]", et_np.shape, "nan%", float(np.isnan(et_np).mean()) if np.issubdtype(et_np.dtype, np.floating) else "NA")
            else:
                print("[dbg full_obs type]", type(full_obs))
        if greedy_only:
            eta_pred_np = _to_numpy_if_tensor(eta_pred)
            eta_sigma_np = _to_numpy_if_tensor(eta_sigma)
            if debug_eval:
                print(
                    "[eval][greedy][eta-mu-debug]",
                    f"pred_present={bool(eta_pred is not None)}",
                    f"pred_shape={None if eta_pred_np is None else tuple(int(x) for x in np.asarray(eta_pred_np).shape)}",
                    f"sigma_present={bool(eta_sigma is not None)}",
                    f"sigma_shape={None if eta_sigma_np is None else tuple(int(x) for x in np.asarray(eta_sigma_np).shape)}",
                )

        if greedy_only:
            if progress_debug:
                print("[progress] before greedy action generation", flush=True)
            action = get_greedy_action_from_env(venv)
            if progress_debug:
                print(f"[progress] after greedy action generation action={np.asarray(action).tolist()}", flush=True)
        elif random_policy:
            action = None
            if action_masks is not None:
                action = sample_random_valid_action(action_space, action_masks)
            if action is None:
                try:
                    action = get_greedy_action_from_env(venv)
                except Exception:
                    action = None
            if action is None and isinstance(full_obs, dict) and action_masks is not None:
                eta_fb = as_eta_mat(full_obs.get("eta_tgt"), R=R, O=O)
                if eta_fb is not None:
                    action = greedy_action_from_eta(eta_fb, action_masks, O=O, action_space=action_space)
            if action is None:
                action = _make_wait_action(action_space)
        else:
            if not no_inject:
                inject_eta_pred_to_env(model, venv, eta_hint=eta_pred, eta_sigma_hint=eta_sigma)
            action = action_pi
            if (not force_policy) and decision_step and action_masks is not None:
                risk_action, risk_diag = choose_risk_aware_action(
                    venv=venv,
                    action_space=action_space,
                    action_masks=action_masks,
                    eta_pred=eta_pred,
                    eta_sigma=eta_sigma,
                    alpha_eta_mu=float(alpha_eta_mu),
                    beta_eta_sigma=float(beta_eta_sigma),
                )
                if risk_action is not None:
                    action = risk_action

            if action is None and isinstance(full_obs, dict) and action_masks is not None:
                eta_fb = as_eta_mat(full_obs.get("eta_tgt"), R=R, O=O)
                if eta_fb is not None:
                    action = greedy_action_from_eta(eta_fb, action_masks, O=O, action_space=action_space)

            if action is None:
                action = _make_wait_action(action_space)

        chosen_action_raw = to_env_action(action, action_space)
        greedy_action = None
        try:
            greedy_action = get_greedy_action_from_env(venv)
        except Exception:
            greedy_action = None
        chosen_diag = _inspect_action_for_trace(venv, chosen_action_raw, require_eta=True)
        greedy_diag = _inspect_action_for_trace(venv, greedy_action, require_eta=True) if greedy_action is not None else None
        chosen_action_in_range = bool(chosen_diag.get("action_in_range", False))
        greedy_action_in_range = False if greedy_diag is None else bool(greedy_diag.get("action_in_range", False))
        chosen_action_allowed_by_mask = _action_is_valid_by_mask(action_space, action_masks, chosen_action_raw)
        greedy_action_allowed_by_mask = _action_is_valid_by_mask(action_space, action_masks, greedy_action) if greedy_action is not None else False
        fallback_triggered = False
        fallback_reason = None
        fallback_source = None
        env_action = chosen_action_raw
        executed_diag = chosen_diag

        if not greedy_only:
            needs_fallback = bool((not chosen_action_allowed_by_mask) or (not bool(chosen_diag.get("dispatch_valid", False))))
            if needs_fallback:
                repaired = None
                repaired_source = None
                if greedy_action is not None and bool(greedy_diag.get("dispatch_valid", False)):
                    repaired = greedy_action
                    repaired_source = "greedy_insertion"
                if repaired is None and isinstance(full_obs, dict):
                    eta_fb = as_eta_mat(full_obs.get("eta_tgt"), R=R, O=O)
                    if eta_fb is not None:
                        eta_repaired = greedy_action_from_eta(eta_fb, action_masks, O=O, action_space=action_space)
                        eta_repaired_diag = _inspect_action_for_trace(venv, eta_repaired, require_eta=True)
                        if bool(eta_repaired_diag.get("dispatch_valid", False)):
                            repaired = eta_repaired
                            repaired_source = "eta_greedy"
                if repaired is None:
                    repaired = _make_wait_action(action_space)
                    repaired_source = "wait_fallback"
                env_action = to_env_action(repaired, action_space)
                executed_diag = _inspect_action_for_trace(venv, env_action, require_eta=True)
                fallback_triggered = True
                if not chosen_action_allowed_by_mask:
                    fallback_reason = "mask_invalid_or_out_of_range"
                else:
                    fallback_reason = str(chosen_diag.get("failure_reason", "invalid_dispatch"))
                fallback_source = repaired_source
                if trace or debug_eval:
                    print(
                        "[eval][fallback]",
                        f"t={_trace_env_state(venv)[0]}",
                        f"raw_chosen={np.asarray(chosen_action_raw).tolist()}",
                        f"action_in_range={chosen_action_in_range}",
                        f"action_allowed_by_mask={chosen_action_allowed_by_mask}",
                        f"dispatch_valid={bool(chosen_diag.get('dispatch_valid', False))}",
                        f"failure_reason={fallback_reason}",
                        f"fallback_source={fallback_source}",
                        f"fallback_action={np.asarray(env_action).tolist()}",
                    )

        chosen_decoded = _decode_action_for_trace(venv, chosen_action_raw)
        greedy_decoded = _decode_action_for_trace(venv, greedy_action) if greedy_action is not None else None
        executed_decoded = _decode_action_for_trace(venv, env_action)
        chosen_is_wait = _is_wait_from_decoded(chosen_decoded)
        greedy_is_wait = _is_wait_from_decoded(greedy_decoded)
        chosen_breakdown_pre = get_action_breakdown_from_env(venv, chosen_action_raw)
        executed_breakdown_pre = get_action_breakdown_from_env(venv, env_action)
        greedy_breakdown_pre = get_action_breakdown_from_env(venv, greedy_action) if greedy_action is not None else None
        wait_allowed_reason = get_wait_reason_from_env(venv)

        mae_i = rmse_i = nll_i = float("nan")
        enabled_i = False
        diag_i = {}
       

        if progress_debug:
            print(f"[progress] before venv.step step_cnt={step_cnt} action={np.asarray(env_action).tolist()}", flush=True)
        step_out = venv.step([env_action])
        if progress_debug:
            print(f"[progress] after venv.step step_cnt={step_cnt}", flush=True)
        if isinstance(step_out, tuple) and len(step_out) == 4:
            obs, reward, done, info = step_out
        elif isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = np.logical_or(terminated, truncated)
        else:
            raise RuntimeError(f"Unexpected venv.step return: {type(step_out)} {step_out}")

        if greedy_only:
            full_obs = get_full_obs(venv)
            mae_i, rmse_i, nll_i, enabled_i, diag_i = eval_eta_for_greedy_action(
                venv=venv,
                model=model,
                action_space=action_space,
                env_action=env_action,
                chosen_decoded=chosen_decoded,
                full_obs=full_obs,
                eta_pred_hint=eta_pred,
                eta_sigma_hint=eta_sigma,
            )

        step_reward = float(reward[0])
        ep_reward += step_reward
        step_cnt += 1
        invalid_flag = False
        info0 = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else info
        try:
            invalid_flag = bool(info0.get("invalid", False))
        except Exception:
            invalid_flag = False
        wait_action_penalty = float(info0.get("reward_wait_action", 0.0)) if isinstance(info0, dict) else 0.0
        wait_backlog_penalty = float(info0.get("reward_wait_backlog", 0.0)) if isinstance(info0, dict) else 0.0
        chosen_components = info0.get("reward_components", {}) if isinstance(info0, dict) else {}
        chosen_components_total = float(info0.get("reward_total", step_reward)) if isinstance(info0, dict) else float(step_reward)
        greedy_components_total = float(greedy_breakdown_pre.get("total", np.nan)) if isinstance(greedy_breakdown_pre, dict) else float("nan")
        chosen_eta_mu_selected = float(chosen_components.get("eta_mu_selected", np.nan)) if isinstance(chosen_components, dict) else float("nan")
        chosen_eta_sigma_selected = float(chosen_components.get("eta_sigma_selected", np.nan)) if isinstance(chosen_components, dict) else float("nan")
        chosen_eta_mu_penalty = float(chosen_components.get("eta_mu_penalty", np.nan)) if isinstance(chosen_components, dict) else float("nan")
        chosen_eta_sigma_penalty = float(chosen_components.get("eta_sigma_penalty", np.nan)) if isinstance(chosen_components, dict) else float("nan")
        action_pi_env = None if action_pi is None else to_env_action(action_pi, action_space)
        risk_override = bool(
            (not greedy_only)
            and (not force_policy)
            and action_pi_env is not None
            and np.any(np.asarray(env_action).reshape(-1) != np.asarray(action_pi_env).reshape(-1))
        )

        env0 = _get_env0(venv)
        envu = getattr(env0, "unwrapped", env0)
        base_now = getattr(envu, "base", None)
        sim_time = int(getattr(base_now, "t", getattr(envu, "t", -1)))
        pending_count = int(len(getattr(base_now, "pending_orders", getattr(envu, "pending_orders", []))))
        completed_count = 0
        if base_now is not None:
            completed_count = int(sum(1 for v in getattr(base_now, "order_status", {}).values() if str(v) == "delivered"))
        if progress_debug and (step_cnt % 20 == 0):
            last_progress_line = (
                f"[progress] step_cnt={step_cnt} sim_time={sim_time} pending_count={pending_count} "
                f"completed_count={completed_count} done={bool(done[0])}"
            )
            print(last_progress_line, flush=True)

        if trace and decision_step and trace_lines < 50:
            trace_lines += 1
            t, pending_count = _trace_env_state(venv)
            num_valid = int(np.sum(_mask_1d(action_masks))) if action_masks is not None else -1
            valid_r = valid_o = valid_p = valid_d = -1
            if action_masks is not None and isinstance(action_space, gym.spaces.MultiDiscrete):
                per_dim = decode_mask(action_masks, np.asarray(action_space.nvec, dtype=np.int64))
                if len(per_dim) >= 4:
                    valid_r = len(per_dim[0])
                    valid_o = len(per_dim[1])
                    valid_p = len(per_dim[2])
                    valid_d = len(per_dim[3])
            print(
                "[trace]",
                f"k={trace_lines}",
                f"t={t}",
                f"pending={pending_count}",
                f"num_valid={num_valid}",
                f"num_valid_riders={valid_r}",
                f"num_valid_orders={valid_o}",
                f"num_valid_pickup_idx={valid_p}",
                f"num_valid_dropoff_idx={valid_d}",
                f"raw_chosen={np.asarray(chosen_action_raw).tolist()}",
                f"executed={np.asarray(env_action).tolist()}",
                f"action_in_range={chosen_action_in_range}",
                f"action_allowed_by_mask={chosen_action_allowed_by_mask}",
                f"dispatch_valid={bool(chosen_diag.get('dispatch_valid', False))}",
                f"chosen_is_wait={chosen_is_wait}",
                f"chosen_decoded={chosen_decoded}",
                f"chosen_rider_id={chosen_diag.get('rider_id')}",
                f"chosen_order_id={chosen_diag.get('order_id')}",
                f"chosen_rider_in_bounds={bool(chosen_diag.get('rider_in_bounds', False))}",
                f"chosen_order_in_bounds={bool(chosen_diag.get('order_in_bounds', False))}",
                f"chosen_rider_count={int(chosen_diag.get('rider_count', -1))}",
                f"chosen_order_count={int(chosen_diag.get('order_count', -1))}",
                f"chosen_pair_in_mapping={bool(chosen_diag.get('pair_in_candidate_mapping', False))}",
                f"chosen_pair_dispatchable={bool(chosen_diag.get('pair_dispatchable', False))}",
                f"chosen_failure_reason={chosen_diag.get('failure_reason')}",
                f"chosen_candidate_riders={chosen_diag.get('candidate_rider_ids')}",
                f"chosen_candidate_orders={chosen_diag.get('candidate_order_ids')}",
                f"greedy={None if greedy_action is None else np.asarray(greedy_action).tolist()}",
                f"greedy_action_in_range={greedy_action_in_range}",
                f"greedy_action_allowed_by_mask={greedy_action_allowed_by_mask}",
                f"greedy_dispatch_valid={False if greedy_diag is None else bool(greedy_diag.get('dispatch_valid', False))}",
                f"greedy_is_wait={greedy_is_wait}",
                f"greedy_decoded={greedy_decoded}",
                f"greedy_rider_id={None if greedy_diag is None else greedy_diag.get('rider_id')}",
                f"greedy_order_id={None if greedy_diag is None else greedy_diag.get('order_id')}",
                f"fallback_triggered={fallback_triggered}",
                f"fallback_reason={fallback_reason}",
                f"fallback_source={fallback_source}",
                f"executed_decoded={executed_decoded}",
                f"r={step_reward:.4f}",
                f"wait_action_pen={wait_action_penalty:.4f}",
                f"wait_backlog_pen={wait_backlog_penalty:.4f}",
                f"chosen_comp_pre={chosen_breakdown_pre}",
                f"executed_comp_pre={executed_breakdown_pre}",
                f"chosen_comp={chosen_components}",
                f"chosen_comp_total={chosen_components_total:.4f}",
                f"eta_mu_selected={chosen_eta_mu_selected:.4f}",
                f"eta_sigma_selected={chosen_eta_sigma_selected:.4f}",
                f"eta_mu_penalty={chosen_eta_mu_penalty:.4f}",
                f"eta_sigma_penalty={chosen_eta_sigma_penalty:.4f}",
                f"greedy_comp={greedy_breakdown_pre}",
                f"greedy_comp_total={greedy_components_total:.4f}",
                f"risk_alpha={float(alpha_eta_mu):.4f}",
                f"risk_beta={float(beta_eta_sigma):.4f}",
                f"risk_override={risk_override}",
                f"risk_diag={risk_diag}",
                f"wait_reason={wait_allowed_reason}",
                f"invalid={invalid_flag}",
            )

        # PPO uses post-step full-matrix ETA evaluation.
        # Greedy uses post-step ETA evaluation on the chosen dispatch action.
        if not greedy_only:
            full_obs = get_full_obs(venv)
            mae_i, rmse_i, nll_i, enabled_i, diag_i = eval_eta_for_assignment(
                venv=venv,
                model=model,
                full_obs=full_obs,
                eta_pred_hint=eta_pred,
                eta_sigma_hint=eta_sigma,
            )
        eta_metrics_enabled = eta_metrics_enabled or bool(enabled_i)
        eta_maes.append(float(mae_i))
        eta_rmses.append(float(rmse_i))
        eta_nlls.append(float(nll_i))
        if eta_zero_diag is None and isinstance(diag_i, dict) and diag_i:
            eta_zero_diag = dict(diag_i)

        if greedy_only and (not bool(enabled_i)):
            if debug_eval and isinstance(diag_i, dict) and diag_i:
                print(
                    "[eval][greedy][eta-fail]",
                    f"reason={diag_i.get('reason')}",
                    f"pred_present={diag_i.get('pred_present')}",
                    f"sigma_present={diag_i.get('sigma_present')}",
                    f"gt_present={diag_i.get('gt_present')}",
                    f"mask_present={diag_i.get('mask_present')}",
                    f"selected_ri={diag_i.get('selected_ri')}",
                    f"selected_oi={diag_i.get('selected_oi')}",
                    f"selected_rider_id={diag_i.get('selected_rider_id')}",
                    f"selected_order_id={diag_i.get('selected_order_id')}",
                    f"pred_shape={diag_i.get('pred_shape')}",
                    f"sigma_shape={diag_i.get('sigma_shape')}",
                    f"gt_shape={diag_i.get('gt_shape')}",
                    f"mask_shape={diag_i.get('mask_shape')}",
                )
        if greedy_only and bool(enabled_i):
            pred_eq_gt = bool(diag_i.get("pred_eq_gt", False)) if isinstance(diag_i, dict) else False
            greedy_pred_eq_gt_flags.append(pred_eq_gt)
            if greedy_pred_eq_gt_diag is None and isinstance(diag_i, dict) and diag_i:
                greedy_pred_eq_gt_diag = dict(diag_i)
            if isinstance(diag_i, dict):
                greedy_eta_debug_rows.append(
                    {
                        "pred": float(diag_i["pred_sample"][0]) if diag_i.get("pred_sample") else float("nan"),
                        "gt": float(diag_i["gt_sample"][0]) if diag_i.get("gt_sample") else float("nan"),
                        "sigma": float(diag_i["sigma_sample"][0]) if diag_i.get("sigma_sample") else float("nan"),
                        "ri": diag_i.get("selected_ri"),
                        "oi": diag_i.get("selected_oi"),
                        "action_ri": diag_i.get("action_ri"),
                        "action_oi": diag_i.get("action_oi"),
                        "rider_id": diag_i.get("selected_rider_id"),
                        "order_id": diag_i.get("selected_order_id"),
                        "mask_value": diag_i.get("mask_value"),
                        "pred_shape": diag_i.get("pred_shape"),
                        "sigma_shape": diag_i.get("sigma_shape"),
                        "gt_shape": diag_i.get("gt_shape"),
                        "valid_count": diag_i.get("valid_count"),
                    }
                )
                if debug_eval:
                    print(
                        "[eval][greedy][eta-step]",
                        f"action_indices=({diag_i.get('action_ri')}, {diag_i.get('action_oi')})",
                        f"rider_id={diag_i.get('selected_rider_id')}",
                        f"order_id={diag_i.get('selected_order_id')}",
                        f"eta_row={diag_i.get('selected_ri')}",
                        f"eta_col={diag_i.get('selected_oi')}",
                        f"mask_value={diag_i.get('mask_value')}",
                        f"pred_mu={diag_i.get('pred_sample', [None])[0] if diag_i.get('pred_sample') else None}",
                        f"pred_sigma={diag_i.get('sigma_sample', [None])[0] if diag_i.get('sigma_sample') else None}",
                        f"gt_eta={diag_i.get('gt_sample', [None])[0] if diag_i.get('gt_sample') else None}",
                        f"collected_eta_samples={len(greedy_eta_debug_rows)}",
                    )
        
        if debug_eval and (not greedy_only) and step_cnt <= 5:
            used_greedy = (to_int_action(action) != to_int_action(action_pi))
            print(
                "[dbg]",
                "step", step_cnt,
                "no_inject", bool(no_inject),
                "eta_pred_none", bool(eta_pred is None),
                "used_greedy", bool(used_greedy),
            )

        if step_cnt >= int(max_steps):
            print(
                f"[warn] max_steps reached seed={int(seed)} step_cnt={step_cnt} sim_time={sim_time}",
                flush=True,
            )
            done = np.asarray([True], dtype=np.bool_)
        if (max_sim_time is not None) and (sim_time >= int(max_sim_time)):
            print(
                f"[warn] max_sim_time reached seed={int(seed)} step_cnt={step_cnt} sim_time={sim_time}",
                flush=True,
            )
            done = np.asarray([True], dtype=np.bool_)

    mae, _ = _safe_nan_stats(eta_maes)
    rmse, _ = _safe_nan_stats(eta_rmses)
    nll, _ = _safe_nan_stats(eta_nlls)
    if greedy_only and debug_eval:
        if greedy_eta_debug_rows:
            print(
                "[eval][greedy][eta-debug]",
                f"collected_eta_samples={len(greedy_eta_debug_rows)}",
                f"first_5_pred_mu={[row['pred'] for row in greedy_eta_debug_rows[:5]]}",
                f"first_5_gt_eta={[row['gt'] for row in greedy_eta_debug_rows[:5]]}",
                f"first_5_pred_sigma={[row['sigma'] for row in greedy_eta_debug_rows[:5]]}",
                f"valid_count={sum(int(row.get('valid_count', 0) or 0) for row in greedy_eta_debug_rows)}",
                f"rider_order_ids={[(row['ri'], row['oi']) for row in greedy_eta_debug_rows[:5]]}",
                f"action_indices={[(row['action_ri'], row['action_oi']) for row in greedy_eta_debug_rows[:5]]}",
                f"mask_values={[row['mask_value'] for row in greedy_eta_debug_rows[:5]]}",
                f"pred_shape={greedy_eta_debug_rows[0].get('pred_shape')}",
                f"sigma_shape={greedy_eta_debug_rows[0].get('sigma_shape')}",
                f"gt_shape={greedy_eta_debug_rows[0].get('gt_shape')}",
            )
        else:
            print(
                "[eval][greedy][eta-debug]",
                "collected_eta_samples=0",
                "first_5_pred_mu=[]",
                "first_5_gt_eta=[]",
                "first_5_pred_sigma=[]",
                "valid_count=0",
                "rider_order_ids=[]",
            )
    if greedy_only and eta_metrics_enabled:
        zero_mae = bool(np.isfinite(mae) and np.isclose(mae, 0.0, atol=1e-12))
        zero_rmse = bool(np.isfinite(rmse) and np.isclose(rmse, 0.0, atol=1e-12))
        zero_nll = bool(np.isfinite(nll) and np.isclose(nll, 0.0, atol=1e-12))
        if greedy_pred_eq_gt_flags and all(greedy_pred_eq_gt_flags):
            warnings.warn(
                (
                    "[eval][greedy] sample_pred == sample_gt for all ETA-evaluated greedy steps; "
                    f"pred_source={None if greedy_pred_eq_gt_diag is None else greedy_pred_eq_gt_diag.get('pred_source')} "
                    f"gt_source={None if greedy_pred_eq_gt_diag is None else greedy_pred_eq_gt_diag.get('gt_source')} "
                    f"sigma_source={None if greedy_pred_eq_gt_diag is None else greedy_pred_eq_gt_diag.get('sigma_source')} "
                    f"selected_ri={None if greedy_pred_eq_gt_diag is None else greedy_pred_eq_gt_diag.get('selected_ri')} "
                    f"selected_oi={None if greedy_pred_eq_gt_diag is None else greedy_pred_eq_gt_diag.get('selected_oi')}"
                ),
                RuntimeWarning,
            )
        if zero_mae and zero_rmse and zero_nll:
            warnings.warn(
                (
                    "[eval][greedy] all ETA metrics are zero for this episode; "
                    f"sample_pred={None if eta_zero_diag is None else eta_zero_diag.get('pred_sample')} "
                    f"sample_gt={None if eta_zero_diag is None else eta_zero_diag.get('gt_sample')} "
                    f"sample_sigma={None if eta_zero_diag is None else eta_zero_diag.get('sigma_sample')} "
                    f"valid_count={None if eta_zero_diag is None else eta_zero_diag.get('valid_count')}"
                ),
                RuntimeWarning,
            )
    return ep_reward, mae, rmse, nll, bool(eta_metrics_enabled)

def evaluate(
    cfg: dict | None,
    config_path: str | None,
    model_path: str,
    vecnorm_path: str,
    episodes: int,
    seed_base: int,
    deterministic: bool,
    no_inject: bool,
    force_policy: bool,
    greedy_only: bool = False,
    random_policy: bool = False,
    trace_first_episode: bool = False,
    alpha_eta_mu: float = 0.01,
    beta_eta_sigma: float = 0.01,
    data_dir: str | None = None,
    duration: int | None = None,
    progress_debug: bool = False,
    max_steps: int = 5000,
    max_sim_time: int | None = None,
    greedy_top_k_couriers: int | None = None,
    greedy_top_k_positions: int | None = None,
    profile_hotspots: bool = False,
) -> Tuple[np.ndarray, List[float], List[float], List[float], List[bool], List[int]]:
    use_random = bool(random_policy)
    env_files = _prepare_eval_dataset(cfg, config_path=config_path)
    cfg_duration = int(env_files["duration"])
    cfg_data_dir = str(env_files["data_dir"])
    if duration is None:
        duration = cfg_duration
    if data_dir is None:
        data_dir = cfg_data_dir
    model_path_present = str(model_path).strip().lower() != "none"
    use_greedy = bool(greedy_only) or ((not use_random) and (not model_path_present))
    has_model_for_eta = bool(model_path_present)

    model = None
    model_kind = "plain"
    model_is_dict = True
    model_node_feat_dim = None
    if use_random and has_model_for_eta:
        try:
            model, model_kind = load_any_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {model_path}") from e
        print("[eval] Running RANDOM policy")
        print(f"[eval] ETA model type: {model_kind}, algo: {model.__class__.__name__}")
        model_obs_space = getattr(model, "observation_space", None)
        model_is_dict = isinstance(model_obs_space, gym.spaces.Dict)
        if model_is_dict:
            try:
                node_x_space = model_obs_space.spaces.get("node_x")
                if node_x_space is not None and len(node_x_space.shape) == 2:
                    model_node_feat_dim = int(node_x_space.shape[1])
            except Exception:
                model_node_feat_dim = None
    elif use_random:
        print("[eval] Running RANDOM policy")
        print("[eval] No ETA model available; ETA metrics will be disabled")
    elif (not use_greedy) or has_model_for_eta:
        try:
            model, model_kind = load_any_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {model_path}") from e

        if use_greedy:
            print("[eval] Running GREEDY insertion policy with model ETA head")
        else:
            print("[eval] Running PPO policy")
        print(f"[eval] Model type: {model_kind}, algo: {model.__class__.__name__}")
        model_obs_space = getattr(model, "observation_space", None)
        model_is_dict = isinstance(model_obs_space, gym.spaces.Dict)
        if model_is_dict:
            try:
                node_x_space = model_obs_space.spaces.get("node_x")
                if node_x_space is not None and len(node_x_space.shape) == 2:
                    model_node_feat_dim = int(node_x_space.shape[1])
            except Exception:
                model_node_feat_dim = None
    elif use_greedy:
        print("[eval] Running GREEDY insertion policy (no model ETA available)")

    rewards: List[float] = []
    eta_maes: List[float] = []
    eta_rmses: List[float] = []
    eta_nlls: List[float] = []
    eta_enabled_flags: List[bool] = []
    seeds: List[int] = []

    for i in range(episodes):
        seed = seed_base + i
        set_global_seeds(seed)

        need_dict = True if model is None else isinstance(model.observation_space, gym.spaces.Dict)
        venv = DummyVecEnv(
            [
                lambda: make_env(
                    cfg=cfg,
                    env_files=env_files,
                    flat_compat=not need_dict,
                    node_feat_dim=model_node_feat_dim,
                    debug=False,
                    progress_debug=bool(progress_debug),
                    greedy_top_k_couriers=greedy_top_k_couriers,
                    greedy_top_k_positions=greedy_top_k_positions,
                    use_action_masker=bool(model is not None and model_kind == "maskable"),
                    alpha_eta_mu=float(alpha_eta_mu),
                    beta_eta_sigma=float(beta_eta_sigma),
                    data_dir=None if data_dir is None else str(data_dir),
                    duration=None if duration is None else int(duration),
                    profile_hotspots=bool(profile_hotspots),
                )
            ]
        )
        if trace_first_episode and i == 0:
            print_trace_action_space_info(venv)
       
        if (model is not None) and (not model_is_dict) and vecnorm_path and Path(vecnorm_path).exists():
            try:
                venv = VecNormalize.load(vecnorm_path, venv)
                venv.training = False
                venv.norm_reward = False
            except Exception:
                pass

        if model is not None:
            model.set_env(venv)

        r, eta_mae, eta_rmse, eta_nll, eta_enabled = run_one(
            model=model,
            model_kind=model_kind,
            venv=venv,
            seed=seed,
            deterministic=deterministic,
            no_inject=no_inject,
            force_policy=force_policy,
            alpha_eta_mu=float(alpha_eta_mu),
            beta_eta_sigma=float(beta_eta_sigma),
            progress_debug=bool(progress_debug),
            max_steps=int(max_steps),
            max_sim_time=None if max_sim_time is None else int(max_sim_time),
            greedy_only=use_greedy,
            random_policy=use_random,
            trace=bool(trace_first_episode and i == 0),
        )

        rewards.append(float(r))
        eta_maes.append(float(eta_mae))
        eta_rmses.append(float(eta_rmse))
        eta_nlls.append(float(eta_nll))
        eta_enabled_flags.append(bool(eta_enabled))
        seeds.append(int(seed))

        try:
            venv.close()
        except Exception:
            pass

    return (
        np.asarray(rewards, dtype=np.float64),
        eta_maes,
        eta_rmses,
        eta_nlls,
        eta_enabled_flags,
        seeds,
    )

def run_mask_self_check(cfg: dict | None = None, seed: int = 0, steps: int = 50) -> None:
    env = make_env(cfg=cfg, flat_compat=False, node_feat_dim=None, debug=False, use_action_masker=False)
    obs, _info = env.reset(seed=int(seed))
    _ = obs
    for _k in range(int(steps)):
        mask = np.asarray(env.action_masks())
        nvec = np.asarray(env.action_space.nvec, dtype=np.int64)
        assert mask.shape == (int(np.sum(nvec)),), f"bad mask shape: {mask.shape}"
        segs = decode_mask(mask, nvec)
        assert len(segs) == len(nvec), "bad number of segments"
        for i, seg in enumerate(segs):
            assert seg.size > 0, f"segment {i} empty"
        a = env.greedy_insertion_action()
        env.step(a)
    print(f"[mask-check] ok steps={int(steps)} seed={int(seed)}")

def _safe_nan_stats(arr):
    return _safe_nan_stats_impl(arr)


def main():
    global PROFILE_HOTSPOTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/final_main.yaml", help="Path to YAML config file")
    parser.add_argument("--no_inject", action="store_true")
    parser.add_argument("--model", type=str, default="outputs/ppo.zip")
    parser.add_argument("--vecnorm", type=str, default="outputs/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--force_policy", action="store_true")
    parser.add_argument("--greedy_only", action="store_true")
    parser.add_argument("--random_policy", action="store_true")
    parser.add_argument("--trace_first_episode", action="store_true")
    parser.add_argument("--mask_self_check", action="store_true")
    parser.add_argument("--eta_uncertainty", action="store_true")
    parser.add_argument("--risk_k", type=float, default=0.0)
    parser.add_argument("--alpha_eta_mu", type=float, default=0.01)
    parser.add_argument("--beta_eta_sigma", type=float, default=0.01)
    parser.add_argument("--profile_hotspots", action="store_true")
    args = parser.parse_args()
    PROFILE_HOTSPOTS = bool(args.profile_hotspots)
    print(f"[eval] Using config: {args.config}")
    cfg = load_config(args.config)
    _duration, _seed, _data_dir = _resolve_eval_config(cfg)

    if args.mask_self_check:
        run_mask_self_check(cfg=cfg, seed=int(args.seed_base), steps=50)
        return

    rewards, eta_maes, eta_rmses, eta_nlls, eta_enabled_flags, seeds = evaluate(
        cfg=cfg,
        config_path=args.config,
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        episodes=args.episodes,
        seed_base=args.seed_base,
        deterministic=args.deterministic,
        no_inject=args.no_inject,
        force_policy=args.force_policy,
        greedy_only=args.greedy_only,
        random_policy=args.random_policy,
        trace_first_episode=args.trace_first_episode,
        alpha_eta_mu=args.alpha_eta_mu,
        beta_eta_sigma=args.beta_eta_sigma,
        profile_hotspots=bool(args.profile_hotspots),
    )

    # ===== EXPORT RESULTS (for paper / sharing) =====
    out_dir = Path("outputs/eval_exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_deterministic = bool(args.deterministic or args.greedy_only)
    mode = "deterministic" if effective_deterministic else "stochastic"
    tag = f"_{args.tag}" if args.tag else ""

    # CSV: per-episode rewards
    csv_path = out_dir / f"eval_{mode}{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "seed", "reward", "eta_mae", "eta_rmse", "eta_nll"])
        for i, (s, r, m, e, n) in enumerate(zip(seeds, rewards, eta_maes, eta_rmses, eta_nlls)):
            writer.writerow([i, s, float(r), float(m), float(e), float(n)])

    
   
    # JSON: summary stats
    eta_mae_mean, eta_mae_std = _safe_nan_stats(eta_maes)
    eta_rmse_mean, eta_rmse_std = _safe_nan_stats(eta_rmses)
    eta_nll_mean, eta_nll_std = _safe_nan_stats(eta_nlls)
    eta_metrics_enabled = bool(np.any(np.asarray(eta_enabled_flags, dtype=np.bool_)))

    summary = {
        "model": args.model,
        "vecnormalize": args.vecnorm,
        "episodes": int(len(rewards)),
        "seed_base": int(args.seed_base),
        "mode": mode,
        "alpha_eta_mu": float(args.alpha_eta_mu),
        "beta_eta_sigma": float(args.beta_eta_sigma),
        "eta_metrics_enabled": bool(eta_metrics_enabled),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "eta_mae_mean": eta_mae_mean,
        "eta_mae_std": eta_mae_std,
        "eta_rmse_mean": eta_rmse_mean,
        "eta_rmse_std": eta_rmse_std,
        "eta_nll_mean": eta_nll_mean,
        "eta_nll_std": eta_nll_std,
    }

    json_path = out_dir / f"summary_{mode}{tag}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[EXPORT] CSV saved to: {csv_path}")
    print(f"[EXPORT] JSON saved to: {json_path}")
    # ===============================================

    print(f"episodes: {len(rewards)}")
    if seeds:
        print(f"seeds: {seeds[0]}..{seeds[-1]}")
    print(
        "reward mean/std/min/max: "
        f"{rewards.mean():.3f} / {rewards.std():.3f} / {rewards.min():.3f} / {rewards.max():.3f}"
    )
    print(
    "eta mae mean/std: "
    f"{eta_mae_mean:.6f} / {eta_mae_std:.6f}"
    )
    print(
        "eta rmse mean/std: "
        f"{eta_rmse_mean:.6f} / {eta_rmse_std:.6f}"
    )
    print(
        "eta nll mean/std: "
        f"{eta_nll_mean:.6f} / {eta_nll_std:.6f}"
    )
if __name__ == "__main__":
    main()
