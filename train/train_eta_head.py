# train/train_eta_head.py
from __future__ import annotations

import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.env_dispatch_ppo import DispatchPPOEnv
from models.joint_encoder import NodeSetEncoderConfig, NodeSetJointEncoder


def make_env(R: int, O: int, duration: int, wait_penalty: float, invalid_action_penalty: float):
    return DispatchPPOEnv(
        R=R,
        O=O,
        duration=duration,
        wait_penalty=wait_penalty,
        invalid_action_penalty=invalid_action_penalty,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _obs_to_tensors(obs: Dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = torch.as_tensor(obs["flat"], device=device).float().unsqueeze(0)          # [1, 39]
    node_x = torch.as_tensor(obs["node_x"], device=device).float().unsqueeze(0)      # [1, N, 8]
    eta_tgt = torch.as_tensor(obs["eta_tgt"], device=device).float().unsqueeze(0)    # [1, R, O]
    eta_mask = torch.as_tensor(obs["eta_mask"], device=device).float().unsqueeze(0) # [1, R, O]
    return flat, node_x, eta_tgt, eta_mask


def masked_huber(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    # pred/tgt/mask: [B, R, O]
    mask = (mask > 0.5).float()
    diff = pred - tgt
    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device))
    lin = abs_diff - quad
    loss = 0.5 * quad * quad + delta * lin
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom

def masked_gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).float()
    lv = torch.clamp(log_var, min=-10.0, max=10.0)
    inv_var = torch.exp(-lv)
    diff2 = (tgt - mu) ** 2
    nll = 0.5 * (diff2 * inv_var + lv)
    denom = mask.sum().clamp_min(1.0)
    return (nll * mask).sum() / denom


@torch.no_grad()
def masked_mae(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    mask = (mask > 0.5)
    if mask.sum().item() == 0:
        return float("nan")
    return float((pred[mask] - tgt[mask]).abs().mean().item())

@torch.no_grad()
def masked_rmse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> float:
    mask = (mask > 0.5)
    if mask.sum().item() == 0:
        return float("nan")
    diff = pred[mask] - tgt[mask]
    return float(torch.sqrt((diff * diff).mean()).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000, help="env steps to collect (decision steps)")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--O", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--delta", type=float, default=1.0, help="Huber delta")
    parser.add_argument("--eta_uncertainty", action="store_true")
    parser.add_argument("--out_dir", type=str, default="outputs/eta_head")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    set_global_seeds(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv([lambda: make_env(args.R, args.O, args.duration, wait_penalty=-0.01, invalid_action_penalty=-0.02)])

    cfg = NodeSetEncoderConfig(
        flat_dim=39,
        node_feat_dim=8,
        R=args.R,
        O=args.O,
        present_idx=7,
        flat_hidden=128,
        node_hidden=128,
        fused_hidden=256,
        out_dim=256,
        dropout=0.0,
        eta_uncertainty=bool(args.eta_uncertainty),
    )
    model = NodeSetJointEncoder(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # simple buffer for minibatch SGD
    buf_flat: List[torch.Tensor] = []
    buf_node: List[torch.Tensor] = []
    buf_tgt: List[torch.Tensor] = []
    buf_mask: List[torch.Tensor] = []

    obs = venv.reset()
    # VecEnv returns list-like obs; for Dict it is a dict of arrays with batch dim = n_envs
    def unwrap_vec_obs(vec_obs: Dict) -> Dict:
        return {k: vec_obs[k][0] for k in vec_obs.keys()}

    running_mae = []
    running_rmse = []
    running_nll = []
    running_loss = []

    for step in range(1, args.steps + 1):
        o = unwrap_vec_obs(obs)
        flat, node_x, eta_tgt, eta_mask = _obs_to_tensors(o, device)

        buf_flat.append(flat.squeeze(0))
        buf_node.append(node_x.squeeze(0))
        buf_tgt.append(eta_tgt.squeeze(0))
        buf_mask.append(eta_mask.squeeze(0))

        # random action to move the env forward; Phase 1 only needs supervision signals
        a = [venv.action_space.sample()]
        obs, _r, _done, _info = venv.step(a)

        if len(buf_flat) >= args.batch:
            Xf = torch.stack(buf_flat, dim=0)    # [B, 39]
            Xn = torch.stack(buf_node, dim=0)    # [B, N, 8]
            Y = torch.stack(buf_tgt, dim=0)      # [B, R, O]
            M = torch.stack(buf_mask, dim=0)     # [B, R, O]

            _feats, eta_mu = model({"flat": Xf, "node_x": Xn})
            eta_log_var = getattr(model, "last_eta_log_var", None)
            if eta_log_var is not None:
                loss = masked_gaussian_nll(eta_mu, eta_log_var, Y, M)
                nll = float(loss.item())
            else:
                loss = masked_huber(eta_mu, Y, M, delta=args.delta)
                nll = float("nan")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            mae = masked_mae(eta_mu.detach(), Y, M)
            rmse = masked_rmse(eta_mu.detach(), Y, M)

            running_loss.append(float(loss.item()))
            running_mae.append(float(mae) if not np.isnan(mae) else 0.0)
            running_rmse.append(float(rmse) if not np.isnan(rmse) else 0.0)
            running_nll.append(float(nll) if np.isfinite(nll) else 0.0)

            buf_flat.clear()
            buf_node.clear()
            buf_tgt.clear()
            buf_mask.clear()

        if step % 1000 == 0:
            mean_loss = float(np.mean(running_loss[-50:])) if running_loss else 0.0
            mean_mae = float(np.mean(running_mae[-50:])) if running_mae else 0.0
            mean_rmse = float(np.mean(running_rmse[-50:])) if running_rmse else 0.0
            mean_nll = float(np.mean(running_nll[-50:])) if running_nll else 0.0
            print(f"[train] step={step} loss={mean_loss:.6f} mae={mean_mae:.6f} rmse={mean_rmse:.6f} nll={mean_nll:.6f}")

    ckpt = {
        "cfg": asdict(cfg),
        "state_dict": model.state_dict(),
        "seed": int(args.seed),
    }
    save_path = out_dir / f"eta_head_seed{args.seed}_steps{args.steps}.pt"
    torch.save(ckpt, save_path)
    print(f"[save] {save_path}")

    try:
        venv.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
