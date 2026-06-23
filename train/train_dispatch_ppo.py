from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
except Exception:
    MaskablePPO = None
    ActionMasker = None
    MaskableMultiInputActorCriticPolicy = None

from sim.env_dispatch_ppo import DispatchPPOEnv
from models.joint_encoder import NodeSetEncoderConfig, SB3NodeSetFeaturesExtractor


def make_env(maskable: bool = False, alpha_eta_mu: float = 0.01, beta_eta_sigma: float = 0.01):
    env = DispatchPPOEnv(
        data_dir="outputs",
        duration=180,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
        R=4,
        O=4,
        invalid_action_penalty=-2.0,
        wait_penalty=-0.01,
        debug=False,
        alpha_eta_mu=float(alpha_eta_mu),
        beta_eta_sigma=float(beta_eta_sigma),
    )
    if not maskable:
        return env
    if ActionMasker is None:
        raise RuntimeError(
            "sb3-contrib is required for masked PPO training. "
            "Install it with: pip install sb3-contrib"
        )
    return ActionMasker(env, lambda e: e.action_masks())


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_vecenv(venv, seed: int):
    try:
        return venv.reset(seed=seed)
    except TypeError:
        try:
            venv.seed(seed)
        except Exception:
            pass
        return venv.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--maskable", action="store_true")
    parser.add_argument("--eta_uncertainty", action="store_true")
    parser.add_argument("--risk_k", type=float, default=0.0)
    parser.add_argument("--alpha_eta_mu", type=float, default=0.01)
    parser.add_argument("--beta_eta_sigma", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.maskable and (MaskablePPO is None or MaskableMultiInputActorCriticPolicy is None):
        raise RuntimeError(
            "sb3-contrib is required for masked PPO training. "
            "Install it with: pip install sb3-contrib"
        )

    set_global_seeds(args.seed)

    venv = DummyVecEnv([
        lambda: make_env(
            maskable=bool(args.maskable),
            alpha_eta_mu=float(args.alpha_eta_mu),
            beta_eta_sigma=float(args.beta_eta_sigma),
        )
    ])
    reset_vecenv(venv, args.seed)

    log_dir = out_dir / "tb" / f"seed{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    outputs = ["stdout"]
    if args.tb:
        outputs.append("tensorboard")
    logger = configure(str(log_dir), outputs)

    enc_cfg = NodeSetEncoderConfig(
        flat_dim=39,
        node_feat_dim=9,
        R=4,
        O=4,
        present_idx=7,
        flat_hidden=128,
        node_hidden=128,
        fused_hidden=256,
        out_dim=256,
        dropout=0.0,
        eta_uncertainty=bool(args.eta_uncertainty),
    )

    policy_kwargs = dict(
        features_extractor_class=SB3NodeSetFeaturesExtractor,
        features_extractor_kwargs=dict(cfg=enc_cfg),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if args.maskable:
        model = MaskablePPO(
            policy=MaskableMultiInputActorCriticPolicy,
            env=venv,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            n_epochs=10,
            clip_range=0.2,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=float(args.target_kl),
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=venv,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            n_epochs=10,
            clip_range=0.2,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=float(args.target_kl),
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    model.set_logger(logger)
    model.learn(total_timesteps=int(args.timesteps))

    model_path = out_dir / (
        f"maskable_seed{args.seed}.zip" if args.maskable else f"ppo_seed{args.seed}.zip"
    )
    model.save(str(model_path))

    try:
        venv.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
