# train/train_joint_phase1.py
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.env_dispatch_ppo import DispatchPPOEnv
from models.joint_encoder import SB3NodeSetFeaturesExtractor
from sb3_contrib.common.wrappers import ActionMasker
from train.joint_maskable_ppo import JointMaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from train.joint_maskable_ppo import JointMaskablePPO


def mask_fn(env):
    return env.action_masks()

def make_env(R: int, O: int, duration: int, invalid_pen: float, wait_pen: float):

    def mask_fn(env):
        return env.action_masks()

    env = DispatchPPOEnv(
        data_dir="outputs",
        duration=duration,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
        R=R,
        O=O,
        invalid_action_penalty=invalid_pen,
        wait_penalty=wait_pen,
        decision_mode="event",
        max_wait_advance=60,
        debug=True,
    )

    env = ActionMasker(env, mask_fn)
    return env


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JointPPO(PPO):
    def __init__(
        self,
        *args,
        eta_coef: float = 0.1,
        eta_clip_min: float = 0.0,
        eta_clip_max: float = 300.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eta_coef = float(eta_coef)
        self.eta_clip_min = float(eta_clip_min)
        self.eta_clip_max = float(eta_clip_max)

    def _eta_loss(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "eta_tgt" not in obs or "eta_mask" not in obs:
            return torch.zeros((), device=self.device)

        eta_tgt = obs["eta_tgt"].float().to(self.device)
        eta_mask = obs["eta_mask"].float().to(self.device)

        eta_tgt = torch.clamp(eta_tgt, self.eta_clip_min, self.eta_clip_max)

        extractor = self.policy.features_extractor
        if not hasattr(extractor, "encoder"):
            return torch.zeros((), device=self.device)

        encoder = getattr(extractor, "encoder")
        if not hasattr(encoder, "eta_head"):
            return torch.zeros((), device=self.device)

        _, eta_pred = encoder(obs)

        if eta_pred.shape != eta_tgt.shape:
            return torch.zeros((), device=self.device)

        diff = (eta_pred - eta_tgt) ** 2
        denom = eta_mask.sum().clamp(min=1.0)
        return (diff * eta_mask).sum() / denom

    def train(self) -> None:
        super().train()

        if self.eta_coef <= 0.0:
            return

        self.policy.set_training_mode(True)

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            obs = rollout_data.observations
            if not isinstance(obs, dict):
                continue

            eta_loss = self._eta_loss(obs)
            if torch.isfinite(eta_loss).all():
                self.policy.optimizer.zero_grad(set_to_none=True)
                (self.eta_coef * eta_loss).backward()
                self.policy.optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--eta_coef", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--tb", action="store_true")

    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--O", type=int, default=4)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--invalid_pen", type=float, default=-0.02)
    parser.add_argument("--wait_pen", type=float, default=-0.01)
    args = parser.parse_args()

    

    set_global_seeds(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""
    run_dir = out_dir / "runs" / f"seed{args.seed}_T{args.timesteps}_kl{args.target_kl}_eta{args.eta_coef}{suffix}"

    venv = DummyVecEnv(
        [lambda: make_env(args.R, args.O, args.duration, args.invalid_pen, args.wait_pen)]
    )
    venv.reset()

    suffix = f"_{args.tag}" if args.tag else ""
    run_dir = out_dir / "runs" / f"seed{args.seed}_T{args.timesteps}_kl{args.target_kl}_eta{args.eta_coef}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = ["stdout"]
    if args.tb:
        outputs.append("tensorboard")
    logger = configure(str(run_dir / "tb"), outputs)

    policy_kwargs = dict(
        features_extractor_class=SB3NodeSetFeaturesExtractor,
        features_extractor_kwargs=dict(
            flat_hidden=128,
            node_hidden=128,
            out_dim=256,
            present_idx=7,
            dropout=0.0,
            R=args.R,
            O=args.O,
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    from train.joint_maskable_ppo import JointMaskablePPO
    model = JointMaskablePPO(
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
        eta_coef=float(args.eta_coef),
    )

    model.set_logger(logger)
    model.learn(total_timesteps=int(args.timesteps))

    model_path = run_dir / "model.zip"
    model.save(str(model_path))
    print(f"[SAVE] {model_path}")


if __name__ == "__main__":
    main()