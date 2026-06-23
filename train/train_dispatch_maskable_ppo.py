from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from train.joint_maskable_ppo import JointMaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.env_dispatch_ppo import DispatchPPOEnv
from models.joint_encoder import SB3NodeSetFeaturesExtractor


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(alpha_eta_mu: float = 0.01, beta_eta_sigma: float = 0.01):
    env = DispatchPPOEnv(
        data_dir="outputs",
        duration=180,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
        R=4,
        O=4,
        invalid_action_penalty=-0.02,
        wait_penalty=-0.01,
        alpha_eta_mu=float(alpha_eta_mu),
        beta_eta_sigma=float(beta_eta_sigma),
    )

    def mask_fn(_env: DispatchPPOEnv):
        return _env.action_masks()

    env = ActionMasker(env, mask_fn)
    return env
from stable_baselines3.common.callbacks import BaseCallback

class PrintEtaShapeCallback(BaseCallback):
    def __init__(self, print_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = int(print_every)

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            fe = self.model.policy.features_extractor
            eta = getattr(fe, "last_eta", None)
            if eta is None:
                print("[debug] last_eta is None")
            else:
                print("[debug] last_eta shape:", tuple(eta.shape))
        return True
class PrintEtaShapeCallback(BaseCallback):
    def __init__(self, print_every: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.print_every = int(print_every)

    def _on_step(self) -> bool:
        if self.n_calls % self.print_every == 0:
            fe = self.model.policy.features_extractor
            enc = getattr(fe, "encoder", None)
            eta = getattr(fe, "last_eta", None)

            if eta is None:
                print("[debug] last_eta is None")
            else:
                print("[debug] last_eta shape:", tuple(eta.shape))

            if enc is not None and hasattr(enc, "eta_head"):
                g = 0.0
                for p in enc.eta_head.parameters():
                    if p.grad is not None:
                        g += float(p.grad.detach().pow(2).sum().cpu().item())
                print("[debug] eta_head grad_l2:", g ** 0.5)

        return True

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--out_dir", type=str, default="outputs/runs")
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--eta_uncertainty", action="store_true")
    parser.add_argument("--lambda_eta", type=float, default=0.1)
    parser.add_argument("--eta_coef", type=float, default=None)
    parser.add_argument("--alpha_eta_mu", type=float, default=0.01)
    parser.add_argument("--beta_eta_sigma", type=float, default=0.01)
    args = parser.parse_args()

    set_global_seeds(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip()
    run_name = tag if tag else f"seed{args.seed}_T{args.timesteps}_kl{args.target_kl}_{ts}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = run_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)

    outputs = ["stdout"]
    if args.tb:
        outputs.append("tensorboard")
    logger = configure(str(log_dir), outputs)

    venv = DummyVecEnv([
        lambda: make_env(
            alpha_eta_mu=float(args.alpha_eta_mu),
            beta_eta_sigma=float(args.beta_eta_sigma),
        )
    ])
    venv.reset()

    policy_kwargs = dict(
        features_extractor_class=SB3NodeSetFeaturesExtractor,
        features_extractor_kwargs=dict(
            flat_hidden=128,
            node_hidden=128,
            out_dim=256,
            present_idx=7,
            dropout=0.0,
            eta_uncertainty=bool(args.eta_uncertainty),
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = JointMaskablePPO(
        policy="MultiInputPolicy",
        env=venv,
        lambda_eta=float(args.lambda_eta if args.eta_coef is None else args.eta_coef),
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
    model_path = run_dir / "model.zip"
    model.save(str(model_path))

    print(f"[SAVE] {model_path}")

    export_name = f"maskable_joint_ueta_seed{args.seed}.zip" if args.eta_uncertainty else f"maskable_seed{args.seed}.zip"
    export_path = Path("outputs") / export_name
    model.save(str(export_path))
    print(f"[SAVE] {export_path}")


if __name__ == "__main__":
    main()
