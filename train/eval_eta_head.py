from __future__ import annotations

import argparse
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.env_dispatch_ppo import DispatchPPOEnv
from models.joint_encoder import NodeSetEncoderConfig, NodeSetJointEncoder


def make_env(R: int, O: int, duration: int):
    return DispatchPPOEnv(
        R=R,
        O=O,
        duration=duration,
        wait_penalty=-0.01,
        invalid_action_penalty=-0.02,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
    )


@torch.no_grad()
def masked_metrics(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor):
    m = (mask > 0.5)
    if m.sum().item() == 0:
        return float("nan"), float("nan")
    diff = pred[m] - tgt[m]
    mae = float(diff.abs().mean().item())
    rmse = float(torch.sqrt((diff * diff).mean()).item())
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--O", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = NodeSetEncoderConfig(**ckpt["cfg"])
    model = NodeSetJointEncoder(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    venv = DummyVecEnv([lambda: make_env(args.R, args.O, args.duration)])

    def unwrap(vec_obs):
        return {k: vec_obs[k][0] for k in vec_obs.keys()}

    maes = []
    rmses = []

    printed_scale = False  

    for ep in range(args.episodes):
        obs = venv.reset()
        done = [False]
        while not done[0]:
            o = unwrap(obs)
            flat = torch.as_tensor(o["flat"], device=device).float().unsqueeze(0)
            node_x = torch.as_tensor(o["node_x"], device=device).float().unsqueeze(0)
            eta_tgt = torch.as_tensor(o["eta_tgt"], device=device).float().unsqueeze(0)
            eta_mask = torch.as_tensor(o["eta_mask"], device=device).float().unsqueeze(0)

            _feats, eta_pred = model({"flat": flat, "node_x": node_x})

            # Print once
            if (not printed_scale):
                m = (eta_mask > 0.5)
                if m.sum().item() > 0:
                    t = eta_tgt[m].detach().cpu().numpy()
                    p = eta_pred[m].detach().cpu().numpy()
                    e = p - t
                    print(
                        "[eval-scale] "
                        f"tgt mean/min/max={t.mean():.6f}/{t.min():.6f}/{t.max():.6f} | "
                        f"pred mean/min/max={p.mean():.6f}/{p.min():.6f}/{p.max():.6f} | "
                        f"abs_err mean/p95/max="
                        f"{np.abs(e).mean():.6f}/"
                        f"{np.quantile(np.abs(e), 0.95):.6f}/"
                        f"{np.abs(e).max():.6f}"
                    )
                    printed_scale = True

            mae, rmse = masked_metrics(eta_pred, eta_tgt, eta_mask)
            if not np.isnan(mae):
                maes.append(mae)
                rmses.append(rmse)

            a = [venv.action_space.sample()]
            obs, _r, done, _info = venv.step(a)

    print(f"[eval] pairs={len(maes)} mae={np.mean(maes):.6f}±{np.std(maes):.6f} rmse={np.mean(rmses):.6f}±{np.std(rmses):.6f}")

    try:
        venv.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()