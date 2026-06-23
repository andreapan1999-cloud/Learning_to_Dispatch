from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from train.eval_dispatch_ppo import evaluate, _safe_nan_stats


def pareto_mask(rows):
    if not rows:
        return np.zeros((0,), dtype=np.bool_)

    reward = np.asarray([float(r["reward_mean"]) for r in rows], dtype=np.float64)
    eta_mae = np.asarray([float(r["eta_mae_mean"]) for r in rows], dtype=np.float64)
    eta_nll = np.asarray([float(r["eta_nll_mean"]) for r in rows], dtype=np.float64)

    efficient = np.ones((len(rows),), dtype=np.bool_)
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j:
                continue
            dominates = (
                (reward[j] >= reward[i])
                and (eta_mae[j] <= eta_mae[i])
                and (eta_nll[j] <= eta_nll[i])
                and (
                    (reward[j] > reward[i])
                    or (eta_mae[j] < eta_mae[i])
                    or (eta_nll[j] < eta_nll[i])
                )
            )
            if dominates:
                efficient[i] = False
                break
    return efficient


def maybe_make_plots(rows, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[pareto] matplotlib unavailable, skipping plots: {e!r}")
        return

    efficient = pareto_mask(rows)
    reward = np.asarray([float(r["reward_mean"]) for r in rows], dtype=np.float64)
    eta_mae = np.asarray([float(r["eta_mae_mean"]) for r in rows], dtype=np.float64)
    eta_nll = np.asarray([float(r["eta_nll_mean"]) for r in rows], dtype=np.float64)
    labels = [f"a={r['alpha_eta_mu']:.3f}, b={r['beta_eta_sigma']:.3f}" for r in rows]

    def save_plot(x, y, xlab: str, ylab: str, name: str) -> None:
        plt.figure(figsize=(7, 5))
        plt.scatter(x[~efficient], y[~efficient], c="steelblue", alpha=0.8, label="sweep points")
        plt.scatter(x[efficient], y[efficient], c="crimson", alpha=0.95, label="Pareto-efficient")
        for i, txt in enumerate(labels):
            plt.annotate(txt, (x[i], y[i]), fontsize=7, alpha=0.8)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / name
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"[pareto] plot saved: {out_path}")

    save_plot(eta_mae, reward, "eta_mae_mean", "reward_mean", "pareto_reward_vs_eta_mae.png")
    save_plot(eta_nll, reward, "eta_nll_mean", "reward_mean", "pareto_reward_vs_eta_nll.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/maskable_ueta_seed2.zip")
    parser.add_argument("--vecnorm", type=str, default="outputs/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed_base", type=int, default=2000)
    parser.add_argument("--out_csv", type=str, default="outputs/eval_exports/pareto_phase44_maskable_ueta_seed2.csv")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    alpha_grid = [0.00, 0.01, 0.02, 0.05]
    beta_grid = [0.00, 0.005, 0.01, 0.02]

    rows = []
    for alpha in alpha_grid:
        for beta in beta_grid:
            tag = f"phase44_a{alpha:.3f}_b{beta:.3f}".replace(".", "p")
            print(f"[pareto] evaluating alpha={alpha:.3f}, beta={beta:.3f}")
            rewards, eta_maes, eta_rmses, eta_nlls, eta_enabled_flags, _seeds = evaluate(
                model_path=args.model,
                vecnorm_path=args.vecnorm,
                episodes=int(args.episodes),
                seed_base=int(args.seed_base),
                deterministic=True,
                no_inject=False,
                force_policy=False,
                greedy_only=False,
                trace_first_episode=False,
                alpha_eta_mu=float(alpha),
                beta_eta_sigma=float(beta),
            )

            reward_mean = float(rewards.mean())
            reward_std = float(rewards.std())
            eta_mae_mean, eta_mae_std = _safe_nan_stats(eta_maes)
            eta_rmse_mean, eta_rmse_std = _safe_nan_stats(eta_rmses)
            eta_nll_mean, eta_nll_std = _safe_nan_stats(eta_nlls)

            row = {
                "tag": tag,
                "alpha_eta_mu": float(alpha),
                "beta_eta_sigma": float(beta),
                "episodes": int(args.episodes),
                "seed_base": int(args.seed_base),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "eta_mae_mean": float(eta_mae_mean),
                "eta_mae_std": float(eta_mae_std),
                "eta_rmse_mean": float(eta_rmse_mean),
                "eta_rmse_std": float(eta_rmse_std),
                "eta_nll_mean": float(eta_nll_mean),
                "eta_nll_std": float(eta_nll_std),
                "eta_metrics_enabled": bool(np.any(np.asarray(eta_enabled_flags, dtype=np.bool_))),
            }
            rows.append(row)
            print(
                "[pareto]",
                f"alpha={alpha:.3f}",
                f"beta={beta:.3f}",
                f"reward_mean={reward_mean:.6f}",
                f"eta_mae_mean={eta_mae_mean:.6f}",
                f"eta_nll_mean={eta_nll_mean:.6f}",
            )

    efficient = pareto_mask(rows)
    for row, is_efficient in zip(rows, efficient):
        row["pareto_efficient"] = bool(is_efficient)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag",
        "alpha_eta_mu",
        "beta_eta_sigma",
        "episodes",
        "seed_base",
        "reward_mean",
        "reward_std",
        "eta_mae_mean",
        "eta_mae_std",
        "eta_rmse_mean",
        "eta_rmse_std",
        "eta_nll_mean",
        "eta_nll_std",
        "eta_metrics_enabled",
        "pareto_efficient",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[pareto] summary saved: {out_csv}")

    if not args.no_plots:
        maybe_make_plots(rows, out_csv.parent)


if __name__ == "__main__":
    main()
