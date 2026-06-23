from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


INPUT_GLOB = "summary_deterministic_phase44_*.json"
OUT_DIR = Path("outputs/eval_exports")
OUT_CSV = OUT_DIR / "pareto_phase44_summary.csv"
OUT_PLOT_MAE = OUT_DIR / "pareto_reward_vs_mae.png"
OUT_PLOT_NLL = OUT_DIR / "pareto_reward_vs_nll.png"


def load_rows():
    rows = []
    for path in sorted(OUT_DIR.glob(INPUT_GLOB)):
        with open(path, "r") as f:
            data = json.load(f)
        rows.append(
            {
                "tag": str(path.stem).removeprefix("summary_deterministic_"),
                "alpha_eta_mu": float(data.get("alpha_eta_mu", np.nan)),
                "beta_eta_sigma": float(data.get("beta_eta_sigma", np.nan)),
                "reward_mean": float(data.get("reward_mean", np.nan)),
                "reward_std": float(data.get("reward_std", np.nan)),
                "eta_mae_mean": float(data.get("eta_mae_mean", np.nan)),
                "eta_mae_std": float(data.get("eta_mae_std", np.nan)),
                "eta_rmse_mean": float(data.get("eta_rmse_mean", np.nan)),
                "eta_rmse_std": float(data.get("eta_rmse_std", np.nan)),
                "eta_nll_mean": float(data.get("eta_nll_mean", np.nan)),
                "eta_nll_std": float(data.get("eta_nll_std", np.nan)),
                "source_file": str(path),
            }
        )
    return rows


def pareto_mask_reward_min_metric(rows, metric_key: str):
    if not rows:
        return np.zeros((0,), dtype=np.bool_)

    reward = np.asarray([float(r["reward_mean"]) for r in rows], dtype=np.float64)
    metric = np.asarray([float(r[metric_key]) for r in rows], dtype=np.float64)
    valid = np.isfinite(reward) & np.isfinite(metric)
    efficient = np.zeros((len(rows),), dtype=np.bool_)

    valid_idx = np.where(valid)[0]
    for i in valid_idx:
        dominated = False
        for j in valid_idx:
            if i == j:
                continue
            if (
                reward[j] >= reward[i]
                and metric[j] <= metric[i]
                and (reward[j] > reward[i] or metric[j] < metric[i])
            ):
                dominated = True
                break
        efficient[i] = not dominated
    return efficient


def write_csv(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag",
        "alpha_eta_mu",
        "beta_eta_sigma",
        "reward_mean",
        "reward_std",
        "eta_mae_mean",
        "eta_mae_std",
        "eta_rmse_mean",
        "eta_rmse_std",
        "eta_nll_mean",
        "eta_nll_std",
        "pareto_reward_eta_mae",
        "pareto_reward_eta_nll",
        "source_file",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[phase44-summary] csv saved: {OUT_CSV}")


def save_plot(rows, metric_key: str, out_path: Path, metric_label: str, efficient_key: str) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[phase44-summary] matplotlib unavailable, skipping {out_path.name}: {e!r}")
        return

    x = np.asarray([float(r[metric_key]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["reward_mean"]) for r in rows], dtype=np.float64)
    efficient = np.asarray([bool(r[efficient_key]) for r in rows], dtype=np.bool_)
    labels = [f"({r['alpha_eta_mu']:.3f},{r['beta_eta_sigma']:.3f})" for r in rows]

    valid = np.isfinite(x) & np.isfinite(y)
    plt.figure(figsize=(7, 5))
    plt.scatter(x[valid & ~efficient], y[valid & ~efficient], c="steelblue", s=55, alpha=0.85, label="completed runs")
    plt.scatter(
        x[valid & efficient],
        y[valid & efficient],
        c="crimson",
        marker="D",
        s=75,
        alpha=0.95,
        label="Pareto-efficient",
    )
    for i, txt in enumerate(labels):
        if valid[i]:
            plt.annotate(txt, (x[i], y[i]), fontsize=7, alpha=0.8)
    plt.xlabel(metric_label)
    plt.ylabel("reward_mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[phase44-summary] plot saved: {out_path}")


def main():
    rows = load_rows()
    if not rows:
        print(f"[phase44-summary] no files matched: {OUT_DIR / INPUT_GLOB}")
        return

    pareto_mae = pareto_mask_reward_min_metric(rows, "eta_mae_mean")
    pareto_nll = pareto_mask_reward_min_metric(rows, "eta_nll_mean")
    for row, is_mae, is_nll in zip(rows, pareto_mae, pareto_nll):
        row["pareto_reward_eta_mae"] = bool(is_mae)
        row["pareto_reward_eta_nll"] = bool(is_nll)

    rows.sort(key=lambda r: (float(r["alpha_eta_mu"]), float(r["beta_eta_sigma"])))
    write_csv(rows)
    save_plot(rows, "eta_mae_mean", OUT_PLOT_MAE, "eta_mae_mean", "pareto_reward_eta_mae")
    save_plot(rows, "eta_nll_mean", OUT_PLOT_NLL, "eta_nll_mean", "pareto_reward_eta_nll")


if __name__ == "__main__":
    main()
