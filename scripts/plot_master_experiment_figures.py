from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


METHOD_ORDER = ["greedy", "ablation_norisk", "pareto", "ueta", "random"]
METHOD_LABELS = {
    "greedy": "Greedy",
    "ablation_norisk": "No Risk",
    "pareto": "Pareto",
    "ueta": "UETA",
    "random": "Random",
}
NOISE_ORDER = ["low", "mid", "high"]
SCALE_ORDER = ["small", "main", "large"]


def _import_matplotlib():
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def load_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return float("nan")
    return float(val)


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize(rows: list[dict], filters: dict[str, str], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        keep = True
        for key, expected in filters.items():
            if str(row.get(key, "")).strip() != str(expected).strip():
                keep = False
                break
        if not keep:
            continue
        gkey = tuple(str(row.get(key, "")).strip() for key in group_keys)
        grouped[gkey]["reward"].append(as_float(row, "reward"))
        grouped[gkey]["eta_mae"].append(as_float(row, "eta_mae"))
        grouped[gkey]["eta_rmse"].append(as_float(row, "eta_rmse"))
        grouped[gkey]["eta_nll"].append(as_float(row, "eta_nll"))

    out = []
    for gkey, metrics in grouped.items():
        item = {k: v for k, v in zip(group_keys, gkey)}
        for metric_name, values in metrics.items():
            mean, std = mean_std(values)
            item[f"{metric_name}_mean"] = mean
            item[f"{metric_name}_std"] = std
            item[f"{metric_name}_n"] = int(np.isfinite(np.asarray(values, dtype=np.float64)).sum())
        out.append(item)
    return out


def style_axes(ax) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    ax.tick_params(top=True, right=True)


def plot_noise_two_panel(rows: list[dict], out_path: Path) -> None:
    plt = _import_matplotlib()
    summary = summarize(rows, filters={"setting": "noise"}, group_keys=["noise", "method"])
    methods = [m for m in METHOD_ORDER if m in {row["method"] for row in summary} and m != "random"]
    x = np.arange(len(NOISE_ORDER), dtype=np.float64)
    colors = {
        "greedy": "#1f3c88",
        "ablation_norisk": "#8f2d56",
        "pareto": "#0f8b8d",
        "ueta": "#b83b5e",
    }
    markers = {"greedy": "o", "ablation_norisk": "s", "pareto": "^", "ueta": "D"}

    lookup = {(row["noise"], row["method"]): row for row in summary}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.8), sharex=True)
    metric_specs = [
        ("reward_mean", "reward_std", "Reward"),
        ("eta_mae_mean", "eta_mae_std", "ETA MAE"),
    ]

    for ax, (mean_key, std_key, ylabel) in zip(axes, metric_specs):
        for method in methods:
            means = []
            errs = []
            for noise in NOISE_ORDER:
                row = lookup.get((noise, method), {})
                means.append(float(row.get(mean_key, np.nan)))
                errs.append(float(row.get(std_key, np.nan)))
            ax.errorbar(
                x,
                means,
                yerr=errs,
                color=colors.get(method, "#333333"),
                marker=markers.get(method, "o"),
                linewidth=1.8,
                markersize=5,
                capsize=3,
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_ylabel(ylabel)
        style_axes(ax)

    axes[0].legend(loc="best", ncol=2)
    axes[0].text(0.5, -0.28, "(a)", transform=axes[0].transAxes, ha="center", va="center", fontsize=14)
    axes[1].text(0.5, -0.28, "(b)", transform=axes[1].transAxes, ha="center", va="center", fontsize=14)
    axes[1].set_xticks(x, [name.capitalize() for name in NOISE_ORDER])
    axes[1].set_xlabel("Noise Level")
    fig.tight_layout(h_pad=1.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_dual_axis_scale_bars(rows: list[dict], setting: str, out_path: Path) -> None:
    plt = _import_matplotlib()
    summary = summarize(rows, filters={"setting": setting}, group_keys=["scale", "method"])
    lookup = {(row["scale"], row["method"]): row for row in summary}
    methods = [m for m in METHOD_ORDER if m in {row["method"] for row in summary} and m != "random"]
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.34

    fig, axes = plt.subplots(1, len(SCALE_ORDER), figsize=(13.5, 3.9))
    if len(SCALE_ORDER) == 1:
        axes = [axes]

    reward_face = "#ffffff"
    reward_edge = "#4a4a4a"
    eta_face = "#f1e3b6"
    eta_edge = "#111111"

    legend_handles = None
    legend_labels = None

    for idx, (ax, scale) in enumerate(zip(axes, SCALE_ORDER)):
        ax2 = ax.twinx()
        reward_means = []
        reward_errs = []
        eta_means = []
        eta_errs = []
        for method in methods:
            row = lookup.get((scale, method), {})
            reward_means.append(float(row.get("reward_mean", np.nan)))
            reward_errs.append(float(row.get("reward_std", np.nan)))
            eta_means.append(float(row.get("eta_mae_mean", np.nan)))
            eta_errs.append(float(row.get("eta_mae_std", np.nan)))

        bars_reward = ax.bar(
            x - width / 2,
            reward_means,
            width=width,
            yerr=reward_errs,
            capsize=3,
            facecolor=reward_face,
            edgecolor=reward_edge,
            linewidth=1.5,
            label="Reward",
        )
        bars_eta = ax2.bar(
            x + width / 2,
            eta_means,
            width=width,
            yerr=eta_errs,
            capsize=3,
            facecolor=eta_face,
            edgecolor=eta_edge,
            linewidth=1.3,
            label="ETA MAE",
        )

        ax.set_xticks(x, [METHOD_LABELS.get(method, method) for method in methods], rotation=18)
        ax.set_title(scale.capitalize())
        if idx == 0:
            ax.set_ylabel("Reward")
        if idx == len(SCALE_ORDER) - 1:
            ax2.set_ylabel("ETA MAE")

        style_axes(ax)
        style_axes(ax2)
        ax.text(0.5, -0.33, f"({chr(ord('a') + idx)})", transform=ax.transAxes, ha="center", va="center", fontsize=14)

        if legend_handles is None:
            legend_handles = [bars_reward[0], bars_eta[0]]
            legend_labels = ["Reward", "ETA MAE"]

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"{setting.capitalize()} Experiments", y=1.08, fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make publication-style figures from master_experiment_table.csv")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/william/Desktop/實驗輸出/master_experiment_table.csv",
        help="Path to master_experiment_table.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/paper_figures",
        help="Directory to save generated figures",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = load_rows(input_path)

    plot_noise_two_panel(rows, out_dir / "noise_two_panel.png")
    plot_dual_axis_scale_bars(rows, setting="final", out_path=out_dir / "final_scale_dual_axis.png")
    plot_dual_axis_scale_bars(rows, setting="runtime", out_path=out_dir / "runtime_scale_dual_axis.png")


if __name__ == "__main__":
    main()
