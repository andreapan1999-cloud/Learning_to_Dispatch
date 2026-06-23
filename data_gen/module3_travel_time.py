from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import csv
import random

from .module1_config import load_config

import numpy as np


def load_edges(edges_path: str | Path) -> List[Tuple[int, int, float]]:
    """
    Load edges from edges.csv (u,v,length_km).
    Returns list of (u, v, length_km).
    """
    edges_path = Path(edges_path)
    edges: List[Tuple[int, int, float]] = []

    with edges_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            u = int(row["u"])
            v = int(row["v"])
            length_km = float(row["length_km"])
            edges.append((u, v, length_km))

    if not edges:
        raise ValueError(f"No edges loaded from {edges_path}")

    return edges


def generate_speed_kmph(t_min: int, cfg: Dict[str, Any], rng: random.Random) -> float:
    """
    Piecewise speed profile:
    - peak hours: lower speed
    - off-peak: higher speed
    plus optional noise
    """
    tt_cfg = cfg.get("travel_time", {})
    base = float(tt_cfg.get("base_speed_kmph", 18.0))
    peak = float(tt_cfg.get("peak_speed_kmph", 12.0))
    peak_hours = tt_cfg.get("peak_hours", [[8, 10], [17, 19]])  # in hours
    noise_std = float(tt_cfg.get("speed_noise_std", 0.0))

    hour = (t_min // 60) % 24
    is_peak = any(int(a) <= hour < int(b) for a, b in peak_hours)

    speed = peak if is_peak else base
    if noise_std > 0:
        speed = max(1e-3, float(rng.normal(loc=speed, scale=noise_std)))

    return speed


def traffic_factor(t_min: int) -> float:
    # morning peak
    if 45 <= t_min <= 75:
        return 1.6
    # evening peak
    if 120 <= t_min <= 150:
        return 1.3
    return 1.0


def generate_travel_times(
    edges: List[Tuple[int, int, float]],
    duration_min: int,
    time_step_min: int,
    cfg: Dict[str, Any],
) -> List[Tuple[int, int, int, float]]:
    """
    Returns rows of (t_min, u, v, travel_time_min) for each edge at each time step.
    """

    # seed for reproducibility
    seed = int(cfg.get("simulation", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    rows: List[Tuple[int, int, int, float]] = []

    for t in range(0, duration_min, time_step_min):
        # baseline speed (km/h), may be time-dependent or constant depending on your generator
        speed_kmph = generate_speed_kmph(t, cfg, rng)

        # time-dependent congestion multiplier (bigger => slower)
        factor = traffic_factor(t)

        for u, v, length_km in edges:
            base_tt_min = (length_km / max(1e-6, speed_kmph)) * 60.0

            # small multiplicative noise so edges are not identical
            noise = float(rng.normal(1.0, 0.03))
            noise = max(0.85, min(1.20, noise))

            tt_min = base_tt_min * factor * noise
            rows.append((t, u, v, tt_min))

    return rows


def traffic_factor(t_min: int) -> float:
    # morning peak
    if 45 <= t_min <= 75:
        return 1.6
    # evening peak
    if 120 <= t_min <= 150:
        return 1.3
    return 1.0

    for t in range(0, duration_min, time_step_min):
        speed_kmph = generate_speed_kmph(t, cfg, rng)

    # time-dependent congestion multiplier (bigger => slower)
    factor = traffic_factor(t)

    for u, v, length_km in edges:
        base_tt_min = (length_km / speed_kmph) * 60.0

        # small multiplicative noise so not all edges are identical
        noise = float(rng.normal(1.0, 0.03))
        noise = max(0.85, min(1.20, noise))

        tt_min = base_tt_min * factor * noise
        rows.append((t, u, v, tt_min))

    return rows


def save_travel_times_csv(
    rows: List[Tuple[int, int, int, float]], out_path: str | Path
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_min", "u", "v", "travel_time_min"])
        for t, u, v, tt in rows:
            w.writerow([t, u, v, tt])

    return out_path


def main() -> None:
    cfg = load_config("config.yaml")

    out_dir = Path(cfg.get("output", {}).get("dir", "outputs"))
    edges_path = out_dir / "edges.csv"

    duration = int(cfg.get("simulation", {}).get("duration", 180))

    tt_cfg = cfg.get("travel_time", {})
    time_step = int(tt_cfg.get("time_step_min", 15))
    out_name = tt_cfg.get("filename", "travel_times.csv")
    out_path = out_dir / out_name

    edges = load_edges(edges_path)
    rows = generate_travel_times(edges, duration, time_step, cfg)
    save_travel_times_csv(rows, out_path)

    print("=== Travel times generated ===")
    print("Edges:", len(edges))
    print("Duration(min):", duration, "Step(min):", time_step)
    print("Rows:", len(rows))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()


def traffic_factor(t_min: int) -> float:
    """
    Simple peak-hour congestion profile (paper-friendly).
    """
    # morning peak
    if 45 <= t_min <= 75:
        return 1.6
    # evening peak
    if 120 <= t_min <= 150:
        return 1.3
    return 1.0
