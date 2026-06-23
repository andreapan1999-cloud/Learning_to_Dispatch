# train/eval_dispatch_random.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sim.env_dispatch_ppo import DispatchPPOEnv


def make_env():
    return DispatchPPOEnv(
        data_dir="outputs",
        duration=180,
        use_gnn_eta=True,
        calibrate_gnn=True,
        calib_samples=2000,
        R=4,
        O=4,
        invalid_action_penalty=-2.0,
        wait_penalty=-0.01,
    )


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def reset_vecenv(venv, seed: int):
    try:
        return venv.reset(seed=seed)
    except TypeError:
        try:
            venv.seed(seed)
        except Exception:
            pass
        return venv.reset()


def run_one_episode_random(venv, seed: int) -> float:
    _ = reset_vecenv(venv, seed)

    done = [False]
    ep_reward = 0.0

    while not done[0]:
        action = np.array([venv.action_space.sample()])
        _, reward, done, _info = venv.step(action)
        ep_reward += float(reward[0])

    return ep_reward


def evaluate_random(
    episodes: int,
    seed_base: int,
    vecnorm_path: str,
) -> Tuple[np.ndarray, List[int]]:
    rewards: List[float] = []
    seeds: List[int] = []

    for i in range(episodes):
        seed = seed_base + i
        set_global_seeds(seed)

        venv = DummyVecEnv([make_env])

        if vecnorm_path and Path(vecnorm_path).exists():
            venv = VecNormalize.load(vecnorm_path, venv)
            venv.training = False
            venv.norm_reward = False

        r = run_one_episode_random(venv, seed=seed)
        rewards.append(r)
        seeds.append(seed)

        try:
            venv.close()
        except Exception:
            pass

    return np.asarray(rewards, dtype=np.float64), seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed_base", type=int, default=2000)
    parser.add_argument("--vecnorm", type=str, default="")
    args = parser.parse_args()

    rewards, seeds = evaluate_random(
        episodes=args.episodes,
        seed_base=args.seed_base,
        vecnorm_path=args.vecnorm,
    )

    print(f"episodes: {len(rewards)}")
    if seeds:
        print(f"seeds: {seeds[0]}..{seeds[-1]}")
    print(
        "reward mean/std/min/max: "
        f"{rewards.mean():.3f} / {rewards.std():.3f} / {rewards.min():.3f} / {rewards.max():.3f}"
    )


if __name__ == "__main__":
    main()