import argparse
from collections import Counter

from stable_baselines3 import PPO
try:
    from sb3_contrib import MaskablePPO
    _HAS_MASK = True
except Exception:
    MaskablePPO = None
    _HAS_MASK = False
from stable_baselines3.common.vec_env import DummyVecEnv

from sim.env_dispatch_ppo import DispatchPPOEnv


def make_env():
    return DispatchPPOEnv(
        R=4,
        O=4,
        duration=180,
        wait_penalty=-0.01,
        invalid_action_penalty=-0.02,
    )


def _is_dict_obs_space(model) -> bool:
    return hasattr(model.observation_space, "spaces")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    print(f"[diagnose] Using model: {args.model}")

    # Try MaskablePPO first (models trained with sb3-contrib)
    if _HAS_MASK:
        try:
            model = MaskablePPO.load(args.model)
            algo = "MaskablePPO"
        except Exception:
            model = PPO.load(args.model)
            algo = "PPO"
    else:
        model = PPO.load(args.model)
        algo = "PPO"

    print(f"[diagnose] Loaded algo: {algo}")
    dict_model = _is_dict_obs_space(model)
    print(f"[diagnose] Model obs space type: {'Dict' if dict_model else 'Box'}")

    venv = DummyVecEnv([make_env])
    obs = venv.reset()

    done = [False]
    total_reward = 0.0
    steps = 0
    ctr = Counter()

    while not done[0]:
        obs_in = obs
        if (not dict_model) and isinstance(obs, dict):
            obs_in = obs["flat"]

        action, _ = model.predict(obs_in, deterministic=args.deterministic)
        ctr[int(action[0])] += 1

        obs, reward, done, info = venv.step(action)
        total_reward += float(reward[0])
        steps += 1

    print(f"[diagnose] Episode steps: {steps}")
    print(f"[diagnose] Total reward: {total_reward:.3f}")
    print(f"[diagnose] Mean step reward: {total_reward / max(1, steps):.3f}")
    print(f"[diagnose] Action 0 ratio: {ctr.get(0,0)/max(1,steps):.3f}")
    print(f"[diagnose] Top actions: {ctr.most_common(10)}")


if __name__ == "__main__":
    main()
