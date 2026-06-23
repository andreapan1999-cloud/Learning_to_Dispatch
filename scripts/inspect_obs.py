# scripts/inspect_obs.py

from sim.env_dispatch_ppo import DispatchPPOEnv
import numpy as np
import torch


def describe(x, prefix=""):
    if isinstance(x, dict):
        print(f"{prefix}dict with keys:")
        for k, v in x.items():
            describe(v, prefix=f"{prefix}  [{k}] ")
    elif isinstance(x, np.ndarray):
        print(f"{prefix}ndarray shape={x.shape} dtype={x.dtype}")
    elif torch.is_tensor(x):
        print(f"{prefix}torch.Tensor shape={tuple(x.shape)} dtype={x.dtype}")
    else:
        print(f"{prefix}{type(x)} value={x}")


def main():
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
    )

    obs, info = env.reset()

    print("=== Observation structure ===")
    describe(obs)

    print("\n=== Action space ===")
    print(env.action_space)

    print("\n=== Observation space ===")
    print(env.observation_space)


if __name__ == "__main__":
    main()