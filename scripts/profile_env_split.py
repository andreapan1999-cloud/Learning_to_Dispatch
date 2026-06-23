import time
import numpy as np
from sim.env_dispatch_ppo import DispatchPPOEnv

def main():
    env = DispatchPPOEnv(
        R=4,
        O=4,
        duration=180,
        wait_penalty=-0.01,
        invalid_action_penalty=-0.02,
    )
    env.reset(seed=0)

    n = 100
    t_build = []
    t_base = []

    for i in range(n):
        # --- time base.step(0) ---
        t1 = time.perf_counter()
        _obs2, _r2, term, trunc, _info = env.base.step(0)
        t2 = time.perf_counter()
        t_base.append((t2 - t1) * 1000.0)

        # --- time _build_obs() ---
        t3 = time.perf_counter()
        _ = env._build_obs()
        t4 = time.perf_counter()
        t_build.append((t4 - t3) * 1000.0)

        if term or trunc:
            break

        if i % 10 == 0:
            print("[split] step", i)

    print("base.step avg ms:", float(np.mean(t_base)))
    print("_build_obs avg ms:", float(np.mean(t_build)))

if __name__ == "__main__":
    main()