import time
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

    t_mask = 0.0
    t_step = 0.0
    n = 0

    print("[profile] start...")

    for i in range(100):   # 🔴 先别跑 2000
        t0 = time.perf_counter()
        if hasattr(env, "action_masks"):
            _ = env.action_masks()
        t1 = time.perf_counter()

        a = env.action_space.sample()
        _obs, _r, term, trunc, _info = env.step(a)
        t2 = time.perf_counter()

        t_mask += (t1 - t0)
        t_step += (t2 - t1)
        n += 1

        if i % 10 == 0:
            print(f"[profile] step {i}")

        if term or trunc:
            env.reset(seed=0)

    print("calls:", n)
    print("mask avg ms:", (t_mask / n) * 1000.0)
    print("step avg ms:", (t_step / n) * 1000.0)

if __name__ == "__main__":
    main()