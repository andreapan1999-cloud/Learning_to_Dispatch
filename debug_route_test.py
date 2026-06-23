from sim.env_dispatch import DispatchEnv


def main():
    env = DispatchEnv(data_dir="outputs", duration=60, debug=False)
    env.reset(seed=0)

    while (not env.idle_riders or not env.pending_orders) and env.t < env.duration:
        env.step(0)

    if not env.idle_riders or not env.pending_orders:
        print("no_rider_or_order")
        return

    rider = env.idle_riders[0]
    rid = int(rider["rider_id"])
    order1 = env.pending_orders[0]

    env.routes.setdefault(rid, [])
    env.rider_pos.setdefault(rid, int(rider["init_node"]))

    p1, d1, delta1 = env._best_order_insertion(rider, order1)
    print("order1_insert", p1, d1, round(float(delta1), 4))
    env.dispatch(rider, order1, total_eta_min=0.0)
    print("route_after_order1", env.routes[rid])

    while not env.pending_orders and env.t < env.duration:
        env.step(0)

    if env.pending_orders:
        order2 = env.pending_orders[0]
        p2, d2, delta2 = env._best_order_insertion(rider, order2)
        print("order2_insert", p2, d2, round(float(delta2), 4))
        env.dispatch(rider, order2, total_eta_min=0.0)
        print("route_after_order2", env.routes[rid])

    pos, delta = env.greedy_best_insertion(rid, int(rider["init_node"]))
    print("single_node_insert", pos, round(float(delta), 4))


if __name__ == "__main__":
    main()
