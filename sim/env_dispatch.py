from __future__ import annotations

import bisect
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import math
from typing import Optional



# --- GNN calibration global cache (process-level) ---
_GNN_CALIB_CACHE = {}

class DispatchEnv(gym.Env):
    """
    Dispatch Environment (clean rebuild)

    Observation: [t, #pending_orders, #idle_riders]
    Action:
      0 = do nothing
      1 = dispatch first order to first idle rider
    """

    metadata = {"render_modes": []}

    def __init__(self, data_dir: str, duration: int, use_gnn_eta: bool = False,
             gnn_ckpt: str = "outputs/eta_gnn.pt",
             calibrate_gnn: bool = True, calib_samples: int = 400,
             verbose: bool = False, debug: bool = False,
             orders_file: str = "orders.csv", riders_file: str = "riders.csv",
             nodes_file: str = "nodes.csv", travel_times_file: str = "travel_times.csv",
             edges_file: str = "edges.csv", profile_hotspots: bool = False):

        super().__init__()
        self.verbose = bool(verbose)
        self.debug = bool(debug)
        self.profile_hotspots = bool(profile_hotspots)
        self._profile_stats: Dict[str, Dict[str, float]] = {}

        self.data_dir = Path(data_dir)
        self.duration = int(duration)
        self.orders_path = self.data_dir / str(orders_file)
        self.riders_path = self.data_dir / str(riders_file)
        self.nodes_path = self.data_dir / str(nodes_file)
        self.travel_times_path = self.data_dir / str(travel_times_file)
        self.edges_path = self.data_dir / str(edges_file)

        # ---- load CSV first (needed by gnn init & calib) ----
        self.orders = self._load_orders(self.orders_path)
        self.riders = self._load_riders(self.riders_path)
        self.node_xy = self._load_nodes(self.nodes_path)
        self.travel_time = self._load_travel_times(self.travel_times_path)
        self._travel_time_bucket_times = {
            k: [int(tb) for tb, _tt in entries] for k, entries in self.travel_time.items()
        }
        self.G = self._build_graph(self.edges_path)

        # ---- gnn flags ----
        self.use_gnn_eta = bool(use_gnn_eta)
        self.gnn_ckpt = Path(gnn_ckpt)
        self.calibrate_gnn = bool(calibrate_gnn)
        self.calib_samples = int(calib_samples)

        # ---- default calib params ----
        self.gnn_alpha = 1.0
        self.gnn_beta = 0.0
        self.gnn_bucket_scale = None
        self.edge_scale = None

        # routing state
        self.routes = {}              # rider_id -> list of (type, order_id, node)
        self.assigned_orders = {}     # rider_id -> list of order dict
        self.load = {}                # rider_id -> int
        self.order_status = {}        # order_id -> str
        self.default_capacity = 3
        self.capacity = {}
        self.routes = {}  # rid -> list of stops
        self.order_state = {}  # oid -> "waiting"|"assigned"|"picked"|"delivered"
        self.rider_eta_remaining = {}  # rid -> minutes remaining to next stop
       
        # ---- init gnn + calibration (cached) ----
        if self.use_gnn_eta:
            self._init_gnn_eta()

            if self.calibrate_gnn:
                cache_key = (str(self.gnn_ckpt.resolve()), str(self.data_dir.resolve()), int(self.duration), int(self.calib_samples))
                cached = _GNN_CALIB_CACHE.get(cache_key)

                if cached is not None:
                    # reuse cached calibration
                    self.gnn_alpha = cached.get("gnn_alpha", 1.0)
                    self.gnn_beta = cached.get("gnn_beta", 0.0)
                    self.gnn_bucket_scale = cached.get("gnn_bucket_scale", None)
                    self.edge_scale = cached.get("edge_scale", None)
                else:
                    # compute once
                    self._fit_gnn_calibration()
                    _GNN_CALIB_CACHE[cache_key] = {
                        "gnn_alpha": self.gnn_alpha,
                        "gnn_beta": self.gnn_beta,
                        "gnn_bucket_scale": getattr(self, "gnn_bucket_scale", None),
                        "edge_scale": getattr(self, "edge_scale", None),
                    }

        
        self.t = 0
        self.pending_orders: List[Dict[str, Any]] = []
        self.idle_riders: List[Dict[str, Any]] = []
        self.rider_pos: Dict[int, int] = {}

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.array([self.duration, 10000, 10000], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.busy_riders = []              # list of rider dicts
        self.busy_until = {}               # rider_id -> t_min
        self.orders_ep = None
        self.riders_ep = None
        self._orders_by_t = None
        self._riders_by_t = None
        self._sp_cache = {}          
        self._sp_cache_max = 50000    
        self._sp_cache_bucket = 5     

    def _record_hotspot(self, name: str, elapsed_s: float) -> None:
        if not self.profile_hotspots:
            return
        stat = self._profile_stats.setdefault(
            name,
            {"calls": 0.0, "total_s": 0.0, "max_ms": 0.0},
        )
        stat["calls"] += 1.0
        stat["total_s"] += float(elapsed_s)
        stat["max_ms"] = max(float(stat["max_ms"]), float(elapsed_s) * 1e3)
        calls = int(stat["calls"])
        if calls <= 5 or (calls % 500 == 0):
            avg_ms = 1e3 * float(stat["total_s"]) / max(1, calls)
            print(
                "[profile]",
                f"fn={name}",
                f"calls={calls}",
                f"avg_ms={avg_ms:.3f}",
                f"max_ms={float(stat['max_ms']):.3f}",
                flush=True,
            )
       
 
    # ---------------- CSV loaders ----------------
    def _load_orders(self, path: Path) -> List[Dict[str, Any]]:
        orders = []
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                orders.append(
                    {
                        "order_id": int(row["order_id"]),
                        "t_min": int(row["t_min"]),
                        "origin": int(row["origin"]),
                        "dest": int(row["dest"]),
                    }
                )
        return orders

    def _load_riders(self, path: Path) -> List[Dict[str, Any]]:
        riders = []
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                riders.append(
                    {
                        "rider_id": int(row["rider_id"]),
                        "start_min": int(row["start_min"]),
                        "end_min": int(row["end_min"]),
                        "init_node": int(row["init_node"]),
                        "speed_factor": float(row["speed_factor"]),
                    }
                )
        return riders

    def _load_nodes(self, path: Path) -> Dict[int, tuple[float, float]]:
        node_xy: Dict[int, tuple[float, float]] = {}
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                nid = int(row["node_id"])
                node_xy[nid] = (float(row["x_km"]), float(row["y_km"]))
        return node_xy

    def _load_travel_times(self, path: Path):
        """
        travel_time[(u, v)] = list of (t_min, travel_time_min), sorted by t_min
        """
        table: Dict[tuple[int, int], list[tuple[int, float]]] = {}
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                u = int(row["u"])
                v = int(row["v"])
                t = int(float(row["t_min"]))
                tt = float(row["travel_time_min"])
                table.setdefault((u, v), []).append((t, tt))

        for k in table:
            table[k].sort(key=lambda x: x[0])
        return table
    def _build_graph(self, edges_path: Path) -> nx.DiGraph:
        G = nx.DiGraph()
        with edges_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                u = int(row["u"])
                v = int(row["v"])
                # store edge; weight will be time-dependent at query time
                G.add_edge(u, v)
        return G

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            rng = np.random.default_rng(int(seed))
        else:
            rng = np.random.default_rng()

        self._ep_rng = rng

        self.t = 0
        self.pending_orders = []
        self.idle_riders = []
        self.rider_pos = {}
        self.busy_riders = []
        self.busy_until = {}
        self.routes.clear()
        self.order_state.clear()
        self.rider_eta_remaining.clear()
        self.load = {}
        if isinstance(self.capacity, dict):
            self.capacity = {}

        orders = list(self.orders)
        riders = list(self.riders)

        rng.shuffle(orders)
        rng.shuffle(riders)

        self.orders_ep = orders
        self.riders_ep = riders

        orders_by_t = {}
        for o in self.orders_ep:
            t0 = int(o["t_min"])
            orders_by_t.setdefault(t0, []).append(o)
        self._orders_by_t = orders_by_t

        riders_by_t = {}
        for r in self.riders_ep:
            t0 = int(r["start_min"])
            riders_by_t.setdefault(t0, []).append(r)
        self._riders_by_t = riders_by_t

        return self._get_obs(), {}

    def _eta_min(self, a: int, b: int, speed_kmph: float = 18.0) -> float:
        (x1, y1) = self.node_xy[a]
        (x2, y2) = self.node_xy[b]
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return (dist / speed_kmph) * 60.0

    def _edge_eta_time_dependent_lookup(self, u: int, v: int, t_min: int) -> float:
        """
        CSV lookup only (no GNN).
        Finds the latest time bucket <= t_min for edge (u,v).
        """
        key = (u, v)
        entries = self.travel_time.get(key)
        if not entries:
            # fallback if missing
            return 9999.0

        bucket_times = self._travel_time_bucket_times.get(key)
        if not bucket_times:
            return float(entries[0][1])
        idx = bisect.bisect_right(bucket_times, int(t_min)) - 1
        if idx < 0:
            idx = 0
        return float(entries[idx][1])

    def _edge_eta_time_dependent(self, u: int, v: int, t_min: int) -> float:
        # baseline: CSV lookup
        if not getattr(self, "use_gnn_eta", False):
            return float(self._edge_eta_time_dependent_lookup(u, v, t_min))

        # GNN: raw -> calibrated
        raw = float(self._edge_eta_gnn_raw(u, v, t_min))

        bucket_size = 45
        b = int((t_min // bucket_size) * bucket_size)
        scale = getattr(self, "gnn_bucket_scale", {}).get(b, self.gnn_alpha)

        edge_s = getattr(self, "edge_scale", {}).get((u, v), 1.0)
        eta = scale * raw * edge_s

        return max(0.1, float(eta))




    def _edge_eta_gnn_raw(self, u: int, v: int, t_min: int) -> float:
        import torch

        t_feat = torch.tensor(
            [[t_min / max(1.0, float(self.duration))]],
            dtype=torch.float32,
            device=self.torch_device,
        )
        uv = torch.tensor([[u, v]], dtype=torch.long, device=self.torch_device)

        with torch.no_grad():
            pred = self.gnn_model(self.gnn_data.x, self.gnn_data.edge_index, uv, t_feat)
        return float(pred.item())



    def _init_gnn_eta(self) -> None:
        import torch
        from torch_geometric.data import Data
        from models.eta_gnn import EtaGNN

        # device
        self.torch_device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # node features (x,y)
        num_nodes = max(self.node_xy.keys()) + 1
        x = torch.zeros((num_nodes, 2), dtype=torch.float32)
        for nid, (xx, yy) in self.node_xy.items():
            x[nid, 0] = float(xx)
            x[nid, 1] = float(yy)

        # edge_index from graph
        us, vs = [], []
        for (u, v) in self.G.edges():
            us.append(int(u))
            vs.append(int(v))
        edge_index = torch.tensor([us, vs], dtype=torch.long)

        self.gnn_data = Data(x=x, edge_index=edge_index).to(self.torch_device)

        self.gnn_model = EtaGNN(in_dim=2, hid_dim=64).to(self.torch_device)
        self.gnn_model.load_state_dict(
            torch.load(self.gnn_ckpt, map_location=self.torch_device)
        )
        self.gnn_model.eval()

    def _fit_gnn_calibration(self) -> None:
        import random
        from collections import defaultdict

        keys = list(self.travel_time.keys())
        if not keys:
            return

        bucket_size = 45
        T_choices = list(range(0, int(self.duration), 15))
        random.shuffle(keys)

        # -------- pass 1: fit bucket scales --------
        b_num = defaultdict(float)
        b_den = defaultdict(float)

        samples = []
        n = 0
        for (u, v) in keys:
            for t in T_choices:
                y_true = float(self._edge_eta_time_dependent_lookup(u, v, t))
                y_pred = float(self._edge_eta_gnn_raw(u, v, t))
                b = int((t // bucket_size) * bucket_size)

                b_num[b] += y_pred * y_true
                b_den[b] += y_pred * y_pred

                samples.append((u, v, t, y_true, y_pred))
                n += 1
                if n >= int(self.calib_samples):
                    break
            if n >= int(self.calib_samples):
                break

        self.gnn_bucket_scale = {}
        for b, den in b_den.items():
            scale = 1.0 if den < 1e-9 else (b_num[b] / den)
            if scale <= 0:
                scale = 1.0
            self.gnn_bucket_scale[int(b)] = float(scale)

        # fallback alpha
        scales = list(self.gnn_bucket_scale.values())
        self.gnn_alpha = float(sum(scales) / max(1, len(scales)))
        self.gnn_beta = 0.0

        # -------- pass 2: fit edge scales using bucket-scaled preds --------
        e_num = defaultdict(float)
        e_den = defaultdict(float)

        for (u, v, t, y_true, y_pred) in samples:
            b = int((t // bucket_size) * bucket_size)
            scale_t = self.gnn_bucket_scale.get(b, self.gnn_alpha)

            e_num[(u, v)] += y_true
            e_den[(u, v)] += (scale_t * y_pred)

        self.edge_scale = {}
        for k, den in e_den.items():
            if den < 1e-9:
                continue
            s = e_num[k] / den
            if s < 0.5:
                s = 0.5
            if s > 2.0:
                s = 2.0
            self.edge_scale[k] = float(s)

        if self.verbose:
            print(
                f"[GNN-CALIB] bucket_scales={self.gnn_bucket_scale} "
                f"(fallback alpha={self.gnn_alpha:.4f}) using {len(samples)} samples | "
                f"edge_scales: {len(self.edge_scale)} edges"
            )
    


    def _shortest_eta(self, src: int, dst: int, t: int) -> float:
        """
        Time-dependent shortest path ETA from src to dst at time t.
        Cached by (src,dst,time-bucket) to speed up training.
        """
        t0 = time.perf_counter() if self.profile_hotspots else 0.0
        if src == dst:
            return 0.0

        tb = (int(t) // int(self._sp_cache_bucket)) * int(self._sp_cache_bucket)
        key = (int(src), int(dst), int(tb))
        hit = self._sp_cache.get(key)
        if hit is not None:
            return float(hit)

        def w(u: int, v: int, d: Dict[str, Any]) -> float:
            return float(self._edge_eta_time_dependent(u, v, tb))

        try:
            val = float(nx.shortest_path_length(self.G, src, dst, weight=w, method="dijkstra"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            val = float(self._eta_min(src, dst))

        # very cheap cache management
        if len(self._sp_cache) >= int(self._sp_cache_max):
            self._sp_cache.clear()

        self._sp_cache[key] = float(val)
        self._record_hotspot("_shortest_eta", time.perf_counter() - t0)
        return float(val)

    def _get_obs(self):
        return np.array(
            [self.t, len(self.pending_orders), len(self.idle_riders)],
            dtype=np.float32,
        )

    def compute_route_cost(self, route) -> float:
        total = 0.0
        if route is None:
            return 0.0
        if len(route) < 2:
            return 0.0
        for i in range(len(route) - 1):
            u = int(route[i])
            v = int(route[i + 1])
            total += float(self._shortest_eta(u, v, self.t))
        return float(total)

    def _route_node_sequence(self, rider_id: int, rider: Optional[Dict[str, Any]] = None) -> List[int]:
        rid = int(rider_id)
        if rider is None:
            rider = next((r for r in (self.idle_riders + self.busy_riders) if int(r["rider_id"]) == rid), None)
        cur = int(self.rider_pos.get(rid, int(rider["init_node"]) if rider is not None else 0))
        route = self.routes.get(rid, [])
        nodes = [cur]
        for stop in route:
            if isinstance(stop, tuple) and len(stop) >= 3:
                nodes.append(int(stop[2]))
            else:
                nodes.append(int(stop))
        return nodes

    def greedy_best_insertion(self, rider_id, new_node):
        rid = int(rider_id)
        route_nodes = self._route_node_sequence(rid)
        if len(route_nodes) == 0:
            return 0, 0.0

        best_delta = float("inf")
        best_pos = 0
        old_cost = self.compute_route_cost(route_nodes)

        for pos in range(1, len(route_nodes) + 1):
            candidate = route_nodes[:pos] + [int(new_node)] + route_nodes[pos:]
            new_cost = self.compute_route_cost(candidate)
            delta = float(new_cost - old_cost)
            if delta < best_delta:
                best_delta = delta
                # convert node-seq insertion pos -> stop-list insertion index
                best_pos = int(pos - 1)

        return int(best_pos), float(best_delta)

    def _best_order_insertion(self, rider: Dict[str, Any], order: Dict[str, Any]) -> Tuple[int, int, float]:
        rid = int(rider["rider_id"])
        route = list(self.routes.get(rid, []))
        n = len(route)
        old_cost = self.compute_route_cost(self._route_nodes_from_stops(rider, route))
        max_pos = getattr(self, "greedy_top_k_positions", None)
        if max_pos is not None:
            max_pos = max(1, int(max_pos))

        best_delta = float("inf")
        best_pick_idx = 0
        best_drop_idx = 1
        oid = int(order.get("order_id", -1))
        pstop = ("pickup", oid, int(order["origin"]))
        dstop = ("dropoff", oid, int(order["dest"]))

        pick_positions = range(0, n + 1)
        if max_pos is not None:
            pick_positions = range(0, min(n + 1, max_pos))
        for p in pick_positions:
            drop_stop = n + 2
            if max_pos is not None:
                drop_stop = min(n + 2, p + 1 + max_pos)
            for d in range(p + 1, drop_stop):
                cand = list(route)
                cand.insert(p, pstop)
                cand.insert(d, dstop)
                if not self._is_capacity_feasible_for_stops(rid, cand):
                    continue
                new_cost = self.compute_route_cost(self._route_nodes_from_stops(rider, cand))
                delta = float(new_cost - old_cost)
                if delta < best_delta:
                    best_delta = delta
                    best_pick_idx = int(p)
                    best_drop_idx = int(d)

        return int(best_pick_idx), int(best_drop_idx), float(best_delta)

    def _route_nodes_from_stops(self, rider: Dict[str, Any], stops: List[Tuple[str, int, int]]) -> List[int]:
        rid = int(rider["rider_id"])
        cur = int(self.rider_pos.get(rid, int(rider.get("init_node", 0))))
        nodes = [cur]
        for stop in stops:
            nodes.append(int(stop[2]))
        return nodes

    def _rider_capacity(self, rider_id: int) -> int:
        rid = int(rider_id)
        if isinstance(self.capacity, dict):
            return int(self.capacity.get(rid, self.default_capacity))
        return int(self.capacity)

    def _is_capacity_feasible_for_stops(self, rider_id: int, stops: List[Tuple[str, int, int]]) -> bool:
        rid = int(rider_id)
        cap = int(self._rider_capacity(rid))
        load_sim = int(self.load.get(rid, 0))
        if load_sim < 0 or load_sim > cap:
            return False
        for stop_type, _oid, _node in stops:
            st = str(stop_type)
            if st == "pickup":
                load_sim += 1
                if load_sim > cap:
                    return False
            elif st == "dropoff":
                load_sim -= 1
                if load_sim < 0:
                    return False
        return True

    def _order_insertion_delta_for_positions(
        self,
        rider: Dict[str, Any],
        order: Dict[str, Any],
        pickup_idx: int,
        dropoff_idx: int,
    ) -> float:
        t0 = time.perf_counter() if self.profile_hotspots else 0.0
        rid = int(rider["rider_id"])
        route = list(self.routes.get(rid, []))
        n = len(route)
        p = int(pickup_idx)
        d = int(dropoff_idx)
        if p < 0 or p > n:
            raise RuntimeError(f"Invalid pickup insertion index: {p}, route_len={n}")
        if d <= p or d > (n + 1):
            raise RuntimeError(f"Invalid dropoff insertion index: {d}, route_len={n}, pickup_idx={p}")

        old_cost = self.compute_route_cost(self._route_nodes_from_stops(rider, route))
        cand = list(route)
        oid = int(order.get("order_id", -1))
        cand.insert(p, ("pickup", oid, int(order["origin"])))
        cand.insert(d, ("dropoff", oid, int(order["dest"])))
        if not self._is_capacity_feasible_for_stops(rid, cand):
            raise RuntimeError(
                f"Infeasible capacity for rider {rid} at insertion (pickup={p}, dropoff={d})"
            )
        new_cost = self.compute_route_cost(self._route_nodes_from_stops(rider, cand))
        self._record_hotspot("_order_insertion_delta_for_positions", time.perf_counter() - t0)
        return float(new_cost - old_cost)
    
    def dispatch(
        self,
        rider: Dict[str, Any],
        order: Dict[str, Any],
        total_eta_min: float,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        rid = int(rider["rider_id"])
        oid = int(order.get("order_id", -1))
        # per-rider capacity/load setup
        if isinstance(self.capacity, dict):
            self.capacity.setdefault(rid, int(self.default_capacity))
        self.load.setdefault(rid, 0)

        # debug (first few only)
        if self.debug:
            self._dbg_load_inc = getattr(self, "_dbg_load_inc", 0) + 1
            if self._dbg_load_inc <= 10:
                print("[dbg load++]", "t", int(self.t), "rid", rid,
                    "load", self.load[rid], "cap", int(self._rider_capacity(rid)))

        if order in self.pending_orders:
            self.pending_orders.remove(order)

        if rider in self.idle_riders:
            self.idle_riders.remove(rider)

        if rider not in self.busy_riders:
            self.busy_riders.append(rider)

        self.order_status[oid] = "assigned"
        self.assigned_orders.setdefault(rid, []).append(order)

        origin = int(order["origin"])
        dest = int(order["dest"])

        self.routes.setdefault(rid, [])
        if meta and ("pickup_insert_idx" in meta) and ("dropoff_insert_idx" in meta):
            pickup_idx = int(meta["pickup_insert_idx"])
            dropoff_idx = int(meta["dropoff_insert_idx"])
            route_delta = self._order_insertion_delta_for_positions(rider, order, pickup_idx, dropoff_idx)
        else:
            pickup_idx, dropoff_idx, route_delta = self._best_order_insertion(rider, order)
            if not np.isfinite(route_delta):
                raise RuntimeError(f"No feasible insertion for rider {rid} under capacity constraints")
        self.routes[rid].insert(int(pickup_idx), ("pickup", oid, origin))
        self.routes[rid].insert(int(dropoff_idx), ("dropoff", oid, dest))

        rider["last_order_id"] = oid
        rider["last_dispatch_t"] = int(self.t)
        rider["last_route_delta"] = float(route_delta)

        if meta:
            rider["last_dispatch_meta"] = dict(meta)
            rider["last_dispatch_meta"]["route_delta"] = float(route_delta)
        
        if float(self.rider_eta_remaining.get(rid, 0.0)) <= 0.0:
            cur = int(self.rider_pos.get(rid, int(rider.get("init_node", 0))))
            first_node = int(self.routes[rid][0][2])
            tt = float(self._shortest_eta(cur, first_node, self.t))
            tt /= max(1e-6, float(rider["speed_factor"]))
            self.rider_eta_remaining[rid] = float(max(1.0, tt))

    def step(self, action: int):
        if self._orders_by_t is None or self._riders_by_t is None:
            self.reset()

        reward = 0.0
        self.t += 1

        # new orders
        for o in self._orders_by_t.get(self.t, []):
            self.pending_orders.append(o)
            oid = int(o.get("order_id", -1))
            if oid >= 0:
                self.order_state[oid] = "waiting"
            oid = int(o.get("order_id", -1))
            self.order_status[oid] = "pending"

        # new riders
        for r in self._riders_by_t.get(self.t, []):
            self.idle_riders.append(r)
            rid = int(r["rider_id"])
            self.rider_pos[rid] = int(r["init_node"])
            self.routes.setdefault(rid, [])
            self.assigned_orders.setdefault(rid, [])
            self.load.setdefault(rid, 0)
            if isinstance(self.capacity, dict):
                self.capacity.setdefault(rid, int(self.default_capacity))

        if int(action) == 1 and self.pending_orders and self.idle_riders:
            rider = self.idle_riders[0]
            order = self.pending_orders[0]
            rid = int(rider["rider_id"])
            cur = int(self.rider_pos.get(rid, int(rider.get("init_node", 0))))
            o = int(order["origin"])
            d = int(order["dest"])
            speed = max(1e-6, float(rider.get("speed_factor", 1.0)))
            total_eta = (
                float(self._shortest_eta(cur, o, self.t)) + float(self._shortest_eta(o, d, self.t))
            ) / speed
            self.dispatch(rider, order, total_eta)
            reward += -float(rider.get("last_route_delta", 0.0))
        # after self.t += 1 and after new riders/orders are added

    
        def _is_off_duty(r):
            return int(r.get("end_min", 10**9)) <= int(self.t)

        off_idle = [r for r in self.idle_riders if _is_off_duty(r)]
        off_busy = [r for r in self.busy_riders if _is_off_duty(r)]

        if off_idle:
            self.idle_riders = [r for r in self.idle_riders if r not in off_idle]
        if off_busy:
            self.busy_riders = [r for r in self.busy_riders if r not in off_busy]

        for r in off_idle + off_busy:
            rid = int(r["rider_id"])
            self.routes.pop(rid, None)
            self.rider_eta_remaining.pop(rid, None)
            self.load.pop(rid, None)
            if isinstance(self.capacity, dict):
                self.capacity.pop(rid, None)
            self.busy_until.pop(rid, None)
            self.rider_pos.pop(rid, None)
            self.assigned_orders.pop(rid, None)

        # route execution (advance one minute)
        all_riders = list(self.idle_riders) + list(self.busy_riders)

        for r in all_riders:
            rid = int(r["rider_id"])
            route = self.routes.get(rid, [])

            if not route:
                self.rider_eta_remaining[rid] = 0.0
                continue

            eta_rem = float(self.rider_eta_remaining.get(rid, 0.0))

            # traveling
            if eta_rem > 0.0:
                self.rider_eta_remaining[rid] = float(max(0.0, eta_rem - 1.0))
                continue

            # arrived at next stop
            stop_type, oid, node = route[0]
            node = int(node)
            oid = int(oid)

            self.rider_pos[rid] = node
            route.pop(0)

            if stop_type == "pickup":
                assert int(self.load.get(rid, 0)) < int(self._rider_capacity(rid))
                self.load[rid] = int(self.load.get(rid, 0)) + 1
                self.order_status[oid] = "picked"

            elif stop_type == "dropoff":
                self.order_status[oid] = "delivered"
                self.assigned_orders[rid] = [
                    o for o in self.assigned_orders.get(rid, [])
                    if int(o.get("order_id", -1)) != oid
                ]
                prev = int(self.load.get(rid, 0))
                if prev > 0:
                    self.load[rid] = prev - 1
                    if self.debug:
                        self._dbg_load_dec = getattr(self, "_dbg_load_dec", 0) + 1
                        if self._dbg_load_dec <= 20:
                            print("[dbg load--]", "t", int(self.t), "rid", rid, "load", self.load[rid])

            # schedule next leg if any
            if route:
                next_type, next_oid, next_node = route[0]
                cur = int(self.rider_pos.get(rid, int(r.get("init_node", 0))))
                tt = float(self._shortest_eta(cur, int(next_node), self.t))
                tt /= max(1e-6, float(r["speed_factor"]))
                tt = max(1.0, tt)
                self.rider_eta_remaining[rid] = float(tt)
            else:
                self.rider_eta_remaining[rid] = 0.0

            self.routes[rid] = route

        new_busy = []
        new_idle = []
        for r in self.idle_riders + self.busy_riders:
            rid = int(r["rider_id"])
            if (
                self.load.get(rid, 0) > 0
                or len(self.routes.get(rid, [])) > 0
                or float(self.rider_eta_remaining.get(rid, 0.0)) > 0.0
            ):
                new_busy.append(r)
            else:
                new_idle.append(r)

        self.busy_riders = new_busy
        self.idle_riders = new_idle

        reward -= 0.05 * float(len(self.pending_orders))

        terminated = self.t >= self.duration
        truncated = False
        info = {}

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

