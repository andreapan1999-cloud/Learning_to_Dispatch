from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from sim.env_dispatch import DispatchEnv


class DispatchPPOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_dir: str = "outputs",
        duration: int = 180,
        orders_file: str = "orders.csv",
        riders_file: str = "riders.csv",
        nodes_file: str = "nodes.csv",
        travel_times_file: str = "travel_times.csv",
        edges_file: str = "edges.csv",
        use_gnn_eta: bool = True,
        calibrate_gnn: bool = True,
        calib_samples: int = 2000,
        R: int = 4,
        O: int = 4,
        invalid_action_penalty: float = -0.02,
        wait_penalty: float = -0.01,
        wait_action_penalty: float = -0.05,
        wait_pending_scale: float = 0.02,
        mask_top_k: int = 20,
        decision_mode: str = "event",
        max_wait_advance: int = 60,
        debug: bool = False,
        node_feat_dim: int | None = None,
        alpha_eta_mu: float = 0.01,
        beta_eta_sigma: float = 0.01,
        progress_debug: bool = False,
        greedy_top_k_couriers: int | None = None,
        greedy_top_k_positions: int | None = None,
        profile_hotspots: bool = False,
    ):
        super().__init__()
        self.debug = debug    
        self.profile_hotspots = bool(profile_hotspots)
        self._profile_stats: Dict[str, Dict[str, float]] = {}
        if self.debug:
            print("[dbg __init__] got debug arg =", debug, "=> self.debug =", self.debug)

        self.base = DispatchEnv(
            data_dir=data_dir,
            duration=duration,
            orders_file=orders_file,
            riders_file=riders_file,
            nodes_file=nodes_file,
            travel_times_file=travel_times_file,
            edges_file=edges_file,
            use_gnn_eta=use_gnn_eta,
            calibrate_gnn=calibrate_gnn,
            calib_samples=calib_samples,
            verbose=False,
            debug=self.debug,
            profile_hotspots=self.profile_hotspots,
        )

        self.R = int(R)
        self.O = int(O)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.wait_penalty = float(wait_penalty)
        self.wait_action_penalty = float(wait_action_penalty)
        self.wait_pending_scale = float(wait_pending_scale)
        self.mask_top_k = int(mask_top_k)
        self.alpha_eta_mu = float(alpha_eta_mu)
        self.beta_eta_sigma = float(beta_eta_sigma)
        self.progress_debug = bool(progress_debug)
        self.greedy_top_k_couriers = None if greedy_top_k_couriers is None else int(greedy_top_k_couriers)
        self.greedy_top_k_positions = None if greedy_top_k_positions is None else int(greedy_top_k_positions)

        self.flat_dim = 39
        self.node_feat_dim = int(node_feat_dim) if node_feat_dim is not None else 9
        if self.node_feat_dim < 8:
            raise ValueError(f"node_feat_dim must be >= 8, got {self.node_feat_dim}")
        self.n_nodes = int(self.R + self.O)

        self.decision_mode = str(decision_mode)
        self.max_wait_advance = int(max_wait_advance)

        self.observation_space = spaces.Dict(
            {
                "flat": spaces.Box(low=-1e9, high=1e9, shape=(self.flat_dim,), dtype=np.float32),
                "node_x": spaces.Box(
                    low=-1e9, high=1e9, shape=(self.n_nodes, self.node_feat_dim), dtype=np.float32
                ),
                "eta_tgt": spaces.Box(
                    low=-1e9, high=1e9, shape=(self.R, self.O), dtype=np.float32
                ),
                "eta_mask": spaces.Box(
                    low=0.0, high=1.0, shape=(self.R, self.O), dtype=np.float32
                ),
            }
        )

        self.max_route_length = int(2 * self.O + 2)
        # MultiDiscrete with sentinel "wait" values on rider/order dimensions.
        # action = [r_idx_or_wait, o_idx_or_wait, pickup_insert_idx, dropoff_insert_idx]
        self.action_space = spaces.MultiDiscrete(
            np.asarray([self.R + 1, self.O + 1, self.max_route_length, self.max_route_length], dtype=np.int64)
        )
        
        # ---- action mask buffer + cache (for MaskablePPO) ----
        self._mask_buf = np.zeros((int(np.sum(self.action_space.nvec)),), dtype=np.bool_)
        self._mask_cache_key = None
        self._last_eta = None
        self._last_has_pair = False
        self._eta_pred_cache = None  # numpy array [R, O] injected by policy
        self._eta_sigma_cache = None  # numpy array [R, O] injected by policy
        self._eta_supervise_every = 10  
        self._eta_supervise_step = 0
        self._debug_force_no_fallback = False
        self._sigma_missing_warned = False
        setattr(self.base, "greedy_top_k_positions", self.greedy_top_k_positions)

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
        if calls <= 5 or (calls % 100 == 0):
            avg_ms = 1e3 * float(stat["total_s"]) / max(1, calls)
            print(
                "[profile]",
                f"fn={name}",
                f"calls={calls}",
                f"avg_ms={avg_ms:.3f}",
                f"max_ms={float(stat['max_ms']):.3f}",
                flush=True,
            )
       

    def set_eta_pred(self, eta_mat):
        """
        eta_mat: numpy array-like with shape (R, O)
        """
        if eta_mat is None:
            self._eta_pred_cache = None
            return
        arr = np.asarray(eta_mat, dtype=np.float32)
        if arr.shape != (self.R, self.O):
            self._eta_pred_cache = None
            return
        self._eta_pred_cache = arr

    def set_eta_sigma(self, eta_sigma_mat):
        if eta_sigma_mat is None:
            self._eta_sigma_cache = None
            return
        arr = np.asarray(eta_sigma_mat, dtype=np.float32)
        if arr.shape != (self.R, self.O):
            self._eta_sigma_cache = None
            return
        arr = np.nan_to_num(arr, nan=1e-6, posinf=1e6, neginf=1e-6)
        self._eta_sigma_cache = np.maximum(arr, 1e-6)

    def _rider_feat(self, rider: Dict[str, Any], t: int) -> List[float]:
        rid = int(rider["rider_id"])
        node = int(self.base.rider_pos.get(rid, int(rider["init_node"])))
        x, y = self.base.node_xy[node]
        return [
            t / max(1.0, float(self.base.duration)),
            float(x),
            float(y),
            float(rider["speed_factor"]),
        ]

    def _order_feat(self, order: Dict[str, Any], t: int) -> List[float]:
        age = max(0, t - int(order["t_min"]))
        ox, oy = self.base.node_xy[int(order["origin"])]
        dx, dy = self.base.node_xy[int(order["dest"])]
        return [
            float(age) / 180.0,
            float(ox),
            float(oy),
            float(dx),
            float(dy),
        ]

    def _get_candidates(self):
        # candidate riders: eligible busy first, then idle
        load = getattr(self.base, "load", {})

        busy_all = list(getattr(self.base, "busy_riders", []))
        eligible_busy = []
        seen_busy = set()
        for r in busy_all:
            rid = int(r["rider_id"])
            if rid in seen_busy:
                continue
            seen_busy.add(rid)
            eligible_busy.append(r)

        eligible_busy.sort(
            key=lambda r: (
                -int(load.get(int(r["rider_id"]), 0)),
                int(r["rider_id"]),
            )
        )

        seen = set()
        cand_riders = []
        for r in eligible_busy + list(self.base.idle_riders):
            rid = int(r["rider_id"])
            if rid in seen:
                continue
            seen.add(rid)
            cand_riders.append(r)
            if len(cand_riders) >= self.R:
                break

        cand_orders = list(self.base.pending_orders[: self.O])

        if self.debug:
            self._dbg_cand = getattr(self, "_dbg_cand", 0) + 1
            if self._dbg_cand <= 20 or (self._dbg_cand % 20 == 0):
                rid_load_pairs = [
                    (int(rr["rider_id"]), int(load.get(int(rr["rider_id"]), 0)))
                    for rr in cand_riders
                ]
                print(
                    "[dbg candidates]",
                    "t", int(self.base.t),
                    "idle", len(self.base.idle_riders),
                    "busy", len(getattr(self.base, "busy_riders", [])),
                    "pending", len(self.base.pending_orders),
                    "cand_r", len(cand_riders),
                    "cand_o", len(cand_orders),
                    "rid_load", rid_load_pairs,
                    "cap", getattr(self.base, "capacity", 1),
                )

        return cand_riders, cand_orders

    def _has_decision(self) -> bool:
        riders, orders = self._get_candidates()
        return (len(riders) > 0) and (len(orders) > 0)

    def _obs_flat(self) -> np.ndarray:
        t = int(self.base.t)
        riders, orders = self._get_candidates()

        t_norm = t / max(1.0, float(self.base.duration))
        p_norm = len(self.base.pending_orders) / 100.0
        i_norm = len(self.base.idle_riders) / 100.0

        vec: List[float] = [t_norm, p_norm, i_norm]

        for k in range(self.R):
            if k < len(riders):
                vec.extend(self._rider_feat(riders[k], t))
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0])

        for k in range(self.O):
            if k < len(orders):
                vec.extend(self._order_feat(orders[k], t))
            else:
                vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.asarray(vec, dtype=np.float32)

    def _eta_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        t = int(self.base.t)
        riders, orders = self._get_candidates()

        eta = np.zeros((self.R, self.O), dtype=np.float32)
        mask = np.zeros((self.R, self.O), dtype=np.float32)

        for ri in range(self.R):
            if ri >= len(riders):
                continue
            rider = riders[ri]
            rid = int(rider["rider_id"])
            rider_node = int(self.base.rider_pos.get(rid, int(rider["init_node"])))
            speed = float(rider["speed_factor"])

            for oi in range(self.O):
                if oi >= len(orders):
                    continue
                order = orders[oi]

                if self.debug and ri == 0 and oi == 0:
                    self._dbg_eta = getattr(self, "_dbg_eta", 0) + 1
                    if self._dbg_eta <= 5:
                        print(
                            "[dbg eta_targets]",
                            "t", t,
                            "len(riders)", len(riders),
                            "len(orders)", len(orders),
                            "r0", riders[0],
                            "o0", orders[0],
                        )
                                    
                o = int(order["origin"])
                d = int(order["dest"])

                eta1 = float(self.base._shortest_eta(rider_node, o, self.base.t)) / max(1e-6, speed)
                eta2 = float(self.base._shortest_eta(o, d, self.base.t)) / max(1e-6, speed)
                eta[ri, oi] = float(eta1 + eta2)
                mask[ri, oi] = 1.0

        return eta, mask

    def _build_obs(self) -> Dict[str, np.ndarray]:
        t = int(self.base.t)
        riders, orders = self._get_candidates()

        flat = self._obs_flat()



        node_x = np.zeros((self.n_nodes, self.node_feat_dim), dtype=np.float32)

        # ---- rider nodes: 0..R-1
        for k in range(self.R):
            if k < len(riders):
                r = riders[k]
                rid = int(r["rider_id"])
                cap_attr = getattr(self.base, "capacity", 1)
                cap = float(cap_attr.get(rid, getattr(self.base, "default_capacity", 1))) if isinstance(cap_attr, dict) else float(cap_attr)
                cur_load = float(getattr(self.base, "load", {}).get(rid, 0))
                if self.node_feat_dim > 8:
                    node_x[k, 8] = cur_load / max(1.0, cap)

                node = int(self.base.rider_pos.get(rid, int(r["init_node"])))
                x, y = self.base.node_xy[node]

                node_x[k, 0] = 1.0   # rider flag
                node_x[k, 1] = 0.0   # order flag
                node_x[k, 2] = float(x)
                node_x[k, 3] = float(y)
                node_x[k, 6] = float(r["speed_factor"])
                node_x[k, 7] = 1.0   # valid

        # ---- order nodes: R..R+O-1
        for k in range(self.O):
            idx = self.R + k
            if k < len(orders):
                o = orders[k]
                age = max(0, t - int(o["t_min"]))
                age_norm = float(age) / 180.0

                ox, oy = self.base.node_xy[int(o["origin"])]
                dx, dy = self.base.node_xy[int(o["dest"])]

                node_x[idx, 0] = 0.0  # rider flag
                node_x[idx, 1] = 1.0  # order flag
                node_x[idx, 2] = float(ox)
                node_x[idx, 3] = float(oy)
                node_x[idx, 4] = float(dx)
                node_x[idx, 5] = float(dy)
                node_x[idx, 6] = float(age_norm)
                node_x[idx, 7] = 1.0  # valid

        has_pair = (len(riders) > 0) and (len(orders) > 0)

        if self.debug and has_pair:
            self._dbg_build = getattr(self, "_dbg_build", 0) + 1
            print(
                "[dbg build]",
                "k", self._dbg_build,
                "t", t,
                "idle", len(self.base.idle_riders),
                "pending", len(self.base.pending_orders),
                "cand_r", len(riders),
                "cand_o", len(orders),
            )

        # ---- eta supervision targets
        if has_pair:
            eta_tgt, eta_mask = self._eta_targets()
            self._last_eta = eta_tgt
            self._last_has_pair = True
        else:
            eta_tgt = np.zeros((self.R, self.O), dtype=np.float32)
            eta_mask = np.zeros((self.R, self.O), dtype=np.float32)
            self._last_has_pair = False

       

        return {
            "flat": flat,
            "node_x": node_x,
            "eta_tgt": eta_tgt,
            "eta_mask": eta_mask,
        }
        
        
       
  

    def _advance_until_decision(self) -> None:
        if self.decision_mode != "event":
            return
        for _ in range(max(1, self.max_wait_advance)):
            if self._has_decision():
                return
            _obs2, _r2, terminated, truncated, _info = self.base.step(0)
            if bool(terminated) or bool(truncated):
                return

    def _normalize_action(self, action) -> np.ndarray:
        act = np.asarray(action, dtype=np.int64).reshape(-1)
        if act.size == 0:
            act = np.asarray([self.R, self.O, 0, 1], dtype=np.int64)
        if act.size == 1:
            if int(act[0]) == 0:
                act = np.asarray([self.R, self.O, 0, 1], dtype=np.int64)
            else:
                act = np.asarray([0, 0, 0, 0], dtype=np.int64)
        if act.size < 4:
            pad = np.zeros((4 - act.size,), dtype=np.int64)
            act = np.concatenate([act, pad], axis=0)
        return act[:4]

    def inspect_action(self, action, require_eta: bool = False) -> Dict[str, Any]:
        riders, orders = self._get_candidates()
        act = self._normalize_action(action)
        ri = int(act[0])
        oi = int(act[1])
        pickup_idx = int(act[2])
        dropoff_idx = int(act[3])
        rider_ids = [int(r["rider_id"]) for r in riders]
        order_ids = [int(o["order_id"]) for o in orders]
        nvec = np.asarray(self.action_space.nvec, dtype=np.int64).reshape(-1)
        action_in_range = bool(act.size == nvec.size and np.all((act >= 0) & (act < nvec)))
        is_wait = bool(ri == self.R or oi == self.O)
        rider_in_bounds = bool(0 <= ri < len(riders))
        order_in_bounds = bool(0 <= oi < len(orders))
        rider_id = int(riders[ri]["rider_id"]) if rider_in_bounds else None
        order_id = int(orders[oi]["order_id"]) if order_in_bounds else None
        pair_in_candidate_mapping = bool(rider_in_bounds and order_in_bounds and rider_id is not None and order_id is not None)

        route_len = None
        valid_pick = False
        valid_drop = False
        delta_ins = None
        pair_dispatchable = False
        failure_reason = "ok"
        dispatch_error = None

        if is_wait:
            failure_reason = "wait_action"
        elif not rider_in_bounds:
            failure_reason = "rider_index_out_of_bounds"
        elif not order_in_bounds:
            failure_reason = "order_index_out_of_bounds"
        else:
            rider = riders[ri]
            order = orders[oi]
            rid = int(rider["rider_id"])
            route_len = int(len(getattr(self.base, "routes", {}).get(rid, [])))
            valid_pick = bool(0 <= pickup_idx <= route_len)
            valid_drop = bool(pickup_idx + 1 <= dropoff_idx <= route_len + 1)
            if not valid_pick:
                failure_reason = "pickup_index_invalid"
            elif not valid_drop:
                failure_reason = "dropoff_index_invalid"
            else:
                try:
                    delta_ins = float(self.base._order_insertion_delta_for_positions(
                        rider, order, int(pickup_idx), int(dropoff_idx)
                    ))
                    pair_dispatchable = bool(np.isfinite(delta_ins))
                    if not pair_dispatchable:
                        failure_reason = "pair_not_dispatchable"
                except Exception as ex:
                    dispatch_error = repr(ex)
                    failure_reason = "pair_not_dispatchable"

        fb_total_eta = None
        pred_total_eta = None
        eta_mu_selected = None
        eta_sigma_selected = None
        eta_sigma_source = "missing"
        eta_sigma_shape = None
        eta_fallback_available = False
        eta_pred_available = False
        eta_sigma_available = False

        if pair_in_candidate_mapping:
            if (
                self._last_eta is not None
                and bool(self._last_has_pair)
                and ri < int(self._last_eta.shape[0])
                and oi < int(self._last_eta.shape[1])
            ):
                fb_val = float(self._last_eta[ri, oi])
                if np.isfinite(fb_val):
                    fb_total_eta = fb_val
                    eta_fallback_available = True
            if (
                self._eta_pred_cache is not None
                and ri < int(self._eta_pred_cache.shape[0])
                and oi < int(self._eta_pred_cache.shape[1])
            ):
                pred_val = float(self._eta_pred_cache[ri, oi])
                if np.isfinite(pred_val):
                    pred_total_eta = pred_val
                    eta_pred_available = True
            if (
                self._eta_sigma_cache is not None
                and ri < int(self._eta_sigma_cache.shape[0])
                and oi < int(self._eta_sigma_cache.shape[1])
            ):
                eta_sigma_shape = tuple(int(x) for x in self._eta_sigma_cache.shape)
                sigma_val = float(np.nan_to_num(float(self._eta_sigma_cache[ri, oi]), nan=1e-6, posinf=1e6, neginf=1e-6))
                eta_sigma_selected = float(max(1e-6, sigma_val))
                eta_sigma_source = "eta_sigma_cache"
                eta_sigma_available = bool(np.isfinite(eta_sigma_selected))

        eta_total_available = bool(eta_pred_available or eta_fallback_available)
        if pred_total_eta is not None and fb_total_eta is not None:
            eta_mu_selected = float(np.clip(pred_total_eta, 0.5 * fb_total_eta, 1.5 * fb_total_eta))
        elif pred_total_eta is not None:
            eta_mu_selected = float(pred_total_eta)
        elif fb_total_eta is not None:
            eta_mu_selected = float(fb_total_eta)

        eta_missing_when_required = bool(require_eta and pair_dispatchable and (not eta_total_available))
        if failure_reason == "ok" and eta_missing_when_required:
            failure_reason = "eta_missing"

        dispatch_valid = bool(
            action_in_range
            and (not is_wait)
            and pair_in_candidate_mapping
            and valid_pick
            and valid_drop
            and pair_dispatchable
            and (not eta_missing_when_required)
        )

        return {
            "action": [int(x) for x in act.tolist()],
            "ri": int(ri),
            "oi": int(oi),
            "pickup_idx": int(pickup_idx),
            "dropoff_idx": int(dropoff_idx),
            "is_wait": bool(is_wait),
            "action_in_range": bool(action_in_range),
            "dispatch_valid": bool(dispatch_valid),
            "failure_reason": str(failure_reason),
            "rider_count": int(len(riders)),
            "order_count": int(len(orders)),
            "candidate_rider_ids": rider_ids,
            "candidate_order_ids": order_ids,
            "rider_in_bounds": bool(rider_in_bounds),
            "order_in_bounds": bool(order_in_bounds),
            "rider_id": rider_id,
            "order_id": order_id,
            "pair_in_candidate_mapping": bool(pair_in_candidate_mapping),
            "route_len": route_len,
            "valid_pick": bool(valid_pick),
            "valid_drop": bool(valid_drop),
            "pair_dispatchable": bool(pair_dispatchable),
            "dispatch_error": dispatch_error,
            "delta_ins": float(delta_ins) if delta_ins is not None else None,
            "eta_required": bool(require_eta and pair_dispatchable),
            "eta_total_available": bool(eta_total_available),
            "eta_pred_available": bool(eta_pred_available),
            "eta_fallback_available": bool(eta_fallback_available),
            "eta_missing_when_required": bool(eta_missing_when_required),
            "pred_total_eta": float(pred_total_eta) if pred_total_eta is not None else None,
            "fb_total_eta": float(fb_total_eta) if fb_total_eta is not None else None,
            "eta_mu_selected": float(eta_mu_selected) if eta_mu_selected is not None else None,
            "eta_sigma_selected": float(eta_sigma_selected) if eta_sigma_selected is not None else None,
            "eta_sigma_available": bool(eta_sigma_available),
            "eta_sigma_source": str(eta_sigma_source),
            "eta_sigma_shape": eta_sigma_shape,
        }

    def is_real_dispatch_valid(self, action, require_eta: bool = False) -> bool:
        return bool(self.inspect_action(action, require_eta=require_eta).get("dispatch_valid", False))

    def decode_action(self, action):
        """
        Decode action into candidate-space indices and mapped rider/order ids.
        """
        diag = self.inspect_action(action, require_eta=False)
        return {
            "ri": int(diag["ri"]),
            "oi": int(diag["oi"]),
            "pickup_idx": int(diag["pickup_idx"]),
            "dropoff_idx": int(diag["dropoff_idx"]),
            "rider_id": diag["rider_id"],
            "order_id": diag["order_id"],
            "is_wait": bool(diag["is_wait"]),
            "action_in_range": bool(diag["action_in_range"]),
            "dispatch_valid": bool(diag["dispatch_valid"]),
            "rider_in_bounds": bool(diag["rider_in_bounds"]),
            "order_in_bounds": bool(diag["order_in_bounds"]),
            "rider_count": int(diag["rider_count"]),
            "order_count": int(diag["order_count"]),
            "pair_in_candidate_mapping": bool(diag["pair_in_candidate_mapping"]),
            "candidate_rider_ids": list(diag["candidate_rider_ids"]),
            "candidate_order_ids": list(diag["candidate_order_ids"]),
            "failure_reason": str(diag["failure_reason"]),
        }

    def greedy_insertion_action(self):
        """
        Greedy baseline action for MultiDiscrete policy space:
        [rider_idx_or_wait, order_idx_or_wait, pickup_insert_idx, dropoff_insert_idx]
        """
        t0 = time.perf_counter() if self.profile_hotspots else 0.0
        if self.progress_debug:
            print("[progress] entering greedy_insertion_action", flush=True)
        riders, orders = self._get_candidates()
        if len(riders) == 0 or len(orders) == 0:
            self._record_hotspot("greedy_insertion_action", time.perf_counter() - t0)
            if self.progress_debug:
                print("[progress] exiting greedy_insertion_action action=wait no_candidates", flush=True)
            return np.asarray([self.R, self.O, 0, 1], dtype=np.int64)

        best = None
        rider_cap = self.R if self.greedy_top_k_couriers is None else min(self.R, int(self.greedy_top_k_couriers))
        setattr(self.base, "greedy_top_k_positions", self.greedy_top_k_positions)
        for ri, rider in enumerate(riders[: rider_cap]):
            for oi, order in enumerate(orders[: self.O]):
                try:
                    p, d, delta = self.base._best_order_insertion(rider, order)
                except Exception:
                    continue
                if not np.isfinite(delta):
                    continue

                p = int(p)
                d = int(d)
                if p < 0 or d <= p:
                    continue
                if p >= self.max_route_length or d >= self.max_route_length:
                    continue

                item = (float(delta), int(ri), int(oi), p, d)
                if best is None or item[0] < best[0]:
                    best = item

        if best is None:
            self._record_hotspot("greedy_insertion_action", time.perf_counter() - t0)
            if self.progress_debug:
                print("[progress] exiting greedy_insertion_action action=wait no_feasible", flush=True)
            return np.asarray([self.R, self.O, 0, 1], dtype=np.int64)

        _, ri, oi, p, d = best
        self._record_hotspot("greedy_insertion_action", time.perf_counter() - t0)
        if self.progress_debug:
            print(f"[progress] exiting greedy_insertion_action action={[ri, oi, p, d]}", flush=True)
        return np.asarray([ri, oi, p, d], dtype=np.int64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _obs, info = self.base.reset(seed=seed)
        self._eta_supervise_step = 0
        self._eta_pred_cache = None
        self._eta_sigma_cache = None
        self._sigma_missing_warned = False
        self._last_eta = None
        self._last_has_pair = False
        self._prev_action = None
        self._mask_cache_key = None
        self._advance_until_decision()
        return self._build_obs(), info
     


    def step(self, action):
        self._eta_supervise_step = getattr(self, "_eta_supervise_step", 0) + 1
        used_pred = False
        invalid_action = False
        reward = 0.0
        reward_wait_action = 0.0
        reward_wait_backlog = 0.0
        reward_invalid = 0.0
        reward_dispatch = 0.0
        reward_route = 0.0
        reward_repeat = 0.0
        reward_dispatch_extra = 0.0
        reward_eta_mu_penalty = 0.0
        reward_eta_sigma_penalty = 0.0
        riders, orders = self._get_candidates()
        has_pair_at_decision = (len(riders) > 0) and (len(orders) > 0)
        pending_now = int(len(self.base.pending_orders))

        dispatched = False
        total_eta = None
        eta_mu_selected = None
        eta_sigma_selected = None
        eta_sigma_source = "missing"
        eta_sigma_shape = None

        act_diag = self.inspect_action(action, require_eta=True)
        act = np.asarray(act_diag["action"], dtype=np.int64)
        ri = int(act_diag["ri"])
        oi = int(act_diag["oi"])
        pickup_idx = int(act_diag["pickup_idx"])
        dropoff_idx = int(act_diag["dropoff_idx"])

        if ri == self.R or oi == self.O:
            reward_wait_action = float(self.wait_action_penalty) * (1.0 + float(self.wait_pending_scale) * float(pending_now))
            reward += reward_wait_action
        else:
            if not bool(act_diag.get("dispatch_valid", False)):
                reward += self.invalid_action_penalty
                reward_invalid += self.invalid_action_penalty
                invalid_action = True
            else:
                rider = riders[ri]
                order = orders[oi]
                delta_ins = act_diag.get("delta_ins")
                pred_total_eta = act_diag.get("pred_total_eta")
                fb_total_eta = act_diag.get("fb_total_eta")
                total_eta = act_diag.get("eta_mu_selected")
                eta_mu_selected = act_diag.get("eta_mu_selected")
                eta_sigma_selected = act_diag.get("eta_sigma_selected")
                eta_sigma_source = str(act_diag.get("eta_sigma_source", "missing"))
                eta_sigma_shape = act_diag.get("eta_sigma_shape")
                used_pred = bool(act_diag.get("eta_pred_available", False))
                if eta_sigma_selected is None:
                    eta_sigma_selected = 0.0

                total_eta = float(max(1e-3, float(total_eta)))

                self.base.dispatch(
                    rider,
                    order,
                    total_eta,
                    meta={
                        "used_pred": bool(used_pred),
                        "pred_total_eta": float(pred_total_eta) if pred_total_eta is not None else None,
                        "fb_total_eta": float(fb_total_eta) if fb_total_eta is not None else None,
                        "pickup_insert_idx": int(pickup_idx),
                        "dropoff_insert_idx": int(dropoff_idx),
                    },
                )

                dispatched = True
                route_delta = float(delta_ins if delta_ins is not None else 0.0)
                reward_dispatch += 1.0
                reward_route += -0.01 * route_delta
                reward_eta_mu_penalty += -self.alpha_eta_mu * float(eta_mu_selected if eta_mu_selected is not None else 0.0)
                reward_eta_sigma_penalty += -self.beta_eta_sigma * float(eta_sigma_selected if eta_sigma_selected is not None else 0.0)
                reward += reward_dispatch + reward_route + reward_eta_mu_penalty + reward_eta_sigma_penalty
                if eta_sigma_source == "missing" and bool(getattr(self, "debug", False)) and (not bool(getattr(self, "_sigma_missing_warned", False))):
                    self._sigma_missing_warned = True
                    print(
                        "[risk][eta-sigma-missing]",
                        f"sigma_source={eta_sigma_source}",
                        f"sigma_shape={eta_sigma_shape}",
                        f"chosen_indices=({ri}, {oi})",
                        "log_var_exists=False",
                    )


        if self._last_has_pair:
            prev_a = getattr(self, "_prev_action", None)
            act_key = tuple(int(x) for x in act.tolist())
            if prev_a is not None and act_key == prev_a:
                reward_repeat -= 0.01
                reward += reward_repeat
            if dispatched:
                reward_dispatch_extra += 0.02
                reward += reward_dispatch_extra

        self._prev_action = tuple(int(x) for x in act.tolist())
        is_wait = bool(ri == self.R or oi == self.O)
        if is_wait or (not dispatched):
            reward_wait_backlog = self.wait_penalty * float(pending_now)
        else:
            reward_wait_backlog = 0.0
        reward += reward_wait_backlog

        _, _, terminated, truncated, info = self.base.step(0)
        self._advance_until_decision()

        info["used_eta_pred"] = bool(used_pred)
        info["action"] = [int(x) for x in act.tolist()]
        info["has_pair"] = bool(has_pair_at_decision)
        info["eta_cache_none"] = bool(self._eta_pred_cache is None)
        info["eta_cache_mean"] = float(np.mean(self._eta_pred_cache)) if self._eta_pred_cache is not None else float("nan")
        info["eta_sigma_cache_none"] = bool(self._eta_sigma_cache is None)
        info["eta_sigma_cache_mean"] = float(np.mean(self._eta_sigma_cache)) if self._eta_sigma_cache is not None else float("nan")
        info["eta_mu_selected"] = float(eta_mu_selected) if eta_mu_selected is not None else float("nan")
        info["eta_sigma_selected"] = float(eta_sigma_selected) if eta_sigma_selected is not None else float("nan")
        info["eta_sigma_source"] = str(eta_sigma_source)
        info["eta_sigma_shape"] = eta_sigma_shape
        info["dispatch_valid"] = bool(act_diag.get("dispatch_valid", False))
        info["dispatch_failure_reason"] = str(act_diag.get("failure_reason", "ok"))
        info["action_in_range"] = bool(act_diag.get("action_in_range", False))
        info["reward_eta_mu_penalty"] = float(reward_eta_mu_penalty)
        info["reward_eta_sigma_penalty"] = float(reward_eta_sigma_penalty)
        info["invalid"] = bool(invalid_action)
        info["is_wait"] = bool(is_wait)
        info["pending"] = int(pending_now)
        info["reward_wait_action"] = float(reward_wait_action)
        info["reward_wait_backlog"] = float(reward_wait_backlog)
        info["reward_invalid"] = float(reward_invalid)
        info["reward_dispatch"] = float(reward_dispatch)
        info["reward_route"] = float(reward_route)
        info["reward_repeat"] = float(reward_repeat)
        info["reward_dispatch_extra"] = float(reward_dispatch_extra)
        info["reward_total"] = float(reward)
        info["reward_components"] = {
            "dispatch": float(reward_dispatch),
            "route_delta": float(reward_route),
            "dispatch_extra": float(reward_dispatch_extra),
            "eta_mu_penalty": float(reward_eta_mu_penalty),
            "eta_sigma_penalty": float(reward_eta_sigma_penalty),
            "eta_mu_selected": float(eta_mu_selected) if eta_mu_selected is not None else float("nan"),
            "eta_sigma_selected": float(eta_sigma_selected) if eta_sigma_selected is not None else float("nan"),
            "eta_sigma_source": str(eta_sigma_source),
            "repeat": float(reward_repeat),
            "invalid": float(reward_invalid),
            "wait_action": float(reward_wait_action),
            "backlog": float(reward_wait_backlog),
        }
        
        return self._build_obs(), float(reward), bool(terminated), bool(truncated), info
    
    def action_masks(self) -> np.ndarray:
        """
        Factorized action masks for MultiDiscrete action space.
        Cross-dimension constraints (e.g., dropoff_idx > pickup_idx) are checked in step().
        """
        t0 = time.perf_counter() if self.profile_hotspots else 0.0
        mask = self._mask_buf
        mask.fill(False)
        riders, orders = self._get_candidates()
        rr, oo = len(riders), len(orders)

        load = getattr(self.base, "load", {})
        rider_ids_key = tuple(int(r["rider_id"]) for r in riders)
        loads_key = tuple(int(load.get(rid, 0)) for rid in rider_ids_key)
        route_lens_key = tuple(int(len(getattr(self.base, "routes", {}).get(rid, []))) for rid in rider_ids_key)
        key = (int(self.base.t), rr, oo, rider_ids_key, loads_key, route_lens_key)
        if self._mask_cache_key == key:
            self._record_hotspot("action_masks", time.perf_counter() - t0)
            return self._mask_buf

        rids = np.asarray(rider_ids_key, dtype=np.int32)
        loads = np.asarray([int(load.get(int(rid), 0)) for rid in rids], dtype=np.int32)

        n0 = int(self.R + 1)
        n1 = int(self.O + 1)
        n2 = int(self.max_route_length)
        n3 = int(self.max_route_length)

        s0 = 0
        s1 = s0 + n0
        s2 = s1 + n1
        s3 = s2 + n2

        # insertion indices always provide at least one safe value
        mask[s2 + 0] = True
        mask[s3 + min(1, n3 - 1)] = True

        feasible_ri = set()
        feasible_oi = set()
        feasible_pick = set()
        feasible_drop = set()
        use_top_k = bool(self.mask_top_k > 0)
        top_k_heap: List[Tuple[float, int, int, int, int]] = []
        feasible_tuples: List[Tuple[float, int, int, int, int]] = []
        route_cost_cache: Dict[Tuple[int, ...], float] = {}
        rider_cap_eff = max(1, min(rr, self.R))
        order_cap = min(oo, self.O)
        if use_top_k and order_cap > 1:
            order_cap = min(order_cap, max(1, int(np.ceil(float(self.mask_top_k) / float(2 * rider_cap_eff)))))
        pair_eval_cap = None if (not use_top_k) else max(2, min(6, int(np.ceil(float(self.mask_top_k) / float(max(1, rider_cap_eff * max(1, order_cap)))))))
        early_stop_target = None if (not use_top_k) else int(self.mask_top_k)
        feasible_found = 0
        stop_search = False
        rider_order_pairs_evaluated = 0
        insertion_pairs_evaluated = 0

        def route_cost_cached(nodes: List[int]) -> float:
            key_nodes = tuple(int(x) for x in nodes)
            hit = route_cost_cache.get(key_nodes)
            if hit is not None:
                return float(hit)
            val = float(self.base.compute_route_cost(list(key_nodes)))
            route_cost_cache[key_nodes] = float(val)
            return float(val)

        for ri in range(min(rr, self.R)):
            if stop_search:
                break
            rider = riders[ri]
            rid = int(rider["rider_id"])
            route = list(getattr(self.base, "routes", {}).get(rid, []))
            route_len = int(len(getattr(self.base, "routes", {}).get(rid, [])))
            max_pick = min(route_len, n2 - 1)
            max_drop = min(route_len + 1, n3 - 1)
            if max_drop <= 0:
                continue
            old_nodes = self.base._route_nodes_from_stops(rider, route)
            old_cost = route_cost_cached(old_nodes)
            cur_node = int(self.base.rider_pos.get(rid, int(rider.get("init_node", 0))))
            order_indices = list(range(min(oo, self.O)))
            if len(order_indices) > order_cap:
                order_indices = sorted(
                    order_indices,
                    key=lambda oi: (
                        float(self.base._eta_min(cur_node, int(orders[oi]["origin"])))
                        + float(self.base._eta_min(int(orders[oi]["origin"]), int(orders[oi]["dest"])))
                    ),
                )[:order_cap]

            for oi in order_indices:
                if stop_search:
                    break
                rider_order_pairs_evaluated += 1
                order = orders[oi]
                oid = int(order.get("order_id", -1))
                pstop = ("pickup", oid, int(order["origin"]))
                dstop = ("dropoff", oid, int(order["dest"]))
                pair_evals = 0
                for p in range(0, max_pick + 1):
                    if stop_search:
                        break
                    d_start = p + 1
                    if d_start > max_drop:
                        continue
                    for d in range(d_start, max_drop + 1):
                        if pair_eval_cap is not None and pair_evals >= pair_eval_cap:
                            break
                        cand = list(route)
                        cand.insert(int(p), pstop)
                        cand.insert(int(d), dstop)
                        if not self.base._is_capacity_feasible_for_stops(rid, cand):
                            continue
                        insertion_pairs_evaluated += 1
                        new_nodes = self.base._route_nodes_from_stops(rider, cand)
                        delta = float(route_cost_cached(new_nodes) - old_cost)
                        pair_evals += 1
                        feasible_found += 1
                        item = (float(delta), int(ri), int(oi), int(p), int(d))
                        if use_top_k:
                            heap_item = (-float(delta), int(ri), int(oi), int(p), int(d))
                            if len(top_k_heap) < int(self.mask_top_k):
                                heapq.heappush(top_k_heap, heap_item)
                            elif float(delta) < -float(top_k_heap[0][0]):
                                heapq.heapreplace(top_k_heap, heap_item)
                        else:
                            feasible_tuples.append(item)
                        if early_stop_target is not None and feasible_found >= early_stop_target:
                            stop_search = True
                            break
                    if pair_eval_cap is not None and pair_evals >= pair_eval_cap:
                        break

        if use_top_k:
            feasible_tuples = sorted(
                [(-float(neg_delta), int(ri), int(oi), int(p), int(d)) for neg_delta, ri, oi, p, d in top_k_heap],
                key=lambda x: x[0],
            )

        for _delta, ri, oi, p, d in feasible_tuples:
            feasible_ri.add(int(ri))
            feasible_oi.add(int(oi))
            feasible_pick.add(int(p))
            feasible_drop.add(int(d))

        for i in feasible_ri:
            mask[s0 + i] = True
        for i in feasible_oi:
            mask[s1 + i] = True
        for p in feasible_pick:
            if 0 <= p < n2:
                mask[s2 + p] = True
        for d in feasible_drop:
            if 0 <= d < n3:
                mask[s3 + d] = True

        has_nonwait = bool(feasible_ri) and bool(feasible_oi) and bool(feasible_pick) and bool(feasible_drop)
        if has_nonwait:
            mask[s0 + self.R] = False
            mask[s1 + self.O] = False
        else:
            mask[s0 + self.R] = True
            mask[s1 + self.O] = True

        self._mask_cache_key = key

        # optional debug print (very limited)
        if getattr(self, "debug", False):
            self._dbg_capmask = getattr(self, "_dbg_capmask", 0) + 1
            if self._dbg_capmask <= 10:
                print("[dbg cap]", "t", int(self.base.t), "loads", loads.tolist(), "kept", len(feasible_tuples), "topk", int(self.mask_top_k))
        if self.profile_hotspots:
            self._dbg_mask_prof = getattr(self, "_dbg_mask_prof", 0) + 1
            if self._dbg_mask_prof <= 5 or (self._dbg_mask_prof % 50 == 0):
                print(
                    "[profile][action_masks]",
                    f"t={int(self.base.t)}",
                    f"rider_order_pairs={int(rider_order_pairs_evaluated)}",
                    f"insertion_pairs={int(insertion_pairs_evaluated)}",
                    f"feasible_found={int(feasible_found)}",
                    f"kept={int(len(feasible_tuples))}",
                    flush=True,
                )
        self._record_hotspot("action_masks", time.perf_counter() - t0)
        return mask

    def action_reward_breakdown(self, action) -> Dict[str, float]:
        pending_now = int(len(self.base.pending_orders))
        diag = self.inspect_action(action, require_eta=True)
        act = np.asarray(diag["action"], dtype=np.int64)
        is_wait = bool(diag["is_wait"])
        dispatched = bool(diag["dispatch_valid"])
        invalid = bool((not is_wait) and (not dispatched))

        reward_dispatch = 0.0
        reward_route = 0.0
        reward_dispatch_extra = 0.0
        reward_repeat = 0.0
        reward_invalid = 0.0
        reward_wait_action = 0.0
        reward_backlog = 0.0
        reward_eta_mu_penalty = 0.0
        reward_eta_sigma_penalty = 0.0
        eta_mu_selected = None
        eta_sigma_selected = None
        eta_sigma_source = "missing"
        eta_sigma_shape = None

        if is_wait:
            reward_wait_action = float(self.wait_action_penalty) * (1.0 + float(self.wait_pending_scale) * float(pending_now))
        else:
            if not dispatched:
                reward_invalid += float(self.invalid_action_penalty)
            else:
                delta_ins = diag.get("delta_ins")
                eta_mu_selected = diag.get("eta_mu_selected")
                eta_sigma_selected = diag.get("eta_sigma_selected")
                eta_sigma_source = str(diag.get("eta_sigma_source", "missing"))
                eta_sigma_shape = diag.get("eta_sigma_shape")
                if eta_sigma_selected is None:
                    eta_sigma_selected = 0.0

                reward_dispatch += 1.0
                reward_route += -0.01 * float(delta_ins if delta_ins is not None else 0.0)
                reward_dispatch_extra += 0.02 if self._last_has_pair else 0.0
                reward_eta_mu_penalty += -self.alpha_eta_mu * float(eta_mu_selected if eta_mu_selected is not None else 0.0)
                reward_eta_sigma_penalty += -self.beta_eta_sigma * float(eta_sigma_selected if eta_sigma_selected is not None else 0.0)

        prev_a = getattr(self, "_prev_action", None)
        act_key = tuple(int(x) for x in act.tolist())
        if self._last_has_pair and prev_a is not None and act_key == prev_a:
            reward_repeat -= 0.01

        if is_wait or (not dispatched):
            reward_backlog = float(self.wait_penalty) * float(pending_now)

        total = (
            reward_dispatch
            + reward_route
            + reward_dispatch_extra
            + reward_eta_mu_penalty
            + reward_eta_sigma_penalty
            + reward_repeat
            + reward_invalid
            + reward_wait_action
            + reward_backlog
        )
        return {
            "is_wait": float(1.0 if is_wait else 0.0),
            "dispatched": float(1.0 if dispatched else 0.0),
            "invalid": float(1.0 if invalid else 0.0),
            "dispatch": float(reward_dispatch),
            "route_delta": float(reward_route),
            "dispatch_extra": float(reward_dispatch_extra),
            "eta_mu_penalty": float(reward_eta_mu_penalty),
            "eta_sigma_penalty": float(reward_eta_sigma_penalty),
            "eta_mu_selected": float(eta_mu_selected) if eta_mu_selected is not None else float("nan"),
            "eta_sigma_selected": float(eta_sigma_selected) if eta_sigma_selected is not None else float("nan"),
            "eta_sigma_source": str(eta_sigma_source),
            "action_in_range": float(1.0 if diag.get("action_in_range", False) else 0.0),
            "dispatch_valid": float(1.0 if dispatched else 0.0),
            "repeat": float(reward_repeat),
            "invalid_pen": float(reward_invalid),
            "wait_action_pen": float(reward_wait_action),
            "backlog_pen": float(reward_backlog),
            "total": float(total),
        }
    


    
