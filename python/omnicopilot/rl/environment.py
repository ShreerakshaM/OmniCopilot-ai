"""Gymnasium environment for learning a communication policy (Phase 5).

Sequential, budget-coupled formulation (so PPO is the right tool, not bolted on):

- An EPISODE is one OPV2V scenario's frame sequence.
- Each STEP is one frame. The ego agent has candidate detections; the policy outputs a
  per-detection score. Detections are selected greedily by score until the per-step
  transmit cap OR the remaining EPISODE budget is hit.
- Transmitted detections are fused (with the other agents' detections) in the C++
  WorldModel; reward = Δ mAP from the ego's transmission − λ·bandwidth.
- The budget is shared across the whole episode: spending now reduces capacity later,
  creating genuine temporal credit assignment.

The detector (SimulatedDetector), fusion (C++ WorldModel) and metric (compute_map) are the
REAL components — the env is a thin RL wrapper, not a re-simulation.

Honest caveat: reward uses GT mAP as a TRAINING signal (offline). Deployment would act on
features only. Simulated detector, not a CNN.
"""

from __future__ import annotations

import sys
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces


class CommunicationEnv(gym.Env):
    """Communication-policy env over OPV2V scenarios (one scenario = one episode)."""

    metadata: dict[str, Any] = {"render_modes": []}

    # Per-candidate feature count (see _features): conf, safety, range_norm,
    # crowding, novelty, budget_frac_remaining.
    FEATS = 6

    def __init__(
        self,
        data_root: str,
        module_dir: str | None = None,
        num_obs_max: int = 60,
        episode_budget_bytes: int = 20000,
        per_step_cap: int = 20,
        obs_bytes: int = 256,
        lam_bandwidth: float = 0.05,
        min_agents: int = 2,
        frame_stride: int = 5,
        assoc_radius: float = 2.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if module_dir:
            sys.path.insert(0, module_dir)
        import _omnicopilot_cpp as oc  # noqa: PLC0415
        from omnicopilot.data.opv2v import OPV2VDataset  # noqa: PLC0415
        from omnicopilot.perception.simulated_detector import (  # noqa: PLC0415
            DetectorNoiseConfig, SimulatedDetector,
        )

        self._oc = oc
        self._num_obs_max = num_obs_max
        self._episode_budget = episode_budget_bytes
        self._per_step_cap = per_step_cap
        self._obs_bytes = obs_bytes
        self._lam = lam_bandwidth
        self._assoc_radius = assoc_radius
        self._rng = np.random.default_rng(seed)
        self._detector = SimulatedDetector(DetectorNoiseConfig(), seed=seed)

        self._ds = OPV2VDataset(data_root)
        self._ds.load()
        self._scenarios = [s for s in self._ds.scenario_ids()
                           if len(self._ds.get_scenario(s).agent_ids) >= min_agents]
        if not self._scenarios:
            msg = f"No scenario with >= {min_agents} agents under {data_root}"
            raise RuntimeError(msg)
        self._frame_stride = frame_stride

        # Observation: per-candidate features (padded to num_obs_max) + 1 global budget.
        obs_dim = num_obs_max * self.FEATS + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        # Action: a transmit logit per candidate slot. >0 => transmit (centred at 0 so a
        # fresh Gaussian policy sends ~half). Wider range than [0,1] so PPO can push
        # scores clearly positive/negative.
        self.action_space = spaces.Box(-5.0, 5.0, (num_obs_max,), np.float32)

        # Episode state (set in reset).
        self._scenario = None
        self._frames = []
        self._fi_idx = 0
        self._budget_left = 0
        self._cur_candidates: list = []  # list[Detection3D] for ego this frame
        self._cur_frame_data: dict | None = None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _safety_weight(self, obj_class: str) -> float:
        c = (obj_class or "").lower()
        if "ped" in c:
            return 1.0
        if "cycl" in c:
            return 0.95
        if "truck" in c or "bus" in c:
            return 0.65
        return 0.7  # vehicle default

    def _features(self, dets: list, ego_xyz: np.ndarray) -> np.ndarray:
        """(num_obs_max, FEATS) feature matrix for this frame's ego candidates."""
        f = np.zeros((self._num_obs_max, self.FEATS), dtype=np.float32)
        n = min(len(dets), self._num_obs_max)
        if n == 0:
            return f
        centers = np.array([d.position for d in dets[:n]], dtype=np.float64)
        for i in range(n):
            d = dets[i]
            rng = float(np.linalg.norm(centers[i] - ego_xyz))
            crowd = float(np.sum(np.linalg.norm(centers[:, :2] - centers[i, :2], axis=1) <= 8.0) - 1)
            crowd = crowd / max(n - 1, 1)
            # novelty proxy: distance to nearest OTHER ego candidate (far = more unique)
            others = np.delete(centers, i, axis=0)
            nov = float(np.min(np.linalg.norm(others[:, :2] - centers[i, :2], axis=1))) if len(others) else 10.0
            nov = min(nov / 10.0, 1.0)
            f[i, 0] = d.confidence
            f[i, 1] = self._safety_weight("vehicle")  # class not carried on Detection3D v1
            f[i, 2] = min(rng / 100.0, 1.0)
            f[i, 3] = crowd
            f[i, 4] = nov
            f[i, 5] = self._budget_left / max(self._episode_budget, 1)
        return f

    def _build_obs(self) -> npt.NDArray[np.float32]:
        ego = self._cur_frame_data["ego_xyz"]
        feats = self._features(self._cur_candidates, ego).reshape(-1)
        budget = np.array([self._budget_left / max(self._episode_budget, 1)], np.float32)
        return np.concatenate([feats, budget]).astype(np.float32)

    def _load_frame(self) -> None:
        """Populate ego candidates + per-agent detections for the current frame."""
        oc = self._oc
        sid = self._scenario
        fi = self._frames[self._fi_idx]
        frames = self._ds.get_all_agent_frames(sid, fi, load_lidar=False)
        aids = list(frames.keys())
        ego_id = aids[0]  # ego = first agent (fixed per episode)
        per_agent = {}
        for aid, fr in frames.items():
            centers = fr.gt_locations_world()
            dims = np.array([o.dimensions for o in fr.gt_objects]) if fr.gt_objects else np.zeros((0, 3))
            yaws = np.array([o.yaw_rad for o in fr.gt_objects]) if fr.gt_objects else np.zeros(0)
            sxyz = np.asarray(fr.pose)[:3, 3] if np.asarray(fr.pose).shape == (4, 4) else np.zeros(3)
            per_agent[aid] = (self._detector.detect(centers, dims, yaws, sxyz), sxyz)
        # GT union for scoring.
        all_c = [fr.gt_locations_world() for fr in frames.values()]
        gt_union = _dedupe(np.vstack(all_c)) if all_c else np.zeros((0, 3))
        gt7 = np.hstack([gt_union, np.tile([4.5, 2.0, 1.5, 0.0], (len(gt_union), 1))]) if len(gt_union) else np.zeros((0, 7))

        self._cur_candidates = per_agent[ego_id][0]
        self._cur_frame_data = {"ego_id": ego_id, "ego_xyz": per_agent[ego_id][1],
                                "per_agent": per_agent, "gt7": gt7}

    def _fuse_and_map(self, transmit_dets: list) -> float:
        """Fuse (all non-ego agents' detections) + the ego's TRANSMITTED subset; return mAP."""
        from omnicopilot.evaluation.metrics import compute_map  # noqa: PLC0415
        from omnicopilot.perception.detector import OpenCOODDetector  # noqa: PLC0415
        oc = self._oc
        cfg = oc.WorldModelConfig()
        cfg.max_entities = 2000
        cfg.fusion.association_max_distance_m = self._assoc_radius
        wm = oc.WorldModel(cfg)
        fd = self._cur_frame_data
        step_t = 1.0
        for aid, (dets, _) in fd["per_agent"].items():
            use = dets if aid != fd["ego_id"] else transmit_dets
            for j, d in enumerate(use):
                o = oc.Observation()
                o.observation_id = f"{aid}_{j}"; o.agent_id = aid
                o.object_class = oc.ObjectClass.VEHICLE
                o.position.x, o.position.y, o.position.z = map(float, d.position)
                o.confidence = float(d.confidence); o.timestamp_s = step_t
                wm.ingest_observations([o], 0.9)
        wm.tick(step_t + 0.1)
        rows = [[e.position.x, e.position.y, e.position.z, 4.5, 2.0, 1.5, 0.0, float(e.confidence)]
                for e in wm.get_entities()]
        preds = np.array(rows) if rows else np.zeros((0, 8))
        m = compute_map([preds], [fd["gt7"]], iou_thresholds=np.array([0.5, 0.7]))
        return float(m.mean_average_precision)

    # ── gym API ────────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._scenario = self._scenarios[self._rng.integers(len(self._scenarios))]
        sc = self._ds.get_scenario(self._scenario)
        self._frames = list(sc.frame_indices[::self._frame_stride])
        self._fi_idx = 0
        self._budget_left = self._episode_budget
        self._load_frame()
        return self._build_obs(), {}

    def step(self, action: npt.NDArray[np.float32]):
        oc = self._oc
        dets = self._cur_candidates
        n = min(len(dets), self._num_obs_max)
        # Per-detection transmit DECISION: transmit candidate i if action[i] > 0 (SB3's
        # Gaussian policy is centred at 0, so a fresh policy sends ~half -> no "send
        # nothing" or "send everything" collapse). This lets the policy control BOTH which
        # detections AND how many. A hard budget cap still applies: if more than the
        # remaining budget allows pass the threshold, keep the highest-scored ones.
        scores = np.asarray(action, dtype=np.float64)[:n]
        max_by_budget = int(self._budget_left // self._obs_bytes)
        cap = int(min(self._per_step_cap, max_by_budget))
        want = [i for i in range(n) if scores[i] > 0.0]
        if len(want) > cap:  # over budget -> keep highest-scored
            want = list(np.array(want)[np.argsort(-scores[want])][:cap])
        chosen = want
        transmit = [dets[i] for i in chosen]
        bytes_sent = len(transmit) * self._obs_bytes
        self._budget_left = max(0, self._budget_left - bytes_sent)

        # Reward: Δ mAP (ego transmits vs. ego stays silent) − λ·(fraction of candidates
        # sent). The penalty is per-detection-fraction so that transmitting a detection
        # that does NOT raise mAP is net-negative -> the policy must learn to send only
        # useful detections, not everything. map_silent is action-independent (cached).
        if self._cur_frame_data.get("map_silent") is None:
            self._cur_frame_data["map_silent"] = self._fuse_and_map([])
        map_silent = self._cur_frame_data["map_silent"]
        map_tx = self._fuse_and_map(transmit)
        delta = map_tx - map_silent
        frac_sent = len(transmit) / max(n, 1)
        reward = delta - self._lam * frac_sent

        self._fi_idx += 1
        terminated = self._fi_idx >= len(self._frames)
        truncated = False
        info = {"delta_map": delta, "bytes_sent": bytes_sent,
                "budget_left": self._budget_left, "n_transmitted": len(transmit)}
        if not terminated:
            self._load_frame()
            obs = self._build_obs()
        else:
            obs = np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, truncated, info


def _dedupe(centers: np.ndarray, tol: float = 2.0) -> np.ndarray:
    kept: list = []
    for c in centers:
        if all(np.linalg.norm(c[:2] - k[:2]) > tol for k in kept):
            kept.append(c)
    return np.array(kept) if kept else np.zeros((0, 3))
