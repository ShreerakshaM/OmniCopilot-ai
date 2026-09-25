# Task 2.1 Spec — Detector Integration & Achievable Cooperative Gain (Phase 2)

> **STATUS UPDATE (2a blocked → 2b adopted).** The real-detector plan below (2a)
> was blocked on the Kaggle runtime: both OpenCOOD (spconv/cumm) and MMDetection3D
> (mmcv) are 2021–2022 stacks incompatible with Kaggle's **Python 3.12**. Concretely:
> spconv's `cumm` JIT-compile fails (missing `tensorview` headers); mmcv/openmim
> crash on `pkgutil.ImpImporter` (removed in 3.12); Kaggle's env-pin did not lower
> Python; conda is not installed. After time-boxed attempts we adopted **2b: a
> SIMULATED detector** (GT + literature-calibrated noise/dropout) to measure
> cooperative gain through our own C++ world model, on Python 3.12, with zero
> detector dependencies. 2a remains a documented drop-in upgrade for a Python-3.10
> environment (Colab with a data subset, or a local GPU) — the downstream pipeline
> is identical; only the detection source changes.
>
> Implemented for 2b: `perception/simulated_detector.py`,
> `scripts/analyze_detection_cooperation.py`, tests in
> `tests/unit/test_simulated_detector.py`.

**Goal of Phase 2:** replace the ground-truth, fixed-confidence inputs used so far
(Results 01 and 02) with a **real 3D object detector**, so we measure *achievable*
cooperative gain under real detection uncertainty and occlusion — not the theoretical
ceiling. This is the result that supports the project thesis.

**Timebox:** one week. Achievable ONLY IF we use pretrained weights and do NOT train.

---

## The single most important decision: use pretrained weights, do NOT train

Training a 3D detector from scratch on Kaggle's GPU quota would consume the entire
timeline. The plan (correction C1) already commits to OpenCOOD's feature-fusion
backbone. OpenCOOD ships **pretrained checkpoints on OPV2V specifically** — same
dataset, same sensor layout — so we run *inference only*.

| Option | Domain match | Effort | Verdict |
|---|---|---|---|
| Train PointPillars from scratch | — | weeks of GPU | ❌ blows the timeline |
| OpenPCDet pretrained (KITTI/nuScenes) | wrong domain (different sensor/classes) | medium + domain gap | ❌ mAP not comparable to OPV2V papers |
| **OpenCOOD pretrained on OPV2V** | exact | inference only | ✅ **chosen** |

**Decision:** use OpenCOOD's pretrained PointPillar checkpoint(s) on OPV2V for both the
single-agent (no fusion) and cooperative (intermediate feature fusion) configs. Run
inference on Kaggle GPU. No training in Phase 2.

**Known constraint (from earlier decisions):** do NOT run OpenCOOD's full
`setup.py install` — it pins `numba==0.49.0`, which fails on modern Python. Import only
the parts we need (model + pretrained weights + their inference entrypoint), mirroring
how we handled the transform utilities.

---

## Target numbers (from prior_art.md — OPV2V PointPillar, AP@IoU=0.7)

These are the field's published numbers. We report OUR measured values against them.

| Config | Published AP@0.7 (approx) | Our result |
|---|---|---|
| No fusion (single agent) | ~0.60 | TBD (Task 2.3) |
| Intermediate fusion | ~0.82 | TBD (Phase 3) |
| Relative gain | ~+20 AP / +33% | TBD |

We do NOT need to beat these. We need our single-agent baseline to land in a plausible
range, then show cooperation moves it in the expected direction.

---

## Sub-tasks (this week)

### 2.1a — Detector wrapper (`python/omnicopilot/perception/detector.py`)
- Typed interface: `Detector.detect(lidar_points: (N,4)) -> list[Detection]`
  where `Detection = {box: (x,y,z,l,w,h,yaw), score: float, cls: str}`.
- Backend: load OpenCOOD pretrained PointPillar checkpoint, single-agent (no-fusion)
  config first. Inference only, `torch.no_grad()`, move to GPU if available.
- No training code. No optimizer. Weights are frozen.

### 2.1b — Single-agent evaluation (`scripts/eval_single_agent.py`)
- For each agent frame: run detector on that agent's LiDAR → detections.
- Match detections to that agent's GT (Task 2.3 `metrics.py`, IoU/distance based).
- Report single-agent **mAP, precision, recall** aggregated across frames.
- Log config + numbers to a result doc (`docs/results/04_single_agent_baseline.md`).

### 2.1c — Occluded-object recovery metric (`python/omnicopilot/perception/occlusion.py`)
**This is the differentiating result — more compelling than raw mAP.**

OPV2V has no explicit occlusion label, so we DERIVE it from the multi-agent GT we
already load:
- An object is **"missed by agent X due to occlusion/range"** if it is absent from
  agent X's own GT list but present in the fleet union (some other agent sees it).
  (The loader's per-agent `gt_objects` already encodes what each agent can see.)
- **Recovery rate** = of the objects a single ego agent misses, what fraction are
  present in the cooperative fused world model.
- Report per agent-count (2..5), tying back to the density curve in Result 01.

This isolates the exact phenomenon cooperation exists to fix: *seeing what you alone
cannot*. It is computable with GT now (upper bound) and with detector output in Phase 3
(achievable).

### 2.1d — Wire the detector into the C++ world model (extends `feed_observations_demo.py`)
- Replace fixed `confidence=0.9` GT feed with detector `score` per detection.
- Feed detector boxes (world frame) into the C++ `WorldModel` → fused entities.
- Now Result 02's pipeline runs on REAL detections, not GT — the honest achievable gain.

---

## Definition of done for Phase 2

1. `detector.py` loads OpenCOOD pretrained weights and produces detections on one OPV2V
   frame (GPU inference verified on Kaggle).
2. Single-agent baseline mAP measured and recorded (`docs/results/04_...md`), landing in
   a plausible range vs. the ~0.60 published figure.
3. Occluded-object recovery metric implemented and reported per agent-count.
4. `feed_observations_demo.py` runs on detector output (not GT), giving an achievable
   coverage-gain number to sit beside Result 01's 2.45× ceiling.
5. All new Python has unit tests (metrics + occlusion logic testable locally, no GPU).

## Explicitly OUT of scope for this week
- Training / fine-tuning any model.
- The intermediate feature-fusion cooperative detector (that's Phase 3 — this week is the
  single-agent baseline + occlusion metric + real-detection plumbing).
- RL communication policy (Phase 5).

---

## Local-first ordering (what we can build WITHOUT GPU, before tomorrow's Kaggle run)
- `metrics.py` (Task 2.3) — pure Python, unit-testable now.
- `occlusion.py` recovery metric on GT — pure Python, unit-testable now.
- `detector.py` interface + wrapper skeleton — the OpenCOOD load call is the only part
  that needs GPU; the interface and box/score plumbing can be written and reviewed now.
- Then on Kaggle: load weights → run inference → measure. Execution, not exploration.
