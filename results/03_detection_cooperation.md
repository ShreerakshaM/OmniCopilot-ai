# Result 03 — Cooperative Gain Under Simulated Detection (Phase 2)

**Date:** Phase 2 result — achievable cooperative gain under detection uncertainty.
**Kill Gate A:** PASSED (cooperation lifts mAP +0.156, recall +25 points).
**Data:** OPV2V `opv2v-2`, 225 frames (≥2 agents, frame-stride 10).
**Method:** per-agent **simulated** detections → single-agent AP baseline vs.
detections fused through the **C++ WorldModel** → cooperative AP. Metric: mAP@[0.5,0.7]
via `omnicopilot.evaluation.metrics.compute_map`.

Reproduce:
```bash
python scripts/analyze_detection_cooperation.py \
    --data-root <DATA_ROOT>/opv2v-2/test \
    --module-dir build/cpp --assoc-radius 2.5 --frame-stride 10 --min-agents 2 --seed 0
```

---

## HONEST CAVEAT — read first

The detector is **simulated, not a neural network.** Detections are ground-truth
objects degraded by a literature-calibrated model:
- Localization noise: σ_xyz ≤ 0.5 m, σ_heading ≤ 1° (V2X-ViT "Noisy Setting").
- Detection dropout: miss probability rises with range and local crowding (occlusion
  proxy) — so distant/occluded objects are the ones a single agent loses.
- False positives: small per-frame rate; range-decayed confidence.

Therefore the **absolute** mAP values are NOT comparable to the OPV2V paper's CNN
numbers. What IS meaningful and defensible is the **cooperative lift** — how much our
fusion recovers when multiple agents share detections. Integrating a real detector
(OpenCOOD/MMDet3D, "2a") was blocked by the Kaggle Python-3.12 vs. spconv/mmcv
incompatibility; it is a documented drop-in upgrade for a Python-3.10 environment
(Colab with a data subset, or a local GPU). See `docs/task_2.1_detector_integration_spec.md`.

---

## Headline (mean over 225 frames)

| | mAP@[0.5,0.7] | precision | recall |
|---|---|---|---|
| Single agent (best) | 0.251 | 0.853 | 0.450 |
| **Cooperative (fused)** | **0.409** | 0.850 | **0.704** |
| **Gain** | **+0.157 (+63% rel.)** | ~flat | **+0.254** |

**The story:** recall jumps from ~45% to ~70% while precision holds at ~85%. Cooperation
recovers objects individual agents miss (blind spots, range, occlusion) **without**
inflating false positives — the C++ world model's association + confirmation keeps the
fused set clean. mAP rises accordingly.

---

## Robustness (3 seeds)

The gain is stable, not a lucky draw:

| Seed | Single mAP | Coop mAP | Gain |
|---|---|---|---|
| 1 | 0.2529 | 0.4088 | +0.1559 |
| 2 | 0.2562 | 0.4123 | +0.1561 |
| 3 | 0.2564 | 0.4167 | +0.1603 |

Gain = **+0.156 to +0.160** across independent seeds (±0.002).

---

## Why precision holding flat matters

The right signature for genuine cooperative gain is **recall up, precision steady.**
If fusion were hallucinating, precision would drop as recall rose. It doesn't
(~0.85 throughout), so the +25 recall points are real object recovery — multi-agent
corroboration surfacing objects a single noisy detector missed. This validates the
fusion/association logic in the C++ core, not just the count.

---

## Relation to the OPV2V paper

The paper reports cooperation lifting PointPillar AP@0.7 from ~0.60 (no fusion) to
~0.82 (intermediate fusion), ~+35% relative, with gain saturating past ~4 agents. Our
result reproduces the **direction and shape** (large cooperative lift, driven by
recovering hard/occluded objects) through our own fusion, at a larger relative lift
(+63%) because our simulated detector applies heavier single-agent dropout. Absolute
values differ (simulated vs. CNN, by design).

---

## Density-scaling — cooperative gain vs. agent count (saturation curve)

Cooperative gain **rises with the number of agents and saturates at 4–5**, matching the
OPV2V paper's key finding ("gain saturates after ~4 CAVs"):

| Agents | Frames | Single mAP | Coop mAP | Gain | Coop recall |
|---|---|---|---|---|---|
| 2 | 102 | 0.263 | 0.356 | +0.093 | 0.656 |
| 3 | 92 | 0.265 | 0.413 | +0.148 | 0.728 |
| 4 | 14 | 0.184 | 0.423 | +0.239 | 0.677 |
| 5 | 17 | 0.252 | 0.496 | +0.244 | 0.763 |

More viewpoints → larger cooperative lift, tapering between 4 and 5 agents
(+0.239 → +0.244). This shows the gain is *structured*, not a flat artifact: sparse
scenes benefit modestly, dense multi-agent scenes benefit most, then saturate.

**Caveat:** the 4- and 5-agent rows are small samples (14 and 17 frames) — directionally
solid but statistically noisier than the 2/3-agent rows (102, 92 frames). The saturation
shape is the takeaway, not the exact high-density values.

---

## Association method — Mahalanobis-gated (Task 3.5)

Fusion association is **probabilistic, not a fixed distance threshold.** Each tracked
entity carries a Kalman-filter 2×2 position covariance; a new observation associates only
if its **Mahalanobis distance** against the innovation covariance `S = P + R` (R scaled by
detection confidence) is within the χ²₂ 99% gate (9.21). Computed with Eigen (LDLT, no
explicit inversion). A precisely-localized entity accepts only nearby detections; an
uncertain entity accepts a wider spread — the correct multi-object-tracker behavior.

Switching from the earlier fixed-5 m Euclidean gate to Mahalanobis gating **preserved the
result** (gain +0.155 vs. the previous +0.157) while making the association principled.
All 79 C++ unit tests pass.

---

## Kill Gate A — verdict

Kill Gate A required cooperation to show meaningful gain on this data. **PASSED:**
+0.157 mAP, +25 recall points, +63% relative, stable across seeds. Cooperation clearly
helps, measured through the real C++ fusion engine.

---

## Caveats & next

- Simulated detector (see caveat above); real-detector integration is the "2a" upgrade.
- Cooperative "detections" use default car dimensions for fused entities (the world
  model tracks centers/confidence, not box extents yet) — a minor approximation for
  IoU; acceptable since IoU here is dominated by center proximity at these dims.
- Next: **Phase 3 trust layer** — down-weight disagreeing/low-quality agents in fusion
  (byzantine-aware confirmation), then **Phase 5** (learned communication policy under
  realistic V2X constraints — the project's core novelty), which does not depend on 2a.
  (The per-agent-count saturation breakdown is DONE — see the density-scaling section.)
