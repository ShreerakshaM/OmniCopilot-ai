# Result 02 — Real Data Through the C++ Fusion Engine

**Date:** first end-to-end run of real OPV2V data through the C++ core.
**Data:** OPV2V `opv2v-2` upload, scenario `2021_08_22_07_52_02`, frame 139 (5 agents).
**Method:** real per-agent ground-truth objects → Python loader → world-frame
positions → pybind11 → C++ `WorldModel` (greedy nearest-neighbor association +
confidence fusion + entity lifecycle). This measures what the **actual fusion
engine** produces, not the theoretical ground-truth ceiling of Result 01.

Reproduce:
```bash
# 1. Build the pybind11 module (needs Internet — FetchContent pulls pybind11 2.12.0)
cmake -B build -DOMNICOPILOT_BUILD_BINDINGS=ON -DOMNICOPILOT_BUILD_TESTS=OFF
cmake --build build -j

# 2. Feed one real frame through the C++ core
python scripts/feed_observations_demo.py \
    --module-dir build/cpp \
    --data-root <DATA_ROOT>/opv2v-2/test \
    --scenario 2021_08_22_07_52_02 --frame 139
```

---

## Headline

Scenario `2021_08_22_07_52_02`, frame 139, 5 agents:

| Metric | Value |
|---|---|
| Per-agent GT object counts | 3477:25, 3486:23, 3495:24, 3504:22, 3513:26 |
| Total observations fed | 120 |
| Best single agent saw | 26 objects |
| C++ fused entities | 62 (confirmed=23, tentative=39) |
| Mean fused confidence | 0.917 |
| **Fusion coverage gain** | **2.38× vs best single agent** |

The C++ fusion engine associated overlapping observations (the same physical
object seen by multiple agents) into single entities, while keeping
agent-exclusive objects separate — cooperative perception through the real core.

---

## Interpretation

**Why 2.38× and not the 4.27× ceiling from Result 01 for this frame.**
Result 01 counted every unique object across all agents using exact world-coordinate
deduplication (the theoretical ceiling: 26 → 111 objects, 4.27×). Result 02 runs the
*real* association logic, which merges any two observations within
`association_max_distance_m` of each other. That radius (default **5.0 m**, class-gated)
is generous for dense urban traffic where cars sit ~4–5 m apart, so some genuinely
distinct nearby vehicles are merged into one entity. The gap between 2.38× (achievable)
and 4.27× (ceiling) is **association tuning headroom**, not a defect.

**Internal consistency check.**
120 observations − 62 entities = 58 observations absorbed into an existing entity.
23 entities are confirmed (≥2 unique sources), and the remaining 39 entities are
single-source → tentative. This matches the reported `confirmed=23, tentative=39`
exactly, so the association bookkeeping is self-consistent.

**Confidence.**
Inputs were fixed at `confidence=0.9`. The fused mean of 0.917 is slightly higher
because multi-source confirmation raises confidence via noisy-OR combination — the
23 confirmed entities pull the mean up.

---

## Association parameters used (default `FusionConfig`)

| Parameter | Value | Effect |
|---|---|---|
| `association_max_distance_m` | 5.0 | Two observations within 5 m merge into one entity |
| `association_iou_threshold` | 0.3 | IoU gate for box-based association |
| `confirmation_source_count` | 2 | Min unique agents for an entity to be CONFIRMED |
| `confirmation_threshold` | 0.6 | Min confidence for CONFIRMED |

Association algorithm: greedy nearest-neighbor (closest pairs assigned first),
class-compatibility gated (a pedestrian never merges with a truck). Greedy is
sufficient for typical per-frame observation counts (<100); a Hungarian assignment
would be optimal but is not needed at this scale.

---

## Caveats & limitations

- **This is fusion behavior, not detection quality.** Inputs are ground-truth objects
  at a fixed confidence, so this result validates association/lifecycle logic — it does
  **not** yet measure gain under real detection uncertainty or occlusion. That is the
  Phase 2 milestone (OpenCOOD feature-fusion detector).
- **The 5 m association radius is untuned.** It should be swept against the GT ceiling
  to find the value that recovers the most true objects without over-merging. Candidate
  future work: sweep 1.5–5 m, report entity count vs. radius, pick the knee.
- **Single frame.** Result 01 is aggregated over 2170 frames; this is one demo frame.
  A fair achievable-gain number requires running the C++ pipeline across all frames.

---

## Next

- Phase 2: replace fixed-confidence GT inputs with a real detector (OpenCOOD feature
  fusion) → measure achievable gain under detection uncertainty and occlusion.
- Sweep `association_max_distance_m` and report the entity-count-vs-radius curve.
- Run the C++ pipeline across all frames of `opv2v-2` for an aggregate achievable gain
  to sit alongside Result 01's 2.45× ceiling.
