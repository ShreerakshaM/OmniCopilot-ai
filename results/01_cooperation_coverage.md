# Result 01 — Cooperation Coverage (Ground-Truth Ceiling) — CORRECTED

> **CORRECTION NOTICE.** The original version of this result reported a **2.45×**
> coverage gain (up to 4.27× on the best frame). Those numbers were **inflated by a
> coordinate-transform bug**: the loader treated OPV2V `vehicles[].location` as
> ego-frame and applied `world = pose @ location`, when the location is already in the
> WORLD frame. This scattered each agent's objects by its own pose offset (~500 m), so
> the same physical car seen by N agents was counted as N distinct objects — inflating
> the "collective" count and the gain. Fixed in commit `0f7dc3c` (+ `bdde5e4` for the
> `+ center` offset), confirmed against the dataset owner's own code
> (OpenCOOD `box_utils.project_world_objects`). The corrected numbers below are smaller
> and honest. See `docs/results/03_detection_cooperation.md` for the primary Phase 2
> result (detector-based mAP), which supersedes raw coverage as the headline metric.

**Data:** OPV2V `opv2v-2`, 16 scenarios, 2170 frames, mean 2.76 agents/frame.
**Method:** ground-truth object coverage — unique objects the fleet sees collectively
vs. the best-placed single agent. No detector (theoretical CEILING of cooperation).
**Object identity across agents:** world-frame proximity, tolerance 2.5 m (per-agent
keys are not shared; corrected transform makes shared cars coincide within ~2 m).

Reproduce: `scripts/analyze_cooperation_coverage.py`.

---

## Headline (CORRECTED)

| Metric | Value |
|---|---|
| Best single agent sees | 16.5 objects/frame |
| Fleet collectively sees | 18.5 objects/frame |
| **Mean gain vs best single agent** | **1.14×** |
| Median gain | 1.12× |
| Gain vs worst single agent | 1.44× |
| **Extra objects revealed by cooperation** | **+2.0 / frame** |

Corrected reading: on this data, a single well-placed agent already sees most objects
in range; cooperation adds ~2 distinct objects per frame on average. The raw-count gain
is modest — which is expected and is *why raw coverage is the wrong headline metric*.
The value of cooperation shows up in **detection quality on the hard/occluded objects**,
not in raw counts — see Result 03 (recall 0.45 → 0.70, mAP +0.157 through fusion).

---

## Retracted content (do not cite)

The following from the original version are **retracted** as artifacts of the bug and
must not be quoted:
- "2.45× / 3.74× coverage gain" and "+27 objects/frame".
- The density-scaling curve (1.90× → 3.74× by agent count).
- The "4.27×, +85 objects" demo frame (`2021_08_22_07_52_02` frame 139).

A corrected per-agent-count density curve has **not** been recomputed (the aggregate
run only produced the corrected headline). If a density curve is needed later, re-run
`analyze_cooperation_coverage.py` with a per-agent-count breakdown on the corrected
loader. Do not reuse the old curve.

---

## Why this matters for the project (unchanged thesis, corrected magnitude)

More agents still means more *potential* coverage, but also more V2X channel
congestion — the tension the intelligent-communication policy addresses. The corrected,
smaller raw-coverage gain does not weaken that thesis; it reframes it: cooperation's
measurable value is in **detection quality under uncertainty** (Result 03), and the
research question — *what to communicate, to whom, when, under a constrained channel* —
stands.

---

## Kill Gate A verdict (re-assessed on corrected numbers)

Kill Gate A asked whether cooperation shows meaningful gain. Raw coverage alone (+1.14×)
is modest, but the **detector-based result (Result 03)** — cooperation lifting mAP
+0.157 (+63% rel.), recall +25 points, stable across seeds — is the decisive evidence.
**Kill Gate A: PASSED**, on the Result 03 metric (the correct one), not on raw coverage.
