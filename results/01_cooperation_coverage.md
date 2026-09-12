# Result 01 — Cooperation Coverage (Ground-Truth Ceiling)

**Date:** first quantitative result of the project.
**Kill Gate A:** PASSED with margin.
**Data:** OPV2V test scenes (`opv2v-2` upload, 16 scenarios, 2-5 agents).
**Method:** ground-truth object coverage — how many unique objects the fleet sees
collectively vs. the best-placed single agent. No detection model (this is the
theoretical CEILING of cooperative benefit; achievable gain with a real detector is
measured later in Phase 2-3).

Reproduce: `notebooks/03_cooperation_analysis.ipynb` +
`scripts/analyze_cooperation_coverage.py`.

---

## Headline

Across 2170 frames (mean 2.76 agents/frame):

| Metric | Value |
|---|---|
| Best single agent sees | 16.5 objects/frame |
| Fleet collectively sees | 43.7 objects/frame |
| **Gain vs best single agent** | **2.45×** |
| Gain vs worst single agent | 3.11× |
| **Extra objects revealed by cooperation** | **+27.2 / frame** |

Even the best-positioned vehicle is blind to ~27 objects per frame that its neighbors
can see. Cooperation more than doubles perception coverage.

---

## Density-scaling curve (the key figure)

Cooperative gain rises monotonically with the number of agents, and begins to saturate
past ~4 agents (consistent with the OPV2V paper):

| Agents | Mean gain (vs best single) | Frames |
|---|---|---|
| 2 | 1.90× | 994 |
| 3 | 2.66× | 874 |
| 4 | 3.57× | 135 |
| 5 | 3.74× | 167 |

```
gain
3.74×                                    ●  (5)
3.57×                            ●  (4)
2.66×               ●  (3)
1.90×    ●  (2)
        └────┴──────┴──────┴──────┴──── agents
         2    3      4      5
```

Observations:
- Monotonic increase — more viewpoints reveal more of the scene.
- Saturation between 4 and 5 agents (3.57 → 3.74) — beyond ~4 well-placed agents,
  additional agents add little because coverage is already near-complete.

---

## Why this matters for the project (the thesis this motivates)

More agents → more cooperative benefit (this curve). **But** more agents → more V2X
channel congestion. So as density grows, the *potential* benefit grows while the
*communication cost* grows too. Capturing the ~3.7× benefit at high density WITHOUT
drowning the channel is precisely what the intelligent-communication policy + realistic
V2X networking layers address. This curve provides real-data motivation for the project's
central question: **what to communicate, to whom, when — under a constrained channel.**

---

## Honest caveats

- **This is a ceiling, not achievable performance.** It counts ground-truth objects with
  ≥1 LiDAR hit for an agent — i.e. what is *theoretically visible*. A real neural detector
  will miss some (distant, sparse, heavily occluded), so the achievable cooperative gain
  (Phase 2-3) will be lower. The ceiling being 2.45×-3.74× means there is substantial
  headroom for cooperation to exploit — which is what we needed to confirm before building
  the detection pipeline.
- **Object identity across agents is by world-frame proximity** (tolerance 3.0m), since
  OPV2V per-agent object keys are not shared. Validated: shared cars match within ~2m.
- **4- and 5-agent samples are smaller** (135 / 167 frames) — the high-density points are
  directionally solid but based on fewer scenes than the 2-3 agent points.

---

## Kill Gate A verdict

PASSED with margin. Target was ≥ +15 AP-equivalent improvement / clear cooperative gain;
concern threshold was < +5. Result: **+27 objects/frame, 2.45× coverage**, with a clean
monotonic density-scaling curve. The core premise — distributed agents collectively
perceive far more than any individual — is empirically confirmed on real data. Proceed to
Phase 2 (real detector → achievable gain).

---

## Demo scene identified

The strongest cooperation scenes all come from the 5-agent scenario
**`2021_08_22_07_52_02`**, around frames 139-187 and 339-363:

| Frame | Agents | Best single | Collective | Gain | Extra objects |
|---|---|---|---|---|---|
| 139 | 5 | 26 | 111 | 4.27× | +85 |
| 179 | 5 | 26 | 110 | 4.23× | +84 |
| 339 | 5 | 23 | 97 | 4.22× | +74 |
| 185 | 5 | 26 | 108 | 4.15× | +82 |

In this scenario, the best-placed vehicle sees ~26 objects while the 5-vehicle fleet
collectively sees ~111 — the single agent is blind to **~85 objects** its neighbors can
see. **This is the primary demo scene** for visualizing cooperative perception (Phase 10).

Raw data: `docs/results/coverage_summary_opv2v2.json`.
