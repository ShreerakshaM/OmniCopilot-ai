# Result 02 — Real Data Through the C++ Fusion Engine — RETRACTED / SUPERSEDED

> **RETRACTION NOTICE.** This result is retracted. It reported "2.38× fusion coverage
> gain" on scenario `2021_08_22_07_52_02` frame 139 (62 fused entities vs. 26 best
> single). Two problems invalidate it:
>
> 1. **Coordinate-transform bug.** The world-frame positions were computed with the
>    buggy `pose @ location` transform (see Result 01 correction notice). The 62-entity
>    count and the 4.27× "ceiling" it was compared against were both inflated by counting
>    the same physical car once per agent. With the corrected transform, that frame fuses
>    to ~20 entities.
> 2. **Broken metric.** "Fusion coverage gain vs best single agent" divided a
>    **deduplicated** fused count by a **non-deduplicated** single-agent raw count — an
>    apples-to-oranges ratio. This is why the corrected number came out as 0.77×, which
>    is meaningless as stated (a deduplicated collective can be smaller than one agent's
>    raw detection list).
>
> **Superseded by `docs/results/03_detection_cooperation.md`**, which measures the C++
> fusion engine's cooperative value correctly: single-agent AP vs. cooperative-fused AP
> (like-with-like), over 225 frames, giving +0.157 mAP / +25 recall points, stable across
> seeds. Use Result 03 as the real end-to-end fusion result. Do not cite the numbers that
> were previously in this file.

---

## What remains valid from this work

The *plumbing* this result first exercised is sound and still in use:
- Real OPV2V data → Python loader → world-frame positions → pybind11 → C++ `WorldModel`
  (association + confidence fusion + entity lifecycle). That end-to-end path works and is
  the same pipeline Result 03 runs on.
- The fusion engine correctly merges the same object seen by multiple agents into one
  confirmed entity while keeping agent-exclusive objects separate — verified in isolation
  (3 agents seeing one car → 1 entity, 3 sources, confirmed) and at scale in Result 03
  (precision holds ~0.85 while recall rises, i.e. no false-positive flooding).

## Lesson recorded

Two bugs compounded here — a data-transform error and a metric-definition error — and
they cancelled in a way that *looked* like a plausible result (2.38×) until real fusion
was run. This is why Result 03 uses a standard, like-with-like metric (mAP) and why the
transform is now validated against the dataset owner's own code.
