# Result 05 — Adversarial Robustness: does the trust layer defend fusion?

**Question:** cooperation improves perception (Result 03), but is it robust to a *bad*
participant? This measures whether adaptive per-agent trust identifies a faulty/malicious
agent and limits the damage it does to the fused world model.

**Setup:** one OPV2V scenario (`2021_08_22_07_52_02`, 5 agents, 167 frames). Honest agents
run the simulated detector (GT + calibrated noise). One agent (`3513`) is adversarial: it
reports every real object but consistently **offset by 5 m** — close enough to associate,
far enough to disagree with consensus (a mislocalization/spoof attack).

**Two-pass method (the two concerns must be separated):**
- **Pass 1 — learn trust:** per frame, reset entities but keep the reliability tracker
  (`WorldModel::ResetEntities`), so consensus is fresh for a *moving* scene while trust
  accumulates across frames. Agreement with the fused consensus (within
  `agreement_radius_m`) rewards an agent and its corroborators; associating-but-disagreeing
  penalizes the outlier.
- **Pass 2 — score:** fresh world model per frame (clean AP), seeding each agent's trust
  from Pass 1. Compare cooperative mAP/precision WITH learned trust vs. WITHOUT (flat prior).

Reproduce:
```bash
python scripts/adversarial_agent_demo.py \
  --data-root <DATA_ROOT>/opv2v-2/test --module-dir build/cpp \
  --min-agents 3 --seed 0 --agreement-radius 2.5 --adv-offset-m 5.0
```

---

## HONEST CAVEAT — read first

Detections are **simulated** (GT + V2X-ViT-calibrated noise), and the adversary is a
**scripted spoofer**, not a learned attacker. This demonstrates the fusion+trust
mechanism's robustness, not adversarial ML. Absolute mAP is not comparable to a CNN's.

---

## Trust learning (Pass 1)

| Agent | Trust (learned) |
|---|---|
| Honest agents (mean) | **0.883** |
| Adversary `3513` | **0.05** (start 0.096 → end 0.05) |

The trust layer cleanly separates honest from adversarial agents (~18× gap), learned
purely from consensus agreement — no ground-truth labels. Honest agents keep high trust
despite the detector's own localization noise (agreement radius 2.5 m sits above honest
noise, below the 5 m attack).

## Defense (Pass 2 — mean over 167 frames)

| | mAP@[0.5,0.7] | precision | recall |
|---|---|---|---|
| WITH trust | 0.166 | 0.417 | 0.381 |
| WITHOUT trust | 0.122 | 0.385 | 0.354 |
| **Defended by trust** | **+0.044** | **+0.032** | +0.027 |

With the adversary weighted equally (WITHOUT trust), its 5 m-off reports pollute the
fused set and drag down precision and mAP. WITH trust, the adversary is down-weighted and
fusion quality is measurably protected.

---

## Note on magnitude (why the defense is modest, and honest)

An earlier, buggy version of this demo reported a larger defense (+0.106 mAP) — but that
was an **artifact**: honest agents' trust had wrongly collapsed (to ~0.12), which
suppressed noise broadly and inflated the apparent effect. After fixing the two root
causes (below), honest agents correctly keep full weight (0.88), so the measured defense
reflects *only* the adversary being filtered — the true, unexaggerated effect. A
smaller-but-correct number is preferred over a larger artifactual one.

## Root causes fixed (the correct architecture, not tuning)

1. **Stale consensus on a dynamic scene.** A persistent world model over a moving scene
   left the "consensus" position stale (cars move), so honest agents were penalized for
   reporting where cars actually are now. Fixed with `ResetEntities()` — clears entities
   each frame, keeps learned trust.
2. **Ordering bias.** Whichever agent's observation created an entity each frame never
   received a corroboration reward. Fixed by crediting the existing corroborator(s) when a
   later observation agrees, so agreement is symmetric.

Guarded by the C++ regression test `WorldModelTest.MislocalizingAdversaryIsPenalized`
(honest > 0.8, adversary < 0.4). 82 C++ tests pass.

---

## Verdict

The trust layer works: it identifies a mislocalizing adversary from consensus alone,
down-weights it, and measurably defends cooperative detection quality — while leaving
honest agents at full trust. This is the robustness result that complements Result 03
(cooperation helps) with "cooperation stays useful under a bad participant."

Next: Phase 5 — learned communication policy under realistic V2X constraints (the
project's core novelty), which builds on this trust foundation.
