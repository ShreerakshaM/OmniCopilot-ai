# Prior-Art Scan — 6 Cooperative Perception Papers

Analysis of the six papers in `docs/paper/`, mapped against the OmniCopilot design.
Purpose: reuse what exists, avoid landmines, confirm our angle is open, correct mistakes.

Papers (converted to markdown in `docs/paper/md/`):
- **V2VNet** (Wang et al., ECCV 2020) — `v2vnet.md`
- **OPV2V** (Xu et al., ICRA 2022) — `opv2v.md`
- **DiscoNet** (Li et al., NeurIPS 2021) — `disconet.md`
- **V2X-ViT** (Xu et al., ECCV 2022) — `v2x-vit.md`
- **CoBEVT** (Xu et al., CoRL 2022) — `cobevt.md`
- **Where2comm** (Hu et al., NeurIPS 2022) — `where2comm.md`

---

## Per-paper: reuse / assumptions / pitfalls

### OPV2V (our dataset — MUST understand)
- **Reuse:** Concrete baseline numbers (see below). Sensor spec (4 cameras, 1×64-ch LiDAR,
  GPS/IMU with **20mm positional / 2° heading error**). 20Hz streamed, **10Hz recorded**
  (matches our tick rate). Train/val/test = 6764/1981/2719 frames. Broadcast range 70m.
  Eval range x∈[-140,140]m, y∈[-40,40]m.
- **Assumes:** Cooperative fusion (early/late/intermediate). Their "Attentive Intermediate
  Fusion" shares neural FEATURES, not object observations.
- **Pitfall for us:** They project to ego frame using shared poses. Coordinate transforms
  are load-bearing — confirms our Task 3.0 risk is real.

**OPV2V baseline numbers (PointPillar, AP@IoU=0.7, Default towns) — our targets:**
| Fusion | AP@0.7 |
|---|---|
| No Fusion (single agent) | 0.602 |
| Late Fusion | 0.781 |
| Early Fusion | 0.800 |
| Intermediate Fusion | 0.815 |

→ Cooperation lifts AP@0.7 from ~0.60 to ~0.80 = **~+20 points (~+35% relative)**. This
CALIBRATES our success criteria: "cooperation helps" should target this order of gain.
Also: **AP gain saturates after ~4 CAVs** (they distribute around the intersection and
cover the blind spots). Use 3-5 agents; more adds little.

### V2VNet (the origin of feature sharing)
- **Reuse:** The idea of sharing intermediate features + a GNN to fuse. Confirms
  "intermediate fusion" is the sweet spot for bandwidth/accuracy.
- **CRITICAL — they already handle:** **time delay compensation** (CNN warps received
  messages by Δt) AND **imperfect localization** (they simulate pose noise). Both are
  things our Task 3.0 must handle — V2VNet shows it's necessary and how they approach it.
- **Assumes:** Feature-level sharing, unlimited-ish communication (they study compression
  but not a learned "what to send" decision).

### DiscoNet (knowledge distillation)
- **Reuse:** Distillation trick (learn from early-fusion teacher, deploy as intermediate).
  Has a **pose-error regression module** to correct pose errors — relevant to Task 3.0.
- **Assumes:** Fully-connected graph, shares full feature map. One communication round
  (lower latency — a point they make against multi-round methods).

### V2X-ViT (the robustness benchmark — MOST relevant to our networking angle)
- **Reuse / WARNING:** This paper ALREADY does much of what our Phase 8.5 targets. It has
  a **"Noisy Setting"**: pose error (σxyz ∈ [0, 0.5]m, σheading ∈ [0°, 1°]) AND **time
  delay = 100ms**, with a **delay-aware positional encoding**. It explicitly "accounts for
  imperfect real-world conditions."
- **BUT — the gap remains:** V2X-ViT models pose/time NOISE as Gaussian perturbations. It
  does NOT model the actual V2X CHANNEL — no CSMA/CA contention, no congestion control, no
  distance-based path loss, no density-dependent throughput collapse. It also does NOT
  learn a "what to communicate" policy — it shares full feature maps and makes fusion
  robust. **This is precisely the seam our Phase 8.5 + Phase 5 occupy.**
- **Assumes:** Full feature-map sharing, fixed bandwidth, robustness via attention.

### CoBEVT (BEV semantic segmentation)
- **Reuse:** BEV fusion via sparse transformers; models "asynchronization and position
  error" too. Compression-rate ablation (8x/16x) worth mirroring.
- **Assumes:** Segmentation task (not just detection), full feature sharing.

### Where2comm (OUR CLOSEST NEIGHBOR — read carefully)
- **What it does:** A **spatial confidence map** decides WHERE to communicate (only send
  perceptually-critical spatial regions), WHO to communicate (sparse graph), and fuses via
  confidence-aware attention. **Multi-round** communication with a **request map** (an
  agent requests complementary info from others). Adapts to varying bandwidth. Achieves
  ">100,000× lower communication" while beating DiscoNet and V2X-ViT on OPV2V.
- **This overlaps a LOT of our stated ideas:**
  - "communicate only what's valuable" → their spatial confidence map
  - "receiver requests missing info" → their request map / multi-round
  - "who to communicate with" → their sparse communication graph
  - "adapt to bandwidth" → they already do this
- **HOW they differ from us (the surviving gaps):**
  1. Their selection is a **supervised, differentiable spatial mask over FEATURE maps** —
     NOT a **reinforcement-learned policy** optimizing a long-horizon communication reward.
  2. They assume communication is **available and lossless** once chosen — **no realistic
     channel** (no contention, congestion, path loss, density effects). Idealized.
  3. They operate at the **feature level** (share BEV feature tensors); we also reason at
     the **object/world-model level** with explicit trust, existence confidence, and an
     LLM reasoning layer.
  4. **No adversarial/trust modeling** — they assume all agents are honest.

---

## Honest assessment: what this means for OmniCopilot

### Where we were NAIVE / need to correct
1. **"Communicating what matters" is NOT novel by itself.** Where2comm did it (spatial
   confidence). Our novelty must be sharper: *learned (RL) policy* + *realistic channel* +
   *object-level trust/existence* — not "we decide what to send," which is done.
2. **"Receiver requests missing info" is NOT novel by itself.** Where2comm's request map +
   multi-round communication already does active, receiver-driven acquisition. Our active
   acquisition must differentiate: explicit information-value estimation at the
   entity/world-model level, under a realistic request-costs-bandwidth channel.
3. **Pose error and time delay are KNOWN, SOLVED-ish problems.** V2VNet, DiscoNet, V2X-ViT
   all handle them. Our Task 3.0 should REUSE their approaches (Δt warping, pose-error
   handling), not reinvent — and we should NOT claim these as contributions.
4. **Feature-level fusion is the SOTA approach.** All six share neural features, not
   object observations. Our object-level world model is DIFFERENT (more interpretable,
   supports trust/LLM) but likely LESS accurate at raw detection than feature fusion. We
   must be honest: our value is interpretability + trust + realistic-comms + reasoning,
   NOT beating Where2comm on raw mAP.

### Where our angle SURVIVES as genuinely underexplored
1. **Communication as a learned RL policy under a REALISTIC V2X channel** (contention,
   congestion, path loss, density-dependent collapse). Every paper here assumes idealized
   or Gaussian-noise communication. None model the actual MAC/PHY. This is the strongest,
   clearest gap — and it is Developer A's networking specialization (Phase 8.5 + Phase 5).
2. **Object/world-model-level cooperation with explicit trust + adversarial resilience.**
   None of the six model malicious/unreliable agents. Trust + Byzantine resilience (Phase
   7) is open at this level.
3. **LLM/agentic reasoning layer feeding structured events back into cooperation** (Phase
   9). Orthogonal to all six — none touch this.
4. **The integrated system + ablation** across all these under realistic constraints.

### The sharpened contribution statement (replaces earlier vaguer one)
> "Prior collaborative-perception work (Where2comm, V2X-ViT, DiscoNet, V2VNet) optimizes
> WHAT features to share and makes fusion robust to Gaussian pose/time noise — but assumes
> the communication channel itself is idealized. OmniCopilot studies communication as a
> reinforcement-learned, receiver-aware, trust-weighted decision evaluated under a
> REALISTIC V2X channel (contention, congestion control, distance-based loss), and shows
> whether learned policies transfer from idealized training to realistic C-V2X and how the
> advantage scales with vehicle density."

### Concrete design corrections to make
- **Reframe Phase 4/5 messaging** away from "we decide what to communicate" (done by
  Where2comm) toward "we LEARN a communication policy AND evaluate it under a realistic
  channel (done by nobody)."
- **Reframe active acquisition (Phase 6)** as entity/world-model-level information-value
  under a channel that charges for requests — distinct from Where2comm's feature-level
  request map.
- **Task 3.0:** explicitly reuse V2VNet/V2X-ViT techniques for pose error (σ up to 0.5m)
  and time delay (~100ms, delay-aware handling). Do not reinvent; cite as standard.
- **Set success targets from OPV2V numbers:** single-agent AP@0.7 ≈ 0.60; cooperative
  ≈ 0.80. Our "cooperation helps" milestone should target a similar ~+15-20 AP gain on
  occluded objects, and we should EXPECT saturation past ~4 agents.
- **Consider adopting feature-level fusion (via OpenCOOD) for the detection backbone**,
  and layer our object-level world model / trust / comms-policy ON TOP — rather than
  competing with feature fusion on raw mAP. This plays to our strengths and reuses SOTA.
- **Be honest in evaluation:** compare our communication-efficiency and robustness-under-
  realistic-channel against Where2comm as the baseline, NOT raw mAP where they likely win.

### Reusable assets (do not reimplement)
- **OpenCOOD** implements V2VNet, DiscoNet, Where2comm, V2X-ViT, CoBEVT + OPV2V/V2X-Sim
  loaders. Use its dataset loader, coordinate handling, and a feature-fusion backbone.
- OPV2V's eval protocol (AP@0.5/0.7, range x∈[-140,140], y∈[-40,40], 70m broadcast).
- V2VNet/V2X-ViT delay-compensation and pose-error handling approaches.
