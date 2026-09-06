# OmniCopilot — Cooperative Multimodal AI Under Communication Constraints

## One-Sentence Summary

> **OmniCopilot is a multi-agent AI system where distributed agents with partial observations collaboratively build an uncertainty-aware world model, intelligently decide what information to share under resource constraints, and make collectively better decisions than any individual agent.**

The first demonstration domain is cooperative autonomous driving over V2X networks — chosen because it naturally combines every core AI challenge (partial observability, multimodal perception, real-time constraints, safety-critical decisions, and limited communication bandwidth).

The architecture is domain-agnostic. The same intelligence loop applies to multi-robot systems, drone swarms, edge AI, distributed sensor networks, and autonomous infrastructure.

---

## The Problem (Domain-Independent)

Consider any system of multiple intelligent agents:

- Each agent observes only part of the world
- Each agent's observations are noisy and uncertain
- Communication between agents is limited (bandwidth, latency, cost)
- The agents must collectively make better decisions than they could alone

Three fundamental AI questions emerge:

1. **Perception under uncertainty** — How do you fuse noisy, conflicting, multimodal observations from multiple sources into a coherent belief about the world?

2. **Intelligent communication** — When you cannot share everything, how does an AI decide what information is most valuable to transmit, to whom, and when?

3. **Collective reasoning** — Can a group of agents with partial knowledge collectively detect, predict, and respond to events that no single agent could handle alone?

These are open problems across AI, robotics, distributed systems, and multi-agent research.

---

## Why Autonomous Driving as First Domain

| Requirement | Why driving is ideal |
|---|---|
| Partial observability | Occlusions, blind spots, limited sensor range |
| Multimodal perception | Camera, LiDAR, radar, GPS, audio |
| Communication constraints | V2X bandwidth limits, latency, packet loss |
| Safety-critical decisions | Must get it right, quantifiable metrics |
| Multiple agents | Naturally many vehicles in the same area |
| Public data available | OPV2V, DAIR-V2X, nuScenes, CARLA |
| Clear baselines | Single vehicle vs. cooperative — easy to compare |

The domain provides a rich, well-instrumented testbed. It does NOT define the project's identity.

---

## Core AI Concepts Used

This is not a "we used YOLO and GPT" project. The AI concepts are structural:

### Multimodal Perception
Fusing heterogeneous sensor data (vision, depth, spatial, telemetry) into unified object-level representations.

### Uncertainty-Aware Reasoning
Every observation carries confidence. Every belief in the world model is probabilistic. The system explicitly represents what it does not know.

### Multi-Agent Cooperative Intelligence
Agents do not merely share data — they reason about what other agents know, what they need, and what would help them most.

### Active Information Acquisition
The system identifies knowledge gaps, estimates the value of missing information, selects the best source, and requests it. This is active perception / active learning applied to multi-agent systems.

### Information-Theoretic Communication
Under bandwidth constraints, the system uses information value (expected information gain, novelty, safety relevance, receiver knowledge state) to prioritize what to transmit. Communication becomes an AI decision, not a networking decision.

### Receiver-Aware Intelligence
Before transmitting, an agent estimates what the receiver already knows. Redundant information is suppressed. This requires theory-of-mind-like reasoning about other agents' belief states.

### Temporal World Modelling
The shared world model maintains state history, tracks confidence decay over time, and supports trajectory prediction. Stale information degrades gracefully.

### Agent Reliability Estimation
The system learns which agents produce accurate observations and weights their contributions accordingly. Unreliable or adversarial agents are down-weighted over time.

### Predictive Intelligence
The system predicts future states (trajectories, hazard development, collision probability) and uses predictions to pre-emptively acquire information or warn agents.

### Agentic Reasoning (LLM Layer)
Higher-level reasoning agents (investigation, risk assessment, explanation) use LLMs for hypothesis generation, tool selection, evidence aggregation, and natural-language explanation — applied to non-real-time decision layers.

### RAG-Augmented Knowledge
External knowledge (standards, research, domain documentation) is retrieved to support reasoning — particularly for the agentic investigation layer.

### Continual Learning
Agent reliability models, communication policies, and fusion weights improve over time as the system accumulates experience.

---

## The Intelligence Loop

```text
OBSERVE (each agent locally)
    ↓
LOCAL PERCEPTION (detect, classify, track)
    ↓
ESTIMATE UNCERTAINTY (what do I know, how confident?)
    ↓
IDENTIFY KNOWLEDGE GAPS (what am I missing?)
    ↓
ESTIMATE INFORMATION VALUE (who could help, how much would it help?)
    ↓
INTELLIGENT COMMUNICATION (transmit highest-value information under constraints)
    ↓
COOPERATIVE FUSION (combine observations, weigh by confidence and source reliability)
    ↓
UPDATE WORLD MODEL (unified, probabilistic, temporal)
    ↓
PREDICT (trajectories, risks, events)
    ↓
REASON (multi-agent, LLM-assisted for complex situations)
    ↓
DECIDE (what action to take)
    ↓
ACT
    ↓
NEW OBSERVATIONS → LOOP
```

The system is not passive. It actively seeks information, adapts communication, and reasons collectively.

---

## Shared World Model

The world model is the central intelligence artifact.

It represents:

- **Entities** — objects, agents, obstacles (with position, velocity, class, confidence)
- **Events** — detected happenings (braking, hazard, anomaly)
- **Relationships** — spatial and causal connections
- **Evidence** — which observations support which beliefs
- **Uncertainty** — calibrated confidence for every belief
- **Temporal history** — state evolution over time
- **Predictions** — projected future states
- **Agent reliability** — trust score per source

Example state:

```text
ENTITY #17 — Pedestrian
  Position: [124.3, 87.1]
  Velocity: [1.2, 0.3] m/s
  Confidence: 0.96
  Sources: Agent A (0.91), Agent B (0.94)
  Predicted trajectory: crossing intersection in 2.4s
  Last updated: 120ms ago

ENTITY #42 — Unknown obstacle
  Position: [201.7, 44.2]
  Confidence: 0.58
  Sources: Agent C only
  Status: NEEDS CORROBORATION
  → Active request sent to Agent D (best viewpoint)
```

The world model is not a database. It is what the collective intelligence currently believes, with full provenance and uncertainty.

---

## Intelligent Communication — The Key Innovation

Traditional approach:
```text
Agent has observations → Broadcast all → Hope network handles it
```

OmniCopilot approach:
```text
Agent has observations
    ↓
Score each observation:
  - Safety relevance
  - Novelty (does receiver already know this?)
  - Confidence (is it worth transmitting uncertain data?)
  - Time sensitivity (will it be stale on arrival?)
  - Receiver benefit (how much does this reduce THEIR uncertainty?)
    ↓
Rank by information value
    ↓
Check network state (bandwidth, latency, loss rate)
    ↓
Transmit top-N within budget
```

This is where AI meets networking. The communication channel becomes an intelligent, adaptive system — not a dumb pipe.

### What makes this novel

Most cooperative perception papers assume unlimited communication or apply simple compression. OmniCopilot treats communication itself as an AI-driven decision under the following constraints:

- What is most valuable to send?
- Who benefits most from receiving it?
- How does network state affect the policy?
- How should the system degrade when the network degrades?

---

## Active Information Acquisition

When the world model identifies high uncertainty about a critical entity:

```text
World Model:
  "Possible obstacle at [X, Y] — confidence only 0.43"

Information Planner:
  "Agent B has clear line-of-sight to that location"
  "Expected information gain from Agent B: 0.81"

Action:
  Request targeted observation from Agent B

Result:
  Agent B confirms obstacle — confidence rises to 0.94
```

The system doesn't passively wait for observations. It actively queries the agent network to resolve uncertainty. This is active learning applied to a distributed multi-agent system.

---

## Evaluation Strategy

### Baselines

| System | Description |
|---|---|
| Single Agent | No cooperation — each agent decides alone |
| Naive Broadcast | Share everything, no intelligence |
| Rule-Based Sharing | Handwritten priority rules |
| **OmniCopilot** | AI-driven cooperative intelligence |

### Metrics

**Perception quality:**
- mAP, precision, recall
- Detection range (can you see further with cooperation?)
- Occluded object detection rate

**Communication efficiency:**
- Bandwidth consumed
- Messages per second
- Bytes per detected object

**Decision quality:**
- Time-to-detection for hazards
- Collision avoidance rate
- False positive/negative rates

**Robustness:**
- Performance under packet loss (5%, 10%, 20%)
- Performance under latency (50ms, 100ms, 200ms)
- Graceful degradation when agents disconnect
- Resilience to unreliable/adversarial agents

**Key result to demonstrate:**

> Under the same communication budget, OmniCopilot achieves X% better perception accuracy than naive broadcast and Y% better than rule-based sharing.

---

## Key Experiments

### Experiment 1 — Does cooperation help?

Single agent vs. cooperative agents on occluded scenarios.

Expected: cooperative significantly outperforms solo.

### Experiment 2 — Does intelligent communication help?

Naive broadcast vs. AI-prioritized communication at the same bandwidth.

Expected: AI-prioritized achieves similar accuracy with less bandwidth (or better accuracy at same bandwidth).

### Experiment 3 — Does active acquisition help?

Passive observation sharing vs. active information requests.

Expected: active reduces uncertainty faster with fewer total messages.

### Experiment 4 — Robustness under degradation

Introduce packet loss, latency, agent failures.

Expected: graceful degradation rather than cliff-edge failure.

### Experiment 5 — Receiver-aware vs. broadcast

Context-aware communication (suppress known information) vs. blind sharing.

Expected: receiver-aware achieves same accuracy with significantly fewer messages.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        OMNICOPILOT CORE                          │
│         (Domain-agnostic cooperative intelligence)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ World Model │  │ Information      │  │ Multi-Agent      │   │
│  │             │  │ Planner          │  │ Reasoning        │   │
│  │ Entities    │  │                  │  │                  │   │
│  │ Events      │  │ Value estimation │  │ Investigation    │   │
│  │ Uncertainty │  │ Source selection  │  │ Risk assessment  │   │
│  │ Predictions │  │ Active requests  │  │ Verification     │   │
│  │ Evidence    │  │ Budget allocation│  │ Explanation      │   │
│  └──────┬──────┘  └────────┬─────────┘  └────────┬─────────┘   │
│         │                  │                     │              │
│         └──────────────────┼─────────────────────┘              │
│                            │                                    │
│                            ▼                                    │
│              ┌──────────────────────────┐                       │
│              │ Communication Intelligence│                       │
│              │                          │                       │
│              │ Prioritization           │                       │
│              │ Receiver-awareness       │                       │
│              │ Network adaptation       │                       │
│              │ Compression              │                       │
│              │ Reliability              │                       │
│              └────────────┬─────────────┘                       │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │ Domain Adapter │
                    └───────┬───────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    Autonomous         Multi-Robot         Drone Swarm
    Driving            Systems             Operations
         │                  │                  │
    V2X / CARLA        ROS 2 / Sim         MAVLink / Sim
    Camera/LiDAR       Camera/Depth        Camera/GPS
```

The domain adapter is thin. The intelligence core is shared.

---

## Domain Adaptations (Same Core, Different Sensors)

### Autonomous Driving (primary demo)
- Agents: vehicles
- Sensors: camera, LiDAR, radar, GPS
- Communication: V2X (bandwidth-limited, latency-sensitive)
- Scenarios: occlusion, hazards, intersections
- Data: OPV2V, DAIR-V2X, CARLA

### Multi-Robot Systems (future)
- Agents: robots
- Sensors: camera, depth, IMU
- Communication: WiFi/5G (variable quality)
- Scenarios: warehouse, exploration, search-and-rescue
- Data: simulation (Gazebo, Isaac)

### Drone Swarms (future)
- Agents: UAVs
- Sensors: camera, GPS, altimeter
- Communication: mesh network (highly constrained)
- Scenarios: surveillance, mapping, delivery coordination
- Data: simulation (AirSim)

### Edge AI / Telecom (future)
- Agents: network nodes / edge servers
- Sensors: traffic metrics, user behavior, system telemetry
- Communication: backhaul (cost-constrained)
- Scenarios: anomaly detection, load prediction, fault correlation
- Data: synthetic / open telecom datasets

You demonstrate ONE. You claim generality through architecture design. You prove generality only if time allows.

---

## Technology Stack

### AI / ML
- Python, PyTorch
- Object detection (YOLO, DETR, or cooperative perception models)
- Multi-object tracking
- Sensor fusion (early/mid/late fusion approaches)
- Uncertainty estimation (MC Dropout, ensembles, evidential deep learning)
- Trajectory prediction
- Reinforcement learning (for communication policies)

### LLM / Agentic AI
- LLM for high-level reasoning (investigation, explanation, planning)
- Tool-calling agents for complex situation analysis
- RAG for knowledge retrieval (standards, research, documentation)

### Systems
- C++ (performance-critical paths: world model updates, message handling)
- Python (ML, experiments, prototyping)
- gRPC / Protobuf (agent communication)
- Docker (reproducible environments)

### Simulation & Data
- CARLA (multi-vehicle simulation with full sensor suite)
- OPV2V / DAIR-V2X (cooperative perception datasets)
- Custom network simulator (bandwidth, latency, packet loss)

### Evaluation
- MLflow or Weights & Biases (experiment tracking)
- Matplotlib / Plotly (visualization)
- Pytest (testing)

---

## Developer Roles

### Developer A — Systems & Distributed Intelligence

**Owns:**
- World model infrastructure (C++ / Python)
- Communication intelligence layer
- Network simulation (bandwidth, latency, loss)
- Message prioritization algorithms
- Agent reliability tracking
- Active information acquisition logic
- Performance and real-time constraints
- gRPC / Protobuf interfaces
- CI/CD and reproducibility

**Research question:**
> How should distributed agents communicate intelligence under constrained resources to maximize collective decision quality?

### Developer B — AI/ML & Perception

**Owns:**
- Multimodal perception models
- Object detection and tracking
- Cooperative sensor fusion
- Uncertainty estimation and calibration
- Trajectory prediction
- Anomaly detection
- LLM reasoning layer
- RAG knowledge system
- Experiment design and evaluation

**Research question:**
> How can uncertain, heterogeneous observations from multiple imperfect agents be fused into reliable collective knowledge?

### Shared
- Architecture decisions
- Dataset selection and preparation
- Scenario design
- Experiment execution
- Results analysis
- Documentation and write-up

---

## Minimum Viable System (Build This First)

```text
3 simulated agents (vehicles in CARLA or replayed from OPV2V)
    ↓
Each runs local object detection
    ↓
Each produces observations with confidence scores
    ↓
Simple shared world model (spatial grid + entity list)
    ↓
Naive fusion (confidence-weighted averaging)
    ↓
Compare: single agent vs. 3-agent cooperative
    ↓
Measure: detection accuracy on occluded objects
```

**Timeline: 6-8 weeks.**

Then add:
1. Intelligent communication (prioritization) — 4 weeks
2. Network constraints (simulated bandwidth/loss) — 2 weeks
3. Active information acquisition — 4 weeks
4. LLM reasoning layer for complex situations — 3 weeks
5. Full evaluation and write-up — 3 weeks

---

## Technical Summary (per domain framing)

The same system, described through the lens of different technical domains:

**Multi-agent ML / perception:** A multi-agent cooperative perception system with
uncertainty-aware fusion, active information acquisition, and information-theoretic
communication under resource constraints.

**GenAI / agentic reasoning:** A multi-agent reasoning layer combining world modelling,
LLM-based investigation, RAG-augmented knowledge retrieval, and tool-using agents for
complex situation analysis (non-real-time layer only).

**Robotics:** A distributed perception platform where agents with partial observability
cooperatively build a shared world model through intelligent, bandwidth-aware information
exchange.

**Autonomous driving:** Cooperative V2X perception with AI-driven message prioritization,
receiver-aware communication, and uncertainty-weighted multi-vehicle sensor fusion.

**Distributed systems / edge AI:** A distributed intelligence system with
communication-aware inference, active information acquisition, graceful degradation under
network constraints, and adaptive agent reliability estimation.

**Networking / V2X:** Communication policies that maximize information value under
bandwidth, latency, and contention constraints in a multi-agent cooperative system,
evaluated under realistic C-V2X channel mechanics.

---

## What NOT to Build

- Generic chatbot or assistant
- Navigation app
- EV charging optimizer
- Weather dashboard
- "AI wrapper" around an API
- Anything that doesn't serve the core question

Every feature must strengthen:

> **How can distributed agents with partial observations cooperatively understand and act upon a dynamic world under communication constraints?**

---

## One-Paragraph Description

> OmniCopilot is a multi-agent AI system where distributed agents cooperatively perceive
> their environment through intelligent, bandwidth-aware communication. Each agent has
> partial observations with uncertainty. The system decides what information is most
> valuable to share, actively requests missing information from other agents, and fuses
> everything into a shared world model. It is demonstrated on cooperative autonomous
> driving, but the architecture applies to any distributed multi-agent problem — robotics,
> drones, edge AI. The central result target: intelligent communication achieves better
> perception than naive broadcasting while using significantly less bandwidth, and this
> advantage is evaluated under realistic V2X network constraints.

### Key technical points (for documentation / write-up)
- **Communication policy:** information-value scoring, receiver-awareness, network adaptation.
- **AI techniques:** cooperative perception models, uncertainty estimation, RL for
  communication policies, LLM reasoning for complex scenarios (non-real-time), RAG for knowledge.
- **Distinction from prior work:** most cooperative perception assumes idealized/unlimited
  communication and optimizes feature compression; this project treats communication itself
  as a learned decision under realistic constraints, and adds active acquisition — the
  system requests what it needs.
- **Generalization:** the core is domain-agnostic; swapping the sensor layer and domain
  adapter reuses the same intelligence loop.

---

## Project Name

**OmniCopilot** — "Omni" suggests multi-source/multi-modal; "Copilot" suggests
cooperation and assistance; the name does not lock the project to "vehicle" or "V2X."

Subtitle: *Cooperative Multimodal AI Under Communication Constraints.*

---

## The Novel Contribution — Communication as Active Inference

This is the intellectual anchor that separates OmniCopilot from existing cooperative perception work.

### The gap in existing research

Most cooperative perception papers (V2VNet, Where2comm, CoBEVT, DiscoNet) focus on *what representation to share* — features, compressed BEV maps, attention masks. They assume communication is available and ask "how do we compress efficiently?"

Very few ask the deeper question:

> **Should I communicate at all? To whom? Is it worth the cost? Does the receiver even need this?**

### OmniCopilot's contribution

We formulate communication as **active inference** — each agent maintains a model of what other agents believe, estimates how a message would update that belief (measured by KL divergence / mutual information / expected free energy), and transmits only when the expected belief update justifies the communication cost.

In one sentence:

> **Agents model each other's minds and speak only when it changes what others would do.**

This connects to:
- **Active Inference** (Friston) — agents minimize surprise by acting on the world, including acting on other agents' beliefs through communication
- **Information Bottleneck** — compress observations into maximally informative messages
- **Theory of Mind** — modelling what other agents know and need
- **Dec-POMDP** — decentralized decision-making under partial observability

### Why this framing matters

The precise articulation of the contribution is:

> "Most cooperative perception treats communication as a compression problem. We treat it as a decision problem — each agent maintains a belief model of other agents and communicates only when the expected information gain exceeds the cost. We learn this policy end-to-end and discover that it develops strategies no hand-designed system would produce."

---

## Learned Communication Policy (RL)

### Why learn instead of engineer?

A hand-designed prioritization formula works:

```
Importance = safety × uncertainty × distance × novelty × ...
```

But it's static. It can't adapt to:
- Different traffic densities
- Different network conditions moment-to-moment
- Receiver-specific needs
- Temporal dynamics (is this information becoming MORE critical over time?)

### The RL formulation

Treat communication as a **cooperative multi-agent reinforcement learning** problem:

- **State:** Agent's local observations + estimated receiver belief state + network conditions (bandwidth, latency, loss)
- **Action:** Which observations to transmit, to whom, given a bandwidth budget
- **Reward:** Improvement in collective perception accuracy at receiving agents (measured as reduction in detection error or uncertainty)
- **Constraint:** Fixed communication budget per timestep

This is a Dec-POMDP with communication — one of the most studied and challenging problems in multi-agent AI.

### Implementation path

1. First build the hand-designed baseline (works immediately, provides comparison)
2. Frame the RL problem — define state/action/reward precisely
3. Train with PPO or SAC in simulation (CARLA or OPV2V replay)
4. Compare: hand-designed vs. learned policy across all metrics
5. Analyze: what does the learned policy do differently?

### Developer split

- **Developer A (systems):** Builds the environment interface, defines the action space (realistic V2X message slots, bandwidth budgets), ensures the policy executes within real-time constraints
- **Developer B (ML):** Designs the RL formulation, reward shaping, training loop, and policy analysis

---

## Emergent Communication Strategies

### The goal: discover something non-obvious

If the learned policy develops behaviors that a human engineer wouldn't design, that's the project's most memorable result.

### Strategies to look for

**1. Selective silence**
The policy learns that sometimes NOT communicating is optimal — avoids flooding the network with low-value updates, reserves bandwidth for critical moments.

**2. Proactive alerting**
The policy sends early, low-confidence warnings about developing situations BEFORE they become critical — a form of predictive communication.

**3. Complementary reporting**
Agents learn to report observations from modalities or viewpoints that other agents lack, rather than confirming what everyone already sees.

**4. Information cascades**
Agent A tells Agent B something. Agent B, now better informed, recognizes it should tell Agent C something it wouldn't have known to share otherwise. Intelligence propagates through the network.

**5. Strategic redundancy**
Under high packet loss, the policy learns to send critical information to multiple agents simultaneously — trading efficiency for reliability.

**6. Bandwidth hoarding**
The policy learns to save bandwidth during calm periods so it has budget available during sudden high-activity events.

### How to surface these

- Train the policy over many episodes
- Record all communication decisions
- Compare to hand-designed baseline decisions
- Identify cases where the learned policy DISAGREES with the baseline
- Analyze WHY using information-theoretic metrics
- Categorize the emergent strategies
- Measure their impact on collective performance

### Why this is powerful

The significance, stated plainly:

> "We trained a communication policy and discovered that it develops strategies we didn't anticipate. For example, it learned to stay silent during routine driving and hoard bandwidth for sudden events — something our hand-designed system couldn't do because it evaluates each moment independently."

That shows you understand AI deeply enough to be surprised by it.

---

## Adversarial Robustness — Byzantine-Resilient Cooperation

### The trust problem

In any multi-agent system, not all agents can be trusted:

- **Sensor malfunction** → noisy, degraded observations
- **Software bug** → systematic errors in classification
- **Adversarial agent** → deliberately sends misleading information
- **Stale agent** → sends outdated observations without updating timestamps

### The defense: learned trust

OmniCopilot maintains a **per-agent reliability model** that evolves over time:

```text
Agent A: trust = 0.96 (consistently accurate, agrees with consensus)
Agent B: trust = 0.83 (occasionally noisy, but self-consistent)
Agent C: trust = 0.31 (frequent contradictions with physics/consensus)
Agent D: trust = 0.94 → 0.52 (sudden accuracy drop — possible malfunction)
```

### Trust estimation mechanisms

**1. Consensus validation**
Compare each agent's observations against the majority. Persistent disagreement lowers trust.

**2. Physical plausibility**
Flag observations that violate physics (impossible velocities, teleporting objects, objects appearing inside buildings).

**3. Temporal consistency**
Track whether an agent's observations are self-consistent over time. Sudden jumps in reported positions without corresponding velocity = suspicious.

**4. Cross-verification**
When multiple agents observe the same entity, compare their reports. Agents that consistently agree with verified ground truth earn higher trust.

**5. Contextual reliability**
An agent may be reliable in daylight but unreliable in rain (camera degradation). Trust becomes context-dependent.

### Impact on fusion

Observations are weighted by trust during cooperative fusion:

```
fused_confidence = Σ (agent_trust_i × observation_confidence_i) / Σ agent_trust_i
```

Low-trust agents contribute less to the world model. Zero-trust agents are quarantined.

### Adversarial experiments

| Scenario | What happens |
|---|---|
| 1 malicious agent (of 5) | System detects within ~50 interactions, down-weights to near-zero |
| 2 malicious agents (of 5) | System still maintains accurate world model (majority honest) |
| 3 malicious agents (of 5) | System degrades — explores Byzantine fault tolerance limits |
| Gradual drift | Agent slowly becomes inaccurate — test detection sensitivity |
| Coordinated attack | Multiple bad agents agree on false information — hardest case |

This demonstrates:
- You think about failure modes (maturity)
- The system is robust (deployable, not just a demo)
- You understand distributed systems trust problems (relevant across domains)

---

## Ablation Study — Proving What Matters

### Why ablations separate good from great

Most projects: "We built a system and it works."
Great projects: "We built a system, removed each component one at a time, and can tell you exactly what matters and by how much."

### Required ablations

| Experiment | What you remove | Question answered |
|---|---|---|
| No cooperation | Remove all inter-agent communication | Does cooperation help at all? |
| No uncertainty | Replace probabilistic confidence with binary (detected/not) | Does uncertainty-awareness improve fusion? |
| No prioritization | Random message selection within bandwidth budget | Does intelligent communication selection matter? |
| No receiver-awareness | Prioritize without modelling receiver's belief state | Does theory-of-mind improve efficiency? |
| No active acquisition | Never request information — only passively share | Does asking help vs. waiting? |
| No trust modelling | Weight all agents equally regardless of history | Does reliability estimation improve robustness? |
| No learned policy | Use only hand-designed rules | Does learning beat engineering? |
| No temporal modelling | No confidence decay, no trajectory prediction | Does temporal reasoning help? |

### Expected insights

You might discover:
- "Receiver-awareness provides marginal improvement in normal conditions but massive improvement under bandwidth constraints"
- "Active acquisition is the single biggest contributor to occluded object detection"
- "The learned policy only outperforms hand-designed rules when network conditions vary dynamically"

Each of these is a finding. Each tells a story. Each demonstrates scientific thinking.

### Presentation format

```text
              Full System    No X    No Y    No Z
mAP:              0.89       0.72    0.85    0.61
Bandwidth:        2.1MB/s    4.8MB/s 2.3MB/s 1.9MB/s
Hazard detect:    94%        71%     91%     58%
```

One table like this in a presentation says more than 50 slides of architecture diagrams.

---

## Self-Improving System — The System Gets Smarter Over Time

### What makes a project unforgettable

The system improves without code changes. It learns from its own operation.

### Mechanism 1: Communication policy improves with experience

The RL policy continues to refine through simulation episodes:

```text
Episode 100:    Communication efficiency = 0.64
Episode 1000:   Communication efficiency = 0.78
Episode 5000:   Communication efficiency = 0.91
```

Show the learning curve. The system literally gets better at deciding what to say.

### Mechanism 2: Trust calibration improves with interaction history

After 100 interactions, agent reliability estimates are noisy.
After 1000 interactions, the system correctly identifies unreliable agents with >95% accuracy.

```text
Interactions:  100   500   1000  5000
Trust accuracy: 62%   79%   94%   98%
```

### Mechanism 3: Fusion weights become context-aware

The system learns:
- "Camera-based Agent X is unreliable in rain but excellent in daylight"
- "LiDAR Agent Y is reliable at short range but degrades beyond 80m"
- "Agent Z is accurate for vehicles but frequently misclassifies pedestrians"

Over time, fusion becomes contextually intelligent without manual tuning.

### Mechanism 4: Spatial priors from experience

After processing many scenarios at a particular intersection:
- The system develops priors: "pedestrians tend to cross at [X, Y]"
- Uncertainty thresholds adjust: "be more alert in areas with historical pedestrian activity"
- Communication priority adjusts: "observations near historical hotspots are higher value"

### Why this matters

The significance, stated plainly:

> "The system doesn't just work — it gets better. We show that after N episodes, communication efficiency improves by X%, trust calibration accuracy reaches Y%, and the system develops location-specific priors that reduce hazard detection time by Z%."

This demonstrates genuine machine learning — not just "we trained a model once," but "the system continuously learns from its own operation."

---

## Complete Timeline (MVP → Full System)

| Weeks | Phase | Milestone | Maturity |
|---|---|---|---|
| 1-2 | Setup | Environment running (OPV2V loaded OR CARLA multi-vehicle) | Foundation |
| 3-4 | Baseline | Single-agent detection measured, baseline numbers established | Foundation |
| 5-8 | Core | Basic cooperative fusion working. First result: "cooperation helps" | **Core result** |
| 9-10 | Communication | Hand-designed prioritization under bandwidth constraints | Core result |
| 11-14 | Learning | RL-learned communication policy trained and compared | **Learned policy** |
| 15-16 | Active | Active information acquisition module working | Strong |
| 17-18 | Robustness | Adversarial agent experiments + trust module | Strong |
| 19-20 | Science | Full ablation study across all components | Rigorous |
| 21-22 | Discovery | Emergent strategy analysis from learned policy | **Novel finding** |
| 23-24 | Self-improvement | Demonstrate learning curves, contextual adaptation | Advanced |
| 25-26 | Polish | Demo video, write-up, visualizations, open-source release | Complete |

**Key insight:** The core result ships at week 10. Everything after is additive. There is
never a point where the project has nothing demonstrable — each milestone stands on its own.

---

## Deliverable at Each Milestone

### Week 10 — Core result
> A cooperative perception system. Cooperation improves detection by X%. Intelligent
> communication achieves the same accuracy at Y% less bandwidth.

### Week 16 — Learned policy
> A learned communication policy that outperforms hand-designed rules and discovers
> non-obvious strategies like selective silence and proactive alerting.

### Week 22 — Rigorous + novel finding
> Full ablation study identifying the biggest contributor; characterization of when the
> learned policy wins; adversarial resilience quantified up to N compromised agents.

### Week 26 — Complete
> The system improves over time (communication efficiency, trust calibration, spatial
> priors). Cooperative communication formulated as active inference; learned policies
> shown to develop emergent strategies hand-engineered systems cannot replicate. Evaluated
> under realistic V2X constraints.

---

## Summary: What Changed From the Previous Versions

| Problem in OmniWorld | Problem in OmniCopilot 2.0 | Fixed here |
|---|---|---|
| Too abstract, no concrete demo | Too automotive-specific | AI-first framing, automotive as demonstration domain |
| 50 sections of architecture | Clearer but still V2X-locked title/framing | Domain-agnostic core with thin adapters |
| No clear single result to show | Clear demo but narrow framing | Domain-agnostic technical summary |
| OmniForge distraction | Less distraction but optional modules creep | Stripped to essentials, clear MVP |
| Messy scope | Better scope but still long | Prioritized timeline with 6-8 week core |
| No novelty claim | Novelty unclear | "Communication as active inference" — clear contribution |
| No learning story | Static system | Self-improving system with learning curves |
| No scientific rigor | No ablations | Full ablation study proving each component's value |
| No surprising result | Predictable outcomes | Emergent strategy discovery from learned policy |
