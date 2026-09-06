# OmniCopilot — Deep Code Walkthrough

Every file, every function, every design decision explained.

Read this document alongside the code. File paths are relative to the repository root.

---

# Table of Contents

1. [The Big Picture — How Data Flows](#1-the-big-picture)
2. [Entry Point — The Agent](#2-entry-point--the-agent)
3. [Data Types — Entity, Observation, Evidence](#3-data-types)
4. [Fusion Engine — Matching & Combining Observations](#4-fusion-engine)
5. [Uncertainty Manager — Confidence Decay & Boosting](#5-uncertainty-manager)
6. [Spatial Index — Fast Location Queries](#6-spatial-index)
7. [Temporal Tracker — Kalman Filter](#7-temporal-tracker)
8. [World Model — The Orchestrator](#8-world-model)
9. [Communication: Budget Manager](#9-budget-manager)
10. [Communication: Serializer](#10-serializer)
11. [Communication: Channel — Network Simulation](#11-channel)
12. [Communication: Prioritizer — What to Send](#12-prioritizer)
13. [Communication: Receiver Model — Theory of Mind](#13-receiver-model)
14. [Trust: Reliability Tracker](#14-reliability-tracker)
15. [Proto Definitions — The Interface Contract](#15-proto-definitions)
16. [Python: Data Types & Validation](#16-python-data-types)
17. [Python: Perception Module](#17-python-perception)
18. [Python: Fusion Module](#18-python-fusion)
19. [Python: RL Environment & Policy](#19-python-rl)
20. [Python: Configuration System](#20-python-config)
21. [Build System & Infrastructure](#21-build-system)
22. [How Everything Connects — Full Data Flow](#22-full-data-flow)

---

# 1. The Big Picture

## The Problem

Imagine 5 autonomous cars at an intersection. Each car has cameras and LiDAR sensors. Each car can only see PART of the world — a truck might block Car A's view of a pedestrian, but Car B on the other side can see them clearly.

The question: **How do these cars share intelligence to collectively see more than any individual car can?**

The constraint: **They can't send everything — bandwidth is limited (typical V2X: ~6 Mbps), there's latency (20-100ms), and packets get lost (5-20%).**

## The Solution Architecture

```
PERCEPTION (Python — ML models)
    "I see a pedestrian at (10, 20) with 0.91 confidence"
         ↓
    Observation struct
         ↓
WORLD MODEL (C++ — core intelligence)
    ├── FUSION: "Is this the same pedestrian Entity #17?"
    ├── KALMAN FILTER: "Predict where it is NOW based on velocity"
    ├── UNCERTAINTY: "Confidence decays over time without new observations"
    ├── SPATIAL INDEX: "Fast lookup: what's near this intersection?"
    └── LIFECYCLE: Tentative → Confirmed → Predicted → Stale → Lost
         ↓
COMMUNICATION (C++ — what to share)
    ├── PRIORITIZER: "Pedestrian is more important than a traffic sign"
    ├── BUDGET: "I have 4096 bytes this tick — select top observations"
    ├── CHANNEL: "Simulate 20ms latency, 5% packet loss"
    ├── RECEIVER MODEL: "Does Car B already know about this pedestrian?"
    └── SERIALIZER: "Convert observation to bytes for transmission"
         ↓
TRUST (C++ — who to believe)
    └── RELIABILITY: "Car C has been wrong 40% of the time — weight it lower"
         ↓
OTHER AGENTS receive, deserialize, and feed into THEIR world models
```

## Why C++ for the Core?

The world model updates at 10Hz (10 times per second). With 5 agents each producing 50 observations, that's 500 fusion operations per second. Each involves:
- Distance computations (spatial queries)
- Matrix operations (Kalman filter)
- Map lookups (entity matching)
- Sorting (prioritization)

Python can do this, but with overhead from garbage collection, dynamic typing, and the GIL. C++ gives:
- Deterministic memory layout (cache-friendly)
- No garbage collection pauses
- Zero overhead abstractions (templates, constexpr)
- Direct memory control

The Python side handles ML models (PyTorch), experiment orchestration, and analysis — where Python excels.

## Why the PImpl Pattern (used everywhere)?

Every major class uses this pattern:

```cpp
// In the header (.h)
class WorldModel {
 public:
  WorldModel(Config config);
  void Tick(double time);
 private:
  struct Impl;                    // Forward declaration only
  std::unique_ptr<Impl> impl_;   // Pointer to hidden implementation
};

// In the source (.cc)
struct WorldModel::Impl {
  Config config;
  FusionEngine fusion;            // Actual members defined HERE
  unordered_map<string, Entity> entities;
  // ...
};
```

**Why?**
1. **Compilation speed:** When you change `Impl` internals, only the `.cc` recompiles. Files that `#include "world_model.h"` don't need to recompile because the header didn't change.
2. **API stability:** The public header is a clean interface. Implementation details are invisible.
3. **Dependency hiding:** The header doesn't need to `#include` everything that `Impl` uses. This reduces header dependency chains.

This is standard practice at Google, Waymo, and most large C++ codebases.

---

# 2. Entry Point — The Agent

**Files:**
- `cpp/core/agent/agent.h` — Interface
- `cpp/core/agent/agent.cc` — Implementation

## What It Is

Each car/robot/drone in the system is an `Agent`. It's the top-level object that owns a `WorldModel` and orchestrates the per-tick processing loop.

## The Configuration

```cpp
struct AgentConfig {
  std::string agent_id;              // "car_A" — unique identifier
  Position3D initial_position;        // Where the agent starts
  WorldModelConfig world_model_config; // How the world model behaves
  ChannelConfig channel_config;       // Network characteristics
};
```

Every configurable parameter flows through these config structs. No magic numbers buried in code.

## The Implementation

```cpp
struct Agent::Impl {
  AgentConfig config;
  WorldModel world_model;   // Each agent has its OWN world model
  Position3D position;       // Current position (updated by simulation)
};
```

**Key insight:** Each agent has its OWN world model. They don't share memory. They share information by sending messages through the channel. This is realistic — real cars don't share RAM.

## The Step Function — The Heartbeat

```cpp
void Agent::Step(const std::vector<Observation>& local_observations,
                 const std::vector<ChannelMessage>& incoming_messages,
                 double current_time_s) {
  // 1. Ingest local observations.
  impl_->world_model.IngestObservations(local_observations, 1.0);

  // 2. TODO: Deserialize and ingest incoming cooperative observations.

  // 3. Tick world model.
  impl_->world_model.Tick(current_time_s);

  // 4. TODO: Run communication policy to decide what to share.
}
```

Every simulation tick (100ms at 10Hz), this function runs:
1. **Local observations** come from the perception model (Python). Trust = 1.0 for your own sensors.
2. **Incoming messages** come from other agents via the network. These would be deserialized and ingested with the sender's trust score (from ReliabilityTracker).
3. **Tick** advances the world model — decay confidence, predict positions, manage lifecycle.
4. **Communication policy** (TODO) decides which observations to share with others.

**Why trust = 1.0 for local observations?** You trust your own sensors fully. Other agents might be noisy, malfunctioning, or adversarial — their trust is tracked by the ReliabilityTracker.

## Non-Copyable, Movable

```cpp
Agent(const Agent&) = delete;            // Cannot copy
Agent& operator=(const Agent&) = delete;  // Cannot copy-assign
Agent(Agent&&) noexcept;                  // CAN move
Agent& operator=(Agent&&) noexcept;       // CAN move-assign
```

**Why?** Copying a WorldModel (with all its entities, trackers, spatial index) is expensive and semantically wrong — there should be ONE world model per agent. But moving (transferring ownership) is fine — e.g., when storing agents in a `std::vector`.

---

# 3. Data Types — Entity, Observation, Evidence

**Files:**
- `cpp/core/world_model/entity.h` — Type definitions
- `cpp/core/world_model/entity.cc` — Position/Velocity utility methods

## The Three Levels of Data

### Level 1: `Observation` — Raw detection from ONE agent

```cpp
struct Observation {
  std::string observation_id;   // "obs_42" — unique per detection
  std::string agent_id;         // "car_A" — who detected this
  std::string track_id;         // "track_7" — stable ID from the tracker

  ObjectClass object_class;     // kPedestrian, kVehicle, etc.
  Position3D position;          // World coordinates (meters)
  Velocity3D velocity;          // Speed and direction (m/s)
  double confidence;            // 0.0 to 1.0
  double timestamp_s;           // When observed
  int sensor_type;              // Camera=1, LiDAR=2, Radar=3, Fused=4

  double length, width, height; // Bounding box dimensions (meters)
  double heading;               // Direction facing (radians, [-π, π])
};
```

**What this represents:** One detection from one agent at one moment. When a camera or LiDAR model says "I see a pedestrian here with 91% confidence," that becomes an Observation.

**Why both `observation_id` and `track_id`?**
- `observation_id` is unique PER DETECTION — each frame produces new ones
- `track_id` is stable ACROSS FRAMES for the same physical object from the same agent's tracker
- This distinction matters: if an agent sends 3 observations of the same pedestrian over 3 frames, they have different `observation_id` but the same `track_id`

### Level 2: `TrackedEntity` — Fused belief about a real-world object

```cpp
struct TrackedEntity {
  std::string entity_id;              // "entity_17" — globally unique

  // FUSED classification
  ObjectClass object_class;
  double class_confidence;

  // BEST ESTIMATE of kinematic state (from Kalman filter + fusion)
  Position3D position;
  Velocity3D velocity;
  double heading;
  double heading_rate;

  // FUSED bounding box
  double length, width, height;

  // COMBINED confidence from all sources
  double confidence;

  // Lifecycle
  EntityState state;                  // Tentative/Confirmed/Predicted/Stale/Lost
  double first_seen_s;
  double last_updated_s;
  uint32_t observation_count;         // Total observations ever received
  uint32_t unique_source_count;       // Different agents that contributed

  // Provenance — WHO saw it, WHEN, with WHAT confidence
  std::vector<Evidence> supporting_evidence;

  // Trajectory predictions
  std::vector<PredictedState> predictions;

  // Safety
  ThreatLevel threat_level;
  double time_to_collision_s;         // -1 if not applicable

  // Flags
  bool needs_corroboration;           // Only one source — uncertain
  bool is_safety_critical;            // Close to collision
  bool active_acquisition_pending;    // We've asked another agent for more info
};
```

**Why this is different from Observation:** An Observation is raw, noisy, from one source. A TrackedEntity is the system's BEST BELIEF — combining multiple observations from multiple agents, smoothed through a Kalman filter, with full provenance tracking. Decisions are based on TrackedEntities, never raw Observations.

**The lifecycle state machine:**

```
TENTATIVE (single source, unconfirmed)
    ↓ gets 2+ unique sources AND confidence ≥ threshold
CONFIRMED (multiple independent sources agree)
    ↓ no new observation for stale_timeout_s (default 2.0s)
PREDICTED (running on Kalman prediction only, no fresh data)
    ↓ confidence drops below stale_confidence_threshold
STALE (losing track — confidence low, no observations)
    ↓ no observation for removal_timeout_s (default 5.0s)
LOST → removed from all data structures
```

**Why this lifecycle?**
- TENTATIVE: One car says "pedestrian." Maybe it's a false positive. Don't trust it for safety decisions yet.
- CONFIRMED: TWO independent cars see it. Much more likely to be real.
- PREDICTED: Nobody has seen it for 2 seconds. It's probably still there (things don't vanish), but we're extrapolating — uncertainty grows.
- STALE: Confidence has decayed a lot. Maybe it left, maybe sensors lost it.
- LOST: It's gone. Clean up memory.

### Level 3: `Evidence` — Provenance record

```cpp
struct Evidence {
  std::string agent_id;
  std::string observation_id;
  double confidence;
  double timestamp_s;
  int sensor_type;
};
```

**Why track this?** When Entity #17 has confidence 0.92, you need to know WHY. Evidence tells you: "Agent A (LiDAR, confidence 0.91, at t=1.2s), Agent B (camera, confidence 0.88, at t=1.3s)." This enables:
- **Debugging:** "Why did the system think a pedestrian was there?" → show the evidence chain
- **Adversarial detection:** "All evidence comes from Agent C, who has trust=0.3" → suspicious
- **LLM explanation layer:** "Entity confirmed by 3 independent LiDAR sensors from different vehicles"
- **Active acquisition:** "Only one source — which other agent could corroborate?"

### Helper types

```cpp
struct Position3D {
  double x = 0.0, y = 0.0, z = 0.0;
  double DistanceTo(const Position3D& other) const;  // 3D Euclidean distance
};

struct Velocity3D {
  double vx = 0.0, vy = 0.0, vz = 0.0;
  double Magnitude() const;  // Speed = sqrt(vx² + vy² + vz²)
};
```

**Default initialization to 0.0:** Every field has a default. This means you can never have uninitialized memory — a common source of bugs in C++.

### The enums

```cpp
enum class ObjectClass : int {
  kUnknown = 0,
  kVehicle = 1,
  kPedestrian = 2,
  kCyclist = 3,
  // ...
};
```

**Why `enum class` instead of `enum`?** Strongly typed. You can't accidentally write `if (object_class == entity_state)` — the compiler would reject it. Also, values are scoped: `ObjectClass::kVehicle`, not just `kVehicle` polluting the namespace.

**Why `kUnknown = 0`?** If a detector can't classify an object, it reports `kUnknown`. The fusion engine allows matching unknown objects to any class (since we don't know yet). This prevents unclassified detections from being dropped.

---

# 4. Fusion Engine — Matching & Combining Observations

**Files:**
- `cpp/core/world_model/fusion.h` — Interface
- `cpp/core/world_model/fusion.cc` — Implementation (~220 lines)

This is the most complex component. It answers three questions:
1. "Is this new observation about an EXISTING entity or a NEW one?" (Association)
2. "How do I update an entity with new evidence?" (Fusion)
3. "What if agents disagree?" (Conflict resolution)

## Configuration

```cpp
struct FusionConfig {
  double association_max_distance_m = 5.0;   // Max distance to match obs↔entity
  double trust_weight = 0.5;                 // How much agent trust matters
  double recency_weight = 0.3;               // How much recency matters
  double sensor_weight = 0.2;                // How much sensor type matters
  double conflict_resolution_threshold = 0.6; // Agreement needed to resolve conflicts
};
```

## `Associate()` — Observation-to-Entity Matching

### The Problem

Car A says "pedestrian at (10.0, 20.0)." We already track Entity #17 (pedestrian at (10.3, 19.8)). Are they the same person?

We also track Entity #42 (vehicle at (12.0, 20.0)). That's also close. Which one does the observation belong to? Or is it a brand new entity?

### The Algorithm: Greedy Nearest-Neighbor

```
1. For every (observation, entity) pair:
   - Compute distance
   - If distance > 5.0m → skip (too far)
   - If observation says "pedestrian" and entity is "truck" → skip (class mismatch)
   - Otherwise → add to candidate list

2. Sort candidates by distance (closest first)

3. Greedy assignment:
   - Take the closest pair
   - Mark both as "matched"
   - Take the next closest pair (skipping already-matched)
   - Repeat

4. Any unmatched observations → new entities (entity_id = "")
```

### Why Greedy Instead of Hungarian Algorithm?

The Hungarian algorithm gives the mathematically OPTIMAL assignment (minimum total distance). It runs in O(n³). With 50 observations and 200 entities, that's 50 × 200 = 10,000 candidate pairs and the Hungarian would take ~10,000³ = 10^12 operations per agent per tick — way too slow.

Greedy nearest-neighbor is O(n² log n) due to sorting. It gives near-optimal results because:
- Most observations are clearly closest to one entity (not ambiguous)
- The distance threshold (5m) eliminates most impossible matches
- Class filtering further reduces candidates

Production autonomous driving systems at Waymo and similar use greedy or auction-based methods, not Hungarian.

### Class Compatibility Check

```cpp
if (observations[oi].object_class != ObjectClass::kUnknown &&
    existing_entities[ei].object_class != ObjectClass::kUnknown &&
    observations[oi].object_class != existing_entities[ei].object_class) {
  continue;  // Class mismatch
}
```

**Three conditions must ALL be true to reject:**
1. The observation has a known class (not Unknown)
2. The entity has a known class (not Unknown)
3. They disagree

This means:
- Unknown observation + Vehicle entity → ALLOWED (might get classified later)
- Pedestrian observation + Unknown entity → ALLOWED
- Pedestrian observation + Truck entity → REJECTED

### Association Score

```cpp
double score = 1.0 - (c.distance / config.association_max_distance_m);
```

Score ranges from 1.0 (same position) to 0.0 (at the distance threshold). This score is returned in `AssociationResult` for diagnostics but doesn't affect the greedy assignment — the sort by raw distance handles that.

## `FuseObservation()` — Updating an Entity with New Evidence

When an observation is matched to an existing entity, we need to UPDATE the entity's state.

### Position Update: Weighted Running Average

```cpp
double weight = obs.confidence * agent_trust;
double old_weight = entity.confidence;
double total_weight = old_weight + weight;
double alpha = weight / total_weight;

entity.position.x += alpha * (obs.position.x - entity.position.x);
```

**What this does mathematically:**

```
new_position = old_position + alpha × (observation - old_position)
             = (1 - alpha) × old_position + alpha × observation
             = weighted average
```

Where `alpha = new_weight / (old_weight + new_weight)`.

**Example:**
- Entity at (10.0, 20.0) with confidence 0.8
- New observation at (12.0, 22.0) with confidence 0.7, agent trust 0.9
- weight = 0.7 × 0.9 = 0.63
- alpha = 0.63 / (0.8 + 0.63) = 0.44
- new_x = 10.0 + 0.44 × (12.0 - 10.0) = 10.0 + 0.88 = 10.88
- new_y = 20.0 + 0.44 × (22.0 - 20.0) = 20.0 + 0.88 = 20.88

The position moved 44% toward the new observation. If the new observation had low confidence (0.2), alpha would be ~0.18 and the position would barely move.

**Why not simple average?** If you average all N observations equally, observation #1 from 10 seconds ago has the same weight as the most recent one. The running weighted average naturally weights recent, high-confidence observations more.

### Heading Update: Circular Mean

```cpp
double diff = obs.heading - entity.heading;
while (diff > M_PI) diff -= 2.0 * M_PI;
while (diff < -M_PI) diff += 2.0 * M_PI;
entity.heading += alpha * diff;
```

**Why this is special:** Angles wrap around. 359° and 1° are only 2° apart, but naive subtraction gives 358°. The normalization to [-π, π] ensures we always interpolate the SHORT way around the circle.

**Example:**
- Entity heading: 350° (= -10° = -0.175 rad)
- Observation heading: 5° (= 0.087 rad)
- diff = 0.087 - (-0.175) = 0.262 rad (15°) → correct, short way
- If entity was 350° and observation was 190°: diff would normalize to go the short way too

### Confidence Update: Bayesian Independent Evidence

```cpp
entity.confidence = 1.0 - (1.0 - entity.confidence) * (1.0 - weight);
```

This is the formula for combining independent evidence:

```
P(object exists | evidence A AND evidence B) = 1 - P(not A) × P(not B)
```

**Example:**
- Current confidence: 0.7 (30% chance it's NOT there)
- New evidence: 0.63 (weight = confidence × trust = 0.7 × 0.9)
- Combined: 1 - (1 - 0.7) × (1 - 0.63) = 1 - 0.3 × 0.37 = 1 - 0.111 = 0.889

**Why this formula?**
- It has diminishing returns: going from 0.5 to 0.75 is easy, going from 0.95 to 0.99 is hard. This matches reality — it takes a LOT of evidence to be 99% sure.
- It's commutative: the order observations arrive doesn't matter.
- Two weak sources (0.5 and 0.5) give 0.75 — stronger than either alone. This is the power of multi-agent cooperation.

### Unique Source Tracking

```cpp
bool new_source = true;
for (const auto& ev : entity.supporting_evidence) {
  if (ev.agent_id == obs.agent_id) {
    new_source = false;
    break;
  }
}
if (new_source) {
  entity.unique_source_count++;
}
```

**Why this matters:** An entity seen by 5 different cars is much more trustworthy than an entity seen 5 times by the same car. `unique_source_count` drives the Tentative → Confirmed transition.

### Evidence History Cap

```cpp
constexpr size_t kMaxEvidence = 50;
if (entity.supporting_evidence.size() > kMaxEvidence) {
  entity.supporting_evidence.erase(entity.supporting_evidence.begin());
}
```

**Why 50?** A long-lived entity at 10Hz with 5 agents would accumulate 50 observations/second. After 10 seconds, that's 500 evidence records. Capping at 50 keeps memory bounded while preserving enough history for analysis. The oldest evidence is removed first (FIFO).

## `ResolveConflict()` — When Agents Disagree

### The Problem

Agent A (trust 0.9): "Vehicle at (10, 20), confidence 0.8"
Agent B (trust 0.9): "Vehicle at (10.5, 20), confidence 0.85"
Agent C (trust 0.9): "Pedestrian at (10.2, 20), confidence 0.7"

Two say Vehicle, one says Pedestrian. What do we believe?

### Weighted Majority Vote

```cpp
vote_weight = confidence × trust
```

Agent A: Vehicle vote = 0.8 × 0.9 = 0.72
Agent B: Vehicle vote = 0.85 × 0.9 = 0.765
Agent C: Pedestrian vote = 0.7 × 0.9 = 0.63

Vehicle total: 0.72 + 0.765 = 1.485
Pedestrian total: 0.63
Grand total: 2.115

Agreement ratio: 1.485 / 2.115 = 0.70 (70%)

Since 70% ≥ 60% (threshold) → accept Vehicle classification.

### What If Nobody Gets 60%?

If the vote split was 55%/45%, nobody wins. The system:
1. Cuts `class_confidence` by half (we're unsure)
2. Sets `needs_corroboration = true` (flagged for active acquisition)
3. Keeps the old classification (conservative — don't change on ambiguous evidence)

### Position From Agreeing Observers

Only positions from agents that agreed on the class are averaged. If we accept "Vehicle," we don't include Agent C's pedestrian position in the average — they might be looking at something else entirely.

### Confidence Penalty

```cpp
entity.confidence *= 0.8;
```

Conflict means uncertainty. Even if we resolve it, we penalize confidence by 20%. This is conservative — better to be less confident and correct than overconfident and wrong.

## `CreateEntity()` — Birth of a New Entity

When an observation doesn't match any existing entity, we create a new one:

```cpp
TrackedEntity entity;
entity.entity_id = GenerateEntityId();      // "entity_1", "entity_2", ...
entity.position = obs.position;              // Start at observation position
entity.confidence = obs.confidence * agent_trust;  // Weighted by trust
entity.state = EntityState::kTentative;      // Single source — unconfirmed
entity.needs_corroboration = true;           // Flag for active acquisition
```

**Entity ID generation:**

```cpp
static uint64_t g_next_entity_id = 1;
static std::string GenerateEntityId() {
  return "entity_" + std::to_string(g_next_entity_id++);
}
```

This is a simple monotonic counter. In a multi-threaded system, this would need to be `std::atomic`. Currently single-threaded per agent (by design — documented in world_model.h).

**Why `static`?** Entity IDs are globally unique across all agents. A static counter ensures "entity_1" is never used twice (within a single process). In a distributed system, you'd prefix with agent_id.

---

# 5. Uncertainty Manager — Confidence Decay & Boosting

**Files:**
- `cpp/core/world_model/uncertainty.h` — Interface + Config
- `cpp/core/world_model/uncertainty.cc` — Implementation

## Why Uncertainty Matters

A pedestrian detected 0.1 seconds ago at confidence 0.95 is very reliable. The SAME detection after 5 seconds with no new observations? Much less reliable — the pedestrian could have moved, left, been a false positive.

Traditional systems treat confidence as static. OmniCopilot treats confidence as a LIVING value that decays without evidence and grows with corroboration.

## Configuration

```cpp
struct UncertaintyConfig {
  double confidence_halflife_s = 3.0;   // Confidence halves every 3 seconds
  double min_confidence = 0.01;          // Never decay below this
  double stale_threshold = 0.15;         // Below this = entity is stale
  double corroboration_boost = 0.15;     // Bonus when new source agrees
  double max_confidence = 0.99;          // Never exceed this (nothing is 100% certain)
};
```

**Why max_confidence = 0.99?** Nothing in perception is 100% certain. Even with 10 agents all agreeing, there's always a chance of systematic error (e.g., all cameras fooled by the same visual illusion). Capping at 0.99 prevents overconfidence.

## `Decay()` — Exponential Confidence Decay

```cpp
double UncertaintyManager::Decay(double current_confidence, double elapsed_s) const {
  double decay_factor = std::pow(2.0, -elapsed_s / config_.confidence_halflife_s);
  double decayed = current_confidence * decay_factor;
  return std::max(decayed, config_.min_confidence);
}
```

### The Math

```
confidence(t) = confidence(0) × 2^(-t / halflife)
```

With halflife = 3.0 seconds:
- After 0 seconds: factor = 2^0 = 1.0 (no change)
- After 3 seconds: factor = 2^(-1) = 0.5 (halved)
- After 6 seconds: factor = 2^(-2) = 0.25 (quartered)
- After 9 seconds: factor = 2^(-3) = 0.125

So an entity with confidence 0.9:
- After 3s: 0.9 × 0.5 = 0.45
- After 6s: 0.9 × 0.25 = 0.225
- After 9s: 0.9 × 0.125 = 0.1125

**Why exponential (not linear)?** Linear decay (lose 0.1 per second) would make 0.9 reach 0.0 in exactly 9 seconds — a cliff edge. Exponential decay is smooth and never reaches zero. The min_confidence floor (0.01) provides a safety net.

**Why halflife of 3 seconds?** In driving at city speeds (30-50 km/h), a pedestrian moves ~1.5m/s. In 3 seconds, they've moved ~4.5 meters. That's a significant position change — our Kalman prediction becomes increasingly uncertain. The halflife matches the timescale at which our prediction quality degrades.

## `Boost()` — Confidence Increase from New Evidence

```cpp
double UncertaintyManager::Boost(double current_confidence,
                                 double observation_confidence,
                                 double agent_trust) const {
  double effective_obs = observation_confidence * agent_trust;
  double combined = 1.0 - (1.0 - current_confidence) * (1.0 - effective_obs);
  combined += config_.corroboration_boost * agent_trust;
  return std::clamp(combined, config_.min_confidence, config_.max_confidence);
}
```

### The Math

Step 1: Bayesian combination
```
P(A or B) = 1 - P(not A) × P(not B)
combined = 1 - (1 - 0.7) × (1 - 0.63) = 1 - 0.3 × 0.37 = 0.889
```

Step 2: Corroboration bonus
```
combined += 0.15 × 0.9 = 0.889 + 0.135 = 1.024
```

Step 3: Clamp to [0.01, 0.99]
```
result = 0.99 (capped)
```

**Why the corroboration bonus?** The Bayesian formula is mathematically correct for independent evidence, but in practice, having a NEW agent (not the same agent again) see the entity is EXTRA valuable — it means the entity is visible from a different viewpoint, with different sensors, reducing the chance of systematic error. The bonus rewards this.

## `CombineIndependent()` — Batch Confidence Combination

```cpp
double UncertaintyManager::CombineIndependent(const double* confidences,
                                              const double* trusts,
                                              uint32_t count) const {
  double product_of_complements = 1.0;
  for (uint32_t i = 0; i < count; ++i) {
    double effective = confidences[i] * trusts[i];
    effective = std::clamp(effective, 0.0, 1.0);
    product_of_complements *= (1.0 - effective);
  }
  double combined = 1.0 - product_of_complements;
  return std::clamp(combined, config_.min_confidence, config_.max_confidence);
}
```

**Example:** Three independent sources with confidence 0.7, 0.8, 0.6 (all trust 1.0):

```
Product of complements = (1-0.7) × (1-0.8) × (1-0.6) = 0.3 × 0.2 × 0.4 = 0.024
Combined = 1 - 0.024 = 0.976
```

Even though no individual source is above 0.8, THREE independent sources agreeing pushes confidence to 0.976. This is the mathematical foundation of WHY cooperative perception helps.

**Why raw pointers (`const double*`) instead of `std::vector`?** Performance. This function might be called in tight loops. Passing pointers avoids vector copy overhead and allows calling with data from any container (vector, array, stack). This is a common pattern in numerical C++ code.

## `IsStale()` and `IsSafetyCritical()`

```cpp
bool UncertaintyManager::IsStale(double confidence) const {
  return confidence <= config_.stale_threshold;  // Default: 0.15
}

bool UncertaintyManager::IsSafetyCritical(double confidence,
                                          double time_to_collision_s) const {
  constexpr double kMinConfidenceForSafety = 0.3;
  constexpr double kCriticalTTC_s = 5.0;
  return confidence >= kMinConfidenceForSafety &&
         time_to_collision_s >= 0.0 &&
         time_to_collision_s <= kCriticalTTC_s;
}
```

**Safety-critical logic:** An entity is flagged as safety-critical only if:
1. We're reasonably confident it exists (≥ 0.3) — not noise
2. It has a valid time-to-collision (≥ 0) — not "unknown"
3. The TTC is ≤ 5 seconds — it's close enough to matter

**Why not lower the confidence threshold?** Below 0.3, it's more likely a false positive. Triggering safety responses for false positives (phantom braking) is dangerous — it can cause rear-end collisions.

---

# 6. Spatial Index — Fast Location Queries

**Files:**
- `cpp/core/world_model/spatial_index.h` — Interface
- `cpp/core/world_model/spatial_index.cc` — Implementation (~135 lines)

## Why This Exists

When a new observation arrives at position (10, 20), we need to find all existing entities near that point for association. Without a spatial index, we'd scan ALL entities:

```
for each entity in all_entities:   // 1000 entities
  compute distance(observation, entity)
```

That's O(N) per observation, and with 50 observations per agent: 50 × 1000 = 50,000 distance computations per tick. With a spatial index, we only check entities in nearby grid cells: typically 10-50 distance computations.

## How Grid-Based Spatial Hashing Works

The world is divided into a grid of cells (default 5m × 5m):

```
     0     5    10    15    20    25
  ┌─────┬─────┬─────┬─────┬─────┐
0 │     │     │     │     │     │
  ├─────┼─────┼─────┼─────┼─────┤
5 │     │  A  │     │     │     │
  ├─────┼─────┼─────┼─────┼─────┤
10│     │     │ B,C │     │     │
  ├─────┼─────┼─────┼─────┼─────┤
15│     │     │     │  D  │     │
  └─────┴─────┴─────┴─────┴─────┘
```

Entity A at (7, 6) → cell (1, 1)
Entities B,C at (12, 11) and (13, 12) → cell (2, 2)
Entity D at (18, 16) → cell (3, 3)

**Query "everything within 6m of (10, 10)":**
- Center (10, 10) → cell (2, 2)
- Radius 6m → need to check cells (0,0) through (3,3) — the cells that overlap the query circle
- Only cells (1,1), (2,2), (3,3) have entities
- Check actual distances: B (2.2m ✅), C (3.6m ✅), A (5.0m ✅), D (10.0m ❌)

## The Data Structures

```cpp
struct SpatialIndex::Impl {
  SpatialIndexConfig config;

  // Grid: cell → set of entity IDs
  std::unordered_map<Cell, std::unordered_set<std::string>, CellHash> grid;

  // Reverse lookup: entity_id → (cell, position)
  struct EntityEntry {
    Cell cell;
    Position3D position;
  };
  std::unordered_map<std::string, EntityEntry> entities;
};
```

**Two maps — why?**
- `grid`: Given a cell coordinate, find all entities in it. Used for queries.
- `entities`: Given an entity ID, find its cell and position. Used for updates and removes.

Without the reverse lookup, removing an entity would require scanning the entire grid to find which cell it's in — O(total cells).

## Cell Hash

```cpp
struct CellHash {
  size_t operator()(const std::pair<int, int>& cell) const {
    auto h1 = std::hash<int>{}(cell.first);
    auto h2 = std::hash<int>{}(cell.second);
    return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};
```

**What's `0x9e3779b97f4a7c15`?** It's derived from the golden ratio (φ = 1.618...). Specifically: `2^64 / φ`. This is the Fibonacci hashing constant — it produces well-distributed hash values with minimal collisions. It's a standard technique from Knuth's "The Art of Computer Programming."

**Why not just XOR?** Plain `h1 ^ h2` has terrible collision properties: cells (1,2) and (2,1) would hash to the same value. The multiplication and bit-shifting spread the bits.

## `Upsert()` — Insert or Update

```cpp
void SpatialIndex::Upsert(const std::string& entity_id, Position3D position) {
  Cell new_cell = impl_->ToCell(position);

  auto it = impl_->entities.find(entity_id);
  if (it != impl_->entities.end()) {
    // Existing entity — check if cell changed
    Cell old_cell = it->second.cell;
    if (old_cell != new_cell) {
      // Remove from old cell, add to new cell
      impl_->grid[old_cell].erase(entity_id);
      if (impl_->grid[old_cell].empty()) impl_->grid.erase(old_cell);
      impl_->grid[new_cell].insert(entity_id);
    }
    it->second = {new_cell, position};
  } else {
    // New entity
    impl_->grid[new_cell].insert(entity_id);
    impl_->entities[entity_id] = {new_cell, position};
  }
}
```

**Why check `old_cell != new_cell`?** If an entity moves slightly within the same cell (e.g., from (10.1, 10.2) to (10.3, 10.4) — both in cell (2,2)), we don't need to modify the grid. Only the position in the reverse lookup changes. This is an optimization — grid modifications (hash set insert/erase) are more expensive than just updating a position value.

**Why erase empty cells?** `if (impl_->grid[old_cell].empty()) impl_->grid.erase(old_cell);` — Without this, the grid accumulates empty cell entries over time. Each empty cell wastes memory and slows down iteration.

## `QueryRadius()` — Find Nearby Entities

```cpp
// Determine which cells could contain entities within the radius
int min_ix = static_cast<int>(std::floor((center.x - radius_m) / cell_size));
int max_ix = static_cast<int>(std::floor((center.x + radius_m) / cell_size));
int min_iy = static_cast<int>(std::floor((center.y - radius_m) / cell_size));
int max_iy = static_cast<int>(std::floor((center.y + radius_m) / cell_size));
```

**This computes the bounding box of the query circle in cell coordinates.** If center is (10, 10) and radius is 6m with cell_size 5m:
- min_ix = floor((10-6)/5) = floor(0.8) = 0
- max_ix = floor((10+6)/5) = floor(3.2) = 3
- Similarly for y

We then scan cells (0,0) through (3,3) — 16 cells. Most are empty. Only cells with entities are checked, and those entities are verified with actual distance computation (the grid is a coarse filter, exact distance is the fine filter).

**Results are sorted by distance:** This is important for the fusion engine's greedy association — closest entities are matched first.

## `QueryKNearest()` — K Nearest Neighbors

```cpp
// Brute-force over all entities — acceptable for typical entity counts (<1000)
std::partial_sort(all.begin(), all.begin() + n, all.end(), ...);
```

**Why brute force?** For <1000 entities, computing all distances and partial-sorting is fast (<1ms). A proper KNN algorithm (expanding-ring search) would be faster for millions of entities but adds complexity we don't need.

**`std::partial_sort`** is O(N log K) — it only sorts enough to find the K smallest elements, not the entire array. For K=5 and N=1000, this is ~50× faster than full sorting.

---

# 7. Temporal Tracker — Kalman Filter

**Files:**
- `cpp/core/world_model/temporal_tracker.h` — Interface
- `cpp/core/world_model/temporal_tracker.cc` — Implementation (~230 lines)

## Why This Exists

Between observations, we need to PREDICT where an entity is. If a pedestrian was at (10, 20) moving at (1.0, 0.0) m/s, and 0.5 seconds have passed, they're probably at (10.5, 20).

But real motion is noisy — the pedestrian might speed up, slow down, or turn. The Kalman filter handles this uncertainty mathematically.

## What Is a Kalman Filter?

It's a recursive algorithm that maintains a BELIEF about the state of an object:
- **State:** What we think (position + velocity) — called `x`
- **Uncertainty:** How unsure we are — called `P` (covariance matrix)

Two steps:
1. **Predict:** "Based on current velocity, where will it be next?" (uncertainty grows)
2. **Update:** "We got a new measurement — correct our belief." (uncertainty shrinks)

## Our State Vector

```
x = [px, py, vx, vy]   (4-dimensional)
```

Position (px, py) in meters, velocity (vx, vy) in meters/second.

**Why not 3D (include z)?** In driving, motion is approximately 2D (ground plane). Z-axis variation is tiny (cars don't fly). Including Z in the Kalman state would waste computation. We track Z separately with simple exponential smoothing.

## The Covariance Matrix P

```
P = 4×4 matrix stored as array of 16 doubles (row-major)

P = [σ²_x    σ_xy   σ_xvx  σ_xvy ]
    [σ_xy    σ²_y   σ_yvx  σ_yvy ]
    [σ_xvx   σ_yvx  σ²_vx  σ_vxvy]
    [σ_xvy   σ_yvy  σ_vxvy σ²_vy ]
```

The diagonal elements tell us how uncertain we are:
- `P[0,0]` = σ²_x = uncertainty in x position (m²)
- `P[1,1]` = σ²_y = uncertainty in y position (m²)
- `P[2,2]` = σ²_vx = uncertainty in x velocity ((m/s)²)
- `P[3,3]` = σ²_vy = uncertainty in y velocity ((m/s)²)

The off-diagonal elements capture correlations (e.g., if position uncertainty is correlated with velocity uncertainty).

## Predict Step

### State Transition

```
px' = px + vx × dt
py' = py + vy × dt
vx' = vx            (constant velocity model)
vy' = vy
```

This is the "constant velocity" model — we assume velocity doesn't change. It's the simplest useful model. More complex models (constant acceleration, turn rate) are possible but add complexity.

### The State Transition Matrix F

```
F = [1  0  dt  0 ]
    [0  1  0   dt]
    [0  0  1   0 ]
    [0  0  0   1 ]
```

State prediction: `x' = F × x`

### Covariance Prediction

```
P' = F × P × F^T + Q
```

Where Q is the process noise matrix — "how much could the true state change beyond our model?"

**We apply F analytically instead of doing matrix multiplication:**

```cpp
// P'[i][j] = P[i][j] + dt*P[i+2][j] + dt*P[i][j+2] + dt²*P[i+2][j+2]
```

This is equivalent to F × P × F^T but avoids a full 4×4 matrix multiply (which would be 64 multiplications + 48 additions). The analytical form has only ~16 operations. For a hot loop running at 10Hz × 1000 entities = 10,000 times/second, this matters.

### Process Noise Q

```cpp
Mat4(Q, 0, 0) = dt4 * qv + qp;   // Position variance grows with time⁴ × velocity noise + time × position noise
Mat4(Q, 0, 2) = dt3 * qv;         // Cross-term between position and velocity
Mat4(Q, 2, 2) = dt2 * qv;         // Velocity variance grows with time² × velocity noise
```

**Intuition:** Even with a constant-velocity model, real objects change velocity. Process noise captures this. Higher process noise = the filter trusts predictions less and relies more on observations.

## Update Step

When a new observation arrives:

### Innovation (measurement residual)

```cpp
double y0 = obs.position.x - state[0];  // Difference between measurement and prediction
double y1 = obs.position.y - state[1];
```

"How far off was our prediction from what we actually measured?"

### Measurement Noise R

```cpp
double r = base_noise / confidence;
```

Low confidence observation → high measurement noise → the filter trusts it less.
High confidence observation → low measurement noise → the filter trusts it more.

**Example:**
- Confidence 0.9: R = 2.0 / 0.9 = 2.22 m²
- Confidence 0.3: R = 2.0 / 0.3 = 6.67 m² (much noisier → less trusted)

### Innovation Covariance S

```
S = H × P × H^T + R
```

Where H is the measurement matrix (selects position from state):
```
H = [1  0  0  0]
    [0  1  0  0]
```

Since H just selects the top-left 2×2 block of P:
```
S = P[0:2, 0:2] + R×I
```

### Kalman Gain K

```
K = P × H^T × S^(-1)
```

K is a 4×2 matrix that tells us how much to adjust each state element based on the innovation.

**S is 2×2, so we invert it analytically:**

```cpp
double det = s00 * s11 - s01 * s10;
double inv_det = 1.0 / det;
```

For a 2×2 matrix, the inverse is:
```
[a b]^(-1) = (1/det) × [ d  -b]
[c d]                   [-c   a]
```

This avoids calling a linear algebra library for a trivial inversion.

### State Update

```cpp
state[i] += K[i*2+0] * y0 + K[i*2+1] * y1;
```

"Move our state estimate toward the measurement, proportionally to the Kalman gain."

### Covariance Update

```cpp
// P = (I - K × H) × P
```

The uncertainty SHRINKS after an update (we got new information). The Kalman gain determines how much it shrinks.

## `GetPredictions()` — Future Trajectory

```cpp
for (int i = 1; i <= num_steps; ++i) {
  double t = step * i;
  ps.position = {px + vx * t, py + vy * t, pz + v_z * t};
  ps.confidence = decay * (1.0 / (1.0 + base_uncertainty));
}
```

Constant-velocity projection into the future. Confidence decays with horizon — predicting 0.5s ahead is much more reliable than 5s ahead. These predictions are used for:
- Collision risk assessment (time-to-collision)
- Communication prioritization (fast-moving objects near us are higher priority)
- Active acquisition planning (where to look next)

---

# 8. World Model — The Orchestrator

**Files:**
- `cpp/core/world_model/world_model.h` — Interface
- `cpp/core/world_model/world_model.cc` — Implementation (~250 lines)

Already covered in detail in section 2. The world model OWNS all sub-components and orchestrates the tick:

```
Tick():
  for each entity:
    1. Decay confidence (UncertaintyManager)
    2. Predict position (TemporalTracker)
    3. Update spatial index (SpatialIndex)
    4. Update lifecycle state (state machine)
    5. Check safety criticality (UncertaintyManager)
  6. Purge lost entities

IngestObservations():
  1. Associate observations to entities (FusionEngine)
  2. Fuse matched observations (FusionEngine)
  3. Create new entities for unmatched (FusionEngine)
  4. Update Kalman trackers (TemporalTracker)
  5. Update spatial index (SpatialIndex)
  6. Revive stale entities on re-observation
```

---

# 9. Budget Manager

**Files:**
- `cpp/core/communication/budget_manager.h` — Interface
- `cpp/core/communication/budget_manager.cc` — Implementation (~65 lines)

## Why This Exists

Each agent has limited bandwidth per tick. If the budget is 4096 bytes and each observation is ~256 bytes, you can send at most 16 observations. The BudgetManager tracks how much budget remains and reserves some for emergencies.

## How It Works

```
Total budget per tick: 4096 bytes
Critical reserve: 20% = 819 bytes
Available for normal messages: 3277 bytes
```

**Non-critical messages** (regular observations) can only use the non-reserved portion.
**Critical messages** (emergency braking, collision warning) can dip into the reserve.

This ensures that even when bandwidth is mostly consumed by regular traffic, there's always capacity for life-saving messages.

```cpp
bool BudgetManager::Allocate(uint32_t bytes, bool is_critical) {
  if (is_critical) {
    if (bytes <= critical_remaining_bytes) {
      // Use from reserve
      critical_remaining_bytes -= bytes;
      remaining_bytes -= bytes;
      return true;
    }
  }
  // Non-critical cannot touch reserve
  uint32_t available = remaining_bytes - critical_remaining_bytes;
  if (!is_critical && bytes > available) return false;
  // ...
}
```

---

# 10. Serializer

**Files:**
- `cpp/core/communication/serializer.h` — Interface
- `cpp/core/communication/serializer.cc` — Implementation (~115 lines)

## Why This Exists

To send an Observation from Agent A to Agent B, we need to convert the C++ struct into bytes (serialize) and convert bytes back to a struct (deserialize). The Serializer handles this.

## The Format

Custom binary format (not Protobuf yet — that requires proto code generation to be wired in):

```
[string: observation_id]     // 4-byte length + string bytes
[string: agent_id]
[string: track_id]
[int32: object_class]
[int32: sensor_type]
[double: position.x]         // 8 bytes each
[double: position.y]
[double: position.z]
[double: velocity.vx]
[double: velocity.vy]
[double: velocity.vz]
[double: confidence]
[double: timestamp_s]
[double: length]
[double: width]
[double: height]
[double: heading]
```

**Strings use length-prefix encoding:**
```
[4 bytes: string length] [N bytes: string content]
```

This allows variable-length strings without delimiters.

## Round-Trip Guarantee

```cpp
auto bytes = serializer.SerializeObservation(obs);
auto decoded = serializer.DeserializeObservation(bytes.data(), bytes.size());
// decoded == obs (every field matches)
```

This is verified by the `SerializerTest.RoundTripObservation` test — critical for correctness.

## `EstimateSize()`

```cpp
uint32_t Serializer::EstimateSize(const Observation& obs) const {
  uint32_t fixed = 104;  // 12 doubles (96) + 2 int32s (8) = 104
  uint32_t strings = 12 + obs.observation_id.size() + obs.agent_id.size() + obs.track_id.size();
  return fixed + strings;
}
```

Used by the BudgetManager and Prioritizer to estimate how many observations fit in the bandwidth budget WITHOUT actually serializing them (which would be wasteful).

---

# 11. Channel — Network Simulation

**Files:**
- `cpp/core/communication/channel.h` — Interface
- `cpp/core/communication/channel.cc` — Implementation (~150 lines)

## Why This Exists

In the real world, messages between cars don't arrive instantly. They face:
- **Latency:** 20-100ms delay
- **Jitter:** Variable delay (±10ms)
- **Packet loss:** 5-20% of messages never arrive
- **Bandwidth limits:** ~6 Mbps for C-V2X
- **Queue overflow:** Too many messages → some get dropped

The Channel simulates all of these realistically.

## Loss Models

### Bernoulli (Independent Loss)

```cpp
bool dropped = (random_0_to_1 < loss_rate);
```

Each packet has an independent probability of being dropped. Simple and commonly used.

### Gilbert-Elliott (Bursty Loss)

Real networks don't lose packets independently. They have "good" periods and "bad" periods:

```
Good state: low loss rate (e.g., 2%)
Bad state: high loss rate (e.g., 50%)

Transitions:
  Good → Bad with probability 0.05 per tick
  Bad → Good with probability 0.30 per tick
```

This models WiFi interference, obstructions, and network congestion — losses come in BURSTS, not uniformly.

## Message Lifecycle

```
1. Agent calls channel.Enqueue(message)
2. Check: queue full? → drop (total_dropped_queue++)
3. Check: bandwidth exceeded? → drop (total_dropped_queue++)
4. Check: packet loss? → drop (total_dropped_loss++)
5. Compute delivery time = enqueue_time + latency + random_jitter
6. Message enters in-flight queue

Later:
7. channel.Tick(current_time) called
8. All messages with delivery_time ≤ current_time are delivered
```

## Dynamic Condition Override

```cpp
void Channel::SetConditions(double loss_rate, double latency_ms, double bandwidth_bps);
```

This allows scenarios like:
- "At t=10s, network congestion doubles latency and loss"
- "At t=20s, one agent enters a tunnel — 100% loss"
- Used by the RL environment to train policies under varying conditions

---

# 12. Prioritizer — What to Send

**Files:**
- `cpp/core/communication/prioritizer.h` — Interface
- `cpp/core/communication/prioritizer.cc` — Implementation (~110 lines)

## Why This Exists

An agent has 50 observations but can only send 16 (bandwidth limit). Which 16 are most valuable?

This is the CORE INNOVATION of the project — hand-designed prioritization as a baseline, later replaced by a learned RL policy.

## Scoring Dimensions

Each observation gets 5 scores:

### 1. Safety Relevance (weight: 0.35)

```cpp
switch (obs.object_class) {
  case kPedestrian:      return 1.0;   // Highest — most vulnerable
  case kCyclist:         return 0.95;
  case kMotorcycle:      return 0.85;
  case kVehicle:         return 0.7;
  case kTruck:           return 0.65;
  case kStaticObstacle:  return 0.4;
  case kTrafficSign:     return 0.3;
  default:               return 0.2;
}
```

**Why pedestrians score highest?** They're the most vulnerable road users and the hardest to detect. Missing a pedestrian can be fatal. Missing a traffic sign is annoying but not deadly.

### 2. Confidence Score (weight: 0.15)

```cpp
scored.confidence_score = 1.0 - std::pow(2.0 * obs.confidence - 1.0, 2.0);
```

This is an inverted parabola that peaks at confidence ~0.5-0.7:

```
Confidence: 0.0 → Score: 0.0 (noise, not worth sending)
Confidence: 0.3 → Score: 0.84
Confidence: 0.5 → Score: 1.0 (PEAK — most informative)
Confidence: 0.7 → Score: 0.84
Confidence: 1.0 → Score: 0.0 (receiver likely already knows)
```

**Why peak at 0.5?** Counter-intuitive, but:
- Very low confidence (0.1) = probably noise. Not worth bandwidth.
- Very high confidence (0.95) = the receiver probably already sees it too. Redundant.
- Medium confidence (0.5) = genuinely uncertain. THIS is where sharing helps most.

### 3. Novelty (weight: 0.20)

```cpp
scored.novelty_score = 1.0 - obs.confidence;
```

Low confidence → high novelty (it's likely new/uncertain information).

**This is a rough approximation.** The REAL novelty score comes from the ReceiverModel — "does the receiver already know this?" But as a hand-designed fallback, low confidence correlates with novelty.

### 4. Time Sensitivity (weight: 0.15)

```cpp
scored.time_sensitivity_score = std::clamp(speed / 20.0, 0.0, 1.0);
```

Fast-moving objects are more time-sensitive — their position changes quickly, so stale information is more dangerous.
- Parked car (0 m/s): time_sensitivity = 0.0
- Walking pedestrian (1.5 m/s): time_sensitivity = 0.075
- City-speed car (14 m/s / 50 km/h): time_sensitivity = 0.7
- Highway-speed car (28 m/s / 100 km/h): time_sensitivity = 1.0

### 5. Receiver Benefit (weight: 0.15)

```cpp
scored.receiver_benefit_score = scored.safety_score * (1.0 - obs.confidence);
```

High safety × low confidence = the receiver would benefit most. This is the hand-designed proxy for the full receiver model.

### Total Score and Selection

```cpp
total = 0.35 × safety + 0.15 × confidence + 0.20 × novelty
      + 0.15 × time_sensitivity + 0.15 × receiver_benefit
```

All observations are scored, sorted by total score (descending), and the top-K that fit in the budget are selected.

**The RL policy (Phase 5) will REPLACE this hand-designed scoring** with a learned function. The experiments then compare: does learning beat engineering?

---

# 13. Receiver Model — Theory of Mind

**Files:**
- `cpp/core/communication/receiver_model.h` — Interface
- `cpp/core/communication/receiver_model.cc` — Partially implemented

## Why This Exists

Before sending an observation to Car B, we should ask: "Does Car B already know this?"

If Car B can see the same pedestrian from its own sensors, sending it our observation is REDUNDANT — wasting bandwidth that could carry genuinely new information.

The ReceiverModel maintains an estimate of what each other agent already knows — a form of **theory of mind** for AI agents.

## How It Works

For each receiver agent, we track:
- Their position (do they have line-of-sight?)
- Which entities we've previously told them about (with timestamps)
- Estimated confidence decay (their knowledge gets stale too)

### Novelty Estimation

```cpp
double ReceiverModel::EstimateNovelty(const std::string& receiver_id,
                                      const Observation& obs,
                                      double current_time_s) const {
  // If receiver has no record of this entity → novelty = 1.0 (completely new)
  // If they were told about it recently → novelty depends on how long ago
  double age = current_time_s - known.last_informed_time_s;
  double decay = 1.0 - std::exp(-age / belief_decay_halflife);
  return decay;
}
```

**The exponential decay means:** If we told Car B about the pedestrian 0.1 seconds ago, novelty ≈ 0 (they just learned). If we told them 5 seconds ago, novelty ≈ 0.8 (their information is getting stale). If we never told them, novelty = 1.0.

---

# 14. Reliability Tracker — Agent Trust

**Files:**
- `cpp/core/trust/reliability.h` — Interface
- `cpp/core/trust/reliability.cc` — Implementation (~100 lines)

## Why This Exists

Not all agents can be trusted equally:
- Agent A has great sensors → usually correct
- Agent B has dirty cameras → noisy detections
- Agent C might be malfunctioning → sending garbage
- Agent D might be adversarial → deliberately lying

The ReliabilityTracker learns each agent's reliability from their track record.

## How Trust Updates

### Exponential Moving Average (EMA)

```cpp
// On correct observation:
accuracy_recent = accuracy_recent × (1 - alpha) + alpha × 1.0;

// On incorrect observation:
accuracy_recent = accuracy_recent × (1 - alpha) + alpha × 0.0;
// Simplifies to:
accuracy_recent = accuracy_recent × (1 - alpha);
```

With alpha = 0.05:
- After a correct observation: trust increases slightly
- After an incorrect observation: trust decreases slightly
- Recent observations matter more than old ones (EMA property)

### Trust Formula

```cpp
trust_score = 0.7 × accuracy_recent + 0.3 × accuracy_lifetime;
```

**Why blend recent and lifetime?**
- `accuracy_recent` (EMA) responds quickly to changes: if an agent suddenly starts sending bad data, trust drops fast
- `accuracy_lifetime` (ratio of correct/total) is stable: an agent with 1000 correct observations doesn't become untrusted from 1 mistake

The 70/30 blend prioritizes recent behavior while anchoring to long-term performance.

### Calibration Period

```cpp
bool is_calibrated = (total_observations >= calibration_period);  // Default: 50

bool IsSuspicious(const std::string& agent_id, double threshold) const {
  return record.is_calibrated && record.trust_score < threshold;
}
```

**Why require calibration?** A new agent with 3 observations, 1 of which was wrong, has accuracy 66%. That DOESN'T mean it's unreliable — we just don't have enough data. The calibration period (50 observations) ensures we have statistical confidence before flagging agents.

### Trust Bounds

```cpp
trust_score = std::clamp(trust_score, min_trust, max_trust);
// min_trust = 0.05, max_trust = 0.99
```

**Why min_trust = 0.05 (not 0)?** Even a terrible agent might occasionally produce correct observations. Zero trust would permanently exclude them. The 5% floor allows recovery — if the malfunction is fixed, the agent can slowly regain trust.

**How trust affects the system:**
- When Agent C (trust 0.3) sends an observation, it's fused with weight = confidence × 0.3 — much less influence than Agent A (trust 0.95)
- When the fusion engine combines independent evidence, low-trust sources contribute less
- The prioritizer can weight receiver_benefit by receiver trust (don't waste bandwidth on messages from untrusted sources)

---

# 15. Proto Definitions — The Interface Contract

**Files:**
- `proto/observation.proto`
- `proto/entity.proto`
- `proto/world_state.proto`
- `proto/communication.proto`
- `proto/metrics.proto`

## Why Protobuf?

In a multi-language system (C++ runtime + Python ML), you need a shared type system. Protobuf provides:
- **Language-neutral type definitions:** Define once, generate C++ and Python code
- **Efficient serialization:** Compact binary format, faster than JSON
- **Schema evolution:** Add fields without breaking old code
- **Type safety:** Enums, nested messages, optional fields

## How They Map to C++ Types

The proto types are designed to be the **external** representation (for communication, storage, APIs), while the C++ structs (`Observation`, `TrackedEntity`) are the **internal** representation (for fast in-memory processing).

The Serializer converts between internal ↔ external formats.

**Current state:** Proto files are complete, but proto code generation isn't wired into the build yet (requires protobuf-devel headers). The C++ code uses its own structs internally and a custom binary serializer. When proto generation is set up, the custom serializer will be replaced with Protobuf's.

---

# 16. Python: Data Types & Validation

**Files:**
- `python/omnicopilot/data/schemas.py` — Pydantic validation schemas

## Why Pydantic?

At system boundaries (loading data from disk, receiving API responses, reading configs), data might be invalid. Pydantic validates data at runtime:

```python
class ObservationSchema(BaseModel):
    observation_id: str
    agent_id: str
    confidence: float = Field(ge=0.0, le=1.0)  # Must be between 0 and 1
    timestamp_s: float = Field(ge=0.0)           # Must be non-negative
```

If you try to create an observation with confidence=1.5, Pydantic raises `ValidationError` immediately — not 10 function calls later when something crashes mysteriously.

---

# 17. Python: Perception Module

**Files:**
- `python/omnicopilot/perception/detector.py` — 3D object detection
- `python/omnicopilot/perception/tracker.py` — Multi-object tracking
- `python/omnicopilot/perception/cooperative.py` — Cooperative models
- `python/omnicopilot/perception/export.py` — ONNX export

## The Architecture

```
Raw sensor data (LiDAR point cloud, camera images)
    ↓
BaseDetector.detect() → list[Detection3D]
    ↓
MultiObjectTracker.update() → list[Track]
    ↓
Convert to Observation structs → feed into C++ WorldModel
```

### `Detection3D` — What a detector outputs

```python
@dataclass(frozen=True)
class Detection3D:
    position: NDArray[np.float64]     # (3,) — x, y, z
    dimensions: NDArray[np.float64]   # (3,) — length, width, height
    heading: float
    velocity: NDArray[np.float64]     # (3,) — vx, vy, vz
    class_id: int
    class_scores: NDArray[np.float64] # Per-class probabilities
    confidence: float
    track_id: int = -1
```

**Why `frozen=True`?** Immutable. Once a detection is produced, it can't be modified. This prevents bugs where downstream code accidentally modifies detections.

### `BaseDetector` — Abstract interface

```python
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, sensor_data: dict[str, Any]) -> DetectionResult:
        """Run detection on sensor data."""

    @abstractmethod
    def get_num_classes(self) -> int:
        """Return number of object classes."""
```

**Why abstract?** Different detectors (PointPillars, CenterPoint, V2VNet) have very different internals but the same external interface. Code that uses a detector doesn't need to know WHICH detector — it just calls `detect()`.

### `V2VNetModel` — Cooperative perception

```python
class V2VNetModel(BaseCooperativeModel):
    """V2VNet: share intermediate features between vehicles."""
```

This wraps OpenCOOD's cooperative perception models. Instead of sharing raw observations, V2VNet shares compressed feature maps from the neural network's intermediate layers. This is more bandwidth-efficient but requires the model architecture to be designed for it.

---

# 18. Python: Fusion Module

**Files:**
- `python/omnicopilot/fusion/algorithms.py` — Fusion algorithms
- `python/omnicopilot/fusion/uncertainty.py` — Uncertainty estimation
- `python/omnicopilot/fusion/calibration.py` — Confidence calibration

## `WeightedAverageFusion` — The Working Implementation

```python
class WeightedAverageFusion(BaseFusionAlgorithm):
    def fuse(self, positions, confidences, trusts):
        weights = confidences * trusts
        weight_sum = weights.sum()
        fused_pos = (positions * weights[:, np.newaxis]).sum(axis=0) / weight_sum
        fused_conf = float(1.0 - np.prod(1.0 - confidences * trusts))
        return FusedEstimate(position=fused_pos, confidence=fused_conf, ...)
```

This is the Python equivalent of the C++ fusion — used for research and prototyping. Same math:
- Position: weighted average
- Confidence: 1 - product of complements (Bayesian independent evidence)

**Why have both C++ and Python versions?** The C++ version runs in the hot loop (real-time). The Python version is for:
- Research experiments (try different fusion algorithms quickly)
- Prototyping (test ideas before implementing in C++)
- Evaluation scripts (analyze results)

## `MCDropoutEstimator` — Uncertainty from Neural Networks

Neural networks output a confidence score, but that score is often BADLY CALIBRATED — a network saying "90% confident" might only be right 70% of the time.

MC Dropout estimates true uncertainty by:
1. Enable dropout layers during inference (normally only used during training)
2. Run the same input N times → get N different outputs
3. Measure variance across outputs → high variance = high uncertainty

This gives us HONEST uncertainty estimates to feed into the world model.

## `TemperatureScaling` — Calibrate Confidence

```
raw_confidence = 0.90    # What the model says
calibrated_confidence = 0.73  # What the actual accuracy is
```

Temperature scaling learns a single parameter T such that `softmax(logits / T)` produces well-calibrated probabilities. Simple, effective, standard in production ML systems.

### Expected Calibration Error (ECE)

```python
def expected_calibration_error(confidences, accuracies, num_bins=15):
    for each bin:
        ece += bin_size × |bin_accuracy - bin_confidence|
```

ECE measures how well confidence matches actual accuracy. Perfect calibration = ECE of 0. This is how we verify that our uncertainty estimates are honest.

---

# 19. Python: RL Environment & Policy

**Files:**
- `python/omnicopilot/rl/environment.py` — Gymnasium environment
- `python/omnicopilot/rl/policy.py` — Neural network policy
- `python/omnicopilot/rl/reward.py` — Reward function
- `python/omnicopilot/rl/train.py` — PPO training loop
- `python/omnicopilot/rl/analyze.py` — Emergent strategy analysis

## The RL Formulation

The hand-designed prioritizer uses fixed rules. The RL policy LEARNS what to communicate.

### State (what the policy sees)

```python
obs_dim = (
    num_observations_max * 7  # Per-obs: x, y, z, vx, vy, confidence, class
    + num_receivers * 3       # Per-receiver: x, y, estimated_knowledge
    + 4                       # Network state: bandwidth, latency, loss, queue
)
```

The policy sees: its own observations, what it thinks receivers know, and the current network conditions.

### Action (what the policy decides)

```python
self.action_space = spaces.Box(low=0.0, high=1.0, shape=(num_observations_max,))
```

For each observation, output a score between 0 and 1. Top-K observations (those that fit in the bandwidth budget) are transmitted.

### Reward (what we optimize for)

```python
class PerceptionImprovementReward:
    def compute(self, ...):
        total = (
            perception_weight × delta_mAP           # Did receiver's detection improve?
            - bandwidth_weight × bandwidth_cost       # Penalty for bytes sent
            + novelty_weight × novelty_fraction       # Bonus for genuinely new info
        )
```

The reward directly incentivizes: "send information that IMPROVES the receiver's detection accuracy, using as little bandwidth as possible."

## Policy Network

```python
class CommunicationPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256, 128)):
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
```

A simple MLP (Multi-Layer Perceptron) with LayerNorm. Takes the full state as input, outputs per-observation transmit scores.

**Why MLP and not Transformer?** The input is fixed-size and relatively small (~400 dimensions). MLPs are simpler, faster to train, and sufficient. If we find attention patterns help (e.g., attending to specific observations), we can upgrade later.

**Why Sigmoid output?** We need scores in [0, 1] to represent "transmit probability." Sigmoid naturally bounds the output.

---

# 20. Python: Configuration System

**Files:**
- `python/omnicopilot/config/config.py` — Typed dataclasses
- `python/omnicopilot/config/defaults/*.yaml` — YAML configs

## Why Hydra?

Every experiment needs dozens of parameters (num_agents, bandwidth, loss_rate, perception model, fusion algorithm, ...). Hydra (by Facebook Research) manages this:

```bash
# Run with defaults
python run_experiment.py

# Override one parameter
python run_experiment.py communication.bandwidth_bps=2e6

# Run multiple configurations (sweep)
python run_experiment.py --multirun communication.packet_loss_rate=0.0,0.05,0.1,0.2
```

No more hardcoded constants. No more "which parameters did I use for that experiment?"

## Config Hierarchy

```
base.yaml          ← Default everything
  ├── experiment/
  │   ├── single_agent.yaml      ← Override: num_agents=1, no communication
  │   ├── cooperative_naive.yaml ← Override: unlimited bandwidth
  │   └── cooperative_intelligent.yaml ← Override: constrained bandwidth
  ├── model/
  │   ├── pointpillars.yaml      ← Detection model config
  │   └── v2vnet.yaml            ← Cooperative model config
  ├── communication/
  │   ├── unlimited.yaml         ← No constraints
  │   └── degraded.yaml          ← High loss, high latency
  └── ablation/
      ├── no_uncertainty.yaml    ← Disable uncertainty
      └── no_trust.yaml          ← Disable trust
```

Each YAML file overrides specific fields. Hydra composes them together. This makes ablation studies trivial — just disable one component per config.

---

# 21. Build System & Infrastructure

## CMake (C++)

```cmake
# Top-level: find dependencies, set compiler flags
# cpp/CMakeLists.txt: build libraries and tests

omnicopilot_core    # Static library: world model + communication + trust
omnicopilot_sim     # Static library: simulation interfaces
omnicopilot_tests   # Test executable: links core + sim + gtest
```

**FetchContent:** Downloads gtest and spdlog automatically during cmake configuration. No manual dependency installation needed (except protobuf).

## Makefile (Task Runner)

```bash
make build         # Build C++ + generate protos
make test          # Run all tests (C++ + Python)
make lint          # Run all linters
make train-policy  # Train RL policy
make ablation      # Run ablation suite
make demo          # Launch demo app
```

One command per task. No remembering complex incantations.

## Pre-commit Hooks

Automatically run BEFORE every git commit:
- ruff (Python lint + format)
- mypy (Python type check)
- clang-format (C++ format)
- detect-secrets (prevent accidentally committing API keys)
- no-commit-to-branch (prevent committing to main)

If any hook fails, the commit is blocked. This ensures code quality from day one.

---

# 22. How Everything Connects — Full Data Flow

## One Complete Tick

```
Simulation provides: sensor data for each agent

Agent A:
  1. Python perception model runs on sensor data
     → produces list[Observation]: "pedestrian at (10,20) conf=0.91"

  2. Observations sent to C++ WorldModel.IngestObservations()
     a. FusionEngine.Associate() matches observations to existing entities
     b. For matches: FuseObservation() updates position/confidence/evidence
        - TemporalTracker.Update() runs Kalman filter
     c. For new: CreateEntity() creates TrackedEntity + new TemporalTracker

  3. WorldModel.Tick(current_time)
     a. UncertaintyManager.Decay() on all entities
     b. TemporalTracker.Predict() advances position predictions
     c. SpatialIndex.Upsert() updates positions
     d. Lifecycle state machine runs
     e. IsSafetyCritical() checks
     f. PurgeEntities() removes Lost entities

  4. Communication decision:
     a. Prioritizer.Score() evaluates all observations
     b. BudgetManager checks available bandwidth
     c. Prioritizer.Prioritize() selects top-K within budget
     d. ReceiverModel checks novelty per receiver
     e. Serializer converts selected observations to bytes
     f. Channel.Enqueue() sends messages (may drop due to loss/queue)

Agent B:
  5. Channel.Tick() delivers messages from Agent A (with latency)
  6. Serializer deserializes bytes back to Observations
  7. ReliabilityTracker.GetTrust("agent_A") provides trust score
  8. WorldModel.IngestObservations(cooperative_obs, trust_A)
     → Agent B now "sees" the pedestrian that Agent A detected

Result:
  Agent B's world model contains a pedestrian entity that it CANNOT see
  with its own sensors — cooperative perception in action.
```

This entire loop runs 10 times per second. The C++ core handles steps 2-8. Python handles step 1 (ML inference) and experiment orchestration.

---

# Summary

| Component | Purpose | Key Algorithm |
|---|---|---|
| Agent | Top-level orchestrator per vehicle | Step loop |
| Entity types | Data representation | Observation → TrackedEntity lifecycle |
| FusionEngine | Match & combine observations | Greedy nearest-neighbor + Bayesian confidence |
| UncertaintyManager | Confidence over time | Exponential decay + independent evidence combination |
| SpatialIndex | Fast location queries | Grid-based spatial hashing |
| TemporalTracker | Predict positions | Constant-velocity Kalman filter |
| WorldModel | Wire everything together | Tick + IngestObservations orchestration |
| BudgetManager | Bandwidth allocation | Critical reserve + allocation tracking |
| Serializer | Struct ↔ bytes | Custom binary format |
| Channel | Network simulation | Latency + jitter + Bernoulli/Gilbert-Elliott loss |
| Prioritizer | What to send | Multi-factor weighted scoring |
| ReceiverModel | Theory of mind | Belief state tracking + novelty estimation |
| ReliabilityTracker | Agent trust | EMA accuracy + calibration period |
| Python Perception | ML detection | PointPillars/V2VNet wrappers |
| Python Fusion | Research fusion | Weighted average + Bayesian |
| Python RL | Learned communication | Gymnasium env + PPO policy |
| Hydra Config | Experiment management | Composable YAML configs |



---

# 23. Active Information Acquisition — Asking for What You Need

**Files:**
- `cpp/core/active_acquisition/info_value.h/.cc` — Estimate value of querying an agent
- `cpp/core/active_acquisition/source_selector.h/.cc` — Pick which agent to ask
- `cpp/core/active_acquisition/request_manager.h/.cc` — Track outstanding requests

## Why This Exists

Traditional cooperative perception is PASSIVE — agents share what they see. Active acquisition is ACTIVE — the system recognizes "I don't know enough about Entity #42" and ASKS a specific agent for information.

**Example:**
- World model has Entity #42 (possible obstacle) with confidence 0.43 — too low to act on, too high to ignore
- Agent B is 30 meters away with clear line-of-sight to that location
- Agent D is 200 meters away with no line-of-sight
- System should ask Agent B (not D) for a targeted observation

This is **active learning applied to multi-agent systems** — one of the project's key novel contributions.

## `InfoValueEstimator` — How Valuable Is Asking?

```cpp
struct InfoValueEstimate {
  std::string agent_id;
  double expected_information_gain;  // How much would we learn? [0, 1]
  double expected_uncertainty_reduction;  // How much would uncertainty drop?
  double cost;                       // Communication cost
  double net_value;                  // gain - cost
};
```

For each uncertain entity, we estimate how valuable it would be to query each available agent. Factors:
- **Distance:** Agent closer to the entity → more likely to see it → higher value
- **Viewing angle:** Agent with unobstructed line-of-sight → higher value
- **Sensor type:** LiDAR agent → better at 3D localization than camera-only
- **Agent trust:** High-trust agent → more reliable information → higher value
- **Communication cost:** Querying costs bandwidth → subtract from value

**Status:** Interface defined, stub implementation. Full implementation in Phase 6 (weeks 15-16).

## `SourceSelector` — Pick the Best Agent to Ask

```cpp
struct SourceSelectionConfig {
  uint32_t max_requests_per_agent = 2;   // Don't overwhelm one agent
  uint32_t max_total_requests = 5;        // Limit outstanding requests
  double min_net_value = 0.2;             // Don't ask if gain is too low
};
```

Takes the ranked InfoValueEstimates and selects which agents to actually query, respecting:
- Per-agent request limits (don't spam one agent with 20 questions)
- Total request budget (bandwidth for requests counts too)
- Minimum value threshold (don't waste a request on marginal gain)

**Status:** Interface defined, stub implementation.

## `RequestManager` — Track Questions and Answers

```cpp
struct InfoRequest {
  std::string request_id;
  std::string requester_id;      // Who's asking
  std::string target_agent_id;   // Who's being asked
  std::string target_entity_id;  // About what
  Position3D target_location;     // Where to look
  double sent_time_s;
  double deadline_s;              // Response needed by when
  double expected_gain;           // Why we asked
};
```

The RequestManager handles the lifecycle:
1. `SubmitRequest()` — Send a question (if budget allows)
2. Track pending requests
3. `RecordResponse()` — Got an answer → feed into world model
4. `Tick()` — Expire timed-out requests (no answer after 1 second)

**Why track deadlines?** In a fast-moving scenario, an answer that arrives 3 seconds late is useless — the situation has changed. Timed-out requests are counted as failures for statistics.

**Status:** Interface defined, stub implementation.

---

# 24. Trust: Anomaly Detection

**Files:**
- `cpp/core/trust/anomaly.h/.cc` — Detect suspicious observations

## Why This Exists

Before updating trust scores, we need to identify WHICH observations are wrong. The AnomalyDetector checks incoming observations against physical reality and consensus.

## Anomaly Types

```cpp
enum class AnomalyType : int {
  kPhysicsViolation,       // Object moves at impossible speed
  kConsensusDeviation,     // One agent disagrees with all others
  kTemporalInconsistency,  // Agent contradicts its own previous observations
  kStatisticalOutlier,     // Agent's accuracy suddenly drops
};
```

### Physics Violations

A detection at (10, 20) at t=1.0, then the same track at (500, 20) at t=1.1 → moved 490 meters in 0.1 seconds = 4,900 m/s = 17,640 km/h. Physically impossible. Flag as anomaly.

### Consensus Deviation

5 agents observe Entity #17. 4 agree it's at position (10, 20). Agent C says it's at (50, 80). Agent C is likely wrong.

### Temporal Inconsistency

Agent B reported a vehicle moving north at 50 km/h for 10 frames. Suddenly it reports the same vehicle at a completely different location with no trajectory connecting the two. Inconsistent with its own history.

**Status:** Interface defined, stub implementation. Full implementation in Phase 7 (weeks 17-18).

---

# 25. Trust: Byzantine Fault Tolerance

**Files:**
- `cpp/core/trust/byzantine.h/.cc` — Consensus under adversarial conditions

## Why This Exists

The ReliabilityTracker handles NOISY agents (innocent but inaccurate). Byzantine tolerance handles ADVERSARIAL agents (deliberately lying).

In the Byzantine generals problem, up to 1/3 of participants can lie and the honest majority can still reach correct consensus. We apply this to observations.

## How It Works

```cpp
struct ConsensusResult {
  bool consensus_reached;
  Observation consensus_observation;   // Best estimate from agreeing agents
  double agreement_ratio;              // Fraction of agents that agree
  std::vector<std::string> agreeing_agents;
  std::vector<std::string> disagreeing_agents;
};
```

For safety-critical observations (e.g., "pedestrian in my path"):
1. Collect observations from all agents about the same entity
2. Cluster observations by position (within `position_tolerance_m`)
3. Find the largest cluster
4. If >2/3 of agents are in that cluster → consensus reached
5. Agents NOT in the cluster → flagged as potentially adversarial

**Why 2/3?** This is the theoretical minimum for Byzantine fault tolerance. With N agents, you can tolerate up to (N-1)/3 adversarial agents. With 5 agents, that's 1 adversarial agent.

**Status:** Interface defined, stub implementation. Full implementation in Phase 7.

---

# 26. SystemConfig — Aggregate Configuration

**File:** `cpp/core/agent/config.h`

```cpp
struct SystemConfig {
  WorldModelConfig world_model;
  FusionConfig fusion;
  UncertaintyConfig uncertainty;
  TemporalTrackerConfig temporal;
  ChannelConfig channel;
  BudgetConfig budget;
  PrioritizationWeights prioritization;
  ReceiverModelConfig receiver_model;
  ReliabilityConfig reliability;
  SourceSelectionConfig source_selection;
  RequestManagerConfig request_manager;
};
```

**Why this exists:** Every component has its own config struct. SystemConfig bundles them all together so you can pass ONE object to configure everything. `DefaultSystemConfig()` returns sensible defaults.

**Design principle:** Every tunable parameter lives in a config struct, never as a magic number in code. This makes experiments reproducible — save the config, reproduce the results.

---

# 27. Simulation Layer

**Files:**
- `cpp/sim/scenario_engine.h/.cc` — Orchestrate multi-agent scenarios
- `cpp/sim/network_sim.h/.cc` — Multi-agent network management
- `cpp/sim/opv2v_replay.h/.cc` — Dataset replay as simulation

## ScenarioEngine — Running Experiments

```cpp
struct ScenarioConfig {
  std::string scenario_id;
  std::string dataset;        // "opv2v", "dair_v2x", "carla"
  uint32_t num_agents = 3;
  double duration_s = 30.0;
  double tick_rate_hz = 10.0; // 10 ticks per second
  ChannelConfig network_config;
};
```

The ScenarioEngine manages the simulation loop:
```
Initialize():
  Load dataset → create N agents → set up network

Run():
  while not finished:
    Tick():
      1. Get sensor data for each agent (from dataset replay)
      2. Each agent runs perception → produces observations
      3. Each agent Step() → ingest + tick world model
      4. Each agent produces outgoing messages
      5. Network delivers messages (with latency/loss)
      6. Receiving agents ingest cooperative observations
      7. Advance time
```

**Status:** Interface defined, stub implementation. Full implementation when OPV2V data is loaded (Phase 2-3).

## NetworkSimulator — Multi-Agent Message Routing

Manages channels between ALL agent pairs. With 5 agents, there are 5×4=20 directional channels (A→B, A→C, A→D, A→E, B→A, B→C, ...).

The NetworkSimulator:
- Creates channels for each agent pair
- Routes messages to the correct channel
- Applies global condition changes (e.g., "network congested for everyone")
- Collects deliveries per agent per tick

**Status:** Interface defined, stub implementation.

## OPV2VReplay — Dataset as Simulation

Instead of running a live simulator (CARLA), we replay recorded data:

```cpp
struct OPV2VReplayConfig {
  std::string data_path;       // Path to OPV2V dataset
  std::string split = "test";  // "train", "val", "test"
  uint32_t scene_index = 0;    // Which scene to replay
};
```

For each frame:
- Load LiDAR point cloud for each vehicle
- Load camera images (optional)
- Load ground truth annotations (for evaluation)
- Load vehicle poses (for coordinate transformation)

**Why replay instead of CARLA?** CARLA requires a GPU-intensive simulator running alongside the AI system. Dataset replay needs only disk I/O — it runs on a laptop. We use CARLA only for custom scenarios that aren't in the dataset.

**Status:** Interface defined, stub implementation. Implementation is Task 1.3 (Developer B).

---

# 28. Python-C++ Bindings (pybind11)

**Files:**
- `cpp/bindings/py_world_model.cc` — WorldModel + entity types
- `cpp/bindings/py_communication.cc` — Channel + Prioritizer
- `cpp/bindings/py_agent.cc` — Agent runtime
- `cpp/bindings/py_simulation.cc` — Module entry point

## Why Bindings Exist

The Python ML models produce observations. The C++ world model consumes them. pybind11 bridges the gap:

```python
# Python code can use C++ classes directly:
from omnicopilot._omnicopilot_cpp import WorldModel, WorldModelConfig, Observation

config = WorldModelConfig()
config.max_entities = 500

wm = WorldModel(config)

obs = Observation()
obs.agent_id = "car_a"
obs.position.x = 10.0
obs.position.y = 20.0
obs.confidence = 0.91

wm.ingest_observations([obs], agent_trust=0.95)
stats = wm.get_stats()
print(f"Entities: {stats.total_entities}")
```

## Module Entry Point

```cpp
// py_simulation.cc — the PYBIND11_MODULE macro creates the Python module

PYBIND11_MODULE(_omnicopilot_cpp, m) {
  m.doc() = "OmniCopilot C++ core";
  omnicopilot::BindWorldModel(m);      // Register WorldModel, Entity types
  omnicopilot::BindCommunication(m);   // Register Channel, Prioritizer
  omnicopilot::BindAgent(m);           // Register Agent
}
```

**The underscore prefix `_omnicopilot_cpp`:** Convention for internal C extension modules. The Python package `omnicopilot` imports from `_omnicopilot_cpp` — users never see the underscore name.

## What's Bound

### WorldModel bindings

```cpp
py::class_<WorldModel>(m, "WorldModel")
    .def(py::init<WorldModelConfig>())
    .def("tick", &WorldModel::Tick)
    .def("ingest_observations", &WorldModel::IngestObservations,
         py::arg("observations"), py::arg("agent_trust") = 1.0)
    .def("query_radius", &WorldModel::QueryRadius)
    .def("get_entity", &WorldModel::GetEntity, py::return_value_policy::reference)
    .def("get_stats", &WorldModel::GetStats)
    .def("reset", &WorldModel::Reset);
```

**`py::return_value_policy::reference`** on `get_entity`: Returns a pointer to the C++ entity without copying. The Python object is a VIEW into C++ memory. This is fast but means the pointer becomes invalid if the entity is removed. For read-only queries during a single tick, this is safe and much faster than copying.

**`py::arg("agent_trust") = 1.0`:** Default argument — if Python calls `ingest_observations([obs])` without trust, it defaults to 1.0.

### Enum bindings

```cpp
py::enum_<ObjectClass>(m, "ObjectClass")
    .value("UNKNOWN", ObjectClass::kUnknown)
    .value("VEHICLE", ObjectClass::kVehicle)
    .value("PEDESTRIAN", ObjectClass::kPedestrian)
```

Python can use: `ObjectClass.PEDESTRIAN` — Pythonic naming, mapped to C++ enum.

**Status:** Bindings are written but require pybind11 to compile (CMake option `OMNICOPILOT_BUILD_BINDINGS=ON`). Currently disabled in the build because pybind11 needs to be installed.

---

# 29. Python: Evaluation Module

**Files:**
- `python/omnicopilot/evaluation/metrics.py` — Metric dataclasses + computation
- `python/omnicopilot/evaluation/ablation.py` — Ablation study runner

## Metrics

Three metric categories, each as a dataclass:

```python
@dataclass
class PerceptionMetrics:
    mean_average_precision: float  # THE key metric for detection
    precision: float               # Of detections made, how many were correct?
    recall: float                  # Of real objects, how many were detected?
    f1_score: float                # Harmonic mean of precision and recall
    occluded_detection_rate: float # Key for cooperative perception — can we see hidden objects?
    mean_position_error_m: float   # How far off are our position estimates?

@dataclass
class CommunicationMetrics:
    bandwidth_used_bps: float
    information_efficiency: float  # Value per byte — THE key communication metric
    redundant_message_rate: float  # How many messages were wasted?
    novelty_hit_rate: float        # How many messages provided genuinely new info?

@dataclass
class SafetyMetrics:
    hazard_detection_rate: float
    mean_time_to_detection_s: float  # How quickly do we spot hazards?
    false_alarm_rate: float           # How often do we cry wolf?
```

**Why dataclasses?** They're lightweight, typed, and serializable. W&B can log them directly. They print nicely. They're immutable by convention.

### `compute_3d_iou()` and `compute_map()`

These compute the standard 3D object detection metrics used in the autonomous driving community:
- **3D IoU (Intersection over Union):** How much do predicted and ground-truth bounding boxes overlap in 3D space?
- **mAP (mean Average Precision):** Area under the precision-recall curve, averaged across classes and IoU thresholds

**Status:** Interfaces defined, implementation TODO. These will use standard evaluation code from OpenPCDet/nuScenes devkit.

## Ablation Runner

```python
@dataclass
class AblationResult:
    ablation_name: str           # "no_uncertainty"
    component_removed: str       # What was disabled
    perception: PerceptionMetrics
    communication: CommunicationMetrics
    safety: SafetyMetrics
    delta_map_vs_full: float     # mAP difference from full system
```

The ablation runner:
1. Loads the base (full system) config
2. For each ablation config (no_uncertainty, no_trust, no_prioritization, ...):
   a. Loads the config that disables one component
   b. Runs the full experiment
   c. Collects metrics
3. Ranks components by `delta_map_vs_full` — the component whose removal causes the BIGGEST mAP drop is the most important

**Status:** Interface defined, implementation TODO (Phase 8, weeks 19-20).

---

# 30. Python: Reasoning Module (LLM + RAG)

**Files:**
- `python/omnicopilot/reasoning/investigation.py` — LLM investigation agent
- `python/omnicopilot/reasoning/rag/store.py` — Vector knowledge base

## Investigation Agent

```python
@dataclass
class InvestigationResult:
    hypothesis: str                  # "Likely road hazard — ice detected"
    confidence: float                # 0.87
    evidence: list[str]              # ["Agent A: abnormal braking", "Agent B: low friction"]
    recommended_actions: list[str]   # ["Warn approaching vehicles", "Reduce speed"]
    explanation: str                 # Natural language explanation for humans
    sources_consulted: list[str]     # RAG documents used
```

**When is this triggered?** NOT every tick. Only when the world model detects unusual patterns that simple rules can't handle:
- Multiple agents reporting different anomalies in the same area
- Sudden confidence drops across many entities
- Contradictory observations that conflict resolution can't solve

**How it works (LangGraph pipeline):**
```
1. Receive situation description from world model
2. LLM generates hypotheses ("What could cause these signals?")
3. RAG retrieves relevant knowledge ("Similar patterns in V2X literature...")
4. LLM evaluates hypotheses against evidence
5. Output: structured InvestigationResult
```

**Why LLM here and not everywhere?** LLMs are:
- Slow (~100ms-1s per call) — can't be in the 10Hz perception loop
- Good at reasoning over unstructured information
- Good at explaining decisions in natural language

So we use them ONLY where reasoning and explanation matter, NOT in the real-time path.

**Status:** Interface defined, implementation TODO (Phase 9, weeks 21-22).

## RAG Knowledge Store

```python
class KnowledgeStore:
    def index_documents(self, documents_dir: Path) -> int:
        """Chunk PDFs, embed with BGE, store in Qdrant."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Embed query, find similar chunks, return with scores."""
```

The RAG pipeline:
```
Input: "What causes coordinated braking events at intersections?"
    ↓
Embed query with BGE-small-en (130MB model, runs on CPU)
    ↓
Search Qdrant for top-5 similar document chunks
    ↓
Return chunks with metadata (source paper, section, score)
    ↓
Feed to LLM as context for investigation
```

**Why Qdrant and not ChromaDB?** Qdrant is production-grade (used by companies at scale), has better performance, typed API, and runs as a local file (no server needed for development). ChromaDB is simpler but less reliable.

**Status:** Interface defined, implementation TODO (Phase 9).

---

# 31. Python: OPV2V Data Loader

**File:** `python/omnicopilot/data/opv2v.py`

```python
@dataclass
class OPV2VFrame:
    agent_id: str
    frame_idx: int
    timestamp_s: float
    lidar_points: NDArray[np.float32] | None   # (N, 4) — x, y, z, intensity
    camera_images: list[NDArray[np.uint8]]      # List of H×W×3 arrays
    pose: NDArray[np.float64]                    # 4×4 transformation matrix
    gt_boxes: NDArray[np.float64] | None        # (M, 7) — x,y,z,l,w,h,heading
    gt_labels: NDArray[np.int64] | None         # (M,) class labels
```

**Why typed NDArrays?** `NDArray[np.float32]` tells mypy (and developers) exactly what shape and type to expect. No runtime cost, but catches bugs like passing float64 where float32 is expected.

**The pose matrix:** Each vehicle has a 4×4 transformation matrix that converts from its local coordinate frame to the world frame. This is essential for cooperative perception — Agent A's observations at (10, 20) in ITS coordinates need to be transformed to world coordinates before comparing with Agent B's observations.

```
World position = Pose_A × Local_position_A
```

**Status:** Interface defined, implementation TODO (Task 1.3 — Developer B).

---

# 32. Python: RL Training and Analysis

**Files:**
- `python/omnicopilot/rl/train.py` — PPO training loop
- `python/omnicopilot/rl/analyze.py` — Emergent strategy analysis

## PPO Training Config

```python
@dataclass
class TrainConfig:
    num_envs: int = 4              # Parallel environments for faster training
    num_steps: int = 2048          # Steps per rollout before update
    learning_rate: float = 3e-4    # Adam learning rate
    num_epochs: int = 10           # PPO update epochs per rollout
    minibatch_size: int = 256      # Mini-batch size for PPO update
    gamma: float = 0.99            # Discount factor (value future rewards)
    gae_lambda: float = 0.95       # GAE lambda (bias-variance tradeoff)
    clip_range: float = 0.2        # PPO clip range (prevent too-large updates)
    entropy_coef: float = 0.01     # Encourage exploration
    total_timesteps: int = 1_000_000  # Total training steps
```

**Why these specific values?** These are the standard PPO hyperparameters from the original paper (Schulman et al., 2017) and CleanRL's tested defaults. They work well across most environments without tuning.

**Key hyperparameters explained:**
- `gamma = 0.99`: Future rewards are worth 99% of present rewards. Since communication decisions have delayed effects (message arrives after latency), we need to value future outcomes.
- `clip_range = 0.2`: PPO's key innovation — don't change the policy too much in one update. Prevents catastrophic policy collapse.
- `entropy_coef = 0.01`: Small bonus for random actions. Prevents the policy from collapsing to always sending (or never sending) the same observations.

## Emergent Strategy Analysis

```python
@dataclass
class StrategyPattern:
    name: str              # "selective_silence"
    description: str       # "Policy learns to stay silent during routine driving"
    frequency: float       # How often this pattern occurs
    performance_impact: float  # Δ performance when pattern activates
    example_episodes: list[int]  # Episodes where pattern is visible
```

**How analysis works:**
1. Record all decisions from the learned policy over 1000 episodes
2. Record all decisions from the hand-designed baseline on the same episodes
3. Find DISAGREEMENT POINTS — where the learned policy does something different
4. Cluster disagreements by context (network conditions, traffic density, etc.)
5. For each cluster, measure performance impact
6. Name and describe the emergent strategy

**Why this matters:** If the learned policy discovers "stay silent during routine driving to save bandwidth for emergencies" — that's a FINDING. It shows the RL agent learned something non-obvious. This is the project's potential headline result.

**Status:** Interface defined, implementation TODO (Phase 5 for training, Phase 8 for analysis).

---

# 33. Python: CLI Entry Point

**File:** `python/omnicopilot/__main__.py`

```python
app = typer.Typer(name="omnicopilot", help="OmniCopilot CLI")

@app.command()
def main() -> None:
    """OmniCopilot — Cooperative Multimodal AI."""
    typer.echo("OmniCopilot v0.1.0")
```

**Why Typer?** It's the modern Python CLI framework (by the FastAPI author). Type hints become CLI arguments automatically:

```bash
python -m omnicopilot              # Uses @app.command() main()
omnicopilot                         # Same, via pyproject.toml [project.scripts]
```

Future commands will be added:
```bash
omnicopilot run-experiment --config base.yaml
omnicopilot train-policy --total-timesteps 1000000
omnicopilot evaluate --checkpoint model.pt
omnicopilot demo --port 8501
```

---

# 34. Tests — What and Why

## C++ Tests (73 tests)

### test_world_model.cc (17 tests)

| Test | What it validates | Why it matters |
|---|---|---|
| Position3DTest.DistanceToSelf | distance(A, A) = 0 | Basic sanity |
| Position3DTest.DistanceToOther | distance((0,0), (3,4)) = 5 | Euclidean distance correct |
| UncertaintyTest.DecayHalvesAtHalflife | After 3s, confidence halves | Core uncertainty mechanic |
| UncertaintyTest.DecayNeverBelowMin | Even after 100s, ≥ 0.01 | Prevents entities from reaching zero |
| UncertaintyTest.BoostIncreasesConfidence | New evidence raises confidence | Cooperation has measurable effect |
| UncertaintyTest.CombineIndependentMultipleSources | 3 sources at 0.7/0.8/0.6 → 0.976 | Multi-agent fusion math correct |
| WorldModelTest.IngestCreatesEntity | First observation creates entity | Basic pipeline works |
| WorldModelTest.SecondSourceConfirmsEntity | Two agents → Confirmed state | Lifecycle state machine works |
| WorldModelTest.FarObservationsCreateSeparateEntities | Distant obs → different entities | Association doesn't falsely match |
| WorldModelTest.StaleEntitiesGetPurged | After long time → entity removed | Memory management correct |
| WorldModelTest.GetUncertainEntities | Single-source → needs_corroboration | Active acquisition integration |
| WorldModelTest.MaxEntitiesEnforced | Can't exceed max_entities | Prevents memory exhaustion |

### test_spatial_index.cc (11 tests)

| Test | What it validates |
|---|---|
| EmptyIndex | No crash on empty queries |
| InsertAndQuery | Basic insert + find |
| QueryMiss | Don't find entities that aren't there |
| MultipleEntities | Results sorted by distance |
| UpdatePosition | Entity moves → found at new position, not old |
| Remove + RemoveNonexistent | Cleanup works, no crash on missing ID |
| CrossCellBoundaryQuery | Entity near cell edge found from neighboring cell |

**CrossCellBoundaryQuery is critical:** Without this test, a bug where entities at cell boundaries are missed would go undetected. In driving, objects at grid boundaries are common.

### test_fusion.cc (8 tests)

| Test | What it validates |
|---|---|
| CreateEntityFromObservation | New entity has correct fields, TENTATIVE state |
| AssociateMatchesCloseObservation | Nearby observation matched to entity |
| AssociateCreatesNewForFarObservation | Distant observation → new entity |
| AssociateRejectsClassMismatch | Pedestrian obs doesn't match Truck entity |
| FuseObservationUpdatesPosition | Position moves toward new observation |
| FuseObservationIncreasesConfidence | Confidence grows with more evidence |
| ResolveConflictMajorityWins | 2-vs-1 class vote resolves correctly |

### test_communication.cc (14 tests)

| Test | What it validates |
|---|---|
| BudgetManager: 4 tests | Reset, allocate, critical reserve protection, critical access |
| Serializer: 3 tests | Single round-trip, batch round-trip, size estimation |
| Channel: 5 tests | Basic deliver (with latency), 100% loss drops, queue overflow, reset, condition override |
| Prioritizer: 3 tests | Pedestrian > traffic sign, budget limits selection, empty input |

**Serializer round-trip is essential:** If serialize→deserialize changes any field, the entire system silently corrupts data. The round-trip test catches this.

### test_trust.cc (12 tests)

| Test | What it validates |
|---|---|
| UnknownAgentGetsDefaultTrust | New agents start at initial_trust, not zero |
| CorrectObservationsIncreaseTrust | Good behavior rewarded |
| IncorrectObservationsDecreaseTrust | Bad behavior penalized |
| TrustNeverBelowMin / NeverAboveMax | Bounds respected |
| IsSuspiciousOnlyAfterCalibration | Don't flag agents with too few observations |
| AutoRegisterOnRecord | RecordCorrect auto-creates agent if not registered |

## Python Tests (3 files)

### tests/conftest.py — Shared Fixtures

```python
@pytest.fixture
def sample_observation() -> ObservationSchema:
    """A sample observation for testing."""
    return ObservationSchema(
        observation_id="obs_001",
        confidence=0.92,
        timestamp_s=1.0,
        ...
    )
```

**Why fixtures?** Multiple test files need the same test data. Fixtures are defined once and injected by pytest automatically. Change the fixture → all tests update.

### test_data_loading.py — Pydantic Validation

Tests that invalid data is REJECTED:
- confidence > 1.0 → ValidationError
- negative timestamp → ValidationError
- num_agents > 20 → ValidationError

**Why test rejection?** If bad data gets through validation, it can cause subtle bugs deep in the system. These tests ensure the boundary checks work.

### test_fusion.py — WeightedAverageFusion

Tests the ONLY fully-implemented Python fusion algorithm:
- Single observation returns itself
- Two equal observations average to midpoint
- Higher trust pulls estimate toward that observation
- Zero trust agent has no influence

### test_rl_environment.py — Gymnasium Environment

Tests that the RL environment follows Gymnasium conventions:
- Observation space and action space are defined
- reset() returns valid observation + info dict
- step() returns (obs, reward, terminated, truncated, info)
- Action space bounds are [0, 1]

**Why these tests?** If the environment violates Gymnasium conventions, RL training will silently fail or produce garbage results. These tests catch that before training starts.

---

# 35. CI/CD — What Gets Checked

## ci-cpp.yml

```yaml
strategy:
  matrix:
    compiler: [gcc-13, clang-17]
    build_type: [Release, Debug]
```

**Matrix build:** Tests all 4 combinations (gcc-Release, gcc-Debug, clang-Release, clang-Debug). Different compilers catch different bugs — GCC and Clang have different warnings and different optimizers.

**Debug build with sanitizers:** AddressSanitizer (detects memory errors) and UndefinedBehaviorSanitizer (detects UB) are enabled in Debug mode. These find bugs that tests alone miss — like buffer overflows or signed integer overflow.

**Static analysis:** clang-tidy runs during the build and catches patterns like:
- Use of raw `new` without smart pointers
- Missing `const` on methods that don't modify state
- Performance anti-patterns (unnecessary copies)

## ci-python.yml

**Lint → Type check → Unit tests → Integration tests → Security**

```yaml
- name: Security lint (bandit rules via ruff)
  run: uv run ruff check python/ --select S

- name: Check for known vulnerabilities
  run: uv run pip-audit
```

**pip-audit:** Checks if any installed package has known security vulnerabilities. If a dependency has a CVE, the build fails.

---

# 36. Docker — Multi-Stage Build

## Dockerfile.runtime

```dockerfile
# Stage 1: Build C++ with all dev tools
FROM ubuntu:24.04 AS cpp-builder
# Install cmake, ninja, g++, protobuf, pybind11
# Build C++ in Release mode

# Stage 2: Runtime with Python
FROM ubuntu:24.04 AS python-env
# Only install Python runtime + uv
# Copy COMPILED C++ bindings from Stage 1
# Install Python dependencies
# Generate Python protobuf code
```

**Why multi-stage?** Stage 1 has all the build tools (cmake, g++, headers) — ~2GB. Stage 2 only has the compiled library + Python runtime — ~500MB. The final image is 4× smaller.

## Dockerfile.ml

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.02-py3 AS base
```

**NVIDIA's official PyTorch container:** Includes CUDA, cuDNN, NCCL, and PyTorch — all pre-compiled and optimized for NVIDIA GPUs. Building CUDA from scratch takes hours and is error-prone. This gives us a working GPU environment in one line.

## docker-compose.yml

```yaml
services:
  runtime:     # C++ core + Python
  ml:          # GPU training (nvidia runtime)
  qdrant:      # Vector database for RAG
  prometheus:  # Metrics collection
  grafana:     # Metrics visualization
  otel-collector:  # Distributed tracing
```

**Why so many services?** Each is independent and optional:
- `docker compose up runtime` — just run the core system
- `docker compose up ml` — GPU training only
- `docker compose --profile observability up` — add monitoring

The `profiles` feature means observability services only start when explicitly requested.

---

# 37. pyproject.toml — Dependency Choices

Key decisions in the Python dependencies:

| Dependency | Why this one | Why not the alternative |
|---|---|---|
| `torch>=2.2` | Industry standard for ML | TensorFlow: less flexible, worse ecosystem for research |
| `lightning>=2.2` | Clean training framework | Raw PyTorch: too much boilerplate for experiments |
| `gymnasium>=0.29` | RL environment standard | OpenAI gym: deprecated in favor of Gymnasium |
| `hydra-core>=1.3` | Config management | argparse: doesn't compose, no sweeps, no reproducibility |
| `wandb>=0.16` | Experiment tracking | MLflow: harder to use, less features for free tier |
| `pydantic>=2.6` | Data validation | dataclasses: no runtime validation |
| `qdrant-client>=1.8` | Vector database | ChromaDB: less production-ready |
| `structlog>=24.1` | Structured logging | stdlib logging: no structured output |
| `ruff>=0.3` | Linting + formatting | pylint + black: two tools, 100× slower |
| `mypy>=1.8` | Type checking | pyright: less mature for strict mode |

### Version Pinning Strategy

```toml
"torch>=2.2,<3.0"
```

**Lower bound (`>=2.2`):** We need features from 2.2+ (torch.compile improvements).
**Upper bound (`<3.0`):** Protect against breaking changes in major version bumps.

This is the standard approach — neither too loose (any version) nor too strict (exact pin).

### Optional Dependencies

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "mypy>=1.8", "ruff>=0.3", ...]
carla = ["carla>=0.9.15"]
demo = ["rerun-sdk>=0.14", "streamlit>=1.31"]
distributed = ["ray[rllib]>=2.9"]
observability = ["prometheus-client>=0.20", "opentelemetry-api>=1.23"]
```

**Why optional groups?** Not everyone needs everything:
- CI/CD: `uv sync --extra dev`
- ML researcher: `uv sync --extra dev --extra notebooks`
- Full install: `uv sync --extra all`
- CARLA: `uv sync --extra carla` (only when using CARLA)

This keeps the base install small and fast.

---

# 38. Design Decisions — Why This, Not That

## Why C++17, Not C++20?

C++20 adds concepts, modules, coroutines, and ranges. We don't use them because:
- **Compiler support:** GCC 8 (common on RHEL/CentOS) doesn't support C++20 modules. C++17 works everywhere.
- **Complexity:** Concepts and coroutines add learning curve without clear benefit for this project.
- **C++17 has everything we need:** `std::optional`, `std::variant`, structured bindings, `constexpr`, `if constexpr`.

## Why Greedy Association, Not Hungarian?

- Hungarian: O(n³), optimal assignment. For 50 obs × 200 entities = too slow per tick.
- Greedy: O(n² log n), near-optimal. Fast enough at 10Hz.
- In practice, most matches are unambiguous (one observation clearly closest to one entity), so greedy and Hungarian give the same result 95%+ of the time.

## Why Grid Spatial Index, Not R-tree?

- R-tree: O(log n) queries, handles non-uniform distributions. More complex to implement.
- Grid: O(1) average queries, simple, cache-friendly. Works well when entities are roughly uniformly distributed (driving scenarios).
- For <1000 entities, the implementation complexity of R-tree isn't justified. Grid is simpler and fast enough.

## Why Custom Binary Serializer, Not Protobuf Everywhere?

Protobuf needs code generation (protoc) and the protobuf development library. Not every build environment has these. The custom serializer:
- Has zero external dependencies
- Is simple and auditable
- Round-trip tested
- Will be replaced by Protobuf when proto generation is integrated

## Why EMA Trust, Not Bayesian Trust?

EMA (Exponential Moving Average) is:
- Simple to implement and debug
- Responds to changes (adapts when an agent starts malfunctioning)
- Computationally trivial (one multiply-add per update)

Bayesian trust (Beta distribution) would be more principled but:
- More complex
- Harder to tune (prior selection)
- Not clearly better for this use case

We start with EMA. If experiments show it's insufficient, we upgrade.

## Why Separate C++ and Python Fusion?

- C++ fusion runs in the hot loop (10Hz × 1000 entities). Speed matters.
- Python fusion is for RESEARCH — try different algorithms, validate math in notebooks, compare approaches.
- Once a Python fusion algorithm is validated, it can be ported to C++.
- This dual approach is standard in ML systems engineering (prototype in Python, deploy in C++).

---

# 39. What's NOT Implemented Yet (and When)

| Component | Status | When | Who |
|---|---|---|---|
| OPV2V data loader (Python) | Interface only | Phase 1 (now) | Developer B |
| Perception model integration | Interface only | Phase 2 | Developer B |
| Multi-agent simulation loop | Stub | Phase 3 | Developer A |
| Anomaly detection logic | Stub | Phase 7 | Developer B |
| Byzantine consensus logic | Stub | Phase 7 | Developer A |
| Active acquisition (info value, source selection, request manager) | Stubs | Phase 6 | Both |
| Network simulator routing | Stub | Phase 3 | Developer A |
| Agent communication policy integration | Stub | Phase 4-5 | Developer A |
| RL training loop | Interface only | Phase 5 | Developer B |
| Emergent strategy analysis | Interface only | Phase 5 | Both |
| LLM investigation agent | Interface only | Phase 9 | Developer A |
| RAG knowledge store | Interface only | Phase 9 | Developer B |
| 3D IoU + mAP computation | Interface only | Phase 3 | Developer B |
| Ablation runner | Interface only | Phase 8 | Developer A |
| Demo application | Not started | Phase 10 | Developer A |
| pybind11 compilation | Written, not compiled | Phase 2 | Developer A |
| Proto code generation | CMake ready, needs protobuf-devel | Phase 2 | Developer A |

---

# 40. Reading Order Recommendation

If you're reading the code for the first time, follow this order:

1. **`entity.h`** — Understand the data types first
2. **`entity.cc`** — Trivial implementations, builds confidence
3. **`uncertainty.h` + `.cc`** — Simple math, fundamental concept
4. **`spatial_index.h` + `.cc`** — Useful data structure, self-contained
5. **`fusion.h` + `.cc`** — The core algorithm, most complex
6. **`temporal_tracker.h` + `.cc`** — Kalman filter, math-heavy but well-commented
7. **`world_model.h` + `.cc`** — How everything wires together
8. **`agent.h` + `.cc`** — The entry point that uses everything above
9. **`channel.h` + `.cc`** — Network simulation, interesting but independent
10. **`prioritizer.h` + `.cc`** — Communication policy, ties to the project's core question
11. **`reliability.h` + `.cc`** — Trust tracking
12. **`budget_manager.h` + `.cc` + `serializer.h` + `.cc`** — Support infrastructure
13. **`receiver_model.h` + `.cc`** — Theory of mind
14. **Python `schemas.py`** — Pydantic validation
15. **Python `algorithms.py`** — WeightedAverageFusion
16. **Python `environment.py` + `policy.py` + `reward.py`** — RL formulation
17. **Python `config.py` + YAML files** — Configuration system
18. **Tests** — Verify your understanding matches the implementation

For each file: read the HEADER first (interface + comments), then the IMPLEMENTATION (how it works), then the TESTS (what properties are guaranteed).
