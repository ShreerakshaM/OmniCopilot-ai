# Task 1.3 — OPV2V Data Loading Spec (OpenCOOD-Wrapper Approach)

Blueprint for `python/omnicopilot/data/opv2v.py`.
Owner: Developer B. Prerequisite: OPV2V test data available + OpenCOOD cloned.

> **DECISION (C1b): Do NOT hand-roll the OPV2V parser or coordinate transforms.**
> We verified empirically that hand-rolling the coordinate transform produced a 70m
> error. OPV2V's authors ship a correct, tested loader (OpenCOOD). This spec wraps it.
> Task 1.3 is now a THIN ADAPTER around OpenCOOD — not a from-scratch parser.

---

## 0. Why this changed (context)

Earlier this spec described building a parser from scratch. During data exploration we hit
a **70-meter coordinate error** and discovered:
- GT `location` is in the **CARLA global frame**, not ego frame (we assumed wrong).
- Cross-agent projection MUST invert the destination pose:
  `transform = inv(x_to_world(x2)) @ x_to_world(x1)` (OpenCOOD's `x1_to_x2`).
- Angles are in **degrees**; `extent` is **half-dimensions**; YAML needs `unsafe_load`.
- Object dict keys are **per-agent** and do NOT identify the same car across agents.

OpenCOOD (`github.com/DerrickXuNu/OpenCOOD`, by the OPV2V authors) already handles ALL of
this correctly and is tested. Reusing it eliminates the entire coordinate-bug class.

---

## 1. What OpenCOOD gives us (reuse, do not reimplement)

| OpenCOOD component | What it does | We use it for |
|---|---|---|
| `opencood.utils.transformation_utils.x_to_world(pose)` | `[x,y,z,roll,yaw,pitch]`(deg) → 4×4 local→world | Any pose → matrix |
| `opencood.utils.transformation_utils.x1_to_x2(x1, x2)` | agent x1 frame → agent x2 frame (`inv(x2_to_world)@x1_to_world`) | All cross-agent projection |
| `opencood.data_utils.datasets.BaseDataset` | Builds scenario DB (scenario→agent→timestamp→file paths); loads raw per-CAV data | Data indexing + raw loading |
| `IntermediateFusionDataset` / `LateFusionDataset` / `EarlyFusionDataset` | Project GT + LiDAR to ego frame, preprocess to voxels/BEV, format labels | Feature-fusion backbone (per C1) |
| Preprocessor / Postprocessor | raw points → voxel/BEV features; bbox formatting | Detection pipeline |

**Rule:** any time you need a coordinate transform, call OpenCOOD's functions. Never write
your own rotation matrix or projection.

---

## 2. The OPV2V data facts (confirmed against real data — for reference)

```
Data root: .../opv2v/test/
└── <scenario>/                       e.g. testoutput_CAV_data_2022-03-15-09-54-40_0
    └── <agent_id>/                   numeric CAV folders: 0, 1, ...
        ├── 000041.pcd                LiDAR (n,4): x,y,z,intensity — in that CAV's LiDAR frame
        ├── 000041.yaml               metadata (see below)
        └── 000041_cameraN.png        4 cameras (optional; skip for LiDAR-only start)
```

YAML fields (confirmed):
- `lidar_pose`: agent pose. Official format = `[x,y,z,roll,yaw,pitch]` (degrees). Our data
  variant may store a pre-built 4×4 matrix (= `x_to_world` output). Handle both.
- `true_ego_pos` / `predicted_ego_pos`: true vs GPS-estimated localization (pose-error study).
- `ego_speed`: km/h.
- `vehicles`: dict of surrounding objects **with ≥1 LiDAR hit from this agent** (so each
  agent lists a DIFFERENT subset — this is the cooperation signal).
  - `location`: `[x,y,z]` center in **CARLA GLOBAL frame**.
  - `angle`: `[roll, yaw, pitch]` in **degrees** (yaw = `angle[1]`).
  - `extent`: `[l/2, w/2, h/2]` **HALF-dimensions** → full L/W/H = 2× each.
  - `center`: small offset from bbox center to vehicle front axis (usually ~0).
  - `ass_id`: cross-agent association id (often -1 = not associated).
  - `obj_type`: e.g. `Car`.
- Must load YAML with `yaml.unsafe_load` (numpy objects embedded).

---

## 3. Implementation: `opv2v.py` as an OpenCOOD adapter

### 3a. Add OpenCOOD as a dependency
- Clone OpenCOOD into the environment (or vendor it under `third_party/OpenCOOD`).
- Add its import path. Document its install in `docs/setup.md`.
- Note license: OpenCOOD is TDG-Attribution-NonCommercial-NoDistrib — fine for a research
  project; record it in our NOTICE/attribution.

### 3b. Two integration options — pick based on need

**Option 1 (recommended to start): use OpenCOOD's dataset classes directly.**
Configure an OpenCOOD `hypes_yaml` for OPV2V, instantiate `IntermediateFusionDataset` (for
the feature-fusion backbone, per C1) or `LateFusionDataset` (if you want per-agent object
outputs to feed our object-level world model). Let it do all loading + projection.

**Option 2 (adapter for our world model): thin wrapper exposing OUR types.**
Wrap OpenCOOD's raw loader to emit our `OPV2VFrame` / `Observation` structs, using
OpenCOOD's transforms for coordinates:

```python
import sys; sys.path.append("third_party/OpenCOOD")
from opencood.utils.transformation_utils import x_to_world, x1_to_x2
import numpy as np, yaml, math

def load_yaml(path):
    with open(path) as f:
        return yaml.unsafe_load(f)          # NOT safe_load

def pose_matrix(lidar_pose):
    """Return 4x4 local->world. Handle both 6-vector and pre-built 4x4 variants."""
    arr = np.array(lidar_pose)
    if arr.shape == (4, 4):
        return arr                          # our data variant: already a matrix
    return x_to_world(list(arr))            # official 6-vector [x,y,z,roll,yaw,pitch] deg

def gt_boxes_world(meta):
    """GT boxes already in GLOBAL frame; return [x,y,z, L,W,H, yaw_rad] + obj ids."""
    boxes, ids = [], []
    for vid, v in meta["vehicles"].items():
        loc = v["location"]                 # GLOBAL frame
        ext = v["extent"]                   # HALF dims
        yaw = math.radians(v["angle"][1])   # degrees -> radians
        boxes.append([loc[0], loc[1], loc[2],
                      2*ext[0], 2*ext[1], 2*ext[2], yaw])
        ids.append(vid)
    return np.array(boxes), ids

def project_between_agents(point_global, src_pose6, dst_pose6):
    """Move a point between agent frames using OpenCOOD's tested transform."""
    T = x1_to_x2(src_pose6, dst_pose6)      # inv(dst_to_world) @ src_to_world
    p = np.array([*point_global, 1.0])
    return (T @ p)[:3]
```

### 3c. Object identity across agents (important nuance)
- Do NOT match objects by dict key across agents (keys are per-agent).
- Since GT `location` is GLOBAL, the SAME physical car has the SAME global location in every
  agent's YAML that lists it. **Match objects across agents by global location proximity**
  (or by `ass_id` if populated). This is how you find the overlap for the cooperation demo.

---

## 4. Validation (the acid tests)

1. **Transform sanity:** take a GT object listed by two agents (same global location),
   confirm both YAMLs report ~identical global `location` (within pose/GPS error, ~<2m).
   If they differ by tens of meters, something is still wrong — STOP and recheck.
2. **CAV cross-detection:** an agent's `lidar_pose` translation is its true world position.
   If agent A's GT includes agent B as an object, that object's global location should match
   agent B's pose translation. Clean, unambiguous transform check.
3. **LiDAR overlap:** project two agents' point clouds into a common frame with OpenCOOD's
   transform; they should visually align (roads/buildings overlap). Misalignment = bug.

---

## 5. Exploration notebook (`notebooks/01_data_exploration.ipynb`)

Must show:
1. Load one scenario via OpenCOOD; print agents + frames.
2. Overlay two agents' LiDAR in a common frame (OpenCOOD transform) — should align.
3. Plot GT boxes (BEV).
4. **Find the cooperation case:** an object with many LiDAR points from agent B but few
   from agent A → occluded for A, visible to B. Record scenario/frame/object. (Satisfies
   Task 1.5 demo-scene verification.) We already saw this structure: agent 0 saw 2
   vehicles, agent 1 saw 4 — different subsets = cooperation signal.

---

## 6. Definition of done

- [ ] OpenCOOD cloned/vendored; import path + license recorded
- [ ] `opv2v.py` loads OPV2V via OpenCOOD (Option 1 or 2), no hand-rolled transforms
- [ ] `yaml.unsafe_load` used; degrees→radians; half→full extents handled
- [ ] Transform sanity test passes (shared object matches across agents within ~2m)
- [ ] LiDAR from two agents visually aligns in a common frame
- [ ] Cooperation/occlusion scene identified (scenario/frame/object recorded)
- [ ] Dataset versioned (DVC or Kaggle Dataset); stats logged to W&B
- [ ] Output validates against Pydantic schemas (`data/schemas.py`)

---

## 7. Coordination notes

- **With Developer A (Task 3.0):** coordinate transforms now come from OpenCOOD's
  `transformation_utils` — A does NOT implement transforms. A's Task 3.0 shrinks to
  "wrap/adopt OpenCOOD transforms + handle pose-error robustness study." Agree on which
  fusion dataset (Intermediate vs Late) feeds the world model.
- **Per C1:** for the detection BACKBONE use OpenCOOD's `IntermediateFusionDataset` +
  a pretrained model (AttFuse/V2VNet/Where2comm). Our object-level world model layers on top.
- **Start LiDAR-only, single class ("Car").** Add cameras/multi-class later.
- **Environment note:** OpenCOOD may need `spconv`, specific torch/CUDA versions. Sort its
  install on Kaggle/Colab (GPU env) early — it's the fiddliest dependency.
