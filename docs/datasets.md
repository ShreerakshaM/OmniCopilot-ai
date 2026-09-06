# OmniCopilot — Dataset Guide

## Overview

This project requires very little data to start. Download only what you need for the current phase.

**Total storage needed throughout the project: ~18-38 GB**
**Storage needed to start: ~10 GB (one download)**

All datasets live in Google Drive. Your laptop stores nothing.

---

## Primary Dataset: OPV2V

**Source:** UCLA Mobility Lab
**URL:** https://mobility-lab.seas.ucla.edu/opv2v/
**Paper:** OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication (ICRA 2022)
**License:** Creative Commons Attribution-NonCommercial 4.0

### What it contains

- Multi-vehicle cooperative driving scenarios
- 3-5 vehicles per scene, each with full sensor suite
- LiDAR point clouds (64-beam)
- RGB camera images (4 cameras per vehicle)
- GPS / IMU telemetry
- 3D bounding box ground truth annotations
- Vehicle poses (transformation matrices)
- 73 diverse driving scenarios

### Splits

| Split | Scenes | Frames | Approx size | When to download |
|---|---|---|---|---|
| Test | ~15 | ~2,170 | ~10 GB | **Week 1** |
| Validation | ~7 | ~1,981 | ~8 GB | **Week 11** |
| Train | ~44 | ~6,764 | ~32 GB | **Only if fine-tuning perception models** |

### How to get it

1. Go to https://mobility-lab.seas.ucla.edu/opv2v/
2. Fill out the access request form (name, email, affiliation)
3. UCLA approves within 1-3 days
4. They send a Google Drive link
5. Copy **test split only** to your Google Drive

### Data format

```
opv2v/test/
├── scenario_001/
│   ├── vehicle_0/             # Ego vehicle
│   │   ├── 000000.pcd         # LiDAR point cloud
│   │   ├── 000000_camera0.png # Front camera
│   │   ├── 000000_camera1.png # Right camera
│   │   ├── 000000_camera2.png # Rear camera
│   │   ├── 000000_camera3.png # Left camera
│   │   └── 000000.yaml        # Pose + metadata
│   ├── vehicle_1/             # Cooperative vehicle
│   │   ├── ...
│   ├── vehicle_2/
│   │   ├── ...
│   └── ground_truth/
│       ├── 000000.yaml        # 3D bounding box annotations
│       └── ...
├── scenario_002/
│   └── ...
└── ...
```

### Key fields per frame (YAML metadata)

```yaml
# Per-vehicle YAML
lidar_pose:            # 4x4 transformation matrix (vehicle → world)
  - [r11, r12, r13, tx]
  - [r21, r22, r23, ty]
  - [r31, r32, r33, tz]
  - [0, 0, 0, 1]
vehicles_in_scene:     # List of all vehicle IDs in this scene
true_ego_pos:          # GPS coordinates

# Ground truth YAML
vehicles:              # Dict of annotated objects
  object_001:
    location: [x, y, z]
    extent: [length/2, width/2, height/2]
    angle: [roll, pitch, yaw]
    class: "car"       # or "pedestrian", "cyclist"
```

### OpenCOOD compatibility

OPV2V is the primary dataset for OpenCOOD framework. If you use OpenCOOD for cooperative perception models, the data loaders are already built:

```python
# OpenCOOD handles OPV2V loading natively
from opencood.data_utils.datasets import build_dataset
dataset = build_dataset(cfg, split="test")
```

---

## Alternative Primary Dataset: V2XSet

**Source:** OpenCOOD authors (same team as OPV2V)
**URL:** Available through OpenCOOD repo — https://github.com/DerrickXuNu/OpenCOOD
**Paper:** V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer (ECCV 2022)

### Why consider this

- Lighter than OPV2V (~15 GB total)
- Same data format (drop-in replacement)
- Created in CARLA with more controlled scenarios
- Good fallback if OPV2V download is slow

| Split | Approx size |
|---|---|
| Test | ~5 GB |
| Validation | ~3 GB |
| Train | ~7 GB |
| **Total** | **~15 GB** |

### When to use

- If OPV2V access takes too long
- If you want a second dataset for cross-dataset evaluation
- If storage is extremely tight (smaller than OPV2V)

---

## Secondary Dataset: DAIR-V2X (Optional — Phase 10)

**Source:** Tsinghua University + Baidu
**URL:** https://thudair.baai.ac.cn/index
**Paper:** DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection (CVPR 2022)
**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0

### What it contains

- Real-world data (not simulation) from Beijing roads
- Vehicle-to-Infrastructure cooperation (vehicle ↔ roadside camera)
- LiDAR + camera from both vehicle and infrastructure
- 3D annotations

### Why use it

- **Real-world** (vs OPV2V which is simulated in CARLA)
- **Different cooperation paradigm** (vehicle ↔ infrastructure, not vehicle ↔ vehicle)
- Demonstrates your architecture generalizes across cooperation types
- Strong claim for paper: "Works on both V2V and V2I, simulated and real"

### Splits

| Component | Approx size |
|---|---|
| Vehicle-side | ~10 GB |
| Infrastructure-side | ~10 GB |
| Cooperative (both) | ~20 GB |

### When to download

**Only in Phase 10 (weeks 23-26)** — when writing the paper and you want to show generalization. Not before.

---

## RAG Knowledge Base Documents (Phase 9)

Not a formal dataset — just PDFs and documents you collect manually.

### What to collect

| Category | Examples | Approx size |
|---|---|---|
| Cooperative perception papers | V2VNet, Where2comm, CoBEVT, DiscoNet, OPV2V paper | ~100 MB |
| V2X standards | ETSI ITS, C-V2X specs, SAE J2735 | ~50 MB |
| Safety guidelines | ISO 26262 summaries, SOTIF | ~30 MB |
| Project documentation | Your own architecture docs, experiment logs | ~10 MB |

### Where to find papers

- **arXiv:** Search "cooperative perception" or "V2X perception"
- **Semantic Scholar:** Good for finding related work chains
- **OpenCOOD repo:** Lists all relevant papers in their README

### Storage

~200 MB total. Store in `data/raw/knowledge_base/` on Google Drive.

---

## Download Timeline

| Week | Action | Cumulative storage |
|---|---|---|
| 1 | Request OPV2V access from UCLA | 0 GB |
| 1-2 | Copy OPV2V **test split** to Google Drive | **10 GB** |
| 2-10 | Work with test split only | 10 GB |
| 11 | Download OPV2V **validation split** | **18 GB** |
| 11-20 | Work with test + validation | 18 GB |
| 21 | Collect PDFs for RAG knowledge base | **~18.3 GB** |
| 23 | (Optional) Download DAIR-V2X cooperative split | **~38 GB** |

---

## Storage Plan

### Free tier (Google Drive 15 GB)

Enough for OPV2V test split (10 GB) + code + docs. Tight but workable for Phases 1-4.

When you need validation split (Phase 5+), either:
- Upgrade Google Drive to 200 GB ($3/month)
- Use wife's university Google Workspace (often unlimited)
- Use university storage if available

### Recommended ($3/month)

Google Drive 200 GB plan. Holds everything comfortably through the entire project including optional DAIR-V2X.

### Where data lives

```
Google Drive/
└── OmniCopilot/
    └── data/
        ├── opv2v/
        │   ├── test/           ← Week 1 (10 GB)
        │   └── validate/       ← Week 11 (8 GB)
        ├── dair_v2x/           ← Week 23, optional (20 GB)
        └── knowledge_base/     ← Week 21 (200 MB)
            ├── papers/
            └── standards/
```

Your laptop stores **zero** dataset files. Everything in cloud.

---

## Datasets You Do NOT Need

| Dataset | Size | Why skip |
|---|---|---|
| OPV2V train split | ~32 GB | Not training perception models from scratch |
| nuScenes | ~300 GB (full) | Single-vehicle, not cooperative |
| Waymo Open Dataset | ~1+ TB | Too large, not cooperative, complex license |
| KITTI | ~80 GB | Old, single-vehicle, no cooperation |
| Argoverse | ~1 TB | Not cooperative |
| OPV2V pre-trained models | ~2 GB | Use OpenCOOD models instead (newer, better) |

---

## Quick Reference

**First download:** OPV2V test split → Google Drive → ~10 GB
**That's enough for:** Phases 1 through 4 (first 10 weeks)
**Total ever needed:** ~18 GB (without DAIR-V2X) or ~38 GB (with DAIR-V2X)
