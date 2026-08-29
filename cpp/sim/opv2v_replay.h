// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// OPV2VReplay — Replay OPV2V dataset as a multi-agent simulation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Configuration for OPV2V dataset replay.
struct OPV2VReplayConfig {
  std::string data_path;          // Path to OPV2V dataset.
  std::string split = "test";     // "train", "val", "test".
  uint32_t scene_index = 0;       // Which scene to replay.
};

/// Replays OPV2V dataset frames as observation batches per agent.
class OPV2VReplay {
 public:
  explicit OPV2VReplay(OPV2VReplayConfig config);
  ~OPV2VReplay();

  /// Load the dataset. Must be called before GetFrame.
  void Load();

  /// Get the number of frames in this scene.
  uint32_t GetFrameCount() const;

  /// Get the number of agents in this scene.
  uint32_t GetAgentCount() const;

  /// Get agent IDs.
  std::vector<std::string> GetAgentIds() const;

  /// Get observations for a specific agent at a specific frame.
  std::vector<Observation> GetFrame(const std::string& agent_id,
                                    uint32_t frame_index) const;

  /// Get ground truth entities for a frame (for evaluation).
  std::vector<TrackedEntity> GetGroundTruth(uint32_t frame_index) const;

  /// Get agent position at a frame.
  Position3D GetAgentPosition(const std::string& agent_id,
                              uint32_t frame_index) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot
