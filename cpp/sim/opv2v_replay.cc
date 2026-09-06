// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/sim/opv2v_replay.h"

namespace omnicopilot {

struct OPV2VReplay::Impl {
  OPV2VReplayConfig config;
};

OPV2VReplay::OPV2VReplay(OPV2VReplayConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

OPV2VReplay::~OPV2VReplay() = default;

void OPV2VReplay::Load() {
  // TODO: Implement dataset loading.
}

uint32_t OPV2VReplay::GetFrameCount() const { return 0; }
uint32_t OPV2VReplay::GetAgentCount() const { return 0; }
std::vector<std::string> OPV2VReplay::GetAgentIds() const { return {}; }

std::vector<Observation> OPV2VReplay::GetFrame(const std::string& agent_id,
                                               uint32_t frame_index) const {
  return {};
}

std::vector<TrackedEntity> OPV2VReplay::GetGroundTruth(
    uint32_t frame_index) const {
  return {};
}

Position3D OPV2VReplay::GetAgentPosition(const std::string& agent_id,
                                         uint32_t frame_index) const {
  return {};
}

}  // namespace omnicopilot
