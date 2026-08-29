// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Agent configuration types and defaults.

#pragma once

#include "cpp/core/active_acquisition/request_manager.h"
#include "cpp/core/active_acquisition/source_selector.h"
#include "cpp/core/communication/budget_manager.h"
#include "cpp/core/communication/channel.h"
#include "cpp/core/communication/prioritizer.h"
#include "cpp/core/communication/receiver_model.h"
#include "cpp/core/trust/reliability.h"
#include "cpp/core/world_model/fusion.h"
#include "cpp/core/world_model/temporal_tracker.h"
#include "cpp/core/world_model/uncertainty.h"
#include "cpp/core/world_model/world_model.h"

namespace omnicopilot {

/// Full system configuration (all components).
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

/// Create a default system configuration.
inline SystemConfig DefaultSystemConfig() {
  return SystemConfig{};
}

}  // namespace omnicopilot
