// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/active_acquisition/source_selector.h"

namespace omnicopilot {

struct SourceSelector::Impl {
  SourceSelectionConfig config;
};

SourceSelector::SourceSelector(SourceSelectionConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

SourceSelector::~SourceSelector() = default;

std::vector<std::string> SourceSelector::Select(
    const std::vector<InfoValueEstimate>& estimates) const {
  // TODO: Implement source selection.
  return {};
}

}  // namespace omnicopilot
