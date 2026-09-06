// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/communication/budget_manager.h"

#include <algorithm>

namespace omnicopilot {

struct BudgetManager::Impl {
  BudgetConfig config;

  uint32_t remaining_bytes = 0;
  uint32_t critical_remaining_bytes = 0;
  uint32_t total_for_tick = 0;
  uint32_t rollover_bytes = 0;
};

BudgetManager::BudgetManager(BudgetConfig config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

BudgetManager::~BudgetManager() = default;

uint32_t BudgetManager::NewTick() {
  uint32_t base = impl_->config.budget_per_tick_bytes;
  uint32_t rollover = impl_->config.allow_budget_rollover
                          ? std::min(impl_->rollover_bytes,
                                     impl_->config.max_rollover_bytes)
                          : 0;

  impl_->total_for_tick = base + rollover;
  impl_->remaining_bytes = impl_->total_for_tick;
  impl_->critical_remaining_bytes = static_cast<uint32_t>(
      base * impl_->config.critical_reserve_fraction);
  impl_->rollover_bytes = 0;

  return impl_->total_for_tick;
}

bool BudgetManager::Allocate(uint32_t bytes, bool is_critical) {
  if (is_critical) {
    // Critical messages draw from the reserved pool first, then general.
    if (bytes <= impl_->critical_remaining_bytes) {
      impl_->critical_remaining_bytes -= bytes;
      impl_->remaining_bytes -= bytes;
      return true;
    }
    // If critical reserve is exhausted, fall through to general budget.
  }

  // Non-critical messages cannot use the critical reserve.
  uint32_t available =
      impl_->remaining_bytes - impl_->critical_remaining_bytes;
  if (!is_critical && bytes > available) {
    return false;
  }

  if (bytes > impl_->remaining_bytes) {
    return false;
  }

  impl_->remaining_bytes -= bytes;
  return true;
}

uint32_t BudgetManager::GetRemaining() const {
  return impl_->remaining_bytes;
}

uint32_t BudgetManager::GetCriticalRemaining() const {
  return impl_->critical_remaining_bytes;
}

double BudgetManager::GetUtilization() const {
  if (impl_->total_for_tick == 0) return 0.0;
  uint32_t used = impl_->total_for_tick - impl_->remaining_bytes;
  return static_cast<double>(used) / impl_->total_for_tick;
}

}  // namespace omnicopilot
