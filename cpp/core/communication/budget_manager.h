// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// BudgetManager — Bandwidth budget allocation and tracking.

#pragma once

#include <cstdint>
#include <memory>

namespace omnicopilot {

/// Configuration for bandwidth budget management.
struct BudgetConfig {
  /// Total bandwidth budget per tick (bytes).
  uint32_t budget_per_tick_bytes = 4096;

  /// Fraction of budget reserved for critical/emergency messages.
  double critical_reserve_fraction = 0.2;

  /// Whether to carry over unused budget to next tick.
  bool allow_budget_rollover = false;

  /// Maximum rollover (prevents hoarding).
  uint32_t max_rollover_bytes = 2048;
};

/// Manages communication bandwidth budget per agent per tick.
class BudgetManager {
 public:
  explicit BudgetManager(BudgetConfig config);
  ~BudgetManager();

  /// Reset budget for a new tick. Returns available bytes.
  uint32_t NewTick();

  /// Attempt to allocate bytes from the budget.
  /// @return true if allocation succeeded, false if insufficient budget.
  bool Allocate(uint32_t bytes, bool is_critical = false);

  /// Get remaining budget (bytes).
  uint32_t GetRemaining() const;

  /// Get remaining critical reserve (bytes).
  uint32_t GetCriticalRemaining() const;

  /// Get utilization fraction for this tick [0, 1].
  double GetUtilization() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot
