// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Channel — Network channel model for V2X communication simulation.
// Simulates realistic bandwidth, latency, packet loss, and congestion.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace omnicopilot {

/// Loss model type.
enum class LossModel : int {
  kBernoulli = 0,        // Independent random loss.
  kGilbertElliott = 1,   // Bursty loss (good/bad state Markov chain).
};

/// Channel configuration.
struct ChannelConfig {
  double max_bandwidth_bps = 6e6;     // 6 Mbps (typical C-V2X).
  double base_latency_ms = 20.0;      // One-way base latency.
  double latency_jitter_ms = 10.0;    // Jitter (uniform random).
  double packet_loss_rate = 0.05;     // 5% baseline loss.
  uint32_t max_queue_depth = 100;     // Max messages in queue.

  LossModel loss_model = LossModel::kBernoulli;
  double gilbert_good_to_bad = 0.05;  // P(good→bad).
  double gilbert_bad_to_good = 0.3;   // P(bad→good).
  double gilbert_bad_loss_rate = 0.5; // Loss rate in bad state.
};

/// A message in transit through the channel.
struct ChannelMessage {
  std::string message_id;
  std::string sender_id;
  std::string receiver_id;
  uint32_t size_bytes = 0;
  double enqueue_time_s = 0.0;
  double delivery_time_s = 0.0;  // Scheduled delivery time.
  std::vector<uint8_t> payload;  // Serialized message bytes.
};

/// Current channel state (observable by the communication policy).
struct ChannelState {
  double available_bandwidth_bps = 0.0;
  double current_latency_ms = 0.0;
  double current_loss_rate = 0.0;
  double utilization = 0.0;          // [0, 1].
  uint32_t queue_depth = 0;
  uint32_t messages_in_flight = 0;
};

/// Network channel simulator.
///
/// Models a V2X communication channel with configurable:
/// - Bandwidth limits (messages queued and scheduled).
/// - Latency (base + jitter).
/// - Packet loss (Bernoulli or Gilbert-Elliott).
/// - Congestion (queue overflow → drop).
class Channel {
 public:
  explicit Channel(ChannelConfig config);
  ~Channel();

  Channel(const Channel&) = delete;
  Channel& operator=(const Channel&) = delete;
  Channel(Channel&&) noexcept;
  Channel& operator=(Channel&&) noexcept;

  /// Attempt to enqueue a message for transmission.
  /// @return true if accepted, false if dropped (queue full or bandwidth exceeded).
  bool Enqueue(ChannelMessage message);

  /// Advance channel time. Delivers messages whose delivery_time <= current_time.
  /// @param current_time_s Current simulation time.
  /// @return Messages delivered at this time step.
  std::vector<ChannelMessage> Tick(double current_time_s);

  /// Get the current observable channel state.
  ChannelState GetState() const;

  /// Dynamically update channel conditions (for scenario-driven degradation).
  void SetConditions(double loss_rate, double latency_ms, double bandwidth_bps);

  /// Get cumulative statistics.
  struct Stats {
    uint64_t total_enqueued = 0;
    uint64_t total_delivered = 0;
    uint64_t total_dropped_loss = 0;
    uint64_t total_dropped_queue = 0;
    uint64_t total_bytes_transmitted = 0;
    double mean_latency_ms = 0.0;
  };
  Stats GetStats() const;

  /// Reset channel state and statistics.
  void Reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace omnicopilot
