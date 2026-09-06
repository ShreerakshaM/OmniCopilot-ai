// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/communication/channel.h"

#include <algorithm>
#include <deque>
#include <random>

namespace omnicopilot {

struct Channel::Impl {
  ChannelConfig config;

  // Messages waiting to be delivered (sorted by delivery time).
  std::deque<ChannelMessage> in_flight;

  // Gilbert-Elliott state: true = good, false = bad.
  bool ge_good_state = true;

  // Random number generator for loss and jitter.
  std::mt19937 rng{42};
  std::uniform_real_distribution<double> uniform{0.0, 1.0};

  // Cumulative statistics.
  uint64_t total_enqueued = 0;
  uint64_t total_delivered = 0;
  uint64_t total_dropped_loss = 0;
  uint64_t total_dropped_queue = 0;
  uint64_t total_bytes = 0;
  double total_latency_ms = 0.0;

  // Bandwidth tracking for current time window.
  double window_start_s = 0.0;
  uint64_t window_bytes = 0;

  // Dynamic overrides (for scenario-driven degradation).
  double override_loss_rate = -1.0;      // Negative = use config.
  double override_latency_ms = -1.0;
  double override_bandwidth_bps = -1.0;

  double GetLossRate() const {
    return (override_loss_rate >= 0.0) ? override_loss_rate
                                       : config.packet_loss_rate;
  }

  double GetLatencyMs() const {
    return (override_latency_ms >= 0.0) ? override_latency_ms
                                        : config.base_latency_ms;
  }

  double GetBandwidthBps() const {
    return (override_bandwidth_bps >= 0.0) ? override_bandwidth_bps
                                           : config.max_bandwidth_bps;
  }

  // Determine if a message is dropped due to loss.
  bool IsDroppedByLoss() {
    double loss_rate;

    if (config.loss_model == LossModel::kGilbertElliott) {
      // Transition state.
      double p = uniform(rng);
      if (ge_good_state) {
        if (p < config.gilbert_good_to_bad) {
          ge_good_state = false;
        }
      } else {
        if (p < config.gilbert_bad_to_good) {
          ge_good_state = true;
        }
      }
      loss_rate = ge_good_state ? GetLossRate() : config.gilbert_bad_loss_rate;
    } else {
      loss_rate = GetLossRate();
    }

    return uniform(rng) < loss_rate;
  }

  // Compute delivery time for a message.
  double ComputeDeliveryTime(double enqueue_time_s) {
    double latency_ms = GetLatencyMs();
    double jitter_ms = config.latency_jitter_ms * (uniform(rng) * 2.0 - 1.0);
    double total_ms = std::max(latency_ms + jitter_ms, 0.0);
    return enqueue_time_s + total_ms / 1000.0;
  }
};

Channel::Channel(ChannelConfig config) : impl_(std::make_unique<Impl>()) {
  impl_->config = std::move(config);
}

Channel::~Channel() = default;
Channel::Channel(Channel&&) noexcept = default;
Channel& Channel::operator=(Channel&&) noexcept = default;

bool Channel::Enqueue(ChannelMessage message) {
  // Check queue depth.
  if (impl_->in_flight.size() >= impl_->config.max_queue_depth) {
    impl_->total_dropped_queue++;
    return false;
  }

  // Check bandwidth: rough check over 1-second window.
  double bw = impl_->GetBandwidthBps();
  double now = message.enqueue_time_s;

  // Reset window if more than 1 second has passed.
  if (now - impl_->window_start_s > 1.0) {
    impl_->window_start_s = now;
    impl_->window_bytes = 0;
  }

  uint64_t msg_bits = static_cast<uint64_t>(message.size_bytes) * 8;
  if (bw > 0.0 && (impl_->window_bytes * 8 + msg_bits) >
                       static_cast<uint64_t>(bw)) {
    impl_->total_dropped_queue++;
    return false;  // Bandwidth exceeded.
  }

  // Check packet loss.
  if (impl_->IsDroppedByLoss()) {
    impl_->total_dropped_loss++;
    impl_->total_enqueued++;
    return false;
  }

  // Schedule delivery.
  message.delivery_time_s = impl_->ComputeDeliveryTime(message.enqueue_time_s);

  impl_->window_bytes += message.size_bytes;
  impl_->total_enqueued++;
  impl_->total_bytes += message.size_bytes;

  impl_->in_flight.push_back(std::move(message));

  return true;
}

std::vector<ChannelMessage> Channel::Tick(double current_time_s) {
  std::vector<ChannelMessage> delivered;

  while (!impl_->in_flight.empty()) {
    auto& front = impl_->in_flight.front();
    if (front.delivery_time_s <= current_time_s) {
      double latency =
          (front.delivery_time_s - front.enqueue_time_s) * 1000.0;
      impl_->total_latency_ms += latency;
      impl_->total_delivered++;
      delivered.push_back(std::move(front));
      impl_->in_flight.pop_front();
    } else {
      break;  // Remaining messages are not yet due (deque is roughly ordered).
    }
  }

  return delivered;
}

ChannelState Channel::GetState() const {
  ChannelState state;
  state.available_bandwidth_bps = impl_->GetBandwidthBps();
  state.current_latency_ms = impl_->GetLatencyMs();
  state.current_loss_rate = impl_->GetLossRate();
  state.queue_depth = static_cast<uint32_t>(impl_->in_flight.size());
  state.messages_in_flight = state.queue_depth;

  double bw = impl_->GetBandwidthBps();
  state.utilization =
      (bw > 0.0) ? static_cast<double>(impl_->window_bytes * 8) / bw : 0.0;
  state.utilization = std::clamp(state.utilization, 0.0, 1.0);

  return state;
}

void Channel::SetConditions(double loss_rate, double latency_ms,
                            double bandwidth_bps) {
  impl_->override_loss_rate = loss_rate;
  impl_->override_latency_ms = latency_ms;
  impl_->override_bandwidth_bps = bandwidth_bps;
}

Channel::Stats Channel::GetStats() const {
  Stats s;
  s.total_enqueued = impl_->total_enqueued;
  s.total_delivered = impl_->total_delivered;
  s.total_dropped_loss = impl_->total_dropped_loss;
  s.total_dropped_queue = impl_->total_dropped_queue;
  s.total_bytes_transmitted = impl_->total_bytes;
  s.mean_latency_ms =
      (impl_->total_delivered > 0)
          ? (impl_->total_latency_ms / impl_->total_delivered)
          : 0.0;
  return s;
}

void Channel::Reset() {
  impl_->in_flight.clear();
  impl_->ge_good_state = true;
  impl_->total_enqueued = 0;
  impl_->total_delivered = 0;
  impl_->total_dropped_loss = 0;
  impl_->total_dropped_queue = 0;
  impl_->total_bytes = 0;
  impl_->total_latency_ms = 0.0;
  impl_->window_start_s = 0.0;
  impl_->window_bytes = 0;
  impl_->override_loss_rate = -1.0;
  impl_->override_latency_ms = -1.0;
  impl_->override_bandwidth_bps = -1.0;
}

}  // namespace omnicopilot
