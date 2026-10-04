// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the Communication layer.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/core/communication/channel.h"
#include "cpp/core/communication/prioritizer.h"
#include "cpp/core/communication/receiver_model.h"

namespace py = pybind11;

namespace omnicopilot {

void BindCommunication(py::module_& m) {
  // ChannelConfig
  py::class_<ChannelConfig>(m, "ChannelConfig")
      .def(py::init<>())
      .def_readwrite("max_bandwidth_bps", &ChannelConfig::max_bandwidth_bps)
      .def_readwrite("base_latency_ms", &ChannelConfig::base_latency_ms)
      .def_readwrite("latency_jitter_ms", &ChannelConfig::latency_jitter_ms)
      .def_readwrite("packet_loss_rate", &ChannelConfig::packet_loss_rate)
      .def_readwrite("max_queue_depth", &ChannelConfig::max_queue_depth);

  // ChannelState
  py::class_<ChannelState>(m, "ChannelState")
      .def(py::init<>())
      .def_readwrite("available_bandwidth_bps", &ChannelState::available_bandwidth_bps)
      .def_readwrite("current_latency_ms", &ChannelState::current_latency_ms)
      .def_readwrite("current_loss_rate", &ChannelState::current_loss_rate)
      .def_readwrite("utilization", &ChannelState::utilization)
      .def_readwrite("queue_depth", &ChannelState::queue_depth);

  // ChannelMessage
  py::class_<ChannelMessage>(m, "ChannelMessage")
      .def(py::init<>())
      .def_readwrite("message_id", &ChannelMessage::message_id)
      .def_readwrite("sender_id", &ChannelMessage::sender_id)
      .def_readwrite("receiver_id", &ChannelMessage::receiver_id)
      .def_readwrite("size_bytes", &ChannelMessage::size_bytes)
      .def_readwrite("enqueue_time_s", &ChannelMessage::enqueue_time_s)
      .def_readwrite("delivery_time_s", &ChannelMessage::delivery_time_s);

  // Channel
  py::class_<Channel>(m, "Channel")
      .def(py::init<ChannelConfig>())
      .def("enqueue", &Channel::Enqueue)
      .def("tick", &Channel::Tick)
      .def("get_state", &Channel::GetState)
      .def("get_stats", &Channel::GetStats)
      .def("set_conditions", &Channel::SetConditions)
      .def("reset", &Channel::Reset);

  // Channel::Stats
  py::class_<Channel::Stats>(m, "ChannelStats")
      .def(py::init<>())
      .def_readonly("total_enqueued", &Channel::Stats::total_enqueued)
      .def_readonly("total_delivered", &Channel::Stats::total_delivered)
      .def_readonly("total_dropped_loss", &Channel::Stats::total_dropped_loss)
      .def_readonly("total_dropped_queue", &Channel::Stats::total_dropped_queue)
      .def_readonly("total_bytes_transmitted", &Channel::Stats::total_bytes_transmitted)
      .def_readonly("mean_latency_ms", &Channel::Stats::mean_latency_ms);

  // PrioritizationWeights
  py::class_<PrioritizationWeights>(m, "PrioritizationWeights")
      .def(py::init<>())
      .def_readwrite("safety_relevance", &PrioritizationWeights::safety_relevance)
      .def_readwrite("confidence", &PrioritizationWeights::confidence)
      .def_readwrite("novelty", &PrioritizationWeights::novelty)
      .def_readwrite("time_sensitivity", &PrioritizationWeights::time_sensitivity)
      .def_readwrite("receiver_benefit", &PrioritizationWeights::receiver_benefit);

  // ScoredObservation
  py::class_<ScoredObservation>(m, "ScoredObservation")
      .def(py::init<>())
      .def_readwrite("observation", &ScoredObservation::observation)
      .def_readwrite("total_score", &ScoredObservation::total_score)
      .def_readwrite("safety_score", &ScoredObservation::safety_score)
      .def_readwrite("confidence_score", &ScoredObservation::confidence_score)
      .def_readwrite("novelty_score", &ScoredObservation::novelty_score)
      .def_readwrite("time_sensitivity_score", &ScoredObservation::time_sensitivity_score)
      .def_readwrite("receiver_benefit_score", &ScoredObservation::receiver_benefit_score)
      .def_readwrite("target_receiver", &ScoredObservation::target_receiver);

  // PrioritizationResult
  py::class_<PrioritizationResult>(m, "PrioritizationResult")
      .def(py::init<>())
      .def_readwrite("selected", &PrioritizationResult::selected)
      .def_readwrite("suppressed", &PrioritizationResult::suppressed)
      .def_readwrite("total_selected_value", &PrioritizationResult::total_selected_value)
      .def_readwrite("total_suppressed_value", &PrioritizationResult::total_suppressed_value);

  // Prioritizer
  py::class_<Prioritizer>(m, "Prioritizer")
      .def(py::init<PrioritizationWeights>())
      .def("prioritize", &Prioritizer::Prioritize,
           py::arg("observations"), py::arg("budget_bytes"),
           py::arg("bytes_per_observation") = 256)
      .def("score", &Prioritizer::Score);
}

}  // namespace omnicopilot
