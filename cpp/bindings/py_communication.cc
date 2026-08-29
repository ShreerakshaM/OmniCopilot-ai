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

  // Channel
  py::class_<Channel>(m, "Channel")
      .def(py::init<ChannelConfig>())
      .def("enqueue", &Channel::Enqueue)
      .def("tick", &Channel::Tick)
      .def("get_state", &Channel::GetState)
      .def("set_conditions", &Channel::SetConditions)
      .def("reset", &Channel::Reset);

  // PrioritizationWeights
  py::class_<PrioritizationWeights>(m, "PrioritizationWeights")
      .def(py::init<>())
      .def_readwrite("safety_relevance", &PrioritizationWeights::safety_relevance)
      .def_readwrite("confidence", &PrioritizationWeights::confidence)
      .def_readwrite("novelty", &PrioritizationWeights::novelty)
      .def_readwrite("time_sensitivity", &PrioritizationWeights::time_sensitivity)
      .def_readwrite("receiver_benefit", &PrioritizationWeights::receiver_benefit);

  // Prioritizer
  py::class_<Prioritizer>(m, "Prioritizer")
      .def(py::init<PrioritizationWeights>())
      .def("prioritize", &Prioritizer::Prioritize)
      .def("score", &Prioritizer::Score);
}

}  // namespace omnicopilot
