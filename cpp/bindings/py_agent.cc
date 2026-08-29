// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the Agent runtime.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/core/agent/agent.h"

namespace py = pybind11;

namespace omnicopilot {

void BindAgent(py::module_& m) {
  py::class_<AgentConfig>(m, "AgentConfig")
      .def(py::init<>())
      .def_readwrite("agent_id", &AgentConfig::agent_id)
      .def_readwrite("initial_position", &AgentConfig::initial_position);

  py::class_<Agent>(m, "Agent")
      .def(py::init<AgentConfig>())
      .def("step", &Agent::Step)
      .def("get_outgoing_messages", &Agent::GetOutgoingMessages)
      .def("get_world_model", &Agent::GetWorldModel, py::return_value_policy::reference)
      .def("get_position", &Agent::GetPosition)
      .def("set_position", &Agent::SetPosition)
      .def("get_id", &Agent::GetId);
}

}  // namespace omnicopilot
