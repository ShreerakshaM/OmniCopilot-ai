// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the simulation layer.
// This is the main pybind11 module entry point.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace omnicopilot {

// Forward declarations of binding functions.
void BindWorldModel(py::module_& m);
void BindCommunication(py::module_& m);
void BindAgent(py::module_& m);

}  // namespace omnicopilot

PYBIND11_MODULE(_omnicopilot_cpp, m) {
  m.doc() = "OmniCopilot C++ core — world model, communication, and agent runtime";

  omnicopilot::BindWorldModel(m);
  omnicopilot::BindCommunication(m);
  omnicopilot::BindAgent(m);
}
