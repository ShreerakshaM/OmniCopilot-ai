# Copyright 2024 OmniCopilot Authors
# SPDX-License-Identifier: Apache-2.0

# ─── Core Library ────────────────────────────────────────────────────────────

add_library(omnicopilot_core STATIC
  # World Model
  core/world_model/world_model.cc
  core/world_model/entity.cc
  core/world_model/spatial_index.cc
  core/world_model/temporal_tracker.cc
  core/world_model/fusion.cc
  core/world_model/uncertainty.cc

  # Communication
  core/communication/channel.cc
  core/communication/prioritizer.cc
  core/communication/receiver_model.cc
  core/communication/budget_manager.cc
  core/communication/serializer.cc

  # Trust
  core/trust/reliability.cc
  core/trust/anomaly.cc
  core/trust/byzantine.cc

  # Active Acquisition
  core/active_acquisition/info_value.cc
  core/active_acquisition/source_selector.cc
  core/active_acquisition/request_manager.cc

  # Agent Runtime
  core/agent/agent.cc
)

target_include_directories(omnicopilot_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(omnicopilot_core
  PUBLIC
    omnicopilot_proto
    spdlog::spdlog
)

# ─── Simulation Library ──────────────────────────────────────────────────────

add_library(omnicopilot_sim STATIC
  sim/network_sim.cc
  sim/opv2v_replay.cc
  sim/scenario_engine.cc
)

target_include_directories(omnicopilot_sim
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(omnicopilot_sim
  PUBLIC
    omnicopilot_core
)

# ─── Python Bindings ─────────────────────────────────────────────────────────

if(OMNICOPILOT_BUILD_BINDINGS)
  pybind11_add_module(_omnicopilot_cpp
    bindings/py_world_model.cc
    bindings/py_communication.cc
    bindings/py_agent.cc
    bindings/py_simulation.cc
  )

  target_link_libraries(_omnicopilot_cpp
    PRIVATE
      omnicopilot_core
      omnicopilot_sim
  )

  # Install the compiled module to the Python package
  install(TARGETS _omnicopilot_cpp
    LIBRARY DESTINATION python/omnicopilot
  )
endif()

# ─── Unit Tests ──────────────────────────────────────────────────────────────

if(OMNICOPILOT_BUILD_TESTS)
  add_executable(omnicopilot_tests
    tests/test_world_model.cc
    tests/test_fusion.cc
    tests/test_communication.cc
    tests/test_trust.cc
    tests/test_spatial_index.cc
  )

  target_link_libraries(omnicopilot_tests
    PRIVATE
      omnicopilot_core
      omnicopilot_sim
      GTest::gtest_main
      GTest::gmock
  )

  include(GoogleTest)
  gtest_discover_tests(omnicopilot_tests)
endif()
