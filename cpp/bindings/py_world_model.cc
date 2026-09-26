// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Python bindings for the WorldModel and related types.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/core/world_model/entity.h"
#include "cpp/core/world_model/world_model.h"

namespace py = pybind11;

namespace omnicopilot {

void BindWorldModel(py::module_& m) {
  // Position3D
  py::class_<Position3D>(m, "Position3D")
      .def(py::init<>())
      .def_readwrite("x", &Position3D::x)
      .def_readwrite("y", &Position3D::y)
      .def_readwrite("z", &Position3D::z)
      .def("distance_to", &Position3D::DistanceTo);

  // Velocity3D
  py::class_<Velocity3D>(m, "Velocity3D")
      .def(py::init<>())
      .def_readwrite("vx", &Velocity3D::vx)
      .def_readwrite("vy", &Velocity3D::vy)
      .def_readwrite("vz", &Velocity3D::vz)
      .def("magnitude", &Velocity3D::Magnitude);

  // ObjectClass enum
  py::enum_<ObjectClass>(m, "ObjectClass")
      .value("UNKNOWN", ObjectClass::kUnknown)
      .value("VEHICLE", ObjectClass::kVehicle)
      .value("PEDESTRIAN", ObjectClass::kPedestrian)
      .value("CYCLIST", ObjectClass::kCyclist)
      .value("TRUCK", ObjectClass::kTruck);

  // EntityState enum
  py::enum_<EntityState>(m, "EntityState")
      .value("TENTATIVE", EntityState::kTentative)
      .value("CONFIRMED", EntityState::kConfirmed)
      .value("PREDICTED", EntityState::kPredicted)
      .value("STALE", EntityState::kStale)
      .value("LOST", EntityState::kLost);

  // Observation
  py::class_<Observation>(m, "Observation")
      .def(py::init<>())
      .def_readwrite("observation_id", &Observation::observation_id)
      .def_readwrite("agent_id", &Observation::agent_id)
      .def_readwrite("track_id", &Observation::track_id)
      .def_readwrite("object_class", &Observation::object_class)
      .def_readwrite("position", &Observation::position)
      .def_readwrite("velocity", &Observation::velocity)
      .def_readwrite("confidence", &Observation::confidence)
      .def_readwrite("timestamp_s", &Observation::timestamp_s);

  // TrackedEntity
  py::class_<TrackedEntity>(m, "TrackedEntity")
      .def(py::init<>())
      .def_readwrite("entity_id", &TrackedEntity::entity_id)
      .def_readwrite("object_class", &TrackedEntity::object_class)
      .def_readwrite("position", &TrackedEntity::position)
      .def_readwrite("velocity", &TrackedEntity::velocity)
      .def_readwrite("confidence", &TrackedEntity::confidence)
      .def_readwrite("position_uncertainty_m",
                     &TrackedEntity::position_uncertainty_m)
      .def_readwrite("position_covariance_xx",
                     &TrackedEntity::position_covariance_xx)
      .def_readwrite("position_covariance_xy",
                     &TrackedEntity::position_covariance_xy)
      .def_readwrite("position_covariance_yy",
                     &TrackedEntity::position_covariance_yy)
      .def_readwrite("state", &TrackedEntity::state)
      .def_readwrite("unique_source_count", &TrackedEntity::unique_source_count)
      .def_readwrite("observation_count", &TrackedEntity::observation_count)
      .def_readwrite("is_safety_critical", &TrackedEntity::is_safety_critical)
      .def_readwrite("needs_corroboration", &TrackedEntity::needs_corroboration);

  // WorldModelConfig
  // FusionConfig — association/fusion parameters.
  py::class_<FusionConfig>(m, "FusionConfig")
      .def(py::init<>())
      .def_readwrite("association_iou_threshold",
                     &FusionConfig::association_iou_threshold)
      .def_readwrite("association_max_distance_m",
                     &FusionConfig::association_max_distance_m)
      .def_readwrite("association_mahalanobis_threshold",
                     &FusionConfig::association_mahalanobis_threshold)
      .def_readwrite("conflict_resolution_threshold",
                     &FusionConfig::conflict_resolution_threshold);

  py::class_<WorldModelConfig>(m, "WorldModelConfig")
      .def(py::init<>())
      .def_readwrite("max_entities", &WorldModelConfig::max_entities)
      .def_readwrite("stale_confidence_threshold", &WorldModelConfig::stale_confidence_threshold)
      .def_readwrite("stale_timeout_s", &WorldModelConfig::stale_timeout_s)
      .def_readwrite("removal_timeout_s", &WorldModelConfig::removal_timeout_s)
      .def_readwrite("confirmation_threshold", &WorldModelConfig::confirmation_threshold)
      .def_readwrite("confirmation_source_count", &WorldModelConfig::confirmation_source_count)
      .def_readwrite("spatial_cell_size_m", &WorldModelConfig::spatial_cell_size_m)
      .def_readwrite("fusion", &WorldModelConfig::fusion);

  // WorldModelStats
  py::class_<WorldModelStats>(m, "WorldModelStats")
      .def(py::init<>())
      .def_readonly("total_entities", &WorldModelStats::total_entities)
      .def_readonly("confirmed", &WorldModelStats::confirmed)
      .def_readonly("tentative", &WorldModelStats::tentative)
      .def_readonly("predicted", &WorldModelStats::predicted)
      .def_readonly("stale", &WorldModelStats::stale)
      .def_readonly("mean_confidence", &WorldModelStats::mean_confidence)
      .def_readonly("tick", &WorldModelStats::tick);

  // WorldModel
  py::class_<WorldModel>(m, "WorldModel")
      .def(py::init<WorldModelConfig>())
      .def("tick", &WorldModel::Tick)
      .def("ingest_observations", &WorldModel::IngestObservations,
           py::arg("observations"), py::arg("agent_trust") = 1.0)
      // Query methods return pointers INTO the world model's internal storage.
      // Use reference_internal so pybind11 does NOT take ownership / free them,
      // and keeps the parent WorldModel alive while the returned objects exist.
      .def("query_radius", &WorldModel::QueryRadius,
           py::return_value_policy::reference_internal)
      .def("get_entity", &WorldModel::GetEntity,
           py::return_value_policy::reference_internal)
      .def("get_entities", &WorldModel::GetEntities,
           py::arg("min_confidence") = 0.0, py::arg("object_class") = -1,
           py::return_value_policy::reference_internal)
      .def("get_uncertain_entities", &WorldModel::GetUncertainEntities,
           py::return_value_policy::reference_internal)
      .def("get_stats", &WorldModel::GetStats)
      .def("reset", &WorldModel::Reset);
}

}  // namespace omnicopilot
