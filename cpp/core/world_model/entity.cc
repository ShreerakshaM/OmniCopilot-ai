// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include "cpp/core/world_model/entity.h"

#include <cmath>

namespace omnicopilot {

double Position3D::DistanceTo(const Position3D& other) const {
  double dx = x - other.x;
  double dy = y - other.y;
  double dz = z - other.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double Velocity3D::Magnitude() const {
  return std::sqrt(vx * vx + vy * vy + vz * vz);
}

}  // namespace omnicopilot
