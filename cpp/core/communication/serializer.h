// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Serializer — Serialize/deserialize observations and messages for transmission.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cpp/core/world_model/entity.h"

namespace omnicopilot {

/// Serialization format.
enum class SerializationFormat : int {
  kProtobuf = 0,
  kFlatBuffers = 1,  // Future: zero-copy for hot path.
};

/// Message serializer/deserializer.
class Serializer {
 public:
  explicit Serializer(SerializationFormat format = SerializationFormat::kProtobuf);
  ~Serializer();

  /// Serialize an observation to bytes.
  std::vector<uint8_t> SerializeObservation(const Observation& obs) const;

  /// Deserialize bytes to an observation.
  Observation DeserializeObservation(const uint8_t* data, size_t size) const;

  /// Serialize a batch of observations.
  std::vector<uint8_t> SerializeBatch(const std::vector<Observation>& batch) const;

  /// Deserialize a batch.
  std::vector<Observation> DeserializeBatch(const uint8_t* data, size_t size) const;

  /// Estimate serialized size of an observation (for budget planning).
  uint32_t EstimateSize(const Observation& obs) const;

 private:
  SerializationFormat format_;
};

}  // namespace omnicopilot
