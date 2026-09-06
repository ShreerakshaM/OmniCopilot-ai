// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0
//
// Lightweight binary serializer for observations.
// Uses a simple custom format for now — Protobuf integration deferred until
// proto code generation is wired into the build for the runtime path.
// Format: fixed-size header + variable strings.

#include "cpp/core/communication/serializer.h"

#include <cstring>
#include <stdexcept>

namespace omnicopilot {

Serializer::Serializer(SerializationFormat format) : format_(format) {}
Serializer::~Serializer() = default;

// Simple binary helpers.
namespace {

void WriteDouble(std::vector<uint8_t>& buf, double val) {
  size_t offset = buf.size();
  buf.resize(offset + sizeof(double));
  std::memcpy(buf.data() + offset, &val, sizeof(double));
}

double ReadDouble(const uint8_t* data, size_t& offset) {
  double val;
  std::memcpy(&val, data + offset, sizeof(double));
  offset += sizeof(double);
  return val;
}

void WriteInt32(std::vector<uint8_t>& buf, int32_t val) {
  size_t offset = buf.size();
  buf.resize(offset + sizeof(int32_t));
  std::memcpy(buf.data() + offset, &val, sizeof(int32_t));
}

int32_t ReadInt32(const uint8_t* data, size_t& offset) {
  int32_t val;
  std::memcpy(&val, data + offset, sizeof(int32_t));
  offset += sizeof(int32_t);
  return val;
}

void WriteString(std::vector<uint8_t>& buf, const std::string& s) {
  WriteInt32(buf, static_cast<int32_t>(s.size()));
  size_t offset = buf.size();
  buf.resize(offset + s.size());
  if (!s.empty()) {
    std::memcpy(buf.data() + offset, s.data(), s.size());
  }
}

std::string ReadString(const uint8_t* data, size_t& offset) {
  int32_t len = ReadInt32(data, offset);
  if (len < 0) return "";
  std::string s(reinterpret_cast<const char*>(data + offset),
                static_cast<size_t>(len));
  offset += static_cast<size_t>(len);
  return s;
}

}  // namespace

std::vector<uint8_t> Serializer::SerializeObservation(
    const Observation& obs) const {
  std::vector<uint8_t> buf;
  buf.reserve(256);

  // Strings.
  WriteString(buf, obs.observation_id);
  WriteString(buf, obs.agent_id);
  WriteString(buf, obs.track_id);

  // Class + sensor type.
  WriteInt32(buf, static_cast<int32_t>(obs.object_class));
  WriteInt32(buf, obs.sensor_type);

  // Position.
  WriteDouble(buf, obs.position.x);
  WriteDouble(buf, obs.position.y);
  WriteDouble(buf, obs.position.z);

  // Velocity.
  WriteDouble(buf, obs.velocity.vx);
  WriteDouble(buf, obs.velocity.vy);
  WriteDouble(buf, obs.velocity.vz);

  // Scalar fields.
  WriteDouble(buf, obs.confidence);
  WriteDouble(buf, obs.timestamp_s);
  WriteDouble(buf, obs.length);
  WriteDouble(buf, obs.width);
  WriteDouble(buf, obs.height);
  WriteDouble(buf, obs.heading);

  return buf;
}

Observation Serializer::DeserializeObservation(const uint8_t* data,
                                               size_t size) const {
  Observation obs;
  size_t offset = 0;

  obs.observation_id = ReadString(data, offset);
  obs.agent_id = ReadString(data, offset);
  obs.track_id = ReadString(data, offset);

  obs.object_class = static_cast<ObjectClass>(ReadInt32(data, offset));
  obs.sensor_type = ReadInt32(data, offset);

  obs.position.x = ReadDouble(data, offset);
  obs.position.y = ReadDouble(data, offset);
  obs.position.z = ReadDouble(data, offset);

  obs.velocity.vx = ReadDouble(data, offset);
  obs.velocity.vy = ReadDouble(data, offset);
  obs.velocity.vz = ReadDouble(data, offset);

  obs.confidence = ReadDouble(data, offset);
  obs.timestamp_s = ReadDouble(data, offset);
  obs.length = ReadDouble(data, offset);
  obs.width = ReadDouble(data, offset);
  obs.height = ReadDouble(data, offset);
  obs.heading = ReadDouble(data, offset);

  return obs;
}

std::vector<uint8_t> Serializer::SerializeBatch(
    const std::vector<Observation>& batch) const {
  std::vector<uint8_t> buf;
  buf.reserve(batch.size() * 256);

  WriteInt32(buf, static_cast<int32_t>(batch.size()));
  for (const auto& obs : batch) {
    auto obs_bytes = SerializeObservation(obs);
    WriteInt32(buf, static_cast<int32_t>(obs_bytes.size()));
    size_t offset = buf.size();
    buf.resize(offset + obs_bytes.size());
    std::memcpy(buf.data() + offset, obs_bytes.data(), obs_bytes.size());
  }

  return buf;
}

std::vector<Observation> Serializer::DeserializeBatch(const uint8_t* data,
                                                      size_t size) const {
  size_t offset = 0;
  int32_t count = ReadInt32(data, offset);

  std::vector<Observation> batch;
  batch.reserve(static_cast<size_t>(count));

  for (int32_t i = 0; i < count; ++i) {
    int32_t obs_size = ReadInt32(data, offset);
    batch.push_back(DeserializeObservation(data + offset,
                                           static_cast<size_t>(obs_size)));
    offset += static_cast<size_t>(obs_size);
  }

  return batch;
}

uint32_t Serializer::EstimateSize(const Observation& obs) const {
  // Fixed fields: 3 doubles (pos) + 3 doubles (vel) + 6 doubles (scalars)
  //             + 2 int32s (class, sensor) = 12*8 + 2*4 = 104 bytes
  // Variable: 3 strings with 4-byte length prefix each.
  uint32_t fixed = 104;
  uint32_t strings = 12 + static_cast<uint32_t>(obs.observation_id.size() +
                                                  obs.agent_id.size() +
                                                  obs.track_id.size());
  return fixed + strings;
}

}  // namespace omnicopilot
