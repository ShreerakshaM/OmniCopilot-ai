// Copyright 2024 OmniCopilot Authors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>

#include "cpp/core/world_model/temporal_tracker.h"

namespace omnicopilot {
namespace {

Observation MakeObservation(double x, double y, double confidence,
                            double timestamp_s = 0.0) {
  Observation observation;
  observation.agent_id = "agent";
  observation.position = {x, y, 0.0};
  observation.velocity = {0.0, 0.0, 0.0};
  observation.confidence = confidence;
  observation.timestamp_s = timestamp_s;
  return observation;
}

TEST(TemporalTrackerTest, TrustedUpdateReducesPositionUncertainty) {
  TemporalTracker tracker(TemporalTrackerConfig{});
  tracker.Initialize(MakeObservation(0.0, 0.0, 1.0));
  double before = tracker.GetPositionUncertainty();

  tracker.Update(MakeObservation(0.1, 0.1, 1.0, 0.1));

  EXPECT_LT(tracker.GetPositionUncertainty(), before);
}

TEST(TemporalTrackerTest, LowConfidenceUpdateHasLessPositionalInfluence) {
  TemporalTracker trusted(TemporalTrackerConfig{});
  TemporalTracker untrusted(TemporalTrackerConfig{});
  trusted.Initialize(MakeObservation(0.0, 0.0, 1.0));
  untrusted.Initialize(MakeObservation(0.0, 0.0, 1.0));

  trusted.Update(MakeObservation(10.0, 0.0, 1.0, 0.1));
  untrusted.Update(MakeObservation(10.0, 0.0, 0.1, 0.1));

  EXPECT_GT(trusted.GetPosition().x, untrusted.GetPosition().x);
}

TEST(TemporalTrackerTest, InnovationDistanceUsesCovariance) {
  TemporalTracker tracker(TemporalTrackerConfig{});
  tracker.Initialize(MakeObservation(0.0, 0.0, 1.0));

  double distance =
      tracker.InnovationMahalanobisDistance(MakeObservation(2.0, 0.0, 1.0));

  EXPECT_GT(distance, 0.0);
  EXPECT_TRUE(std::isfinite(distance));
}

}  // namespace
}  // namespace omnicopilot
