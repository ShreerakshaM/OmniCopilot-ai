# Phase 3 — Trust-Aware Fusion and Agent-Count Saturation

Phase 3 adds three upgrades to the Phase 2 cooperative-perception pipeline:

- **Trust-aware fusion:** observation evidence is weighted by detector confidence
  and the reporting agent's trust score. Low-trust measurements contribute less to
  existence confidence, class selection, dimensions, and Kalman measurement
  confidence.
- **Byzantine consensus:** critical multi-agent observations can require a
  trust-weighted supermajority (>2/3), while reporting disagreeing agents for
  downstream reliability updates.
- **Uncertainty-aware tracking:** the temporal tracker owns position and velocity
  fusion. Association uses an uncertainty-normalized squared distance with a
  configurable Mahalanobis threshold and retains a hard distance ceiling for
  numerical and operational safety. Eigen's LDLT decomposition is used for
  innovation inversion, and the Kalman covariance update uses Joseph form.

The Result 03 analysis now also reports mAP and recall grouped by the number of
agents present in each frame. This produces the expected saturation view around
four agents without changing the headline aggregate metric.

## Reproduce

```bash
python scripts/analyze_detection_cooperation.py \
  --data-root <DATA_ROOT>/opv2v-2/test \
  --module-dir build/cpp \
  --assoc-radius 2.5 \
  --frame-stride 10 \
  --min-agents 2 \
  --seed 0 \
  --out results/phase3_agent_count.json
```

The JSON output contains `agent_count_breakdown`, keyed by agent count, with
frame count, single-agent mAP, cooperative mAP, absolute gain, and recall.
Absolute values retain the Phase 2 caveat: detections are simulated
ground-truth-plus-noise, not a neural detector.
