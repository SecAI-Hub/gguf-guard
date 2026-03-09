# Scoring Methodology

## Composite Score

The scan produces a score from 0.0 (clean) to 1.0 (highly suspicious). The score combines four independent analysis layers:

```
score = max(layered_score, raw_score)
```

Where `layered_score` is a weighted sum of four layers, and `raw_score` is a simple accumulation from anomaly severities as a floor.

## Layer Breakdown

### 1. Tensor-Local (weight: 0.30)

Per-tensor checks on dequantized statistics:

| Check | Threshold | Severity |
|-------|-----------|----------|
| Abnormal mean | \|mean\| > 10.0 | warning |
| Extreme kurtosis | kurtosis > 100.0 | critical |
| Constant tensor | variance < 1e-12 (non-zero values) | warning |
| High sparsity | zero fraction > 0.99 | warning |
| High outlier ratio | > 10% beyond 3 sigma | warning |
| NaN values | any | critical |
| Inf values | any | critical |

### 2. Role-Group (weight: 0.25)

Cross-layer consistency within tensor role groups (e.g., all `attn_q.weight` tensors across layers):

- Groups tensors by role (strips layer number from name)
- Requires 3+ members for meaningful comparison
- Checks both standard z-score and robust z-score (median/MAD)
- Flags if deviation > 5.0 standard deviations from group
- Checks mean consistency and variance consistency separately

The robust z-score (using MAD) catches outliers in heavy-tailed groups where the standard z-score would miss them. When MAD is near-zero (all values effectively identical), the robust check is skipped to avoid false positives.

### 3. Model-Global (weight: 0.20)

Anomaly concentration analysis:

- If anomalies are concentrated in a few tensors (ratio > 3.0 anomalies per affected tensor), adds 0.2 to this layer
- Rationale: targeted attacks modify few tensors heavily, while legitimate variation is spread evenly

### 4. Reference (weight: 0.25)

Deviation from a known-good reference profile:

- Compares each tensor's mean, variance, and kurtosis against reference ranges
- Only active when a reference profile is provided
- Each out-of-range check produces a warning anomaly

## Severity Weights

| Severity | Score contribution |
|----------|--------------------|
| Critical | +0.30 |
| Warning | +0.10 |
| Info | +0.02 |

## Thresholds

| Score range | Verdict |
|-------------|---------|
| 0.0 - 0.2 | PASS |
| 0.2 - 0.5 | WARN |
| 0.5 - 1.0 | FAIL |

## Quant-Aware Scoring

Block-level quantization anomalies are incorporated into the main score:

| Check | Threshold | Severity |
|-------|-----------|----------|
| Low scale entropy | < 1.0 bits (10+ blocks) | warning |
| Extreme scale ratio | max/min > 1e6 | warning |
| Low code entropy | < 2.0 bits (10+ blocks) | warning |
| High saturation (low/high) | > 50% at bin extreme | warning |
| Repeated blocks | > 10% identical | critical |
| Excessive zero scales | > 50% near-zero | warning |

## Cross-Layer Z-Score Mathematics

For N samples, the maximum z-score a single outlier can achieve is sqrt(N-1). This means:

- N=3: max z = 1.41
- N=10: max z = 3.0
- N=26: max z = 5.0
- N=31: max z = 5.48

The default threshold of 5.0 requires 26+ layers to flag a single outlier, which prevents false positives on small models while catching anomalies in production-sized models.

## Lineage Scoring

When comparing a source model against a quantized derivative, each tensor's statistical distance is compared against a per-format loss budget:

| Quant Type | Max Distance | Max Variance Shift | Max Kurtosis Shift |
|------------|-------------|-------------------|-------------------|
| F32 | 0.0 | 0.0 | 0.0 |
| F16 | 0.5 | 0.1 | 1.0 |
| BF16 | 1.0 | 0.2 | 2.0 |
| Q8_0 | 2.0 | 0.5 | 5.0 |
| Q4_K | 5.0 | 1.5 | 15.0 |
| Q4_0 | 8.0 | 2.0 | 20.0 |

Verdicts: `consistent` (all within budget), `suspicious` (unexpected drift), `incompatible` (different structure).
