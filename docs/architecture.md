# Architecture

## Overview

gguf-guard is a pure-Go static analysis tool for GGUF model files. It operates entirely on the binary structure — no inference, no GPU, no model loading. The analysis pipeline runs in a fixed order, from cheapest to most expensive.

## Analysis Pipeline

```
GGUF File
  │
  ├─1─ Parse (gguf/parser.go)
  │     Read header, metadata, tensor info table
  │
  ├─2─ Structural Policy (analysis/policy.go)         [fast]
  │     Version check, offset validation, overlap detection,
  │     metadata consistency, tensor shape sanity
  │
  ├─3─ Family Matching (analysis/families.go)          [fast]
  │     Score against known architectures (llama, mistral, etc.)
  │     by tensor naming patterns and shape rules
  │
  ├─4─ Fingerprint (analysis/fingerprint.go)           [fast]
  │     Deterministic structural hash from tensor names,
  │     shapes, types, and metadata
  │
  ├─5─ Dequantized Statistics (analysis/stats.go)      [medium]
  │     Per-tensor: mean, variance, stddev, kurtosis, skewness,
  │     percentiles, outlier ratio, zero fraction, NaN/Inf count
  │
  ├─6─ Block Analysis (gguf/block.go + analysis/quant.go)  [medium]
  │     Raw quantization block inspection: scale entropy,
  │     code entropy, saturation, repeated blocks, zero scales
  │
  ├─7─ Anomaly Detection (analysis/anomaly.go)         [fast]
  │     Layered scoring from tensor stats and block stats
  │     Cross-layer consistency, robust z-scores
  │
  └─8─ Reference Comparison (analysis/reference.go)    [fast]
        Compare against known-good reference profile
        Tiered confidence: high/medium/low/insufficient-data
```

## Package Responsibilities

### `gguf/` — Binary Parsing

- **parser.go**: Reads GGUF v2/v3 format. Extracts header, metadata key-value pairs, tensor info table, and data offset. Does not load tensor data into memory until explicitly requested.
- **types.go**: GGML type definitions (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K through Q8_K). Includes type size, block size, and support flags.
- **dequant.go**: Converts raw tensor bytes to float32 for statistical analysis. Supports F32, F16, BF16, Q8_0, Q4_0, Q4_1. Optional sampling for large tensors.
- **block.go**: Extracts block-level statistics directly from quantized data without dequantizing. Computes scale distributions, code entropy, saturation ratios, block repetition. Supports Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0.

### `analysis/` — Analysis Engine

- **stats.go**: Computes per-tensor statistics from dequantized float32 values.
- **anomaly.go**: Core anomaly detection with four scoring layers (tensor-local, role-group, model-global, reference). Uses both standard and robust z-scores.
- **robust.go**: Outlier-resistant statistics — median, MAD (Median Absolute Deviation), trimmed mean, Tukey fences with IQR. Designed to reduce false positives on naturally heavy-tailed tensor distributions.
- **quant.go**: Quant-format-aware checks on block-level statistics. Detects anomalies invisible to dequantized analysis (e.g., uniform scales with varied codes).
- **reference.go**: Reference profile management. Single-sample or multi-sample (merged). Tracks provenance (converter version, llama.cpp commit). Tiered confidence matching.
- **compare.go**: Tensor-by-tensor comparison between two models, computing statistical distances.
- **lineage.go**: Compares source (FP/BF16) model against a quantized derivative. Uses per-format quantization loss budgets to distinguish expected drift from suspicious drift.
- **fingerprint.go**: Deterministic structural hash from tensor names, shapes, and types. Quick identity check without reading data.
- **manifest.go**: Per-tensor SHA-256 hash manifest with Merkle tree root. Generates, saves, loads, and verifies.
- **policy.go**: Structural validation rules — version, offsets, overlaps, metadata keys, tensor shapes, architecture consistency.
- **families.go**: Model family definitions with expected tensor roles and shape rules for llama, mistral, mixtral, qwen2, gemma, phi.

### `cmd/gguf-guard/` — CLI

- **main.go**: Entry point with 10 subcommands. Orchestrates the analysis pipeline and handles output formatting.

## Design Decisions

1. **No inference dependency**: All analysis is static. This means gguf-guard can run in restricted environments (CI, containers) without GPU or model runtime.

2. **Block-level analysis**: Quantized types pack weights into blocks with per-block scales. A malicious actor could manipulate scales while keeping dequantized statistics normal. Block-level analysis catches this by inspecting the raw quantization structure.

3. **Robust statistics**: Neural network tensor distributions are often heavy-tailed (high kurtosis, long tails). Standard z-scores produce false positives. Median/MAD-based scoring is more resistant to legitimate outliers.

4. **Layered scoring**: Single-metric scoring misses targeted attacks. The four-layer approach (local, group, global, reference) catches both broad corruption and surgical modifications.

5. **Quantization loss budgets**: When comparing a quantized model against its FP source, some statistical drift is expected. The budgets (per quant type) normalize this, so only unexpected drift triggers alerts.
