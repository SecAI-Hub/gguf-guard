# gguf-guard

Static analysis and integrity verification for GGUF model files. Detects weight-level anomalies, structural tampering, and quantization-aware attacks without running inference.

## Why

GGUF files from public model hubs can be tampered with — trojaned weights, corrupted quantization blocks, swapped tensors. `gguf-guard` catches these by analyzing the raw binary structure and statistical properties of every tensor, comparing them against known-good baselines.

## Features

- **Full anomaly scan** — per-tensor statistical analysis (mean, variance, kurtosis, outlier ratio, sparsity) with layered scoring
- **Quant-aware block analysis** — inspects raw quantization blocks (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0) for scale entropy anomalies, repeated blocks, bin saturation
- **Robust statistics** — median/MAD, trimmed means, Tukey fences to reduce false positives on heavy-tailed distributions
- **Structural policy checks** — validates GGUF version, tensor offsets/overlaps, metadata consistency, known architecture patterns
- **Model family matching** — identifies architecture (llama, mistral, mixtral, qwen2, gemma, phi) by tensor naming and shape patterns
- **Lineage verification** — compares source (FP/BF16) model against quantized GGUF, normalizing for expected quantization loss per format
- **Per-tensor integrity manifests** — SHA-256 hash per tensor with Merkle tree root for efficient verification
- **Reference profiles** — build from multiple clean samples, merge for wider ranges, compare with tiered confidence (high/medium/low)
- **Fingerprinting** — deterministic structural hash for quick identity checks
- **CI-friendly** — `--quiet` mode outputs `PASS/WARN/FAIL score=N.NN`, exit code 2 on failure

## Install

```bash
go install github.com/SecAI-Hub/gguf-guard/cmd/gguf-guard@latest
```

Or build from source:

```bash
git clone https://github.com/SecAI-Hub/gguf-guard.git
cd gguf-guard
go build -o gguf-guard ./cmd/gguf-guard
```

## Quick Start

```bash
# Full scan
gguf-guard scan model.gguf

# CI mode (one-line output, exit code 2 on FAIL)
gguf-guard scan --quiet model.gguf

# Scan with reference profile
gguf-guard scan --reference llama-7b-ref.json model.gguf

# Structural inspection only (fast)
gguf-guard inspect model.gguf

# Show model metadata
gguf-guard info model.gguf

# Generate integrity manifest
gguf-guard manifest model.gguf --output manifest.json

# Verify against manifest
gguf-guard verify-manifest model.gguf manifest.json

# Compare two models
gguf-guard compare baseline.gguf candidate.gguf

# Lineage check (source FP model vs quantized)
gguf-guard lineage source-f32.gguf candidate-q4k.gguf

# Build reference from multiple clean samples
gguf-guard build-reference clean1.gguf clean2.gguf --output ref.json
```

## Commands

| Command | Description |
|---------|-------------|
| `scan` | Full anomaly scan with layered scoring |
| `fingerprint` | Generate structural fingerprint |
| `compare` | Compare two GGUF files tensor-by-tensor |
| `profile` | Generate reference profile from a single model |
| `build-reference` | Build merged reference from multiple samples |
| `lineage` | Verify quantized model against FP/BF16 source |
| `manifest` | Generate per-tensor SHA-256 integrity manifest |
| `verify-manifest` | Verify model against integrity manifest |
| `inspect` | Run structural policy checks and family matching |
| `info` | Show model metadata and tensor type summary |

Run `gguf-guard <command> --help` for command-specific flags.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | PASS — no significant anomalies |
| 1 | Error (parse failure, missing file) |
| 2 | FAIL — score exceeds threshold or policy violation |

## Scoring

The scan produces a composite score from 0.0 (clean) to 1.0 (highly suspicious), built from four layers:

| Layer | Weight | What it checks |
|-------|--------|----------------|
| Tensor-local | 0.30 | Per-tensor moments, outlier ratios, NaN/Inf |
| Role-group | 0.25 | Cross-layer consistency (z-score + robust z-score) |
| Model-global | 0.20 | Anomaly concentration pattern |
| Reference | 0.25 | Deviation from known-good reference profile |

Thresholds: `score > 0.2` = WARN, `score > 0.5` = FAIL.

See [docs/scoring.md](docs/scoring.md) for details.

## Supported Quantization Types

Dequantization: F32, F16, BF16, Q8_0, Q4_0, Q4_1

Block-level analysis: Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0

## Project Structure

```
gguf/
  parser.go       GGUF v2/v3 binary parser
  types.go        GGML type definitions
  dequant.go      Dequantization routines
  block.go        Block-level quantization analysis

analysis/
  stats.go        Tensor statistics (mean, variance, kurtosis, percentiles)
  anomaly.go      Layered anomaly detection and scoring
  robust.go       Robust statistics (median, MAD, trimmed mean, Tukey)
  quant.go        Quant-format-aware anomaly checks
  reference.go    Reference profiles (single/multi-sample, provenance)
  compare.go      Model-to-model comparison
  lineage.go      Lineage diff (source vs quantized)
  fingerprint.go  Structural fingerprinting
  manifest.go     Per-tensor integrity manifests (SHA-256 + Merkle)
  policy.go       Structural policy validation
  families.go     Model family matching heuristics

cmd/gguf-guard/
  main.go         CLI entry point
```

## Documentation

- [Architecture](docs/architecture.md) — design and analysis pipeline
- [Scoring](docs/scoring.md) — anomaly scoring methodology
- [CLI Reference](docs/cli-reference.md) — all commands and flags

## Testing

```bash
go test ./... -v
```

73 tests covering parser, dequantization, block analysis, statistics, anomaly detection, robust statistics, manifests, policy checks, family matching, lineage, and quant-aware analysis.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
