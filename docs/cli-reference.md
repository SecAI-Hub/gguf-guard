# CLI Reference

## Global

```
gguf-guard <command> [flags] [arguments]
gguf-guard version
gguf-guard help
```

## scan

Full anomaly scan with layered scoring.

```
gguf-guard scan [flags] model.gguf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--reference` | | Path to reference profile JSON |
| `--output` | stdout | Write JSON output to file |
| `--max-tensors` | 0 (all) | Limit number of tensors to analyze |
| `--stats` | false | Include per-tensor statistics in output |
| `--quiet` | false | One-line output: `PASS/WARN/FAIL score=N.NN summary` |

Output includes: fingerprint, structural policy report, family match, anomaly report with layered scores, quant block report.

Exit code 2 if score > 0.5.

## fingerprint

Generate a deterministic structural fingerprint.

```
gguf-guard fingerprint [flags] model.gguf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | stdout | Write JSON output to file |

## compare

Compare two GGUF files tensor-by-tensor.

```
gguf-guard compare [flags] baseline.gguf candidate.gguf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | stdout | Write JSON output to file |
| `--max-tensors` | 0 (all) | Limit tensors to analyze |

## profile

Generate a reference profile from a single model.

```
gguf-guard profile [flags] model.gguf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | stdout | Write JSON output to file |
| `--margin` | 3.0 | Standard deviations for acceptable range |
| `--max-tensors` | 0 (all) | Limit tensors |

## build-reference

Build a merged reference profile from multiple clean samples.

```
gguf-guard build-reference [flags] model1.gguf [model2.gguf ...]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | stdout | Write merged reference JSON |
| `--margin` | 3.0 | Standard deviations for range |
| `--max-tensors` | 0 (all) | Limit tensors |

Multi-sample references produce wider, more robust ranges. The merged profile tracks sample count and source hashes.

## lineage

Verify a quantized model against its FP/BF16 source.

```
gguf-guard lineage [flags] source.gguf candidate.gguf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | stdout | Write JSON output to file |
| `--max-tensors` | 0 (all) | Limit tensors |

Verdicts: `consistent`, `suspicious`, `incompatible`. Exit code 2 on `suspicious`.

## manifest

Generate a per-tensor SHA-256 integrity manifest.

```
gguf-guard manifest [flags] model.gguf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | stdout | Write manifest JSON |

Output includes per-tensor hashes and a Merkle tree root.

## verify-manifest

Verify a model against a previously generated manifest.

```
gguf-guard verify-manifest model.gguf manifest.json
```

Prints `PASS: all tensors match manifest` or lists mismatches. Exit code 2 on mismatch.

## inspect

Run structural policy checks and model family matching.

```
gguf-guard inspect [flags] model.gguf
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | stdout | Write JSON output to file |

Exit code 2 on policy violation.

## info

Show model metadata and tensor type summary.

```
gguf-guard info model.gguf
```

Outputs: path, GGUF version, architecture, quant type, tensor count, parameter count, file size, metadata, type distribution.
