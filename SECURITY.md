# Security Policy

## Scope

gguf-guard is a defensive static analysis tool for GGUF model files. It does not execute model code, run inference, or load models into memory beyond what is needed for binary parsing.

## Reporting a Vulnerability

If you discover a security issue in gguf-guard, please report it privately:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email: Open a [private security advisory](https://github.com/SecAI-Hub/gguf-guard/security/advisories/new) on GitHub
3. Include: description, reproduction steps, affected versions, and impact assessment

We will acknowledge receipt within 48 hours and provide a timeline for a fix.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x | Yes |
| < 0.2 | No |

## Security Considerations

- **Parser safety**: The GGUF parser validates magic bytes, version, and bounds-checks all offsets before reading tensor data. Malformed files should produce parse errors, not crashes.
- **No code execution**: gguf-guard never executes embedded code, scripts, or arbitrary metadata values from GGUF files.
- **Memory bounds**: Tensor data reads are bounded by declared sizes and file size. The `--max-tensors` flag limits memory usage for large models.
- **Hash algorithms**: Uses SHA-256 for all integrity hashing (manifests, fingerprints).
