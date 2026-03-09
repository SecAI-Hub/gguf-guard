# Contributing to gguf-guard

## Getting Started

```bash
git clone https://github.com/SecAI-Hub/gguf-guard.git
cd gguf-guard
go test ./...
```

## Development

- Go 1.21+
- No external dependencies (stdlib only)
- Run `go vet ./...` before submitting

## Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests for new functionality
4. Ensure all tests pass: `go test ./... -count=1`
5. Submit a pull request with a clear description

## Adding Quantization Type Support

To add support for a new GGUF quantization type:

1. Add the type constant to `gguf/types.go`
2. Add dequantization in `gguf/dequant.go`
3. Add block extraction in `gguf/block.go` (if block-quantized)
4. Add tests for each
5. Update `analysis/quant.go` `isQuantizedType()` if block-quantized

## Adding a Model Family

To add a new model architecture family:

1. Add a `ModelFamily` entry in `analysis/families.go` `KnownFamilies`
2. Define expected tensor roles and shape rules
3. Add tensor name patterns to `getKnownTensorPatterns()` in `analysis/policy.go`
4. Add tests

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`)
- Keep functions focused and testable
- Use table-driven tests where appropriate
- Document exported types and functions
