# Security audit — 2026-08-02

## Scope

This audit covered GGUF parsing, tensor reads and hashing, statistical analysis,
reference profiles, integrity manifests, CLI output, tests, containers, and the
GitHub Actions supply chain. The review assumed model files and sidecar JSON are
attacker-controlled.

## Remediated findings

- Added strict bounds for headers, strings, counts, nested metadata arrays,
  tensor reads, and statistical samples.
- Rejects symlinks and other non-regular inputs, invalid UTF-8, duplicate keys
  and tensor names, invalid booleans and types, arithmetic overflow, malformed
  quantization blocks, unaligned/out-of-bounds ranges, and overlapping tensors.
- Pins reads and streaming SHA-256 operations to the file identity accepted by
  the parser and detects size or modification-time changes.
- Manifest verification now checks the full-file and structure hashes, summary
  fields, exact tensor set and metadata, every tensor hash, and the Merkle root.
- Reference and manifest JSON is bounded, rejects unknown/trailing data, and is
  written atomically without following output symlinks.
- Reference profiles now require source and structure provenance, valid ranges,
  and exact model-layout matching. They can tighten but cannot relax built-in
  limits, and uncovered analyzed tensors are reported as findings.
- Tensor and quantization analysis no longer skips read/dequantization failures;
  critical anomalies always fail the CLI regardless of aggregate score.
- Bounded statistical and block analysis now samples complete windows across
  each large tensor, including its beginning and end, rather than inspecting
  only a prefix. Derived score layers and summaries are recomputed after every
  policy and quantization finding.
- Hardened the public sampler against adversarial caller limits: encoded sample
  reads are capped at 64 MiB before multiplication or full-tensor allocation.
  Independent policy-layout arithmetic rejects fabricated overflowing offsets.
- Replaced quadratic overlap checking with a sorted O(n log n) implementation
  at adversarially high tensor counts.
- Added a non-root, digest-pinned container and hardened CI, CodeQL, dependency
  updates, and signed/provenanced release workflows with immutable action pins.
- Gated every release publisher on a read-only check that requires a
  v-prefixed strict SemVer annotated tag resolving exactly to the workflow
  commit, with that commit contained in `origin/main`. Write, package, OIDC,
  and attestation permissions remain confined to the publishing jobs.

## Verification completed

- `go test -race -count=1 ./...`
- `go vet ./...`
- `gosec -severity medium -confidence medium ./...`: no findings
- `govulncheck ./...`: no reachable vulnerabilities
- Trivy source/config/secret scan: no HIGH or CRITICAL findings
- Trivy final-image scan: no HIGH or CRITICAL findings
- Gitleaks full-history and working-tree scans: no findings
- Actionlint and a read-only/capability-dropped container smoke test
- Release dependency/permission graph checks, strict SemVer positive/negative
  fixtures, and synthetic annotated/lightweight/off-main Git ancestry cases

## Residual operational risks

- Statistical checks are detection signals, not proof that a model is benign.
- A manifest authenticates content only when the manifest itself comes from a
  trusted, authenticated channel. Sign or attest manifests in the surrounding
  release/registry workflow.
- An actor that can modify a model concurrently with scanning can still create
  availability failures. Scan immutable, access-controlled artifacts and verify
  the final full-file digest at promotion time.
- Very large valid models remain intentionally CPU- and I/O-intensive. Apply
  job-level CPU, memory, file-size, and wall-clock limits around the CLI.
