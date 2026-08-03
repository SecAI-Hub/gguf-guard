package analysis

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

const maxManifestBytes = 64 * 1024 * 1024

// Manifest is a sidecar integrity document for a GGUF file.
// It contains per-tensor hashes, structural metadata, and an optional
// Merkle root that enables efficient integrity verification.
type Manifest struct {
	Version       string        `json:"manifest_version"`
	FileHash      string        `json:"file_hash"`
	StructureHash string        `json:"structure_hash"`
	Architecture  string        `json:"architecture"`
	QuantType     string        `json:"quant_type"`
	TensorCount   int           `json:"tensor_count"`
	Parameters    uint64        `json:"parameters"`
	FileSize      int64         `json:"file_size"`
	MerkleRoot    string        `json:"merkle_root"`
	Tensors       []TensorEntry `json:"tensors"`
}

// TensorEntry holds per-tensor integrity data.
type TensorEntry struct {
	Name         string   `json:"name"`
	Type         string   `json:"type"`
	Shape        []uint64 `json:"shape"`
	Offset       uint64   `json:"offset"`
	DataSize     int64    `json:"data_size"`
	Hash         string   `json:"hash"` // SHA-256 of raw tensor data
	ElementCount uint64   `json:"element_count"`
}

// GenerateManifest creates an integrity manifest for a GGUF file.
// This reads each tensor's raw data and computes its SHA-256 hash.
func GenerateManifest(gf *gguf.File, fp *Fingerprint) (*Manifest, error) {
	if gf == nil || fp == nil {
		return nil, fmt.Errorf("GGUF file and fingerprint are required")
	}
	current, err := GenerateFingerprint(gf)
	if err != nil {
		return nil, fmt.Errorf("fingerprint GGUF file: %w", err)
	}
	if fp.FileHash != current.FileHash || fp.StructureHash != current.StructureHash ||
		fp.Architecture != current.Architecture || fp.QuantType != current.QuantType ||
		fp.TensorCount != current.TensorCount || fp.ParameterCount != current.ParameterCount ||
		fp.FileSize != current.FileSize {
		return nil, fmt.Errorf("fingerprint does not match parsed GGUF file")
	}
	m := &Manifest{
		Version:       "1.0",
		FileHash:      fp.FileHash,
		StructureHash: fp.StructureHash,
		Architecture:  fp.Architecture,
		QuantType:     fp.QuantType,
		TensorCount:   len(gf.Tensors),
		Parameters:    gf.TotalParameters(),
		FileSize:      gf.FileSize,
		Tensors:       make([]TensorEntry, 0, len(gf.Tensors)),
	}

	hashes := make([][]byte, 0, len(gf.Tensors))

	for i := range gf.Tensors {
		ti := &gf.Tensors[i]
		hashStr, err := gguf.HashTensorData(gf, ti)
		if err != nil {
			return nil, fmt.Errorf("read tensor %q: %w", ti.Name, err)
		}
		hashBytes, _ := hex.DecodeString(hashStr)
		hashes = append(hashes, hashBytes)

		m.Tensors = append(m.Tensors, TensorEntry{
			Name:         ti.Name,
			Type:         ti.Type.String(),
			Shape:        ti.Dims,
			Offset:       ti.Offset,
			DataSize:     ti.ByteSize(),
			Hash:         hashStr,
			ElementCount: ti.ElementCount,
		})
	}

	m.MerkleRoot = computeMerkleRoot(hashes)

	return m, nil
}

// VerifyManifest checks a GGUF file against a previously generated manifest.
// Returns a list of mismatched tensors (empty = all verified).
func VerifyManifest(gf *gguf.File, m *Manifest) ([]string, error) {
	if gf == nil {
		return nil, fmt.Errorf("GGUF file is required")
	}
	if err := validateManifest(m); err != nil {
		return nil, err
	}

	fp, err := GenerateFingerprint(gf)
	if err != nil {
		return nil, fmt.Errorf("fingerprint current file: %w", err)
	}

	var mismatches []string
	if m.FileHash != fp.FileHash {
		mismatches = append(mismatches, "file hash mismatch")
	}
	if m.StructureHash != fp.StructureHash {
		mismatches = append(mismatches, "structure hash mismatch")
	}
	if m.Architecture != fp.Architecture {
		mismatches = append(mismatches, "architecture mismatch")
	}
	if m.QuantType != fp.QuantType {
		mismatches = append(mismatches, "quantization type mismatch")
	}
	if m.TensorCount != len(gf.Tensors) || len(m.Tensors) != len(gf.Tensors) {
		mismatches = append(mismatches, "tensor count mismatch")
	}
	if m.Parameters != gf.TotalParameters() {
		mismatches = append(mismatches, "parameter count mismatch")
	}
	if m.FileSize != gf.FileSize {
		mismatches = append(mismatches, "file size mismatch")
	}

	entryMap := make(map[string]TensorEntry, len(m.Tensors))
	for _, te := range m.Tensors {
		entryMap[te.Name] = te
	}

	currentHashes := make([][]byte, 0, len(gf.Tensors))
	matchedEntries := make(map[string]struct{}, len(gf.Tensors))
	for i := range gf.Tensors {
		ti := &gf.Tensors[i]
		expected, ok := entryMap[ti.Name]
		if !ok {
			mismatches = append(mismatches, fmt.Sprintf("%s: not in manifest", ti.Name))
			continue
		}
		matchedEntries[ti.Name] = struct{}{}
		if expected.Type != ti.Type.String() || expected.Offset != ti.Offset || expected.DataSize != ti.ByteSize() || expected.ElementCount != ti.ElementCount || !equalShape(expected.Shape, ti.Dims) {
			mismatches = append(mismatches, fmt.Sprintf("%s: tensor metadata mismatch", ti.Name))
		}

		actual, err := gguf.HashTensorData(gf, ti)
		if err != nil {
			mismatches = append(mismatches, fmt.Sprintf("%s: read error: %v", ti.Name, err))
			continue
		}
		hashBytes, _ := hex.DecodeString(actual)
		currentHashes = append(currentHashes, hashBytes)
		if actual != expected.Hash {
			mismatches = append(mismatches, fmt.Sprintf("%s: hash mismatch", ti.Name))
		}
	}
	for _, entry := range m.Tensors {
		if _, matched := matchedEntries[entry.Name]; !matched {
			mismatches = append(mismatches, fmt.Sprintf("%s: manifest tensor not present in file", entry.Name))
		}
	}
	if len(currentHashes) == len(gf.Tensors) && computeMerkleRoot(currentHashes) != m.MerkleRoot {
		mismatches = append(mismatches, "Merkle root mismatch")
	}

	return mismatches, nil
}

// SaveManifest writes a manifest to a JSON file.
func SaveManifest(path string, m *Manifest) error {
	if err := validateManifest(m); err != nil {
		return err
	}
	return writeJSONAtomic(path, m, 0644)
}

// LoadManifest reads a manifest from a JSON file.
func LoadManifest(path string) (*Manifest, error) {
	var m Manifest
	if err := decodeJSONFile(path, maxManifestBytes, &m); err != nil {
		return nil, err
	}
	if err := validateManifest(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

func validateManifest(m *Manifest) error {
	if m == nil {
		return fmt.Errorf("manifest is required")
	}
	if m.Version != "1.0" {
		return fmt.Errorf("unsupported manifest version %q", m.Version)
	}
	if !validSHA256(m.FileHash) || !validSHA256(m.StructureHash) || !validSHA256(m.MerkleRoot) {
		return fmt.Errorf("manifest contains an invalid SHA-256 digest")
	}
	if m.Architecture == "" || len(m.Architecture) > 256 || m.QuantType == "" || len(m.QuantType) > 64 ||
		m.FileSize <= 0 || m.TensorCount <= 0 || m.TensorCount > 100_000 ||
		m.TensorCount != len(m.Tensors) || m.Parameters == 0 {
		return fmt.Errorf("manifest summary fields are inconsistent")
	}
	seen := make(map[string]struct{}, len(m.Tensors))
	hashes := make([][]byte, 0, len(m.Tensors))
	type manifestRange struct{ start, end uint64 }
	ranges := make([]manifestRange, 0, len(m.Tensors))
	var parameters uint64
	for i, entry := range m.Tensors {
		if entry.Name == "" || len(entry.Name) > 64*1024 || entry.Type == "" || len(entry.Type) > 64 ||
			entry.DataSize <= 0 || entry.ElementCount == 0 || len(entry.Shape) == 0 || len(entry.Shape) > 8 || !validSHA256(entry.Hash) {
			return fmt.Errorf("manifest tensor %d is invalid", i)
		}
		if entry.Offset > uint64(m.FileSize) || uint64(entry.DataSize) > uint64(m.FileSize)-entry.Offset {
			return fmt.Errorf("manifest tensor %q has an out-of-bounds range", entry.Name)
		}
		if _, exists := seen[entry.Name]; exists {
			return fmt.Errorf("duplicate manifest tensor %q", entry.Name)
		}
		seen[entry.Name] = struct{}{}
		product := uint64(1)
		for _, dimension := range entry.Shape {
			if dimension == 0 || product > ^uint64(0)/dimension {
				return fmt.Errorf("manifest tensor %q has invalid shape", entry.Name)
			}
			product *= dimension
		}
		if product != entry.ElementCount {
			return fmt.Errorf("manifest tensor %q element count does not match shape", entry.Name)
		}
		if ^uint64(0)-parameters < entry.ElementCount {
			return fmt.Errorf("manifest parameter count overflows")
		}
		parameters += entry.ElementCount
		ranges = append(ranges, manifestRange{start: entry.Offset, end: entry.Offset + uint64(entry.DataSize)})
		decoded, _ := hex.DecodeString(entry.Hash)
		hashes = append(hashes, decoded)
	}
	if parameters != m.Parameters {
		return fmt.Errorf("manifest parameter summary does not match tensor shapes")
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].start < ranges[j].start })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].start < ranges[i-1].end {
			return fmt.Errorf("manifest contains overlapping tensor ranges")
		}
	}
	if computeMerkleRoot(hashes) != m.MerkleRoot {
		return fmt.Errorf("manifest Merkle root does not match tensor hashes")
	}
	return nil
}

func validSHA256(value string) bool {
	if len(value) != sha256.Size*2 {
		return false
	}
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size
}

func equalShape(left, right []uint64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

// computeMerkleRoot builds a Merkle tree from leaf hashes and returns the root.
func computeMerkleRoot(hashes [][]byte) string {
	if len(hashes) == 0 {
		return ""
	}
	if len(hashes) == 1 {
		return hex.EncodeToString(hashes[0])
	}

	level := make([][]byte, len(hashes))
	copy(level, hashes)

	for len(level) > 1 {
		var next [][]byte
		for i := 0; i < len(level); i += 2 {
			if i+1 < len(level) {
				combined := append(level[i], level[i+1]...)
				h := sha256.Sum256(combined)
				next = append(next, h[:])
			} else {
				// Odd node: hash with itself
				combined := append(level[i], level[i]...)
				h := sha256.Sum256(combined)
				next = append(next, h[:])
			}
		}
		level = next
	}

	return hex.EncodeToString(level[0])
}
