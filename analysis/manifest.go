package analysis

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

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
		data, err := gguf.ReadTensorData(gf, ti, 0)
		if err != nil {
			return nil, fmt.Errorf("read tensor %q: %w", ti.Name, err)
		}

		h := sha256.Sum256(data)
		hashStr := hex.EncodeToString(h[:])
		hashes = append(hashes, h[:])

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
	hashMap := make(map[string]string)
	for _, te := range m.Tensors {
		hashMap[te.Name] = te.Hash
	}

	var mismatches []string
	for i := range gf.Tensors {
		ti := &gf.Tensors[i]
		expected, ok := hashMap[ti.Name]
		if !ok {
			mismatches = append(mismatches, fmt.Sprintf("%s: not in manifest", ti.Name))
			continue
		}

		data, err := gguf.ReadTensorData(gf, ti, 0)
		if err != nil {
			mismatches = append(mismatches, fmt.Sprintf("%s: read error: %v", ti.Name, err))
			continue
		}

		h := sha256.Sum256(data)
		actual := hex.EncodeToString(h[:])
		if actual != expected {
			mismatches = append(mismatches, fmt.Sprintf("%s: hash mismatch (expected %s, got %s)", ti.Name, expected[:16], actual[:16]))
		}
	}

	return mismatches, nil
}

// SaveManifest writes a manifest to a JSON file.
func SaveManifest(path string, m *Manifest) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// LoadManifest reads a manifest from a JSON file.
func LoadManifest(path string) (*Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
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
