package analysis

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

// Fingerprint is a structural identity of a GGUF model.
// Two models with the same structure hash have identical tensor layouts
// (names, shapes, types) regardless of weight values.
type Fingerprint struct {
	FileHash       string        `json:"file_hash"`
	StructureHash  string        `json:"structure_hash"`
	Architecture   string        `json:"architecture"`
	QuantType      string        `json:"quant_type"`
	TensorCount    int           `json:"tensor_count"`
	ParameterCount uint64        `json:"parameter_count"`
	FileSize       int64         `json:"file_size"`
	Version        uint32        `json:"version"`
	LayerCount     int           `json:"layer_count"`
	TensorSummary  []TensorBrief `json:"tensor_summary"`
}

// TensorBrief is a compact tensor descriptor for fingerprinting.
type TensorBrief struct {
	Name  string   `json:"name"`
	Shape []uint64 `json:"shape"`
	Type  string   `json:"type"`
}

// GenerateFingerprint creates a structural fingerprint of a GGUF file.
func GenerateFingerprint(gf *gguf.File) (*Fingerprint, error) {
	fp := &Fingerprint{
		Architecture:   gf.Architecture(),
		QuantType:      gf.QuantType(),
		TensorCount:    len(gf.Tensors),
		ParameterCount: gf.TotalParameters(),
		FileSize:       gf.FileSize,
		Version:        gf.Version,
	}

	// File hash (full SHA256)
	fileHash, err := hashFile(gf.Path)
	if err != nil {
		return nil, fmt.Errorf("file hash: %w", err)
	}
	fp.FileHash = fileHash

	// Structure hash: deterministic hash of (sorted tensor names + shapes + types)
	fp.StructureHash = computeStructureHash(gf.Tensors)

	// Tensor summary and layer count
	layerSet := make(map[string]bool)
	fp.TensorSummary = make([]TensorBrief, len(gf.Tensors))
	for i, t := range gf.Tensors {
		fp.TensorSummary[i] = TensorBrief{
			Name:  t.Name,
			Shape: t.Dims,
			Type:  t.Type.String(),
		}
		// Extract layer identifier (e.g., "blk.0" from "blk.0.attn_q.weight")
		parts := strings.Split(t.Name, ".")
		if len(parts) >= 2 && (parts[0] == "blk" || parts[0] == "layers") {
			layerSet[parts[0]+"."+parts[1]] = true
		}
	}
	fp.LayerCount = len(layerSet)

	return fp, nil
}

func computeStructureHash(tensors []gguf.TensorInfo) string {
	// Build sorted canonical representation
	entries := make([]string, len(tensors))
	for i, t := range tensors {
		dims := make([]string, len(t.Dims))
		for d, v := range t.Dims {
			dims[d] = fmt.Sprintf("%d", v)
		}
		entries[i] = fmt.Sprintf("%s:[%s]:%s", t.Name, strings.Join(dims, ","), t.Type)
	}
	sort.Strings(entries)

	h := sha256.New()
	for _, e := range entries {
		h.Write([]byte(e))
		h.Write([]byte{'\n'})
	}
	return hex.EncodeToString(h.Sum(nil))
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
