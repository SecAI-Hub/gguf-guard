package analysis

import (
	"testing"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

func TestCheckStructuralPolicyClean(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "blk.0.attn_q.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.attn_k.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
	}
	path := createTestGGUF(t, dir, "clean.gguf", tensors, map[string]any{
		"general.architecture": "llama",
	})
	gf, _ := gguf.Parse(path)

	report := CheckStructuralPolicy(gf)
	if !report.Pass {
		t.Errorf("clean model should pass, violations: %v", report.Violations)
	}
}

func TestCheckStructuralPolicyMissingArch(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "weight", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	path := createTestGGUF(t, dir, "noarch.gguf", tensors, map[string]any{})
	gf, _ := gguf.Parse(path)

	report := CheckStructuralPolicy(gf)
	found := false
	for _, v := range report.Violations {
		if v.Type == "missing_architecture" {
			found = true
		}
	}
	if !found {
		t.Error("expected missing_architecture violation")
	}
}

func TestCheckStructuralPolicyZeroElements(t *testing.T) {
	dir := t.TempDir()
	// Can't easily create zero-element tensor with the test helper,
	// but we test the check function directly
	gf := &gguf.File{
		Version: 3,
		Tensors: []gguf.TensorInfo{
			{Name: "empty", NDims: 1, Dims: []uint64{0}, Type: gguf.TypeF32, ElementCount: 0},
		},
		Metadata: map[string]any{"general.architecture": "test"},
		FileSize: 1000,
	}

	report := CheckStructuralPolicy(gf)
	found := false
	for _, v := range report.Violations {
		if v.Type == "zero_element_tensor" {
			found = true
		}
	}
	if !found {
		t.Error("expected zero_element_tensor violation")
	}
	_ = dir
}

func TestCheckStructuralPolicyOverlap(t *testing.T) {
	// Create a file with overlapping tensors
	gf := &gguf.File{
		Version:    3,
		DataOffset: 100,
		FileSize:   10000,
		Tensors: []gguf.TensorInfo{
			{Name: "a", NDims: 1, Dims: []uint64{100}, Type: gguf.TypeF32, ElementCount: 100, Offset: 0},
			{Name: "b", NDims: 1, Dims: []uint64{100}, Type: gguf.TypeF32, ElementCount: 100, Offset: 200}, // overlaps: a is 400 bytes, b starts at 200
		},
		Metadata: map[string]any{"general.architecture": "test"},
	}

	report := CheckStructuralPolicy(gf)
	found := false
	for _, v := range report.Violations {
		if v.Type == "tensor_overlap" {
			found = true
		}
	}
	if !found {
		t.Error("expected tensor_overlap violation")
	}
}
