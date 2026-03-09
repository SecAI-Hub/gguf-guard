package analysis

import (
	"testing"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

func TestMatchFamilyLlama(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "token_embd.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "output.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "output_norm.weight", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
		{Name: "blk.0.attn_q.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.attn_k.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.attn_v.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.attn_output.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.attn_norm.weight", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
		{Name: "blk.0.ffn_gate.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.ffn_up.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.ffn_down.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.ffn_norm.weight", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	path := createTestGGUF(t, dir, "llama.gguf", tensors, map[string]any{
		"general.architecture": "llama",
	})
	gf, _ := gguf.Parse(path)

	best := BestFamilyMatch(gf)
	if best == nil {
		t.Fatal("expected a family match")
	}
	if best.Family != "llama" {
		t.Errorf("family = %q, want llama", best.Family)
	}
	if best.MatchScore < 0.8 {
		t.Errorf("score = %v, want > 0.8", best.MatchScore)
	}
}

func TestMatchFamilyUnknown(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "custom.weight", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	path := createTestGGUF(t, dir, "unknown.gguf", tensors, map[string]any{
		"general.architecture": "custom_arch",
	})
	gf, _ := gguf.Parse(path)

	best := BestFamilyMatch(gf)
	if best != nil {
		t.Errorf("expected no match for unknown arch, got %q (%.2f)", best.Family, best.MatchScore)
	}
}

func TestMatchFamilyAll(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "blk.0.attn_q.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
	}
	path := createTestGGUF(t, dir, "test.gguf", tensors, map[string]any{
		"general.architecture": "llama",
	})
	gf, _ := gguf.Parse(path)

	matches := MatchFamily(gf)
	if len(matches) == 0 {
		t.Fatal("expected family matches")
	}
	// Should include llama at minimum
	foundLlama := false
	for _, m := range matches {
		if m.Family == "llama" {
			foundLlama = true
		}
	}
	if !foundLlama {
		t.Error("expected llama in matches")
	}
}

func TestExtractTensorRole(t *testing.T) {
	tests := []struct {
		name, prefix, want string
	}{
		{"blk.0.attn_q.weight", "blk", "attn_q.weight"},
		{"blk.15.ffn_down.weight", "blk", "ffn_down.weight"},
		{"output.weight", "blk", ""},
		{"blk.0", "blk", ""},
	}
	for _, tt := range tests {
		got := extractTensorRole(tt.name, tt.prefix)
		if got != tt.want {
			t.Errorf("extractTensorRole(%q, %q) = %q, want %q", tt.name, tt.prefix, got, tt.want)
		}
	}
}
