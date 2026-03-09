package analysis

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

func TestGenerateAndVerifyManifest(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
		{Name: "b", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	path := createTestGGUF(t, dir, "model.gguf", tensors, map[string]any{})

	gf, err := gguf.Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	fp, err := GenerateFingerprint(gf)
	if err != nil {
		t.Fatalf("fp: %v", err)
	}

	m, err := GenerateManifest(gf, fp)
	if err != nil {
		t.Fatalf("manifest: %v", err)
	}

	if m.TensorCount != 2 {
		t.Errorf("tensor count = %d, want 2", m.TensorCount)
	}
	if m.MerkleRoot == "" {
		t.Error("merkle root is empty")
	}
	if m.FileHash == "" {
		t.Error("file hash is empty")
	}
	if len(m.Tensors) != 2 {
		t.Fatalf("entries = %d, want 2", len(m.Tensors))
	}
	if m.Tensors[0].Hash == "" {
		t.Error("tensor hash is empty")
	}

	// Verify against same file — should pass
	mismatches, err := VerifyManifest(gf, m)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if len(mismatches) != 0 {
		t.Errorf("unexpected mismatches: %v", mismatches)
	}
}

func TestSaveLoadManifest(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")

	m := &Manifest{
		Version:       "1.0",
		FileHash:      "abc123",
		StructureHash: "xyz",
		TensorCount:   1,
		MerkleRoot:    "root",
		Tensors: []TensorEntry{
			{Name: "weight", Hash: "deadbeef", DataSize: 64},
		},
	}

	if err := SaveManifest(path, m); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := LoadManifest(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if loaded.MerkleRoot != m.MerkleRoot {
		t.Errorf("merkle = %q, want %q", loaded.MerkleRoot, m.MerkleRoot)
	}
	if len(loaded.Tensors) != 1 {
		t.Fatalf("tensors = %d", len(loaded.Tensors))
	}
	if loaded.Tensors[0].Name != "weight" {
		t.Errorf("name = %q", loaded.Tensors[0].Name)
	}
}

func TestLoadManifestNotFound(t *testing.T) {
	_, err := LoadManifest("/nonexistent")
	if err == nil {
		t.Error("expected error")
	}
}

func TestMerkleRootSingle(t *testing.T) {
	hash := []byte("0123456789abcdef0123456789abcdef")
	root := computeMerkleRoot([][]byte{hash})
	if root == "" {
		t.Error("empty root for single hash")
	}
}

func TestMerkleRootEmpty(t *testing.T) {
	root := computeMerkleRoot(nil)
	if root != "" {
		t.Errorf("expected empty root, got %q", root)
	}
}

func TestMerkleRootDeterministic(t *testing.T) {
	h1 := []byte("aaaa")
	h2 := []byte("bbbb")
	r1 := computeMerkleRoot([][]byte{h1, h2})
	r2 := computeMerkleRoot([][]byte{h1, h2})
	if r1 != r2 {
		t.Error("merkle root not deterministic")
	}
}

func TestVerifyManifestMismatch(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	path := createTestGGUF(t, dir, "model.gguf", tensors, map[string]any{})

	gf, _ := gguf.Parse(path)

	// Manifest with wrong hash
	m := &Manifest{
		Tensors: []TensorEntry{
			{Name: "a", Hash: "0000000000000000000000000000000000000000000000000000000000000000"},
		},
	}

	mismatches, err := VerifyManifest(gf, m)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if len(mismatches) == 0 {
		t.Error("expected mismatch")
	}
}

func TestVerifyManifestMissingTensor(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	path := createTestGGUF(t, dir, "model.gguf", tensors, map[string]any{})

	gf, _ := gguf.Parse(path)

	// Empty manifest
	m := &Manifest{Tensors: []TensorEntry{}}

	mismatches, err := VerifyManifest(gf, m)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if len(mismatches) != 1 {
		t.Errorf("mismatches = %d, want 1", len(mismatches))
	}
}

// createTestGGUF for manifest tests - reuse from fingerprint_test.go
func init() {
	// Ensure createTestGGUF and writeStr are available (defined in fingerprint_test.go)
	_ = os.TempDir()
}
