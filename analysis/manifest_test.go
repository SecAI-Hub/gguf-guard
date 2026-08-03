package analysis

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
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

func TestGenerateManifestRejectsMismatchedFingerprint(t *testing.T) {
	dir := t.TempDir()
	path := createTestGGUF(t, dir, "model.gguf", []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}, map[string]any{})
	gf, err := gguf.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	fp, err := GenerateFingerprint(gf)
	if err != nil {
		t.Fatal(err)
	}
	fp.StructureHash = strings.Repeat("0", 64)
	if _, err := GenerateManifest(gf, fp); err == nil {
		t.Fatal("mismatched fingerprint must be rejected")
	}
}

func TestSaveLoadManifest(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")

	m := &Manifest{
		Version:       "1.0",
		FileHash:      strings.Repeat("a", 64),
		StructureHash: strings.Repeat("b", 64),
		Architecture:  "test",
		QuantType:     "F32",
		TensorCount:   1,
		Parameters:    16,
		FileSize:      128,
		Tensors: []TensorEntry{
			{Name: "weight", Type: "F32", Shape: []uint64{4, 4}, Hash: strings.Repeat("c", 64), DataSize: 64, ElementCount: 16},
		},
	}
	hashBytes, _ := hex.DecodeString(m.Tensors[0].Hash)
	m.MerkleRoot = computeMerkleRoot([][]byte{hashBytes})

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

	fp, _ := GenerateFingerprint(gf)
	m, _ := GenerateManifest(gf, fp)
	m.Tensors[0].Hash = strings.Repeat("0", 64)
	badHash, _ := hex.DecodeString(m.Tensors[0].Hash)
	m.MerkleRoot = computeMerkleRoot([][]byte{badHash})

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

	fp, _ := GenerateFingerprint(gf)
	m, _ := GenerateManifest(gf, fp)
	m.Tensors[0].Name = "different"

	mismatches, err := VerifyManifest(gf, m)
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if len(mismatches) != 2 {
		t.Errorf("mismatches = %d, want 2", len(mismatches))
	}
}

func TestVerifyManifestRejectsRemovedTensor(t *testing.T) {
	dir := t.TempDir()
	originalPath := createTestGGUF(t, dir, "original.gguf", []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
		{Name: "b", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}, map[string]any{})
	original, err := gguf.Parse(originalPath)
	if err != nil {
		t.Fatal(err)
	}
	fp, err := GenerateFingerprint(original)
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := GenerateManifest(original, fp)
	if err != nil {
		t.Fatal(err)
	}

	candidatePath := createTestGGUF(t, dir, "candidate.gguf", []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}, map[string]any{})
	candidate, err := gguf.Parse(candidatePath)
	if err != nil {
		t.Fatal(err)
	}
	mismatches, err := VerifyManifest(candidate, manifest)
	if err != nil {
		t.Fatal(err)
	}
	if len(mismatches) == 0 {
		t.Fatal("removed tensor must fail manifest verification")
	}
}

func TestLoadManifestRejectsUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, []byte(`{"manifest_version":"1.0","unexpected":true}`), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadManifest(path); err == nil {
		t.Fatal("unknown manifest field must be rejected")
	}
}

func TestSaveManifestRefusesSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.WriteFile(target, []byte("preserve"), 0644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "manifest.json")
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if err := SaveManifest(link, &Manifest{}); err == nil {
		t.Fatal("symlink output must be rejected")
	}
	data, _ := os.ReadFile(target)
	if string(data) != "preserve" {
		t.Fatal("symlink target was modified")
	}
}

// createTestGGUF for manifest tests - reuse from fingerprint_test.go
func init() {
	// Ensure createTestGGUF and writeStr are available (defined in fingerprint_test.go)
	_ = os.TempDir()
}
