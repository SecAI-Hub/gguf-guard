package analysis

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

// createTestGGUF writes a minimal GGUF file for fingerprint tests.
func createTestGGUF(t *testing.T, dir, name string, tensors []gguf.TensorInfo, meta map[string]any) string {
	t.Helper()
	path := filepath.Join(dir, name)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	f.Write(gguf.Magic[:])
	binary.Write(f, binary.LittleEndian, uint32(3))
	binary.Write(f, binary.LittleEndian, uint64(len(tensors)))
	binary.Write(f, binary.LittleEndian, uint64(len(meta)))

	for key, val := range meta {
		writeStr(f, key)
		switch v := val.(type) {
		case string:
			binary.Write(f, binary.LittleEndian, uint32(gguf.MetaString))
			writeStr(f, v)
		}
	}

	var dataOffset uint64
	for i, ti := range tensors {
		writeStr(f, ti.Name)
		binary.Write(f, binary.LittleEndian, ti.NDims)
		for _, d := range ti.Dims {
			binary.Write(f, binary.LittleEndian, d)
		}
		binary.Write(f, binary.LittleEndian, uint32(ti.Type))
		binary.Write(f, binary.LittleEndian, dataOffset)
		_ = i
		dataOffset += uint64(ti.ByteSize())
	}

	pos, _ := f.Seek(0, 1)
	padding := (32 - (pos % 32)) % 32
	if padding > 0 {
		f.Write(make([]byte, padding))
	}

	for _, ti := range tensors {
		f.Write(make([]byte, ti.ByteSize()))
	}

	return path
}

func writeStr(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	f.Write([]byte(s))
}

func TestGenerateFingerprint(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "blk.0.attn_q.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.0.attn_k.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
		{Name: "blk.1.attn_q.weight", NDims: 2, Dims: []uint64{4, 4}, Type: gguf.TypeF32, ElementCount: 16},
	}
	path := createTestGGUF(t, dir, "model.gguf", tensors, map[string]any{
		"general.architecture": "llama",
	})

	gf, err := gguf.Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	fp, err := GenerateFingerprint(gf)
	if err != nil {
		t.Fatalf("fingerprint: %v", err)
	}

	if fp.Architecture != "llama" {
		t.Errorf("arch = %q, want llama", fp.Architecture)
	}
	if fp.TensorCount != 3 {
		t.Errorf("tensor count = %d, want 3", fp.TensorCount)
	}
	if fp.LayerCount != 2 {
		t.Errorf("layer count = %d, want 2", fp.LayerCount)
	}
	if fp.FileHash == "" {
		t.Error("file hash is empty")
	}
	if fp.StructureHash == "" {
		t.Error("structure hash is empty")
	}
	if fp.ParameterCount != 48 {
		t.Errorf("parameters = %d, want 48", fp.ParameterCount)
	}
}

func TestStructureHashDeterministic(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
		{Name: "b", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}

	// Create the same file twice
	p1 := createTestGGUF(t, dir, "m1.gguf", tensors, map[string]any{})
	p2 := createTestGGUF(t, dir, "m2.gguf", tensors, map[string]any{})

	gf1, _ := gguf.Parse(p1)
	gf2, _ := gguf.Parse(p2)
	fp1, _ := GenerateFingerprint(gf1)
	fp2, _ := GenerateFingerprint(gf2)

	if fp1.StructureHash != fp2.StructureHash {
		t.Error("identical structures should have same structure hash")
	}
}

func TestStructureHashDiffers(t *testing.T) {
	dir := t.TempDir()
	t1 := []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	t2 := []gguf.TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{8}, Type: gguf.TypeF32, ElementCount: 8},
	}

	p1 := createTestGGUF(t, dir, "m1.gguf", t1, map[string]any{})
	p2 := createTestGGUF(t, dir, "m2.gguf", t2, map[string]any{})

	gf1, _ := gguf.Parse(p1)
	gf2, _ := gguf.Parse(p2)
	fp1, _ := GenerateFingerprint(gf1)
	fp2, _ := GenerateFingerprint(gf2)

	if fp1.StructureHash == fp2.StructureHash {
		t.Error("different structures should have different hashes")
	}
}
