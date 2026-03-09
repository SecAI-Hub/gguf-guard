package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeTestGGUF creates a minimal valid GGUF v3 file with the given tensors and metadata.
func writeTestGGUF(t *testing.T, dir string, tensors []TensorInfo, metadata map[string]any) string {
	t.Helper()
	path := filepath.Join(dir, "test.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// Magic
	f.Write(Magic[:])
	// Version 3
	binary.Write(f, binary.LittleEndian, uint32(3))
	// Tensor count
	binary.Write(f, binary.LittleEndian, uint64(len(tensors)))
	// Metadata KV count
	binary.Write(f, binary.LittleEndian, uint64(len(metadata)))

	// Write metadata
	for key, val := range metadata {
		writeGGUFString(f, key)
		switch v := val.(type) {
		case string:
			binary.Write(f, binary.LittleEndian, uint32(MetaString))
			writeGGUFString(f, v)
		case uint32:
			binary.Write(f, binary.LittleEndian, uint32(MetaUint32))
			binary.Write(f, binary.LittleEndian, v)
		case float32:
			binary.Write(f, binary.LittleEndian, uint32(MetaFloat32))
			binary.Write(f, binary.LittleEndian, v)
		case bool:
			binary.Write(f, binary.LittleEndian, uint32(MetaBool))
			if v {
				binary.Write(f, binary.LittleEndian, uint8(1))
			} else {
				binary.Write(f, binary.LittleEndian, uint8(0))
			}
		}
	}

	// Calculate tensor data sizes and offsets
	var dataOffset uint64
	offsets := make([]uint64, len(tensors))
	for i, ti := range tensors {
		offsets[i] = dataOffset
		dataOffset += uint64(ti.ByteSize())
	}

	// Write tensor info entries
	for i, ti := range tensors {
		writeGGUFString(f, ti.Name)
		binary.Write(f, binary.LittleEndian, ti.NDims)
		for _, d := range ti.Dims {
			binary.Write(f, binary.LittleEndian, d)
		}
		binary.Write(f, binary.LittleEndian, uint32(ti.Type))
		binary.Write(f, binary.LittleEndian, offsets[i])
	}

	// Align to 32 bytes for data section
	pos, _ := f.Seek(0, 1)
	padding := (32 - (pos % 32)) % 32
	if padding > 0 {
		f.Write(make([]byte, padding))
	}

	// Write tensor data (zeros for simplicity — individual tests write specific values)
	for _, ti := range tensors {
		f.Write(make([]byte, ti.ByteSize()))
	}

	return path
}

func writeGGUFString(f *os.File, s string) {
	binary.Write(f, binary.LittleEndian, uint64(len(s)))
	f.Write([]byte(s))
}

func TestParseMagic(t *testing.T) {
	dir := t.TempDir()

	// Invalid magic
	bad := filepath.Join(dir, "bad.gguf")
	os.WriteFile(bad, []byte("NOPE"), 0644)
	_, err := Parse(bad)
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestParseVersion(t *testing.T) {
	dir := t.TempDir()

	// Unsupported version (1)
	path := filepath.Join(dir, "v1.gguf")
	f, _ := os.Create(path)
	f.Write(Magic[:])
	binary.Write(f, binary.LittleEndian, uint32(1))
	f.Close()

	_, err := Parse(path)
	if err == nil {
		t.Fatal("expected error for unsupported version")
	}
}

func TestParseMinimal(t *testing.T) {
	dir := t.TempDir()
	ti := TensorInfo{
		Name:         "weight",
		NDims:        2,
		Dims:         []uint64{4, 4},
		Type:         TypeF32,
		ElementCount: 16,
	}
	metadata := map[string]any{
		"general.architecture": "test",
	}
	path := writeTestGGUF(t, dir, []TensorInfo{ti}, metadata)

	gf, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	if gf.Version != 3 {
		t.Errorf("version = %d, want 3", gf.Version)
	}
	if gf.TensorCount != 1 {
		t.Errorf("tensor count = %d, want 1", gf.TensorCount)
	}
	if gf.Architecture() != "test" {
		t.Errorf("architecture = %q, want %q", gf.Architecture(), "test")
	}
	if len(gf.Tensors) != 1 {
		t.Fatalf("len(tensors) = %d, want 1", len(gf.Tensors))
	}
	if gf.Tensors[0].Name != "weight" {
		t.Errorf("tensor name = %q, want %q", gf.Tensors[0].Name, "weight")
	}
	if gf.Tensors[0].ElementCount != 16 {
		t.Errorf("element count = %d, want 16", gf.Tensors[0].ElementCount)
	}
}

func TestParseMultipleTensors(t *testing.T) {
	dir := t.TempDir()
	tensors := []TensorInfo{
		{Name: "blk.0.attn_q.weight", NDims: 2, Dims: []uint64{8, 8}, Type: TypeF32, ElementCount: 64},
		{Name: "blk.0.attn_k.weight", NDims: 2, Dims: []uint64{8, 8}, Type: TypeF16, ElementCount: 64},
		{Name: "blk.1.attn_q.weight", NDims: 2, Dims: []uint64{8, 8}, Type: TypeF32, ElementCount: 64},
	}
	path := writeTestGGUF(t, dir, tensors, map[string]any{"general.architecture": "llama"})

	gf, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(gf.Tensors) != 3 {
		t.Errorf("tensor count = %d, want 3", len(gf.Tensors))
	}
	if gf.TotalParameters() != 192 {
		t.Errorf("total params = %d, want 192", gf.TotalParameters())
	}
}

func TestReadTensorData(t *testing.T) {
	dir := t.TempDir()
	ti := TensorInfo{
		Name:         "test",
		NDims:        1,
		Dims:         []uint64{4},
		Type:         TypeF32,
		ElementCount: 4,
	}
	path := writeTestGGUF(t, dir, []TensorInfo{ti}, map[string]any{})
	gf, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	data, err := ReadTensorData(gf, &gf.Tensors[0], 0)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(data) != 16 { // 4 floats * 4 bytes
		t.Errorf("data len = %d, want 16", len(data))
	}
}

func TestF16ToF32(t *testing.T) {
	tests := []struct {
		name string
		bits uint16
		want float32
	}{
		{"zero", 0x0000, 0.0},
		{"neg_zero", 0x8000, float32(math.Copysign(0, -1))},
		{"one", 0x3C00, 1.0},
		{"neg_one", 0xBC00, -1.0},
		{"half", 0x3800, 0.5},
		{"inf", 0x7C00, float32(math.Inf(1))},
		{"neg_inf", 0xFC00, float32(math.Inf(-1))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := F16ToF32(tt.bits)
			if math.IsInf(float64(tt.want), 0) {
				if !math.IsInf(float64(got), 0) || (tt.want > 0) != (got > 0) {
					t.Errorf("F16ToF32(%#x) = %v, want %v", tt.bits, got, tt.want)
				}
				return
			}
			if got != tt.want {
				t.Errorf("F16ToF32(%#x) = %v, want %v", tt.bits, got, tt.want)
			}
		})
	}
}

func TestF16NaN(t *testing.T) {
	got := F16ToF32(0x7C01) // NaN
	if !math.IsNaN(float64(got)) {
		t.Errorf("F16ToF32(0x7C01) = %v, want NaN", got)
	}
}

func TestBF16ToF32(t *testing.T) {
	tests := []struct {
		bits uint16
		want float32
	}{
		{0x3F80, 1.0},
		{0xBF80, -1.0},
		{0x0000, 0.0},
		{0x4000, 2.0},
	}
	for _, tt := range tests {
		got := BF16ToF32(tt.bits)
		if got != tt.want {
			t.Errorf("BF16ToF32(%#x) = %v, want %v", tt.bits, got, tt.want)
		}
	}
}

func TestQuantType(t *testing.T) {
	dir := t.TempDir()
	tensors := []TensorInfo{
		{Name: "a", NDims: 1, Dims: []uint64{32}, Type: TypeQ4_K, ElementCount: 32},
		{Name: "b", NDims: 1, Dims: []uint64{32}, Type: TypeQ4_K, ElementCount: 32},
		{Name: "c", NDims: 1, Dims: []uint64{4}, Type: TypeF32, ElementCount: 4},
	}
	path := writeTestGGUF(t, dir, tensors, map[string]any{})
	gf, err := Parse(path)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if gf.QuantType() != "Q4_K" {
		t.Errorf("QuantType() = %q, want Q4_K", gf.QuantType())
	}
}
