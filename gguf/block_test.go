package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestExtractBlockStatsQ8_0(t *testing.T) {
	// 2 blocks of Q8_0 (34 bytes each)
	data := make([]byte, 68)

	// Block 0: scale=1.0, codes all 1
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // F16 1.0
	for i := 2; i < 34; i++ {
		data[i] = 1
	}

	// Block 1: scale=2.0, codes all 2
	binary.LittleEndian.PutUint16(data[34:], 0x4000) // F16 2.0
	for i := 36; i < 68; i++ {
		data[i] = 2
	}

	bs := ExtractBlockStats(data, "test", TypeQ8_0)
	if bs == nil {
		t.Fatal("expected non-nil block stats")
	}
	if bs.NumBlocks != 2 {
		t.Errorf("num_blocks = %d, want 2", bs.NumBlocks)
	}
	if bs.ScaleMin != 1.0 {
		t.Errorf("scale_min = %v, want 1.0", bs.ScaleMin)
	}
	if bs.ScaleMax != 2.0 {
		t.Errorf("scale_max = %v, want 2.0", bs.ScaleMax)
	}
	if bs.RepeatedBlocks != 0 {
		t.Errorf("repeated = %d, want 0", bs.RepeatedBlocks)
	}
}

func TestExtractBlockStatsQ4_0(t *testing.T) {
	// 1 block of Q4_0 (18 bytes)
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale 1.0

	bs := ExtractBlockStats(data, "test", TypeQ4_0)
	if bs == nil {
		t.Fatal("nil")
	}
	if bs.QuantType != "Q4_0" {
		t.Errorf("quant = %q", bs.QuantType)
	}
	if bs.NumBlocks != 1 {
		t.Errorf("blocks = %d", bs.NumBlocks)
	}
}

func TestExtractBlockStatsReturnsNilForF32(t *testing.T) {
	bs := ExtractBlockStats([]byte{0, 0, 0, 0}, "test", TypeF32)
	if bs != nil {
		t.Error("expected nil for F32")
	}
}

func TestRepeatedBlocks(t *testing.T) {
	// 3 identical Q8_0 blocks
	block := make([]byte, 34)
	binary.LittleEndian.PutUint16(block[0:], 0x3C00)
	for i := 2; i < 34; i++ {
		block[i] = 5
	}
	data := make([]byte, 102)
	copy(data[0:34], block)
	copy(data[34:68], block)
	copy(data[68:102], block)

	bs := ExtractBlockStats(data, "repeat", TypeQ8_0)
	if bs.RepeatedBlocks != 2 {
		t.Errorf("repeated = %d, want 2", bs.RepeatedBlocks)
	}
}

func TestValidateBlockLayout(t *testing.T) {
	// Valid Q4_0: 32 elements, block size 32, type size 18
	ti := &TensorInfo{Name: "ok", Type: TypeQ4_0, ElementCount: 32, Dims: []uint64{32}, NDims: 1}
	if err := ValidateBlockLayout(ti, 18); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Truncated
	if err := ValidateBlockLayout(ti, 10); err == nil {
		t.Error("expected error for truncated data")
	}

	// Non-divisible element count
	ti2 := &TensorInfo{Name: "bad", Type: TypeQ4_0, ElementCount: 33, Dims: []uint64{33}, NDims: 1}
	if err := ValidateBlockLayout(ti2, 100); err == nil {
		t.Error("expected error for non-divisible element count")
	}
}

func TestShannonEntropy(t *testing.T) {
	// Uniform distribution: max entropy
	counts := make([]int, 4)
	for i := range counts {
		counts[i] = 25
	}
	e := shannonEntropy(counts, 100)
	if math.Abs(e-2.0) > 0.01 {
		t.Errorf("uniform entropy = %v, want 2.0", e)
	}

	// Single bin: zero entropy
	counts2 := []int{100, 0, 0, 0}
	e2 := shannonEntropy(counts2, 100)
	if e2 != 0 {
		t.Errorf("single-bin entropy = %v, want 0", e2)
	}
}

func TestZeroScaleCount(t *testing.T) {
	// 2 Q8_0 blocks, one with zero scale
	data := make([]byte, 68)
	binary.LittleEndian.PutUint16(data[0:], 0x0000)  // zero scale
	binary.LittleEndian.PutUint16(data[34:], 0x3C00) // normal scale

	bs := ExtractBlockStats(data, "test", TypeQ8_0)
	if bs.ZeroScales != 1 {
		t.Errorf("zero_scales = %d, want 1", bs.ZeroScales)
	}
}
