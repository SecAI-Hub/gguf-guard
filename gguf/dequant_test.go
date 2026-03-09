package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestDequantF32(t *testing.T) {
	// 4 float32 values: 1.0, -2.0, 3.5, 0.0
	data := make([]byte, 16)
	binary.LittleEndian.PutUint32(data[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(data[4:], math.Float32bits(-2.0))
	binary.LittleEndian.PutUint32(data[8:], math.Float32bits(3.5))
	binary.LittleEndian.PutUint32(data[12:], math.Float32bits(0.0))

	vals, err := Dequantize(data, TypeF32, 0)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	if len(vals) != 4 {
		t.Fatalf("len = %d, want 4", len(vals))
	}
	expected := []float32{1.0, -2.0, 3.5, 0.0}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("vals[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestDequantF32Sampling(t *testing.T) {
	// 100 float32 values, sample down to 10
	data := make([]byte, 400)
	for i := 0; i < 100; i++ {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)))
	}

	vals, err := Dequantize(data, TypeF32, 10)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	if len(vals) != 10 {
		t.Fatalf("len = %d, want 10", len(vals))
	}
	// First should be 0.0, last should be near 99
	if vals[0] != 0.0 {
		t.Errorf("first = %v, want 0", vals[0])
	}
}

func TestDequantF16(t *testing.T) {
	// 3 F16 values: 1.0 (0x3C00), -1.0 (0xBC00), 0.5 (0x3800)
	data := make([]byte, 6)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00)
	binary.LittleEndian.PutUint16(data[2:], 0xBC00)
	binary.LittleEndian.PutUint16(data[4:], 0x3800)

	vals, err := Dequantize(data, TypeF16, 0)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	expected := []float32{1.0, -1.0, 0.5}
	for i, v := range vals {
		if v != expected[i] {
			t.Errorf("vals[%d] = %v, want %v", i, v, expected[i])
		}
	}
}

func TestDequantBF16(t *testing.T) {
	// BF16: 1.0 = 0x3F80, 2.0 = 0x4000
	data := make([]byte, 4)
	binary.LittleEndian.PutUint16(data[0:], 0x3F80)
	binary.LittleEndian.PutUint16(data[2:], 0x4000)

	vals, err := Dequantize(data, TypeBF16, 0)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	if len(vals) != 2 {
		t.Fatalf("len = %d, want 2", len(vals))
	}
	if vals[0] != 1.0 || vals[1] != 2.0 {
		t.Errorf("vals = %v, want [1.0, 2.0]", vals)
	}
}

func TestDequantQ8_0(t *testing.T) {
	// Q8_0: block = 34 bytes (2 byte scale + 32 int8 weights)
	// scale=1.0, all weights = 1 -> all values = 1.0
	data := make([]byte, 34)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // F16 1.0
	for i := 2; i < 34; i++ {
		data[i] = 1 // int8(1)
	}

	vals, err := Dequantize(data, TypeQ8_0, 0)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	if len(vals) != 32 {
		t.Fatalf("len = %d, want 32", len(vals))
	}
	for i, v := range vals {
		if math.Abs(float64(v)-1.0) > 1e-3 {
			t.Errorf("vals[%d] = %v, want ~1.0", i, v)
		}
	}
}

func TestDequantQ4_0(t *testing.T) {
	// Q4_0: block = 18 bytes (2 byte scale + 16 packed nibbles = 32 values)
	// Verify we get 32 values per block
	data := make([]byte, 18)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale = 1.0
	// nibbles all zero -> all values = (0-8) * scale = -8.0
	vals, err := Dequantize(data, TypeQ4_0, 0)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	if len(vals) != 32 {
		t.Fatalf("len = %d, want 32", len(vals))
	}
}

func TestDequantQ4_1(t *testing.T) {
	// Q4_1: block = 20 bytes
	data := make([]byte, 20)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // scale (F16 1.0)
	binary.LittleEndian.PutUint16(data[2:], 0x0000) // min (F16 0.0)

	vals, err := Dequantize(data, TypeQ4_1, 0)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	if len(vals) != 32 {
		t.Fatalf("len = %d, want 32", len(vals))
	}
}

func TestDequantUnsupported(t *testing.T) {
	_, err := Dequantize([]byte{0, 0}, TypeQ8_K, 0)
	if err == nil {
		t.Fatal("expected error for unsupported type")
	}
}

func TestDequantEmptyData(t *testing.T) {
	vals, err := Dequantize([]byte{}, TypeF32, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vals) != 0 {
		t.Errorf("expected empty, got %d values", len(vals))
	}
}

func TestTypeStringAndInfo(t *testing.T) {
	if TypeF32.String() != "F32" {
		t.Errorf("F32.String() = %q", TypeF32.String())
	}
	if TypeQ4_K.BlockSize() != 256 {
		t.Errorf("Q4_K.BlockSize() = %d", TypeQ4_K.BlockSize())
	}
	if TypeQ4_K.TypeSize() != 144 {
		t.Errorf("Q4_K.TypeSize() = %d", TypeQ4_K.TypeSize())
	}
	if !TypeQ4_K.Supported() {
		t.Error("Q4_K should be supported")
	}
	if TypeQ8_K.Supported() {
		t.Error("Q8_K should not be supported")
	}
	if GGMLType(255).String() != "unknown(255)" {
		t.Errorf("unknown type string = %q", GGMLType(255).String())
	}
}
