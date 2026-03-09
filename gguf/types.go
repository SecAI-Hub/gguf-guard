// Package gguf implements parsing and dequantization of GGUF model files.
//
// GGUF (GGML Universal Format) is the standard format for quantized LLMs
// used by llama.cpp and compatible inference engines.
package gguf

import "fmt"

// GGMLType represents a GGML tensor data type.
type GGMLType uint32

const (
	TypeF32  GGMLType = 0
	TypeF16  GGMLType = 1
	TypeQ4_0 GGMLType = 2
	TypeQ4_1 GGMLType = 3
	TypeQ5_0 GGMLType = 6
	TypeQ5_1 GGMLType = 7
	TypeQ8_0 GGMLType = 8
	TypeQ8_1 GGMLType = 9
	TypeQ2_K GGMLType = 10
	TypeQ3_K GGMLType = 11
	TypeQ4_K GGMLType = 12
	TypeQ5_K GGMLType = 13
	TypeQ6_K GGMLType = 14
	TypeQ8_K GGMLType = 15
	TypeIQ2  GGMLType = 16
	TypeIQ3  GGMLType = 17
	TypeIQ1  GGMLType = 18
	TypeBF16 GGMLType = 30
)

// typeInfo holds block size and byte size for each quantization type.
type typeInfo struct {
	BlockSize int // number of elements per block
	TypeSize  int // bytes per block
}

var typeInfoMap = map[GGMLType]typeInfo{
	TypeF32:  {1, 4},
	TypeF16:  {1, 2},
	TypeBF16: {1, 2},
	TypeQ4_0: {32, 18},
	TypeQ4_1: {32, 20},
	TypeQ5_0: {32, 22},
	TypeQ5_1: {32, 24},
	TypeQ8_0: {32, 34},
	TypeQ8_1: {32, 36},
	TypeQ2_K: {256, 84},
	TypeQ3_K: {256, 110},
	TypeQ4_K: {256, 144},
	TypeQ5_K: {256, 176},
	TypeQ6_K: {256, 210},
	TypeQ8_K: {256, 292},
}

// String returns the human-readable name of a GGML type.
func (t GGMLType) String() string {
	names := map[GGMLType]string{
		TypeF32: "F32", TypeF16: "F16", TypeBF16: "BF16",
		TypeQ4_0: "Q4_0", TypeQ4_1: "Q4_1",
		TypeQ5_0: "Q5_0", TypeQ5_1: "Q5_1",
		TypeQ8_0: "Q8_0", TypeQ8_1: "Q8_1",
		TypeQ2_K: "Q2_K", TypeQ3_K: "Q3_K",
		TypeQ4_K: "Q4_K", TypeQ5_K: "Q5_K",
		TypeQ6_K: "Q6_K", TypeQ8_K: "Q8_K",
		TypeIQ2: "IQ2", TypeIQ3: "IQ3", TypeIQ1: "IQ1",
	}
	if name, ok := names[t]; ok {
		return name
	}
	return fmt.Sprintf("unknown(%d)", t)
}

// BlockSize returns the number of elements per quantization block.
func (t GGMLType) BlockSize() int {
	if info, ok := typeInfoMap[t]; ok {
		return info.BlockSize
	}
	return 0
}

// TypeSize returns the number of bytes per block.
func (t GGMLType) TypeSize() int {
	if info, ok := typeInfoMap[t]; ok {
		return info.TypeSize
	}
	return 0
}

// Supported returns true if dequantization is implemented for this type.
func (t GGMLType) Supported() bool {
	switch t {
	case TypeF32, TypeF16, TypeBF16,
		TypeQ4_0, TypeQ4_1, TypeQ5_0, TypeQ5_1,
		TypeQ8_0, TypeQ4_K, TypeQ5_K, TypeQ6_K:
		return true
	}
	return false
}

// MetadataValueType represents a GGUF metadata value type.
type MetadataValueType uint32

const (
	MetaUint8   MetadataValueType = 0
	MetaInt8    MetadataValueType = 1
	MetaUint16  MetadataValueType = 2
	MetaInt16   MetadataValueType = 3
	MetaUint32  MetadataValueType = 4
	MetaInt32   MetadataValueType = 5
	MetaFloat32 MetadataValueType = 6
	MetaBool    MetadataValueType = 7
	MetaString  MetadataValueType = 8
	MetaArray   MetadataValueType = 9
	MetaUint64  MetadataValueType = 10
	MetaInt64   MetadataValueType = 11
	MetaFloat64 MetadataValueType = 12
)

// Magic bytes at the start of every GGUF file.
var Magic = [4]byte{'G', 'G', 'U', 'F'}

// File represents a parsed GGUF file.
type File struct {
	Path        string
	Version     uint32
	TensorCount uint64
	MetaCount   uint64
	Metadata    map[string]any
	Tensors     []TensorInfo
	DataOffset  int64 // byte offset where tensor data begins
	FileSize    int64
}

// TensorInfo holds metadata about a single tensor in the GGUF file.
type TensorInfo struct {
	Name         string
	NDims        uint32
	Dims         []uint64
	Type         GGMLType
	Offset       uint64 // relative to DataOffset
	ElementCount uint64
}

// Architecture returns the model architecture from metadata (e.g., "llama", "mistral").
func (f *File) Architecture() string {
	if v, ok := f.Metadata["general.architecture"]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return "unknown"
}

// QuantType returns the dominant quantization type across all tensors.
func (f *File) QuantType() string {
	counts := make(map[GGMLType]int)
	for _, t := range f.Tensors {
		counts[t.Type]++
	}
	var maxType GGMLType
	var maxCount int
	for t, c := range counts {
		if c > maxCount {
			maxCount = c
			maxType = t
		}
	}
	return maxType.String()
}

// TotalParameters returns the sum of all tensor element counts.
func (f *File) TotalParameters() uint64 {
	var total uint64
	for _, t := range f.Tensors {
		total += t.ElementCount
	}
	return total
}

// ByteSize returns the size of a tensor's data in bytes.
func (t *TensorInfo) ByteSize() int64 {
	info, ok := typeInfoMap[t.Type]
	if !ok {
		return 0
	}
	if info.BlockSize == 1 {
		return int64(t.ElementCount) * int64(info.TypeSize)
	}
	nBlocks := (int64(t.ElementCount) + int64(info.BlockSize) - 1) / int64(info.BlockSize)
	return nBlocks * int64(info.TypeSize)
}
