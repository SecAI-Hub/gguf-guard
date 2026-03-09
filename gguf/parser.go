package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const maxMetaKVCount = 100_000
const maxTensorCount = 100_000
const maxStringLen = 10 * 1024 * 1024 // 10 MB

// Parse reads and parses a GGUF file, returning the header, metadata, and tensor info.
// It does NOT read tensor data — use ReadTensorData for that.
func Parse(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}

	gf := &File{
		Path:     path,
		Metadata: make(map[string]any),
		FileSize: stat.Size(),
	}

	// Read magic
	var magic [4]byte
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != Magic {
		return nil, fmt.Errorf("invalid magic: %x (expected %x)", magic, Magic)
	}

	// Read version
	if err := binary.Read(f, binary.LittleEndian, &gf.Version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if gf.Version != 2 && gf.Version != 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", gf.Version)
	}

	// Read tensor count and metadata KV count
	if err := binary.Read(f, binary.LittleEndian, &gf.TensorCount); err != nil {
		return nil, fmt.Errorf("read tensor_count: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &gf.MetaCount); err != nil {
		return nil, fmt.Errorf("read meta_count: %w", err)
	}

	if gf.TensorCount > maxTensorCount {
		return nil, fmt.Errorf("tensor count too large: %d", gf.TensorCount)
	}
	if gf.MetaCount > maxMetaKVCount {
		return nil, fmt.Errorf("metadata KV count too large: %d", gf.MetaCount)
	}

	// Parse metadata KV pairs
	for i := uint64(0); i < gf.MetaCount; i++ {
		key, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("metadata key %d: %w", i, err)
		}
		var valType uint32
		if err := binary.Read(f, binary.LittleEndian, &valType); err != nil {
			return nil, fmt.Errorf("metadata type %d: %w", i, err)
		}
		val, err := readValue(f, MetadataValueType(valType))
		if err != nil {
			return nil, fmt.Errorf("metadata value %q: %w", key, err)
		}
		gf.Metadata[key] = val
	}

	// Parse tensor info entries
	gf.Tensors = make([]TensorInfo, 0, gf.TensorCount)
	for i := uint64(0); i < gf.TensorCount; i++ {
		ti, err := readTensorInfo(f)
		if err != nil {
			return nil, fmt.Errorf("tensor info %d: %w", i, err)
		}
		gf.Tensors = append(gf.Tensors, ti)
	}

	// Data starts at the next alignment boundary
	headerEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("seek: %w", err)
	}
	const alignment = 32
	gf.DataOffset = ((headerEnd + alignment - 1) / alignment) * alignment

	return gf, nil
}

// ReadTensorData reads raw bytes for a specific tensor from the GGUF file.
// maxBytes limits the read size (0 = read all).
func ReadTensorData(gf *File, ti *TensorInfo, maxBytes int64) ([]byte, error) {
	f, err := os.Open(gf.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	size := ti.ByteSize()
	if size == 0 {
		return nil, fmt.Errorf("unknown type size for %s", ti.Type)
	}
	if maxBytes > 0 && size > maxBytes {
		size = maxBytes
	}

	buf := make([]byte, size)
	if _, err := f.ReadAt(buf, gf.DataOffset+int64(ti.Offset)); err != nil {
		return nil, fmt.Errorf("read tensor data %q: %w", ti.Name, err)
	}
	return buf, nil
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > uint64(maxStringLen) {
		return "", fmt.Errorf("string too long: %d bytes", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readValue(r io.Reader, vtype MetadataValueType) (any, error) {
	switch vtype {
	case MetaUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaFloat32:
		var v float32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return v != 0, nil
	case MetaString:
		return readString(r)
	case MetaUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaFloat64:
		var v float64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case MetaArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var length uint64
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return nil, err
		}
		if length > maxMetaKVCount {
			return nil, fmt.Errorf("array too long: %d", length)
		}
		arr := make([]any, 0, length)
		for i := uint64(0); i < length; i++ {
			v, err := readValue(r, MetadataValueType(elemType))
			if err != nil {
				return nil, fmt.Errorf("array element %d: %w", i, err)
			}
			arr = append(arr, v)
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unsupported metadata type: %d", vtype)
	}
}

func readTensorInfo(r io.Reader) (TensorInfo, error) {
	var ti TensorInfo

	name, err := readString(r)
	if err != nil {
		return ti, fmt.Errorf("read name: %w", err)
	}
	ti.Name = name

	if err := binary.Read(r, binary.LittleEndian, &ti.NDims); err != nil {
		return ti, fmt.Errorf("read ndims: %w", err)
	}
	if ti.NDims > 8 {
		return ti, fmt.Errorf("too many dimensions: %d", ti.NDims)
	}

	ti.Dims = make([]uint64, ti.NDims)
	ti.ElementCount = 1
	for d := uint32(0); d < ti.NDims; d++ {
		if err := binary.Read(r, binary.LittleEndian, &ti.Dims[d]); err != nil {
			return ti, fmt.Errorf("read dim %d: %w", d, err)
		}
		ti.ElementCount *= ti.Dims[d]
	}

	var dtype uint32
	if err := binary.Read(r, binary.LittleEndian, &dtype); err != nil {
		return ti, fmt.Errorf("read type: %w", err)
	}
	ti.Type = GGMLType(dtype)

	if err := binary.Read(r, binary.LittleEndian, &ti.Offset); err != nil {
		return ti, fmt.Errorf("read offset: %w", err)
	}

	return ti, nil
}

// f16ToF32 converts a 16-bit IEEE 754 half-precision float to float32.
func F16ToF32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits) & 0x3FF

	switch {
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
		return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
	case exp == 0x1F:
		if mant == 0 {
			return math.Float32frombits((sign << 31) | (0xFF << 23)) // Inf
		}
		return math.Float32frombits((sign << 31) | (0xFF << 23) | (mant << 13)) // NaN
	default:
		return math.Float32frombits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
	}
}

// bf16ToF32 converts a bfloat16 to float32 by shifting into the upper 16 bits.
func BF16ToF32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}
