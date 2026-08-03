package gguf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	maxMetaKVCount                = 100_000
	maxTensorCount                = 100_000
	maxStringLen                  = 10 * 1024 * 1024 // 10 MiB
	maxMetadataKeyLen             = 4 * 1024
	maxTensorNameLen              = 64 * 1024
	maxHeaderBytes                = 256 * 1024 * 1024
	maxMetadataArrayDepth         = 8
	maxMetadataArrayValues        = 2_000_000
	maxTensorReadBytes            = 256 * 1024 * 1024
	maxTensorSampleBytes   uint64 = 64 * 1024 * 1024
)

type parseState struct {
	r                 io.Reader
	arrayValuesParsed uint64
}

// Parse reads and parses a GGUF file, returning the header, metadata, and tensor info.
// It does NOT read tensor data — use ReadTensorData for that.
func Parse(path string) (*File, error) {
	pathInfo, err := os.Lstat(path)
	if err != nil {
		return nil, fmt.Errorf("lstat: %w", err)
	}
	if !pathInfo.Mode().IsRegular() {
		return nil, fmt.Errorf("refusing non-regular GGUF input: %s", pathInfo.Mode())
	}

	// #nosec G304 -- path is the caller-selected model input and is pinned to
	// the regular-file identity established by Lstat immediately above.
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	if !stat.Mode().IsRegular() || !os.SameFile(pathInfo, stat) {
		return nil, fmt.Errorf("GGUF input changed while opening")
	}
	if stat.Size() < 24 {
		return nil, fmt.Errorf("file too small for GGUF header: %d bytes", stat.Size())
	}

	gf := &File{
		Path:       path,
		Metadata:   make(map[string]any),
		FileSize:   stat.Size(),
		sourceInfo: stat,
	}
	limited := &io.LimitedReader{R: f, N: maxHeaderBytes + 1}
	state := &parseState{r: limited}

	// Read magic
	var magic [4]byte
	if err := binary.Read(state.r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != Magic {
		return nil, fmt.Errorf("invalid magic: %x (expected %x)", magic, Magic)
	}

	// Read version
	if err := binary.Read(state.r, binary.LittleEndian, &gf.Version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if gf.Version != 2 && gf.Version != 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", gf.Version)
	}

	// Read tensor count and metadata KV count
	if err := binary.Read(state.r, binary.LittleEndian, &gf.TensorCount); err != nil {
		return nil, fmt.Errorf("read tensor_count: %w", err)
	}
	if err := binary.Read(state.r, binary.LittleEndian, &gf.MetaCount); err != nil {
		return nil, fmt.Errorf("read meta_count: %w", err)
	}

	if gf.TensorCount > maxTensorCount {
		return nil, fmt.Errorf("tensor count too large: %d", gf.TensorCount)
	}
	if gf.TensorCount == 0 {
		return nil, fmt.Errorf("GGUF file contains no tensors")
	}
	if gf.MetaCount > maxMetaKVCount {
		return nil, fmt.Errorf("metadata KV count too large: %d", gf.MetaCount)
	}

	// Parse metadata KV pairs
	for i := uint64(0); i < gf.MetaCount; i++ {
		key, err := readStringLimit(state.r, maxMetadataKeyLen)
		if err != nil {
			return nil, fmt.Errorf("metadata key %d: %w", i, err)
		}
		if _, exists := gf.Metadata[key]; exists {
			return nil, fmt.Errorf("duplicate metadata key %q", key)
		}
		if key == "" || strings.ContainsRune(key, '\x00') {
			return nil, fmt.Errorf("invalid metadata key")
		}
		var valType uint32
		if err := binary.Read(state.r, binary.LittleEndian, &valType); err != nil {
			return nil, fmt.Errorf("metadata type %d: %w", i, err)
		}
		val, err := readValue(state, MetadataValueType(valType), 0)
		if err != nil {
			return nil, fmt.Errorf("metadata value %q: %w", key, err)
		}
		gf.Metadata[key] = val
	}

	// Parse tensor info entries
	gf.Tensors = make([]TensorInfo, 0, gf.TensorCount)
	tensorNames := make(map[string]struct{}, gf.TensorCount)
	var totalParameters uint64
	for i := uint64(0); i < gf.TensorCount; i++ {
		ti, err := readTensorInfo(state)
		if err != nil {
			return nil, fmt.Errorf("tensor info %d: %w", i, err)
		}
		if _, exists := tensorNames[ti.Name]; exists {
			return nil, fmt.Errorf("duplicate tensor name %q", ti.Name)
		}
		if ti.Name == "" || strings.ContainsRune(ti.Name, '\x00') {
			return nil, fmt.Errorf("invalid tensor name")
		}
		tensorNames[ti.Name] = struct{}{}
		if ^uint64(0)-totalParameters < ti.ElementCount {
			return nil, fmt.Errorf("total parameter count overflows uint64")
		}
		totalParameters += ti.ElementCount
		gf.Tensors = append(gf.Tensors, ti)
	}

	// Data starts at the next alignment boundary
	headerEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("seek: %w", err)
	}
	if headerEnd > maxHeaderBytes {
		return nil, fmt.Errorf("GGUF header exceeds %d-byte limit", maxHeaderBytes)
	}
	alignment, err := metadataAlignment(gf.Metadata)
	if err != nil {
		return nil, err
	}
	// metadataAlignment limits alignment to 4096 and headerEnd is capped at
	// 256 MiB, so this conversion and addition cannot approach int64 overflow.
	alignment64 := int64(alignment) // #nosec G115 -- metadataAlignment limits alignment to 4096
	gf.DataOffset = ((headerEnd + alignment64 - 1) / alignment64) * alignment64
	if gf.DataOffset > gf.FileSize {
		return nil, fmt.Errorf("tensor data offset %d exceeds file size %d", gf.DataOffset, gf.FileSize)
	}
	if err := validateTensorRanges(gf); err != nil {
		return nil, err
	}
	if after, err := f.Stat(); err != nil || !sourceMatches(gf, after) {
		return nil, fmt.Errorf("GGUF input changed while parsing")
	}

	return gf, nil
}

// ReadTensorData reads raw bytes for a specific tensor from the GGUF file.
// maxBytes limits the read size. Reads are always capped at 256 MiB; use
// HashTensorData for full-tensor integrity checks.
func ReadTensorData(gf *File, ti *TensorInfo, maxBytes int64) ([]byte, error) {
	if gf == nil || ti == nil {
		return nil, fmt.Errorf("nil GGUF file or tensor")
	}
	if maxBytes < 0 {
		return nil, fmt.Errorf("maxBytes must not be negative")
	}
	f, err := os.Open(gf.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	if !sourceMatches(gf, stat) {
		return nil, fmt.Errorf("GGUF input changed after parsing")
	}

	start, size, err := tensorRange(gf, ti, stat.Size())
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, fmt.Errorf("unknown type size for %s", ti.Type)
	}
	limit := int64(maxTensorReadBytes)
	if maxBytes > 0 && maxBytes < limit {
		limit = maxBytes
	}
	if size > limit {
		size = limit
	}

	buf := make([]byte, int(size)) // #nosec G115 -- size is capped at maxTensorReadBytes
	if _, err := f.ReadAt(buf, start); err != nil {
		return nil, fmt.Errorf("read tensor data %q: %w", ti.Name, err)
	}
	if after, err := f.Stat(); err != nil || !sourceMatches(gf, after) {
		return nil, fmt.Errorf("GGUF input changed while reading tensor %q", ti.Name)
	}
	return buf, nil
}

// ReadTensorSample returns complete, evenly distributed storage blocks from a
// tensor. It bounds memory and I/O while covering the beginning, middle, and
// end of large tensors instead of inspecting only an attacker-controlled
// prefix. maxElements must be positive.
func ReadTensorSample(gf *File, ti *TensorInfo, maxElements int) ([]byte, error) {
	if gf == nil || ti == nil {
		return nil, fmt.Errorf("nil GGUF file or tensor")
	}
	if maxElements <= 0 {
		return nil, fmt.Errorf("maxElements must be positive")
	}
	typeInfo, ok := typeInfoMap[ti.Type]
	if !ok || typeInfo.BlockSize <= 0 || typeInfo.TypeSize <= 0 {
		return nil, fmt.Errorf("unsupported tensor type %s", ti.Type)
	}

	f, err := os.Open(gf.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	if !sourceMatches(gf, stat) {
		return nil, fmt.Errorf("GGUF input changed after parsing")
	}
	start, size, err := tensorRange(gf, ti, stat.Size())
	if err != nil {
		return nil, err
	}

	totalBlocks := ti.ElementCount / uint64(typeInfo.BlockSize)
	maxBlocks := boundedSampleBlocks(
		totalBlocks,
		uint64(maxElements),        // #nosec G115 -- maxElements is checked positive above
		uint64(typeInfo.BlockSize), // #nosec G115 -- trusted table value is checked positive above
		uint64(typeInfo.TypeSize),  // #nosec G115 -- trusted table value is checked positive above
	)
	if maxBlocks >= totalBlocks {
		// boundedSampleBlocks only selects the complete-tensor path when its
		// encoded body fits the hard sample-memory limit.
		buf := make([]byte, int(size)) // #nosec G115 -- complete sample is <= maxTensorSampleBytes
		if _, err := f.ReadAt(buf, start); err != nil {
			return nil, fmt.Errorf("read tensor data %q: %w", ti.Name, err)
		}
		if after, err := f.Stat(); err != nil || !sourceMatches(gf, after) {
			return nil, fmt.Errorf("GGUF input changed while reading tensor %q", ti.Name)
		}
		return buf, nil
	}

	windowCount := uint64(64)
	if maxBlocks < windowCount {
		windowCount = maxBlocks
	}
	sampleBytes := maxBlocks * uint64(typeInfo.TypeSize)
	buf := make([]byte, int(sampleBytes)) // #nosec G115 -- boundedSampleBlocks caps bytes at 64 MiB
	baseCount := maxBlocks / windowCount
	extraCount := maxBlocks % windowCount
	outOffset := 0
	for window := uint64(0); window < windowCount; window++ {
		count := baseCount
		if window < extraCount {
			count++
		}
		stratumStart := multiplyDivideSmall(totalBlocks, window, windowCount)
		stratumEnd := multiplyDivideSmall(totalBlocks, window+1, windowCount)
		if width := stratumEnd - stratumStart; count > width {
			count = width
		}
		blockStart := stratumStart + (stratumEnd-stratumStart-count)/2
		if window == 0 {
			blockStart = stratumStart
		} else if window+1 == windowCount {
			blockStart = stratumEnd - count
		}
		byteCount := int(count * uint64(typeInfo.TypeSize))               // #nosec G115 -- count is within the 64 MiB sample cap
		byteOffset := start + int64(blockStart*uint64(typeInfo.TypeSize)) // #nosec G115 -- tensorRange proves the encoded tensor range fits int64
		if _, err := f.ReadAt(buf[outOffset:outOffset+byteCount], byteOffset); err != nil {
			return nil, fmt.Errorf("sample tensor data %q: %w", ti.Name, err)
		}
		outOffset += byteCount
	}
	buf = buf[:outOffset]
	if after, err := f.Stat(); err != nil || !sourceMatches(gf, after) {
		return nil, fmt.Errorf("GGUF input changed while sampling tensor %q", ti.Name)
	}
	return buf, nil
}

// boundedSampleBlocks applies the public API's element request while enforcing
// an invariant independent of caller behavior: a sample can never allocate or
// read more than maxTensorSampleBytes. All type sizes are trusted table values.
func boundedSampleBlocks(totalBlocks, maxElements, blockSize, typeSize uint64) uint64 {
	requested := maxElements / blockSize
	if requested == 0 {
		requested = 1
	}
	byteLimited := maxTensorSampleBytes / typeSize
	if requested > byteLimited {
		requested = byteLimited
	}
	if requested > totalBlocks {
		requested = totalBlocks
	}
	return requested
}

// multiplyDivideSmall computes floor(value*multiplier/divisor) without the
// overflow caused by multiplying an adversarial uint64 value first. The
// multiplier is at most divisor in the sampling caller.
func multiplyDivideSmall(value, multiplier, divisor uint64) uint64 {
	return (value/divisor)*multiplier + ((value%divisor)*multiplier)/divisor
}

// HashFile computes SHA-256 over the exact regular file accepted by Parse.
func HashFile(gf *File) (string, error) {
	if gf == nil {
		return "", fmt.Errorf("nil GGUF file")
	}
	f, err := os.Open(gf.Path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return "", fmt.Errorf("stat: %w", err)
	}
	if !sourceMatches(gf, stat) {
		return "", fmt.Errorf("GGUF input changed after parsing")
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	if after, err := f.Stat(); err != nil || !sourceMatches(gf, after) {
		return "", fmt.Errorf("GGUF input changed while hashing")
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// HashTensorData computes SHA-256 over a tensor without allocating its full body.
func HashTensorData(gf *File, ti *TensorInfo) (string, error) {
	if gf == nil || ti == nil {
		return "", fmt.Errorf("nil GGUF file or tensor")
	}
	f, err := os.Open(gf.Path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return "", fmt.Errorf("stat: %w", err)
	}
	if !sourceMatches(gf, stat) {
		return "", fmt.Errorf("GGUF input changed after parsing")
	}
	start, size, err := tensorRange(gf, ti, stat.Size())
	if err != nil {
		return "", err
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(f, start, size)); err != nil {
		return "", fmt.Errorf("hash tensor data %q: %w", ti.Name, err)
	}
	if after, err := f.Stat(); err != nil || !sourceMatches(gf, after) {
		return "", fmt.Errorf("GGUF input changed while hashing tensor %q", ti.Name)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func sourceMatches(gf *File, stat os.FileInfo) bool {
	if stat == nil || !stat.Mode().IsRegular() || stat.Size() != gf.FileSize {
		return false
	}
	return gf.sourceInfo == nil || (os.SameFile(gf.sourceInfo, stat) && gf.sourceInfo.ModTime().Equal(stat.ModTime()))
}

func readString(r io.Reader) (string, error) {
	return readStringLimit(r, maxStringLen)
}

func readStringLimit(r io.Reader, maxLength uint64) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > maxLength {
		return "", fmt.Errorf("string too long: %d bytes", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	if !utf8.Valid(buf) {
		return "", fmt.Errorf("string is not valid UTF-8")
	}
	return string(buf), nil
}

func readValue(state *parseState, vtype MetadataValueType, depth int) (any, error) {
	if depth > maxMetadataArrayDepth {
		return nil, fmt.Errorf("metadata array nesting exceeds %d", maxMetadataArrayDepth)
	}
	r := state.r
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
		if v > 1 {
			return nil, fmt.Errorf("invalid boolean value %d", v)
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
		if state.arrayValuesParsed > maxMetadataArrayValues-length {
			return nil, fmt.Errorf("aggregate metadata array values exceed %d", maxMetadataArrayValues)
		}
		state.arrayValuesParsed += length
		arr := make([]any, 0, length)
		for i := uint64(0); i < length; i++ {
			v, err := readValue(state, MetadataValueType(elemType), depth+1)
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

func readTensorInfo(state *parseState) (TensorInfo, error) {
	var ti TensorInfo
	r := state.r

	name, err := readStringLimit(r, maxTensorNameLen)
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
	if ti.NDims == 0 {
		return ti, fmt.Errorf("tensor must have at least one dimension")
	}

	ti.Dims = make([]uint64, ti.NDims)
	ti.ElementCount = 1
	for d := uint32(0); d < ti.NDims; d++ {
		if err := binary.Read(r, binary.LittleEndian, &ti.Dims[d]); err != nil {
			return ti, fmt.Errorf("read dim %d: %w", d, err)
		}
		if ti.Dims[d] == 0 {
			return ti, fmt.Errorf("dimension %d is zero", d)
		}
		if ti.ElementCount > ^uint64(0)/ti.Dims[d] {
			return ti, fmt.Errorf("element count overflow at dimension %d", d)
		}
		ti.ElementCount *= ti.Dims[d]
	}

	var dtype uint32
	if err := binary.Read(r, binary.LittleEndian, &dtype); err != nil {
		return ti, fmt.Errorf("read type: %w", err)
	}
	ti.Type = GGMLType(dtype)
	typeInfo, supported := typeInfoMap[ti.Type]
	if !supported || typeInfo.TypeSize == 0 {
		return ti, fmt.Errorf("unsupported tensor type: %d", dtype)
	}
	if typeInfo.BlockSize > 1 && ti.ElementCount%uint64(typeInfo.BlockSize) != 0 {
		return ti, fmt.Errorf("element count %d is not divisible by block size %d", ti.ElementCount, typeInfo.BlockSize)
	}

	if err := binary.Read(r, binary.LittleEndian, &ti.Offset); err != nil {
		return ti, fmt.Errorf("read offset: %w", err)
	}

	return ti, nil
}

func metadataAlignment(metadata map[string]any) (uint64, error) {
	alignment := uint64(32)
	if raw, ok := metadata["general.alignment"]; ok {
		switch value := raw.(type) {
		case uint32:
			alignment = uint64(value)
		case uint64:
			alignment = value
		default:
			return 0, fmt.Errorf("general.alignment must be an unsigned integer")
		}
	}
	if alignment == 0 || alignment > 4096 || alignment&(alignment-1) != 0 {
		return 0, fmt.Errorf("invalid GGUF alignment %d", alignment)
	}
	return alignment, nil
}

func tensorRange(gf *File, ti *TensorInfo, fileSize int64) (int64, int64, error) {
	size := ti.ByteSize()
	if size <= 0 {
		return 0, 0, fmt.Errorf("invalid byte size for tensor %q", ti.Name)
	}
	if ti.Offset > uint64(math.MaxInt64) {
		return 0, 0, fmt.Errorf("tensor %q offset overflows int64", ti.Name)
	}
	relative := int64(ti.Offset)
	if relative > math.MaxInt64-gf.DataOffset {
		return 0, 0, fmt.Errorf("tensor %q start offset overflows", ti.Name)
	}
	start := gf.DataOffset + relative
	if start < gf.DataOffset || size > fileSize-start {
		return 0, 0, fmt.Errorf("tensor %q range [%d,%d) exceeds file size %d", ti.Name, start, start+size, fileSize)
	}
	return start, size, nil
}

func validateTensorRanges(gf *File) error {
	type dataRange struct {
		start int64
		end   int64
		name  string
	}
	ranges := make([]dataRange, 0, len(gf.Tensors))
	alignment, err := metadataAlignment(gf.Metadata)
	if err != nil {
		return err
	}
	for i := range gf.Tensors {
		if gf.Tensors[i].Offset%alignment != 0 {
			return fmt.Errorf("tensor %q offset is not aligned to %d bytes", gf.Tensors[i].Name, alignment)
		}
		start, size, err := tensorRange(gf, &gf.Tensors[i], gf.FileSize)
		if err != nil {
			return err
		}
		ranges = append(ranges, dataRange{start: start, end: start + size, name: gf.Tensors[i].Name})
	}
	sort.Slice(ranges, func(i, j int) bool { return ranges[i].start < ranges[j].start })
	for i := 1; i < len(ranges); i++ {
		if ranges[i].start < ranges[i-1].end {
			return fmt.Errorf("tensor %q overlaps tensor %q", ranges[i].name, ranges[i-1].name)
		}
	}
	return nil
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
