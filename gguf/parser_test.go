package gguf

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func metadataArrayBytes(t *testing.T, element MetadataValueType, count uint64, payload []byte) []byte {
	t.Helper()
	var buffer bytes.Buffer
	for _, value := range []any{uint32(element), count} {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	buffer.Write(payload)
	return buffer.Bytes()
}

func arrayParseState(raw []byte) *parseState {
	return &parseState{r: &io.LimitedReader{R: bytes.NewReader(raw), N: int64(len(raw))}}
}

func writeArrayTestGGUF(t *testing.T, arrays map[string][]byte) string {
	t.Helper()
	var buffer bytes.Buffer
	write := func(value any) {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	text := func(value string) { write(uint64(len(value))); buffer.WriteString(value) }
	buffer.Write(Magic[:])
	write(uint32(3))
	write(uint64(1))
	write(uint64(len(arrays)))
	for key, raw := range arrays {
		text(key)
		write(uint32(MetaArray))
		buffer.Write(raw)
	}
	text("weight")
	write(uint32(1))
	write(uint64(1))
	write(uint32(TypeF32))
	write(uint64(0))
	for buffer.Len()%32 != 0 {
		buffer.WriteByte(0)
	}
	write(float32(0))
	path := filepath.Join(t.TempDir(), "arrays.gguf")
	if err := os.WriteFile(path, buffer.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseLargeTokenizerAndPrimitiveArrays(t *testing.T) {
	const count = 248_320
	tokens := bytes.Repeat([]byte{1, 0, 0, 0, 0, 0, 0, 0, 't'}, count)
	types := make([]byte, count*4)
	for i := range count {
		binary.LittleEndian.PutUint32(types[i*4:], uint32(i%8))
	}
	path := writeArrayTestGGUF(t, map[string][]byte{
		"tokenizer.ggml.tokens":     metadataArrayBytes(t, MetaString, count, tokens),
		"tokenizer.ggml.token_type": metadataArrayBytes(t, MetaInt32, count, types),
	})
	parsed, err := Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	gotTokens := parsed.Metadata["tokenizer.ggml.tokens"].([]any)
	gotTypes := parsed.Metadata["tokenizer.ggml.token_type"].([]any)
	if len(gotTokens) != count || len(gotTypes) != count || gotTokens[count-1] != "t" || gotTypes[count-1] != int32((count-1)%8) {
		t.Fatal("large string/primitive array values did not round trip")
	}
}

func TestMetadataArrayElementLimit(t *testing.T) {
	raw := metadataArrayBytes(t, MetaUint8, maxMetadataArrayElements, bytes.Repeat([]byte{7}, maxMetadataArrayElements))
	value, err := readValue(arrayParseState(raw), MetaArray, 0)
	if err != nil || len(value.([]any)) != maxMetadataArrayElements {
		t.Fatalf("exact element boundary rejected: %v", err)
	}
	for _, count := range []uint64{maxMetadataArrayElements + 1, math.MaxUint64} {
		state := arrayParseState(metadataArrayBytes(t, MetaUint8, count, nil))
		if _, err := readValue(state, MetaArray, 0); err == nil || !strings.Contains(err.Error(), "array too long") {
			t.Fatalf("oversized count %d accepted: %v", count, err)
		}
		if state.arrayValuesParsed != 0 {
			t.Fatal("oversized array consumed a value reservation")
		}
	}
}

func TestMetadataArrayWirePreflight(t *testing.T) {
	for _, element := range []MetadataValueType{MetaUint8, MetaUint64, MetaString, MetaArray} {
		t.Run(elementName(element), func(t *testing.T) {
			width, err := minimumValueBytes(element)
			if err != nil {
				t.Fatal(err)
			}
			raw := metadataArrayBytes(t, element, 2, make([]byte, width))
			state := arrayParseState(raw)
			if _, err := readValue(state, MetaArray, 0); err == nil || !strings.Contains(err.Error(), "remaining") {
				t.Fatalf("truncated array not rejected by wire preflight: %v", err)
			}
			if state.arrayValuesParsed != 0 {
				t.Fatal("truncated declaration reached allocation reservation")
			}
		})
	}
	// Actual Parse must constrain remaining bytes to file size, not only the
	// much larger header allowance, before allocating a million interface slots.
	path := writeArrayTestGGUF(t, map[string][]byte{"values": metadataArrayBytes(t, MetaUint8, maxMetadataArrayElements, nil)})
	if _, err := Parse(path); err == nil || !strings.Contains(err.Error(), "remaining") {
		t.Fatalf("small file bypassed array allocation preflight: %v", err)
	}
	raw := metadataArrayBytes(t, MetaUint64, 2, make([]byte, 16))
	state := &parseState{r: &io.LimitedReader{R: bytes.NewReader(raw), N: 16}}
	if _, err := readValue(state, MetaArray, 0); err == nil {
		t.Fatal("header remaining-byte bound bypassed")
	}
	if err := requireWireBytes(bytes.NewReader(nil), math.MaxUint64, 8); err == nil {
		t.Fatal("wire size multiplication overflow accepted")
	}
}

func elementName(value MetadataValueType) string {
	return map[MetadataValueType]string{MetaUint8: "uint8", MetaUint64: "uint64", MetaString: "string", MetaArray: "nested"}[value]
}

func TestMetadataArrayAggregateAndTypes(t *testing.T) {
	state := arrayParseState(metadataArrayBytes(t, MetaUint8, 3, []byte{1, 2, 3}))
	state.arrayValuesParsed = maxMetadataArrayValues - 3
	if _, err := readValue(state, MetaArray, 0); err != nil || state.arrayValuesParsed != maxMetadataArrayValues {
		t.Fatalf("exact aggregate boundary failed: %v", err)
	}
	state.r = arrayParseState(metadataArrayBytes(t, MetaUint8, 1, []byte{1})).r
	if _, err := readValue(state, MetaArray, 0); err == nil || !strings.Contains(err.Error(), "aggregate") {
		t.Fatal("aggregate boundary bypassed")
	}
	state.arrayValuesParsed = math.MaxUint64
	state.r = arrayParseState(metadataArrayBytes(t, MetaUint8, 0, nil)).r
	if _, err := readValue(state, MetaArray, 0); err == nil {
		t.Fatal("aggregate subtraction overflow accepted")
	}
	for _, count := range []uint64{0, 1} {
		state := arrayParseState(metadataArrayBytes(t, MetadataValueType(math.MaxUint32), count, nil))
		if _, err := readValue(state, MetaArray, 0); err == nil || !strings.Contains(err.Error(), "element type") {
			t.Fatal("invalid element type accepted")
		}
	}
	child := metadataArrayBytes(t, MetaUint8, 2, []byte{1, 2})
	state = arrayParseState(metadataArrayBytes(t, MetaArray, 2, append(append([]byte{}, child...), child...)))
	state.arrayValuesParsed = maxMetadataArrayValues - 5
	if _, err := readValue(state, MetaArray, 0); err == nil || !strings.Contains(err.Error(), "aggregate") {
		t.Fatal("nested arrays bypassed aggregate budget")
	}
	invalid := metadataArrayBytes(t, MetadataValueType(math.MaxUint32), 0, nil)
	if _, err := readValue(arrayParseState(metadataArrayBytes(t, MetaArray, 1, invalid)), MetaArray, 0); err == nil {
		t.Fatal("nested invalid element type accepted")
	}
}

func TestMetadataArrayDepthPreserved(t *testing.T) {
	for _, depth := range []int{maxMetadataArrayDepth, maxMetadataArrayDepth + 1} {
		raw := metadataArrayBytes(t, MetaUint8, 1, []byte{1})
		for i := 1; i < depth; i++ {
			raw = metadataArrayBytes(t, MetaArray, 1, raw)
		}
		_, err := readValue(arrayParseState(raw), MetaArray, 0)
		if (err != nil) != (depth > maxMetadataArrayDepth) {
			t.Fatalf("depth %d: %v", depth, err)
		}
	}
	// Existing semantics allow an empty array at the terminal depth: no child
	// is visited beyond it. Do not turn this compatibility fix into a new rule.
	if _, err := readValue(arrayParseState(metadataArrayBytes(t, MetaArray, 0, nil)), MetaArray, maxMetadataArrayDepth); err != nil {
		t.Fatal(err)
	}
}

func TestMetadataStringAndKVLimitsPreserved(t *testing.T) {
	var buffer bytes.Buffer
	if err := binary.Write(&buffer, binary.LittleEndian, uint64(maxStringLen)); err != nil {
		t.Fatal(err)
	}
	reader := &io.LimitedReader{R: bytes.NewReader(buffer.Bytes()), N: int64(buffer.Len())}
	if _, err := readString(reader); err == nil || !strings.Contains(err.Error(), "remaining") {
		t.Fatal("truncated string length reached data allocation")
	}
	buffer.Reset()
	if err := binary.Write(&buffer, binary.LittleEndian, uint64(maxStringLen+1)); err != nil {
		t.Fatal(err)
	}
	if _, err := readString(bytes.NewReader(buffer.Bytes())); err == nil || !strings.Contains(err.Error(), "too long") {
		t.Fatal("string ceiling changed")
	}
	buffer.Reset()
	buffer.Write(Magic[:])
	for _, value := range []any{uint32(3), uint64(1), uint64(100_001)} {
		if err := binary.Write(&buffer, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "too-many-keys.gguf")
	if err := os.WriteFile(path, buffer.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := Parse(path); err == nil || !strings.Contains(err.Error(), "metadata KV count too large") {
		t.Fatalf("100000-pair ceiling changed: %v", err)
	}
}

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
		dataOffset = (dataOffset + 31) &^ 31
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
	f.Write(make([]byte, dataOffset))

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

func TestParseRejectsSymlink(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, []TensorInfo{{Name: "a", NDims: 1, Dims: []uint64{1}, Type: TypeF32, ElementCount: 1}}, nil)
	link := filepath.Join(dir, "linked.gguf")
	if err := os.Symlink(path, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	if _, err := Parse(link); err == nil {
		t.Fatal("symlink GGUF input must be rejected")
	}
}

func TestParseRejectsModelWithoutTensors(t *testing.T) {
	path := writeTestGGUF(t, t.TempDir(), nil, map[string]any{"general.architecture": "llama"})
	if _, err := Parse(path); err == nil {
		t.Fatal("metadata-only GGUF must not be accepted as a model")
	}
}

func TestParseRejectsOverlappingTensors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "overlap.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	f.Write(Magic[:])
	binary.Write(f, binary.LittleEndian, uint32(3))
	binary.Write(f, binary.LittleEndian, uint64(2))
	binary.Write(f, binary.LittleEndian, uint64(0))
	for _, name := range []string{"a", "b"} {
		writeGGUFString(f, name)
		binary.Write(f, binary.LittleEndian, uint32(1))
		binary.Write(f, binary.LittleEndian, uint64(4))
		binary.Write(f, binary.LittleEndian, uint32(TypeF32))
		binary.Write(f, binary.LittleEndian, uint64(0))
	}
	pos, _ := f.Seek(0, 1)
	padding := (32 - (pos % 32)) % 32
	f.Write(make([]byte, padding+16))
	f.Close()
	if _, err := Parse(path); err == nil {
		t.Fatal("overlapping tensor ranges must be rejected")
	}
}

func TestParseRejectsDimensionOverflow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "overflow.gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	f.Write(Magic[:])
	binary.Write(f, binary.LittleEndian, uint32(3))
	binary.Write(f, binary.LittleEndian, uint64(1))
	binary.Write(f, binary.LittleEndian, uint64(0))
	writeGGUFString(f, "overflow")
	binary.Write(f, binary.LittleEndian, uint32(2))
	binary.Write(f, binary.LittleEndian, ^uint64(0))
	binary.Write(f, binary.LittleEndian, uint64(2))
	binary.Write(f, binary.LittleEndian, uint32(TypeF32))
	binary.Write(f, binary.LittleEndian, uint64(0))
	f.Close()
	if _, err := Parse(path); err == nil {
		t.Fatal("dimension multiplication overflow must be rejected")
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

func TestReadTensorSampleCoversWholeTensor(t *testing.T) {
	dir := t.TempDir()
	ti := TensorInfo{Name: "sample", NDims: 1, Dims: []uint64{256}, Type: TypeF32, ElementCount: 256}
	path := writeTestGGUF(t, dir, []TensorInfo{ti}, map[string]any{})
	initial, err := Parse(path)
	if err != nil {
		t.Fatalf("initial parse: %v", err)
	}
	f, err := os.OpenFile(path, os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 256; i++ {
		var value [4]byte
		binary.LittleEndian.PutUint32(value[:], math.Float32bits(float32(i)))
		if _, err := f.WriteAt(value[:], initial.DataOffset+int64(i*4)); err != nil {
			_ = f.Close()
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	gf, err := Parse(path)
	if err != nil {
		t.Fatalf("parse populated tensor: %v", err)
	}
	data, err := ReadTensorSample(gf, &gf.Tensors[0], 8)
	if err != nil {
		t.Fatalf("sample: %v", err)
	}
	values, err := Dequantize(data, TypeF32, 0)
	if err != nil {
		t.Fatalf("decode sample: %v", err)
	}
	if len(values) != 8 || values[0] != 0 || values[len(values)-1] != 255 {
		t.Fatalf("sample did not cover tensor endpoints: %v", values)
	}
}

func TestBoundedSampleBlocksCapsUntrustedCallerRequest(t *testing.T) {
	got := boundedSampleBlocks(^uint64(0), ^uint64(0), 1, 4)
	want := maxTensorSampleBytes / 4
	if got != want {
		t.Fatalf("sample block cap = %d, want %d", got, want)
	}
	if got*4 > maxTensorSampleBytes {
		t.Fatalf("sample cap permits %d bytes", got*4)
	}
}

func TestParseRejectsOversizedTensorName(t *testing.T) {
	ti := TensorInfo{Name: strings.Repeat("x", maxTensorNameLen+1), NDims: 1, Dims: []uint64{1}, Type: TypeF32, ElementCount: 1}
	path := writeTestGGUF(t, t.TempDir(), []TensorInfo{ti}, nil)
	if _, err := Parse(path); err == nil {
		t.Fatal("oversized tensor name must be rejected")
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
		{Name: "a", NDims: 1, Dims: []uint64{256}, Type: TypeQ4_K, ElementCount: 256},
		{Name: "b", NDims: 1, Dims: []uint64{256}, Type: TypeQ4_K, ElementCount: 256},
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
