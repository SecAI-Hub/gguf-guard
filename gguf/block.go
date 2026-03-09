package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// BlockStats holds statistics extracted from the raw quantization block structure,
// not from dequantized values. This detects anomalies hidden in scale placement,
// bin usage, or block structure that normal dequantized-value analysis would miss.
type BlockStats struct {
	TensorName     string    `json:"tensor_name"`
	QuantType      string    `json:"quant_type"`
	NumBlocks      int       `json:"num_blocks"`
	Scales         []float32 `json:"-"` // raw scale values (not serialized)
	ScaleMin       float32   `json:"scale_min"`
	ScaleMax       float32   `json:"scale_max"`
	ScaleMean      float64   `json:"scale_mean"`
	ScaleStdDev    float64   `json:"scale_std_dev"`
	ScaleEntropy   float64   `json:"scale_entropy"`   // Shannon entropy of scale distribution
	ScaleRatio     float64   `json:"scale_ratio"`     // max/min scale ratio
	CodeEntropy    float64   `json:"code_entropy"`    // entropy of quantized code distribution
	SaturationLow  float64   `json:"saturation_low"`  // fraction of codes at minimum bin
	SaturationHigh float64   `json:"saturation_high"` // fraction of codes at maximum bin
	RepeatedBlocks int       `json:"repeated_blocks"` // count of duplicate blocks
	ZeroScales     int       `json:"zero_scales"`     // blocks with near-zero scale
}

// ExtractBlockStats analyzes the raw quantization block structure of a tensor.
// Returns nil for non-quantized types (F32, F16, BF16).
func ExtractBlockStats(data []byte, name string, qtype GGMLType) *BlockStats {
	switch qtype {
	case TypeQ8_0:
		return extractQ8_0Blocks(data, name)
	case TypeQ4_0:
		return extractQ4_0Blocks(data, name)
	case TypeQ4_1:
		return extractQ4_1Blocks(data, name)
	case TypeQ4_K:
		return extractQ4_KBlocks(data, name)
	case TypeQ5_K:
		return extractQ5_KBlocks(data, name)
	case TypeQ6_K:
		return extractQ6_KBlocks(data, name)
	default:
		return nil // non-quantized or unsupported
	}
}

func extractQ8_0Blocks(data []byte, name string) *BlockStats {
	const blockSize = 34
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil
	}

	scales := make([]float32, nBlocks)
	codeCounts := make([]int, 256) // int8 range mapped to 0-255

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		scales[bi] = F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		for qi := 0; qi < elemsPerBlock; qi++ {
			code := data[off+2+qi]
			codeCounts[code]++
		}
	}

	bs := buildBlockStats(name, "Q8_0", nBlocks, scales)
	bs.CodeEntropy = shannonEntropy(codeCounts, nBlocks*elemsPerBlock)
	bs.SaturationLow = float64(codeCounts[128]) / float64(nBlocks*elemsPerBlock)  // int8 min = -128
	bs.SaturationHigh = float64(codeCounts[127]) / float64(nBlocks*elemsPerBlock) // int8 max = 127
	bs.RepeatedBlocks = countRepeatedBlocks(data, blockSize)
	return bs
}

func extractQ4_0Blocks(data []byte, name string) *BlockStats {
	const blockSize = 18
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil
	}

	scales := make([]float32, nBlocks)
	codeCounts := make([]int, 16) // 4-bit range

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		scales[bi] = F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		for qi := 0; qi < 16; qi++ {
			lo := data[off+2+qi] & 0x0F
			hi := data[off+2+qi] >> 4
			codeCounts[lo]++
			codeCounts[hi]++
		}
	}

	bs := buildBlockStats(name, "Q4_0", nBlocks, scales)
	bs.CodeEntropy = shannonEntropy(codeCounts, nBlocks*elemsPerBlock)
	bs.SaturationLow = float64(codeCounts[0]) / float64(nBlocks*elemsPerBlock)
	bs.SaturationHigh = float64(codeCounts[15]) / float64(nBlocks*elemsPerBlock)
	bs.RepeatedBlocks = countRepeatedBlocks(data, blockSize)
	return bs
}

func extractQ4_1Blocks(data []byte, name string) *BlockStats {
	const blockSize = 20
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil
	}

	scales := make([]float32, nBlocks)
	codeCounts := make([]int, 16)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		scales[bi] = F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		for qi := 0; qi < 16; qi++ {
			lo := data[off+4+qi] & 0x0F
			hi := data[off+4+qi] >> 4
			codeCounts[lo]++
			codeCounts[hi]++
		}
	}

	bs := buildBlockStats(name, "Q4_1", nBlocks, scales)
	bs.CodeEntropy = shannonEntropy(codeCounts, nBlocks*elemsPerBlock)
	bs.SaturationLow = float64(codeCounts[0]) / float64(nBlocks*elemsPerBlock)
	bs.SaturationHigh = float64(codeCounts[15]) / float64(nBlocks*elemsPerBlock)
	bs.RepeatedBlocks = countRepeatedBlocks(data, blockSize)
	return bs
}

func extractQ4_KBlocks(data []byte, name string) *BlockStats {
	const blockSize = 144
	const elemsPerBlock = 256
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil
	}

	// Collect super-block scales (d values)
	scales := make([]float32, nBlocks)
	// Also collect sub-block scales for richer analysis
	var subScales []float32
	codeCounts := make([]int, 16)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		scales[bi] = d
		scalesData := data[off+4 : off+16]

		for j := 0; j < 8; j++ {
			var sc float32
			if j < 4 {
				sc = d * float32(scalesData[j]&63)
			} else {
				scBits := uint8(scalesData[j+4]&0x0F) | uint8((scalesData[j-4]>>6)<<4)
				sc = d * float32(scBits)
			}
			subScales = append(subScales, sc)

			for qi := 0; qi < 16; qi++ {
				qOff := off + 16 + j*16 + qi
				lo := data[qOff] & 0x0F
				hi := data[qOff] >> 4
				codeCounts[lo]++
				codeCounts[hi]++
			}
		}
	}

	bs := buildBlockStats(name, "Q4_K", nBlocks, scales)
	bs.CodeEntropy = shannonEntropy(codeCounts, nBlocks*elemsPerBlock)
	bs.SaturationLow = float64(codeCounts[0]) / float64(nBlocks*elemsPerBlock)
	bs.SaturationHigh = float64(codeCounts[15]) / float64(nBlocks*elemsPerBlock)
	bs.RepeatedBlocks = countRepeatedBlocks(data, blockSize)
	return bs
}

func extractQ5_KBlocks(data []byte, name string) *BlockStats {
	const blockSize = 176
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil
	}

	scales := make([]float32, nBlocks)
	codeCounts := make([]int, 32) // 5-bit range

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		scales[bi] = F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		qh := data[off+16 : off+48]
		qs := data[off+48 : off+176]

		for j := 0; j < 8; j++ {
			for qi := 0; qi < 32; qi++ {
				elemIdx := j*32 + qi
				qOff := j*16 + qi/2
				var q uint8
				if qi%2 == 0 {
					q = qs[qOff] & 0x0F
				} else {
					q = qs[qOff] >> 4
				}
				hiBit := uint8((qh[elemIdx/8] >> uint(elemIdx%8)) & 1)
				q |= hiBit << 4
				codeCounts[q]++
			}
		}
	}

	bs := buildBlockStats(name, "Q5_K", nBlocks, scales)
	bs.CodeEntropy = shannonEntropy(codeCounts, nBlocks*256)
	bs.SaturationLow = float64(codeCounts[0]) / float64(nBlocks*256)
	bs.SaturationHigh = float64(codeCounts[31]) / float64(nBlocks*256)
	bs.RepeatedBlocks = countRepeatedBlocks(data, blockSize)
	return bs
}

func extractQ6_KBlocks(data []byte, name string) *BlockStats {
	const blockSize = 210
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil
	}

	scales := make([]float32, nBlocks)
	codeCounts := make([]int, 64) // 6-bit range

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off+208:]))
		scales[bi] = d
		ql := data[off : off+128]
		qh := data[off+128 : off+192]

		for elemIdx := 0; elemIdx < 256; elemIdx++ {
			qlIdx := elemIdx / 2
			var lo uint8
			if elemIdx%2 == 0 {
				lo = ql[qlIdx] & 0x0F
			} else {
				lo = ql[qlIdx] >> 4
			}
			qhIdx := elemIdx / 4
			qhShift := uint(elemIdx%4) * 2
			hi := (qh[qhIdx] >> qhShift) & 0x03
			q := lo | (hi << 4)
			codeCounts[q]++
		}
	}

	bs := buildBlockStats(name, "Q6_K", nBlocks, scales)
	bs.CodeEntropy = shannonEntropy(codeCounts, nBlocks*256)
	bs.SaturationLow = float64(codeCounts[0]) / float64(nBlocks*256)
	bs.SaturationHigh = float64(codeCounts[63]) / float64(nBlocks*256)
	bs.RepeatedBlocks = countRepeatedBlocks(data, blockSize)
	return bs
}

// buildBlockStats computes scale statistics from extracted per-block scales.
func buildBlockStats(name, quantType string, nBlocks int, scales []float32) *BlockStats {
	bs := &BlockStats{
		TensorName: name,
		QuantType:  quantType,
		NumBlocks:  nBlocks,
		Scales:     scales,
	}

	if len(scales) == 0 {
		return bs
	}

	bs.ScaleMin = scales[0]
	bs.ScaleMax = scales[0]
	var sum float64
	for _, s := range scales {
		sum += float64(s)
		if s < bs.ScaleMin {
			bs.ScaleMin = s
		}
		if s > bs.ScaleMax {
			bs.ScaleMax = s
		}
		if math.Abs(float64(s)) < 1e-10 {
			bs.ZeroScales++
		}
	}
	bs.ScaleMean = sum / float64(len(scales))

	var m2 float64
	for _, s := range scales {
		d := float64(s) - bs.ScaleMean
		m2 += d * d
	}
	bs.ScaleStdDev = math.Sqrt(m2 / float64(len(scales)))

	// Scale entropy: histogram of quantized scale values
	bs.ScaleEntropy = scaleEntropy(scales)

	// Scale ratio
	if math.Abs(float64(bs.ScaleMin)) > 1e-10 {
		bs.ScaleRatio = float64(bs.ScaleMax) / float64(bs.ScaleMin)
	}

	return bs
}

// shannonEntropy computes Shannon entropy from a histogram of counts.
func shannonEntropy(counts []int, total int) float64 {
	if total == 0 {
		return 0
	}
	var entropy float64
	n := float64(total)
	for _, c := range counts {
		if c > 0 {
			p := float64(c) / n
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

// scaleEntropy bins scale values and computes Shannon entropy.
func scaleEntropy(scales []float32) float64 {
	if len(scales) < 2 {
		return 0
	}
	const nBins = 64
	min, max := float64(scales[0]), float64(scales[0])
	for _, s := range scales[1:] {
		v := float64(s)
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	rng := max - min
	if rng < 1e-10 {
		return 0 // all scales identical
	}

	bins := make([]int, nBins)
	for _, s := range scales {
		idx := int(float64(nBins-1) * (float64(s) - min) / rng)
		if idx >= nBins {
			idx = nBins - 1
		}
		if idx < 0 {
			idx = 0
		}
		bins[idx]++
	}
	return shannonEntropy(bins, len(scales))
}

// countRepeatedBlocks counts how many consecutive blocks are byte-identical.
func countRepeatedBlocks(data []byte, blockSize int) int {
	nBlocks := len(data) / blockSize
	if nBlocks < 2 {
		return 0
	}
	count := 0
	for i := 1; i < nBlocks; i++ {
		prev := data[(i-1)*blockSize : i*blockSize]
		curr := data[i*blockSize : (i+1)*blockSize]
		if bytesEqual(prev, curr) {
			count++
		}
	}
	return count
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ValidateBlockLayout checks that tensor data is consistent with the declared
// quantization type — correct block count, no truncation, no overlap.
func ValidateBlockLayout(ti *TensorInfo, dataSize int64) error {
	info, ok := typeInfoMap[ti.Type]
	if !ok {
		return fmt.Errorf("unknown quant type %s", ti.Type)
	}
	expectedSize := ti.ByteSize()
	if expectedSize == 0 {
		return fmt.Errorf("cannot compute expected size for %s", ti.Type)
	}
	if dataSize < expectedSize {
		return fmt.Errorf("tensor %q: data truncated: have %d bytes, need %d",
			ti.Name, dataSize, expectedSize)
	}

	if info.BlockSize > 1 {
		nBlocks := int64(ti.ElementCount) / int64(info.BlockSize)
		remainder := int64(ti.ElementCount) % int64(info.BlockSize)
		if remainder != 0 {
			return fmt.Errorf("tensor %q: element count %d not divisible by block size %d",
				ti.Name, ti.ElementCount, info.BlockSize)
		}
		expectedBytes := nBlocks * int64(info.TypeSize)
		if expectedBytes != expectedSize {
			return fmt.Errorf("tensor %q: size mismatch: %d blocks * %d = %d, but ByteSize() = %d",
				ti.Name, nBlocks, info.TypeSize, expectedBytes, expectedSize)
		}
	}
	return nil
}
