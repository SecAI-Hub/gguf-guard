package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Dequantize converts raw quantized tensor bytes into float32 values.
// For large tensors, pass maxElements > 0 to sample evenly across the data.
func Dequantize(data []byte, qtype GGMLType, maxElements int) ([]float32, error) {
	switch qtype {
	case TypeF32:
		return dequantF32(data, maxElements)
	case TypeF16:
		return dequantF16(data, maxElements)
	case TypeBF16:
		return dequantBF16(data, maxElements)
	case TypeQ8_0:
		return dequantQ8_0(data, maxElements)
	case TypeQ4_0:
		return dequantQ4_0(data, maxElements)
	case TypeQ4_1:
		return dequantQ4_1(data, maxElements)
	case TypeQ5_0:
		return dequantQ5_0(data, maxElements)
	case TypeQ5_1:
		return dequantQ5_1(data, maxElements)
	case TypeQ4_K:
		return dequantQ4_K(data, maxElements)
	case TypeQ5_K:
		return dequantQ5_K(data, maxElements)
	case TypeQ6_K:
		return dequantQ6_K(data, maxElements)
	default:
		return nil, fmt.Errorf("dequantization not supported for type %s", qtype)
	}
}

func dequantF32(data []byte, maxElements int) ([]float32, error) {
	count := len(data) / 4
	if maxElements > 0 && count > maxElements {
		return sampleF32(data, count, maxElements), nil
	}
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

func dequantF16(data []byte, maxElements int) ([]float32, error) {
	count := len(data) / 2
	if maxElements > 0 && count > maxElements {
		return sampleF16(data, count, maxElements), nil
	}
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		out[i] = F16ToF32(binary.LittleEndian.Uint16(data[i*2:]))
	}
	return out, nil
}

func dequantBF16(data []byte, maxElements int) ([]float32, error) {
	count := len(data) / 2
	if maxElements > 0 && count > maxElements {
		return sampleBF16(data, count, maxElements), nil
	}
	out := make([]float32, count)
	for i := 0; i < count; i++ {
		out[i] = BF16ToF32(binary.LittleEndian.Uint16(data[i*2:]))
	}
	return out, nil
}

// Q8_0: 34 bytes per block of 32 (2-byte f16 scale + 32 int8 quants)
func dequantQ8_0(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 34
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q8_0: data too short")
	}

	step := 1
	if maxElements > 0 {
		totalElems := nBlocks * elemsPerBlock
		if totalElems > maxElements {
			step = totalElems / maxElements
			if step < 1 {
				step = 1
			}
		}
	}

	out := make([]float32, 0, nBlocks*elemsPerBlock/step)
	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		for qi := 0; qi < elemsPerBlock; qi++ {
			idx := bi*elemsPerBlock + qi
			if step > 1 && idx%step != 0 {
				continue
			}
			q := int8(data[off+2+qi])
			out = append(out, d*float32(q))
		}
	}
	return out, nil
}

// Q4_0: 18 bytes per block of 32 (2-byte f16 scale + 16 bytes of 4-bit quants)
func dequantQ4_0(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 18
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q4_0: data too short")
	}

	step := 1
	if maxElements > 0 {
		totalElems := nBlocks * elemsPerBlock
		if totalElems > maxElements {
			step = totalElems / maxElements
			if step < 1 {
				step = 1
			}
		}
	}

	out := make([]float32, 0, nBlocks*elemsPerBlock/step)
	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		for qi := 0; qi < elemsPerBlock; qi++ {
			idx := bi*elemsPerBlock + qi
			if step > 1 && idx%step != 0 {
				continue
			}
			byteIdx := qi / 2
			var q uint8
			if qi%2 == 0 {
				q = data[off+2+byteIdx] & 0x0F
			} else {
				q = data[off+2+byteIdx] >> 4
			}
			out = append(out, d*(float32(q)-8.0))
		}
	}
	return out, nil
}

// Q4_1: 20 bytes per block of 32 (2-byte f16 d, 2-byte f16 m, 16 bytes quants)
func dequantQ4_1(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 20
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q4_1: data too short")
	}

	step := blockStepSize(nBlocks, elemsPerBlock, maxElements)
	out := make([]float32, 0, nBlocks*elemsPerBlock/step)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		m := F16ToF32(binary.LittleEndian.Uint16(data[off+2:]))
		for qi := 0; qi < elemsPerBlock; qi++ {
			idx := bi*elemsPerBlock + qi
			if step > 1 && idx%step != 0 {
				continue
			}
			byteIdx := qi / 2
			var q uint8
			if qi%2 == 0 {
				q = data[off+4+byteIdx] & 0x0F
			} else {
				q = data[off+4+byteIdx] >> 4
			}
			out = append(out, d*float32(q)+m)
		}
	}
	return out, nil
}

// Q5_0: 22 bytes per block of 32 (2-byte f16 d, 4-byte high bits, 16 bytes low nibbles)
func dequantQ5_0(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 22
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q5_0: data too short")
	}

	step := blockStepSize(nBlocks, elemsPerBlock, maxElements)
	out := make([]float32, 0, nBlocks*elemsPerBlock/step)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		qh := binary.LittleEndian.Uint32(data[off+2:])
		for qi := 0; qi < elemsPerBlock; qi++ {
			idx := bi*elemsPerBlock + qi
			if step > 1 && idx%step != 0 {
				continue
			}
			byteIdx := qi / 2
			var q uint8
			if qi%2 == 0 {
				q = data[off+6+byteIdx] & 0x0F
			} else {
				q = data[off+6+byteIdx] >> 4
			}
			// Add the 5th bit from qh
			hiBit := uint8((qh >> uint(qi)) & 1)
			q |= hiBit << 4
			out = append(out, d*(float32(q)-16.0))
		}
	}
	return out, nil
}

// Q5_1: 24 bytes per block of 32 (2-byte d, 2-byte m, 4-byte high bits, 16 bytes)
func dequantQ5_1(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 24
	const elemsPerBlock = 32
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q5_1: data too short")
	}

	step := blockStepSize(nBlocks, elemsPerBlock, maxElements)
	out := make([]float32, 0, nBlocks*elemsPerBlock/step)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		m := F16ToF32(binary.LittleEndian.Uint16(data[off+2:]))
		qh := binary.LittleEndian.Uint32(data[off+4:])
		for qi := 0; qi < elemsPerBlock; qi++ {
			idx := bi*elemsPerBlock + qi
			if step > 1 && idx%step != 0 {
				continue
			}
			byteIdx := qi / 2
			var q uint8
			if qi%2 == 0 {
				q = data[off+8+byteIdx] & 0x0F
			} else {
				q = data[off+8+byteIdx] >> 4
			}
			hiBit := uint8((qh >> uint(qi)) & 1)
			q |= hiBit << 4
			out = append(out, d*float32(q)+m)
		}
	}
	return out, nil
}

// Q4_K: 144 bytes per super-block of 256 elements
// Layout: 2-byte d, 2-byte dmin, 12-byte scales, 128-byte quants
func dequantQ4_K(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 144
	const elemsPerBlock = 256
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q4_K: data too short")
	}

	step := blockStepSize(nBlocks, elemsPerBlock, maxElements)
	out := make([]float32, 0, nBlocks*elemsPerBlock/step)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		dmin := F16ToF32(binary.LittleEndian.Uint16(data[off+2:]))
		scales := data[off+4 : off+16] // 12 bytes

		for j := 0; j < 8; j++ { // 8 sub-blocks of 32
			var sc, m float32
			if j < 4 {
				sc = d * float32(scales[j]&63)
				m = dmin * float32(scales[j+4]&63)
			} else {
				scBits := uint8(scales[j+4]&0x0F) | uint8((scales[j-4]>>6)<<4)
				mBits := uint8(scales[j+4]>>4) | uint8((scales[j]>>6)<<4)
				sc = d * float32(scBits)
				m = dmin * float32(mBits)
			}

			for qi := 0; qi < 32; qi++ {
				idx := bi*elemsPerBlock + j*32 + qi
				if step > 1 && idx%step != 0 {
					continue
				}
				qOff := off + 16 + j*16 + qi/2
				var q uint8
				if qi%2 == 0 {
					q = data[qOff] & 0x0F
				} else {
					q = data[qOff] >> 4
				}
				out = append(out, sc*float32(q)-m)
			}
		}
	}
	return out, nil
}

// Q5_K: 176 bytes per super-block of 256 elements
// Layout: 2-byte d, 2-byte dmin, 12-byte scales, 32-byte high bits, 128-byte quants
func dequantQ5_K(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 176
	const elemsPerBlock = 256
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q5_K: data too short")
	}

	step := blockStepSize(nBlocks, elemsPerBlock, maxElements)
	out := make([]float32, 0, nBlocks*elemsPerBlock/step)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		d := F16ToF32(binary.LittleEndian.Uint16(data[off:]))
		dmin := F16ToF32(binary.LittleEndian.Uint16(data[off+2:]))
		scales := data[off+4 : off+16]  // 12 bytes
		qh := data[off+16 : off+48]     // 32 bytes (high bits)
		qs := data[off+48 : off+176]    // 128 bytes (low nibbles)

		for j := 0; j < 8; j++ {
			var sc, m float32
			if j < 4 {
				sc = d * float32(scales[j]&63)
				m = dmin * float32(scales[j+4]&63)
			} else {
				scBits := uint8(scales[j+4]&0x0F) | uint8((scales[j-4]>>6)<<4)
				mBits := uint8(scales[j+4]>>4) | uint8((scales[j]>>6)<<4)
				sc = d * float32(scBits)
				m = dmin * float32(mBits)
			}

			for qi := 0; qi < 32; qi++ {
				elemIdx := j*32 + qi
				idx := bi*elemsPerBlock + elemIdx
				if step > 1 && idx%step != 0 {
					continue
				}
				qOff := j*16 + qi/2
				var q uint8
				if qi%2 == 0 {
					q = qs[qOff] & 0x0F
				} else {
					q = qs[qOff] >> 4
				}
				// 5th bit from qh
				hiBit := uint8((qh[elemIdx/8] >> uint(elemIdx%8)) & 1)
				q |= hiBit << 4
				out = append(out, sc*float32(q)-m)
			}
		}
	}
	return out, nil
}

// Q6_K: 210 bytes per super-block of 256 elements
// Layout: 128-byte ql (low 4 bits), 64-byte qh (high 2 bits), 16-byte scales, 2-byte d
func dequantQ6_K(data []byte, maxElements int) ([]float32, error) {
	const blockSize = 210
	const elemsPerBlock = 256
	nBlocks := len(data) / blockSize
	if nBlocks == 0 {
		return nil, fmt.Errorf("Q6_K: data too short")
	}

	step := blockStepSize(nBlocks, elemsPerBlock, maxElements)
	out := make([]float32, 0, nBlocks*elemsPerBlock/step)

	for bi := 0; bi < nBlocks; bi++ {
		off := bi * blockSize
		ql := data[off : off+128]
		qh := data[off+128 : off+192]
		scalesRaw := data[off+192 : off+208] // int8 scales for 16 sub-blocks of 16
		d := F16ToF32(binary.LittleEndian.Uint16(data[off+208:]))

		for j := 0; j < 16; j++ { // 16 sub-blocks of 16 elements
			sc := d * float32(int8(scalesRaw[j]))
			for qi := 0; qi < 16; qi++ {
				elemIdx := j*16 + qi
				idx := bi*elemsPerBlock + elemIdx
				if step > 1 && idx%step != 0 {
					continue
				}

				// Low 4 bits from ql (128 bytes, 2 values per byte)
				qlIdx := elemIdx / 2
				var lo uint8
				if elemIdx%2 == 0 {
					lo = ql[qlIdx] & 0x0F
				} else {
					lo = ql[qlIdx] >> 4
				}

				// High 2 bits from qh (64 bytes, 4 values per byte)
				qhIdx := elemIdx / 4
				qhShift := uint(elemIdx%4) * 2
				hi := (qh[qhIdx] >> qhShift) & 0x03

				q := lo | (hi << 4)
				out = append(out, sc*(float32(q)-32.0))
			}
		}
	}
	return out, nil
}

// blockStepSize calculates the sampling step to limit output to maxElements.
func blockStepSize(nBlocks, elemsPerBlock, maxElements int) int {
	if maxElements <= 0 {
		return 1
	}
	total := nBlocks * elemsPerBlock
	if total <= maxElements {
		return 1
	}
	step := total / maxElements
	if step < 1 {
		return 1
	}
	return step
}

// sampleF32 samples evenly from F32 data to produce at most maxElements values.
func sampleF32(data []byte, count, maxElements int) []float32 {
	step := count / maxElements
	if step < 1 {
		step = 1
	}
	out := make([]float32, 0, maxElements)
	for i := 0; i < count; i += step {
		v := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			continue
		}
		out = append(out, v)
	}
	return out
}

func sampleF16(data []byte, count, maxElements int) []float32 {
	step := count / maxElements
	if step < 1 {
		step = 1
	}
	out := make([]float32, 0, maxElements)
	for i := 0; i < count; i += step {
		v := F16ToF32(binary.LittleEndian.Uint16(data[i*2:]))
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			continue
		}
		out = append(out, v)
	}
	return out
}

func sampleBF16(data []byte, count, maxElements int) []float32 {
	step := count / maxElements
	if step < 1 {
		step = 1
	}
	out := make([]float32, 0, maxElements)
	for i := 0; i < count; i += step {
		v := BF16ToF32(binary.LittleEndian.Uint16(data[i*2:]))
		if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
			continue
		}
		out = append(out, v)
	}
	return out
}
