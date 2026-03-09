// Package analysis provides statistical analysis, fingerprinting, and anomaly
// detection for GGUF model weights.
package analysis

import (
	"math"
	"sort"
)

// TensorStats holds computed statistics for a single tensor.
type TensorStats struct {
	Name         string             `json:"name"`
	Type         string             `json:"type"`
	Shape        []uint64           `json:"shape"`
	ElementCount uint64             `json:"element_count"`
	Mean         float64            `json:"mean"`
	Variance     float64            `json:"variance"`
	StdDev       float64            `json:"std_dev"`
	Skewness     float64            `json:"skewness"`
	Kurtosis     float64            `json:"kurtosis"`
	Min          float64            `json:"min"`
	Max          float64            `json:"max"`
	AbsMean      float64            `json:"abs_mean"`
	ZeroFraction float64            `json:"zero_fraction"`
	OutlierRatio float64            `json:"outlier_ratio"`
	Percentiles  map[string]float64 `json:"percentiles"`
	Samples      int                `json:"samples"`
	InfCount     int                `json:"inf_count"`
	NaNCount     int                `json:"nan_count"`
}

// ComputeStats calculates comprehensive statistics from dequantized float32 values.
func ComputeStats(values []float32, name, typeName string, shape []uint64, elementCount uint64) *TensorStats {
	s := &TensorStats{
		Name:         name,
		Type:         typeName,
		Shape:        shape,
		ElementCount: elementCount,
	}

	if len(values) == 0 {
		return s
	}

	// Filter out NaN/Inf, counting them
	clean := make([]float64, 0, len(values))
	for _, v := range values {
		f := float64(v)
		if math.IsNaN(f) {
			s.NaNCount++
		} else if math.IsInf(f, 0) {
			s.InfCount++
		} else {
			clean = append(clean, f)
		}
	}

	n := len(clean)
	s.Samples = n
	if n == 0 {
		return s
	}

	// Mean, AbsMean, Min, Max, ZeroCount
	var sum, absSum float64
	s.Min = clean[0]
	s.Max = clean[0]
	var zeroCount int
	for _, v := range clean {
		sum += v
		absSum += math.Abs(v)
		if v < s.Min {
			s.Min = v
		}
		if v > s.Max {
			s.Max = v
		}
		if v == 0 {
			zeroCount++
		}
	}
	nf := float64(n)
	s.Mean = sum / nf
	s.AbsMean = absSum / nf
	s.ZeroFraction = float64(zeroCount) / nf

	// Variance, Skewness, Kurtosis (single-pass with central moments)
	var m2, m3, m4 float64
	for _, v := range clean {
		d := v - s.Mean
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		m4 += d2 * d2
	}
	s.Variance = m2 / nf
	s.StdDev = math.Sqrt(s.Variance)

	if s.Variance > 1e-30 {
		s.Skewness = (m3 / nf) / math.Pow(s.Variance, 1.5)
		s.Kurtosis = (m4/nf)/(s.Variance*s.Variance) - 3.0 // excess kurtosis
	}

	// Outlier ratio (beyond 3 sigma)
	if s.StdDev > 0 {
		var outliers int
		threshold := 3.0 * s.StdDev
		for _, v := range clean {
			if math.Abs(v-s.Mean) > threshold {
				outliers++
			}
		}
		s.OutlierRatio = float64(outliers) / nf
	}

	// Percentiles (on sorted data)
	sorted := make([]float64, n)
	copy(sorted, clean)
	sort.Float64s(sorted)
	s.Percentiles = map[string]float64{
		"p1":  percentile(sorted, 0.01),
		"p5":  percentile(sorted, 0.05),
		"p25": percentile(sorted, 0.25),
		"p50": percentile(sorted, 0.50),
		"p75": percentile(sorted, 0.75),
		"p95": percentile(sorted, 0.95),
		"p99": percentile(sorted, 0.99),
	}

	return s
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := p * float64(len(sorted)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi || hi >= len(sorted) {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}
