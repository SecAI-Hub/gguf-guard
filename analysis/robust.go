package analysis

import (
	"math"
	"sort"
)

// RobustStats holds outlier-resistant statistics that reduce false positives
// on naturally non-Gaussian tensors (e.g., attention weights, embeddings).
type RobustStats struct {
	Median       float64 `json:"median"`
	MAD          float64 `json:"mad"`           // Median Absolute Deviation
	TrimmedMean  float64 `json:"trimmed_mean"`  // 10% trimmed mean
	TukeyLowerQ1 float64 `json:"tukey_lower"`   // Q1 - 1.5*IQR
	TukeyUpperQ3 float64 `json:"tukey_upper"`   // Q3 + 1.5*IQR
	TukeyOutliers int    `json:"tukey_outliers"` // count outside Tukey fences
	IQR          float64 `json:"iqr"`
	P25          float64 `json:"p25"`
	P75          float64 `json:"p75"`
}

// ComputeRobustStats calculates outlier-resistant statistics from float64 values.
func ComputeRobustStats(values []float64) *RobustStats {
	if len(values) == 0 {
		return &RobustStats{}
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	n := len(sorted)
	rs := &RobustStats{}

	// Median
	rs.Median = medianSorted(sorted)

	// MAD (Median Absolute Deviation)
	absDevs := make([]float64, n)
	for i, v := range sorted {
		absDevs[i] = math.Abs(v - rs.Median)
	}
	sort.Float64s(absDevs)
	rs.MAD = medianSorted(absDevs)

	// Trimmed mean (10% from each end)
	trimFrac := 0.10
	trimCount := int(float64(n) * trimFrac)
	if trimCount >= n/2 {
		trimCount = 0 // not enough data to trim
	}
	var trimSum float64
	trimN := n - 2*trimCount
	for i := trimCount; i < n-trimCount; i++ {
		trimSum += sorted[i]
	}
	if trimN > 0 {
		rs.TrimmedMean = trimSum / float64(trimN)
	}

	// Quartiles and IQR
	rs.P25 = percentileSorted(sorted, 0.25)
	rs.P75 = percentileSorted(sorted, 0.75)
	rs.IQR = rs.P75 - rs.P25

	// Tukey fences
	rs.TukeyLowerQ1 = rs.P25 - 1.5*rs.IQR
	rs.TukeyUpperQ3 = rs.P75 + 1.5*rs.IQR

	// Count Tukey outliers
	for _, v := range sorted {
		if v < rs.TukeyLowerQ1 || v > rs.TukeyUpperQ3 {
			rs.TukeyOutliers++
		}
	}

	return rs
}

// ComputeRobustStatsF32 is a convenience wrapper for float32 slices.
func ComputeRobustStatsF32(values []float32) *RobustStats {
	f64 := make([]float64, len(values))
	for i, v := range values {
		f64[i] = float64(v)
	}
	return ComputeRobustStats(f64)
}

func medianSorted(sorted []float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return sorted[n/2]
}

func percentileSorted(sorted []float64, p float64) float64 {
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

// RobustZScore computes the modified z-score using median and MAD.
// This is more resistant to outliers than (x-mean)/std.
// Uses the conventional scaling factor 0.6745 for consistency with normal distribution.
func RobustZScore(value, median, mad float64) float64 {
	if mad < 1e-10 {
		if math.Abs(value-median) < 1e-10 {
			return 0
		}
		return math.Copysign(100, value-median) // effectively infinite
	}
	return 0.6745 * (value - median) / mad
}
