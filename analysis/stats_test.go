package analysis

import (
	"math"
	"testing"
)

func TestComputeStatsEmpty(t *testing.T) {
	s := ComputeStats(nil, "empty", "F32", []uint64{0}, 0)
	if s.Samples != 0 {
		t.Errorf("samples = %d, want 0", s.Samples)
	}
	if s.Mean != 0 {
		t.Errorf("mean = %v, want 0", s.Mean)
	}
}

func TestComputeStatsBasic(t *testing.T) {
	// Values: 1, 2, 3, 4, 5 -> mean=3, variance=2
	values := []float32{1, 2, 3, 4, 5}
	s := ComputeStats(values, "test", "F32", []uint64{5}, 5)

	if s.Samples != 5 {
		t.Errorf("samples = %d, want 5", s.Samples)
	}
	if math.Abs(s.Mean-3.0) > 1e-6 {
		t.Errorf("mean = %v, want 3.0", s.Mean)
	}
	if math.Abs(s.Variance-2.0) > 1e-6 {
		t.Errorf("variance = %v, want 2.0", s.Variance)
	}
	if math.Abs(s.StdDev-math.Sqrt(2.0)) > 1e-6 {
		t.Errorf("stddev = %v, want %v", s.StdDev, math.Sqrt(2.0))
	}
	if s.Min != 1.0 {
		t.Errorf("min = %v, want 1", s.Min)
	}
	if s.Max != 5.0 {
		t.Errorf("max = %v, want 5", s.Max)
	}
}

func TestComputeStatsZeros(t *testing.T) {
	values := make([]float32, 100)
	s := ComputeStats(values, "zeros", "F32", []uint64{100}, 100)

	if s.ZeroFraction != 1.0 {
		t.Errorf("zero_fraction = %v, want 1.0", s.ZeroFraction)
	}
	if s.Variance != 0 {
		t.Errorf("variance = %v, want 0", s.Variance)
	}
}

func TestComputeStatsNaN(t *testing.T) {
	values := []float32{1, float32(math.NaN()), 3, float32(math.Inf(1)), 5}
	s := ComputeStats(values, "mixed", "F32", []uint64{5}, 5)

	if s.NaNCount != 1 {
		t.Errorf("nan_count = %d, want 1", s.NaNCount)
	}
	if s.InfCount != 1 {
		t.Errorf("inf_count = %d, want 1", s.InfCount)
	}
	if s.Samples != 3 {
		t.Errorf("samples = %d, want 3 (clean)", s.Samples)
	}
}

func TestComputeStatsPercentiles(t *testing.T) {
	values := make([]float32, 1000)
	for i := range values {
		values[i] = float32(i)
	}
	s := ComputeStats(values, "range", "F32", []uint64{1000}, 1000)

	p50 := s.Percentiles["p50"]
	if math.Abs(p50-499.5) > 1.0 {
		t.Errorf("p50 = %v, want ~499.5", p50)
	}
	p1 := s.Percentiles["p1"]
	if p1 > 20 {
		t.Errorf("p1 = %v, want <20", p1)
	}
	p99 := s.Percentiles["p99"]
	if p99 < 980 {
		t.Errorf("p99 = %v, want >980", p99)
	}
}

func TestComputeStatsOutliers(t *testing.T) {
	// 100 normal values near 0, plus one extreme outlier
	values := make([]float32, 100)
	for i := range values {
		values[i] = float32(i%3) * 0.1
	}
	values[99] = 1000.0 // extreme outlier

	s := ComputeStats(values, "outlier", "F32", []uint64{100}, 100)
	if s.OutlierRatio == 0 {
		t.Error("expected nonzero outlier ratio")
	}
}

func TestComputeStatsSymmetric(t *testing.T) {
	// Symmetric distribution: skewness should be ~0
	values := []float32{-3, -2, -1, 0, 1, 2, 3}
	s := ComputeStats(values, "sym", "F32", []uint64{7}, 7)

	if math.Abs(s.Skewness) > 0.1 {
		t.Errorf("skewness = %v, want ~0 for symmetric data", s.Skewness)
	}
	if math.Abs(s.Mean) > 1e-6 {
		t.Errorf("mean = %v, want 0", s.Mean)
	}
}
