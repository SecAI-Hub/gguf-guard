package analysis

import (
	"math"
	"testing"
)

func TestComputeRobustStatsBasic(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	rs := ComputeRobustStats(values)

	if math.Abs(rs.Median-5.5) > 0.01 {
		t.Errorf("median = %v, want 5.5", rs.Median)
	}
	if rs.IQR == 0 {
		t.Error("IQR should not be zero")
	}
	if rs.TukeyOutliers != 0 {
		t.Errorf("tukey outliers = %d, want 0", rs.TukeyOutliers)
	}
}

func TestComputeRobustStatsWithOutlier(t *testing.T) {
	values := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 100}
	rs := ComputeRobustStats(values)

	// 100 should be a Tukey outlier
	if rs.TukeyOutliers == 0 {
		t.Error("expected Tukey outlier for value 100")
	}

	// Trimmed mean should be much less affected by the outlier
	if rs.TrimmedMean > 20 {
		t.Errorf("trimmed mean = %v, should be < 20", rs.TrimmedMean)
	}
}

func TestComputeRobustStatsEmpty(t *testing.T) {
	rs := ComputeRobustStats(nil)
	if rs.Median != 0 {
		t.Errorf("empty median = %v", rs.Median)
	}
}

func TestComputeRobustStatsMAD(t *testing.T) {
	// Symmetric data: MAD = median of |x - median|
	values := []float64{1, 2, 3, 4, 5}
	rs := ComputeRobustStats(values)
	// Median = 3, deviations = [2, 1, 0, 1, 2], MAD = 1.0
	if math.Abs(rs.MAD-1.0) > 0.01 {
		t.Errorf("MAD = %v, want 1.0", rs.MAD)
	}
}

func TestRobustZScore(t *testing.T) {
	// Normal case
	z := RobustZScore(10.0, 5.0, 2.0)
	expected := 0.6745 * 5.0 / 2.0
	if math.Abs(z-expected) > 0.001 {
		t.Errorf("z = %v, want %v", z, expected)
	}

	// Zero MAD, same value
	z2 := RobustZScore(5.0, 5.0, 0.0)
	if z2 != 0 {
		t.Errorf("same value with zero MAD = %v, want 0", z2)
	}

	// Zero MAD, different value
	z3 := RobustZScore(10.0, 5.0, 0.0)
	if z3 != 100 {
		t.Errorf("diff with zero MAD = %v, want 100", z3)
	}
}

func TestComputeRobustStatsF32(t *testing.T) {
	values := []float32{1, 2, 3, 4, 5}
	rs := ComputeRobustStatsF32(values)
	if math.Abs(rs.Median-3.0) > 0.01 {
		t.Errorf("median = %v, want 3.0", rs.Median)
	}
}
