package analysis

import (
	"math"
	"testing"
)

func TestCompareIdentical(t *testing.T) {
	stats := []*TensorStats{
		{Name: "a", Mean: 0.01, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
		{Name: "b", Mean: 0.02, Variance: 0.2, StdDev: 0.447, Kurtosis: 1.5},
	}
	fp := &Fingerprint{FileHash: "abcd1234abcd1234abcd1234abcd1234", StructureHash: "xyz"}

	result := Compare(stats, stats, fp, fp)
	if !result.SameStruct {
		t.Error("identical models should have same structure")
	}
	if result.MaxDistance != 0 {
		t.Errorf("max distance = %v, want 0", result.MaxDistance)
	}
	if len(result.Diffs) != 0 {
		t.Errorf("diffs = %d, want 0", len(result.Diffs))
	}
	if len(result.MissingIn) != 0 || len(result.ExtraIn) != 0 {
		t.Error("no missing or extra expected")
	}
}

func TestCompareDifferent(t *testing.T) {
	baseStats := []*TensorStats{
		{Name: "a", Mean: 0.0, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
	}
	candStats := []*TensorStats{
		{Name: "a", Mean: 5.0, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
	}
	fp1 := &Fingerprint{FileHash: "aaaa1111bbbb2222cccc3333dddd4444", StructureHash: "same"}
	fp2 := &Fingerprint{FileHash: "eeee5555ffff6666aaaa7777bbbb8888", StructureHash: "same"}

	result := Compare(baseStats, candStats, fp1, fp2)
	if result.MaxDistance == 0 {
		t.Error("expected nonzero distance for different means")
	}
	// Mean shifted by 5.0 with stddev 0.316 -> normalized ~15.8, should be significant
	if len(result.Diffs) == 0 {
		t.Error("expected significant diffs")
	}
}

func TestCompareMissingExtra(t *testing.T) {
	baseStats := []*TensorStats{
		{Name: "a", Mean: 0.0, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
		{Name: "b", Mean: 0.0, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
	}
	candStats := []*TensorStats{
		{Name: "a", Mean: 0.0, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
		{Name: "c", Mean: 0.0, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
	}
	fp1 := &Fingerprint{FileHash: "aaaa1111bbbb2222cccc3333dddd4444", StructureHash: "x"}
	fp2 := &Fingerprint{FileHash: "eeee5555ffff6666aaaa7777bbbb8888", StructureHash: "y"}

	result := Compare(baseStats, candStats, fp1, fp2)
	if !result.SameStruct {
		// StructureHash differs so this should be false
		t.Log("correctly detected different structures")
	}
	if len(result.MissingIn) != 1 || result.MissingIn[0] != "b" {
		t.Errorf("missing = %v, want [b]", result.MissingIn)
	}
	if len(result.ExtraIn) != 1 || result.ExtraIn[0] != "c" {
		t.Errorf("extra = %v, want [c]", result.ExtraIn)
	}
}

func TestCompareStructureDifference(t *testing.T) {
	fp1 := &Fingerprint{FileHash: "aaaa1111bbbb2222cccc3333dddd4444", StructureHash: "struct_a"}
	fp2 := &Fingerprint{FileHash: "eeee5555ffff6666aaaa7777bbbb8888", StructureHash: "struct_b"}

	result := Compare(nil, nil, fp1, fp2)
	if result.SameStruct {
		t.Error("different structure hashes should yield SameStruct=false")
	}
}

func TestSqNorm(t *testing.T) {
	// Normal case: (2/4)^2 = 0.25
	got := sqNorm(2.0, 4.0)
	if math.Abs(got-0.25) > 1e-10 {
		t.Errorf("sqNorm(2,4) = %v, want 0.25", got)
	}

	// Near-zero scale with near-zero delta -> 0
	got = sqNorm(1e-15, 1e-15)
	if got != 0 {
		t.Errorf("sqNorm(tiny, tiny) = %v, want 0", got)
	}

	// Near-zero scale with real delta -> raw
	got = sqNorm(3.0, 1e-15)
	if math.Abs(got-9.0) > 1e-10 {
		t.Errorf("sqNorm(3, tiny) = %v, want 9", got)
	}
}
