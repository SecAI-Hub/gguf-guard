package analysis

import (
	"testing"
)

func TestCompareLineageConsistent(t *testing.T) {
	sourceStats := []*TensorStats{
		{Name: "a", Mean: 0.01, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
		{Name: "b", Mean: 0.02, Variance: 0.2, StdDev: 0.447, Kurtosis: 1.5},
	}
	// Candidate with small drift (within Q4_K budget)
	candStats := []*TensorStats{
		{Name: "a", Mean: 0.02, Variance: 0.11, StdDev: 0.331, Kurtosis: 2.1},
		{Name: "b", Mean: 0.03, Variance: 0.21, StdDev: 0.458, Kurtosis: 1.6},
	}

	srcFP := &Fingerprint{FileHash: "aaaa1111bbbb2222cccc3333dddd4444", StructureHash: "same", QuantType: "F32"}
	candFP := &Fingerprint{FileHash: "eeee5555ffff6666aaaa7777bbbb8888", StructureHash: "same", QuantType: "Q4_K"}

	result := CompareLineage(sourceStats, candStats, srcFP, candFP)

	if result.Verdict != "consistent" {
		t.Errorf("verdict = %q, want consistent", result.Verdict)
	}
	if result.AnalyzedTensors != 2 {
		t.Errorf("analyzed = %d, want 2", result.AnalyzedTensors)
	}
}

func TestCompareLineageSuspicious(t *testing.T) {
	sourceStats := []*TensorStats{
		{Name: "a", Mean: 0.01, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0},
		{Name: "b", Mean: 0.02, Variance: 0.2, StdDev: 0.447, Kurtosis: 1.5},
	}
	// Candidate with extreme drift
	candStats := []*TensorStats{
		{Name: "a", Mean: 50.0, Variance: 10.0, StdDev: 3.16, Kurtosis: 200.0},
		{Name: "b", Mean: 50.0, Variance: 10.0, StdDev: 3.16, Kurtosis: 200.0},
	}

	srcFP := &Fingerprint{FileHash: "aaaa1111bbbb2222cccc3333dddd4444", StructureHash: "same", QuantType: "F32"}
	candFP := &Fingerprint{FileHash: "eeee5555ffff6666aaaa7777bbbb8888", StructureHash: "same", QuantType: "Q4_K"}

	result := CompareLineage(sourceStats, candStats, srcFP, candFP)

	if result.Verdict != "suspicious" {
		t.Errorf("verdict = %q, want suspicious", result.Verdict)
	}
	if len(result.UnexpectedDrift) == 0 {
		t.Error("expected unexpected drift entries")
	}
}

func TestCompareLineageNoMatch(t *testing.T) {
	srcFP := &Fingerprint{FileHash: "aaaa1111bbbb2222cccc3333dddd4444", StructureHash: "x", QuantType: "F32"}
	candFP := &Fingerprint{FileHash: "eeee5555ffff6666aaaa7777bbbb8888", StructureHash: "y", QuantType: "Q4_K"}

	result := CompareLineage(nil, nil, srcFP, candFP)
	if result.Verdict != "incompatible" {
		t.Errorf("verdict = %q, want incompatible", result.Verdict)
	}
}

func TestDefaultQuantLossBudgets(t *testing.T) {
	// F32 should have zero tolerance
	b := DefaultQuantLossBudgets["F32"]
	if b.MaxDistance != 0 {
		t.Errorf("F32 max distance = %v, want 0", b.MaxDistance)
	}

	// Q4_K should have more tolerance than Q8_0
	q4 := DefaultQuantLossBudgets["Q4_K"]
	q8 := DefaultQuantLossBudgets["Q8_0"]
	if q4.MaxDistance <= q8.MaxDistance {
		t.Errorf("Q4_K budget (%v) should be > Q8_0 (%v)", q4.MaxDistance, q8.MaxDistance)
	}
}
