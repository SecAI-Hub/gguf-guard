package analysis

import (
	"fmt"
	"math"
)

// LineageResult describes the comparison between a trusted source model
// and a candidate GGUF, accounting for expected quantization loss.
type LineageResult struct {
	SourceHash     string        `json:"source_hash"`
	CandidateHash  string        `json:"candidate_hash"`
	SameStructure  bool          `json:"same_structure"`
	SourceQuant    string        `json:"source_quant"`
	CandidateQuant string        `json:"candidate_quant"`
	TotalTensors   int           `json:"total_tensors"`
	AnalyzedTensors int          `json:"analyzed_tensors"`
	ExpectedDrift  []TensorDrift `json:"expected_drift,omitempty"`
	UnexpectedDrift []TensorDrift `json:"unexpected_drift,omitempty"`
	Summary        string        `json:"summary"`
	Verdict        string        `json:"verdict"` // "consistent", "suspicious", "incompatible"
}

// TensorDrift describes whether a tensor's deviation from the source
// is within expected quantization loss or anomalous.
type TensorDrift struct {
	Name            string  `json:"name"`
	MeanDelta       float64 `json:"mean_delta"`
	VarDelta        float64 `json:"variance_delta"`
	KurtDelta       float64 `json:"kurtosis_delta"`
	Distance        float64 `json:"distance"`
	ExpectedMaxDist float64 `json:"expected_max_distance"`
	Verdict         string  `json:"verdict"` // "expected", "suspicious"
}

// QuantLossBudget defines the expected statistical drift for a quantization type.
// These are approximate upper bounds derived from empirical observation of
// well-formed quantizations.
type QuantLossBudget struct {
	MeanTolerance float64 // expected max |mean_delta| / |original_mean|
	VarTolerance  float64 // expected max |var_delta| / original_var
	KurtTolerance float64 // expected max |kurtosis_delta|
	MaxDistance    float64 // max normalized distance for "expected" drift
}

// DefaultQuantLossBudgets maps quantization type names to their expected loss budgets.
var DefaultQuantLossBudgets = map[string]QuantLossBudget{
	"F32":  {0.0, 0.0, 0.0, 0.0},    // identical
	"F16":  {0.001, 0.005, 0.5, 0.5}, // very small loss
	"BF16": {0.005, 0.02, 1.0, 1.0},
	"Q8_0": {0.01, 0.05, 2.0, 2.0},
	"Q6_K": {0.02, 0.10, 3.0, 3.0},
	"Q5_K": {0.03, 0.15, 4.0, 4.0},
	"Q5_0": {0.03, 0.15, 4.0, 4.0},
	"Q5_1": {0.03, 0.15, 4.0, 4.0},
	"Q4_K": {0.05, 0.20, 5.0, 5.0},
	"Q4_0": {0.05, 0.25, 6.0, 6.0},
	"Q4_1": {0.05, 0.25, 6.0, 6.0},
}

// CompareLineage compares a source (trusted) model against a candidate GGUF,
// normalizing for expected quantization loss. This distinguishes "this tensor
// drifted because of Q4_K quantization" from "this tensor drifted more than
// Q4_K should cause."
func CompareLineage(
	sourceStats, candStats []*TensorStats,
	sourceFP, candFP *Fingerprint,
) *LineageResult {
	result := &LineageResult{
		SourceHash:     sourceFP.FileHash[:16],
		CandidateHash:  candFP.FileHash[:16],
		SameStructure:  sourceFP.StructureHash == candFP.StructureHash,
		SourceQuant:    sourceFP.QuantType,
		CandidateQuant: candFP.QuantType,
		TotalTensors:   len(candStats),
	}

	// Determine the loss budget based on the candidate's quantization
	budget, ok := DefaultQuantLossBudgets[candFP.QuantType]
	if !ok {
		// Conservative fallback for unknown quant types
		budget = QuantLossBudget{0.10, 0.30, 8.0, 8.0}
	}

	// Build lookup for source stats
	sourceMap := make(map[string]*TensorStats)
	for _, s := range sourceStats {
		sourceMap[s.Name] = s
	}

	for _, cs := range candStats {
		ss, ok := sourceMap[cs.Name]
		if !ok {
			continue
		}
		result.AnalyzedTensors++

		drift := TensorDrift{
			Name:            cs.Name,
			MeanDelta:       cs.Mean - ss.Mean,
			VarDelta:        cs.Variance - ss.Variance,
			KurtDelta:       cs.Kurtosis - ss.Kurtosis,
			ExpectedMaxDist: budget.MaxDistance,
		}

		// Compute normalized distance
		drift.Distance = math.Sqrt(
			sqNormLineage(drift.MeanDelta, ss.StdDev) +
				sqNormLineage(drift.VarDelta, ss.Variance) +
				sqNormLineage(drift.KurtDelta, math.Max(math.Abs(ss.Kurtosis), 1.0)),
		)

		if drift.Distance <= budget.MaxDistance {
			drift.Verdict = "expected"
			result.ExpectedDrift = append(result.ExpectedDrift, drift)
		} else {
			drift.Verdict = "suspicious"
			result.UnexpectedDrift = append(result.UnexpectedDrift, drift)
		}
	}

	// Generate verdict
	unexpectedCount := len(result.UnexpectedDrift)
	if result.AnalyzedTensors == 0 {
		result.Verdict = "incompatible"
		result.Summary = "no matching tensors between source and candidate"
	} else if unexpectedCount == 0 {
		result.Verdict = "consistent"
		result.Summary = fmt.Sprintf(
			"all %d tensors within expected %s quantization drift",
			result.AnalyzedTensors, candFP.QuantType)
	} else {
		suspiciousFrac := float64(unexpectedCount) / float64(result.AnalyzedTensors)
		if suspiciousFrac > 0.1 {
			result.Verdict = "suspicious"
		} else {
			result.Verdict = "consistent" // a few outliers is normal
		}
		result.Summary = fmt.Sprintf(
			"%d/%d tensors exceed expected %s drift (%.1f%%)",
			unexpectedCount, result.AnalyzedTensors, candFP.QuantType, suspiciousFrac*100)
	}

	return result
}

func sqNormLineage(delta, scale float64) float64 {
	if math.Abs(scale) < 1e-10 {
		if math.Abs(delta) < 1e-10 {
			return 0
		}
		return delta * delta
	}
	n := delta / scale
	return n * n
}
