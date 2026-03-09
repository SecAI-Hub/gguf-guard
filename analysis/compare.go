package analysis

import (
	"fmt"
	"math"
)

// CompareResult holds the comparison between two models.
type CompareResult struct {
	Baseline    string           `json:"baseline"`
	Candidate   string           `json:"candidate"`
	SameStruct  bool             `json:"same_structure"`
	Diffs       []TensorDiff     `json:"diffs,omitempty"`
	MissingIn   []string         `json:"missing_in_candidate,omitempty"`
	ExtraIn     []string         `json:"extra_in_candidate,omitempty"`
	MaxDistance float64           `json:"max_distance"`
	MeanDistance float64          `json:"mean_distance"`
	Summary     string           `json:"summary"`
}

// TensorDiff shows how a tensor's statistics differ between two models.
type TensorDiff struct {
	Name         string  `json:"name"`
	MeanDelta    float64 `json:"mean_delta"`
	VarDelta     float64 `json:"variance_delta"`
	KurtDelta    float64 `json:"kurtosis_delta"`
	Distance     float64 `json:"distance"`
	Significant  bool    `json:"significant"`
}

// Compare analyzes the statistical differences between two sets of tensor stats.
func Compare(baseStats, candStats []*TensorStats, baseFP, candFP *Fingerprint) *CompareResult {
	result := &CompareResult{
		Baseline:   baseFP.FileHash[:16],
		Candidate:  candFP.FileHash[:16],
		SameStruct: baseFP.StructureHash == candFP.StructureHash,
	}

	// Build lookup maps
	baseMap := make(map[string]*TensorStats)
	for _, s := range baseStats {
		baseMap[s.Name] = s
	}
	candMap := make(map[string]*TensorStats)
	for _, s := range candStats {
		candMap[s.Name] = s
	}

	// Find missing and extra tensors
	for name := range baseMap {
		if _, ok := candMap[name]; !ok {
			result.MissingIn = append(result.MissingIn, name)
		}
	}
	for name := range candMap {
		if _, ok := baseMap[name]; !ok {
			result.ExtraIn = append(result.ExtraIn, name)
		}
	}

	// Compare shared tensors
	var totalDist float64
	var count int
	for name, bs := range baseMap {
		cs, ok := candMap[name]
		if !ok {
			continue
		}

		diff := TensorDiff{
			Name:      name,
			MeanDelta: cs.Mean - bs.Mean,
			VarDelta:  cs.Variance - bs.Variance,
			KurtDelta: cs.Kurtosis - bs.Kurtosis,
		}

		// Euclidean distance in the (normalized) statistical space
		diff.Distance = math.Sqrt(
			sqNorm(diff.MeanDelta, bs.StdDev) +
				sqNorm(diff.VarDelta, bs.Variance) +
				sqNorm(diff.KurtDelta, math.Max(math.Abs(bs.Kurtosis), 1.0)),
		)

		// Flag as significant if distance > 3.0 (roughly 3 sigma)
		diff.Significant = diff.Distance > 3.0

		if diff.Distance > result.MaxDistance {
			result.MaxDistance = diff.Distance
		}
		totalDist += diff.Distance
		count++

		if diff.Significant {
			result.Diffs = append(result.Diffs, diff)
		}
	}

	if count > 0 {
		result.MeanDistance = totalDist / float64(count)
	}

	sigCount := len(result.Diffs)
	if sigCount == 0 && len(result.MissingIn) == 0 && len(result.ExtraIn) == 0 {
		result.Summary = fmt.Sprintf("models are statistically consistent (mean dist: %.4f)", result.MeanDistance)
	} else {
		result.Summary = fmt.Sprintf(
			"%d significant diffs, %d missing, %d extra (max dist: %.4f)",
			sigCount, len(result.MissingIn), len(result.ExtraIn), result.MaxDistance,
		)
	}

	return result
}

// sqNorm computes (delta/scale)^2 for normalized distance.
func sqNorm(delta, scale float64) float64 {
	if scale < 1e-10 {
		if math.Abs(delta) < 1e-10 {
			return 0
		}
		return delta * delta // can't normalize, use raw
	}
	n := delta / scale
	return n * n
}
