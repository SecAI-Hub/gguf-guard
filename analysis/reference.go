package analysis

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// ReferenceProfile defines expected statistical ranges for a model architecture.
// Generated from known-good models, used for comparison-based anomaly detection.
type ReferenceProfile struct {
	Name           string                    `json:"name"`
	Architecture   string                    `json:"architecture"`
	Family         string                    `json:"family"`
	QuantType      string                    `json:"quant_type"`
	StructureHash  string                    `json:"structure_hash"`
	ParameterCount uint64                    `json:"parameter_count"`
	Thresholds     *Thresholds               `json:"thresholds,omitempty"`
	TensorProfiles map[string]*TensorProfile `json:"tensor_profiles"`
	CreatedFrom    string                    `json:"created_from"`
	SourceHash     string                    `json:"source_hash"`
}

// TensorProfile holds acceptable ranges for a tensor's statistics.
type TensorProfile struct {
	MeanRange     [2]float64 `json:"mean_range"`
	VarianceRange [2]float64 `json:"variance_range"`
	KurtosisRange [2]float64 `json:"kurtosis_range"`
}

// LoadReference reads a reference profile from a JSON file.
func LoadReference(path string) (*ReferenceProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read reference: %w", err)
	}
	var ref ReferenceProfile
	if err := json.Unmarshal(data, &ref); err != nil {
		return nil, fmt.Errorf("parse reference: %w", err)
	}
	return &ref, nil
}

// SaveReference writes a reference profile to a JSON file.
func SaveReference(path string, ref *ReferenceProfile) error {
	data, err := json.MarshalIndent(ref, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal reference: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// ProfileFromStats generates a reference profile from computed tensor statistics.
// The margin parameter (e.g., 3.0) defines how many standard deviations
// around the measured values to use as the acceptable range.
func ProfileFromStats(stats []*TensorStats, fp *Fingerprint, margin float64) *ReferenceProfile {
	ref := &ReferenceProfile{
		Name:           fp.Architecture + "-" + fp.QuantType,
		Architecture:   fp.Architecture,
		QuantType:      fp.QuantType,
		StructureHash:  fp.StructureHash,
		ParameterCount: fp.ParameterCount,
		TensorProfiles: make(map[string]*TensorProfile),
		CreatedFrom:    fp.FileHash,
		SourceHash:     fp.FileHash,
	}

	// Group tensors by role for computing per-role statistics
	roleStats := make(map[string][]*TensorStats)
	for _, s := range stats {
		role := extractRole(s.Name)
		if role != "" {
			roleStats[role] = append(roleStats[role], s)
		}
	}

	for _, s := range stats {
		role := extractRole(s.Name)
		group := roleStats[role]

		tp := &TensorProfile{}

		if len(group) >= 3 {
			// Use group statistics to define range (more robust)
			means := make([]float64, len(group))
			vars := make([]float64, len(group))
			kurts := make([]float64, len(group))
			for i, g := range group {
				means[i] = g.Mean
				vars[i] = g.Variance
				kurts[i] = g.Kurtosis
			}
			tp.MeanRange = rangeFromValues(means, margin)
			tp.VarianceRange = rangeFromValues(vars, margin)
			tp.KurtosisRange = rangeFromValues(kurts, margin)
		} else {
			// Single tensor: use percentage-based range
			tp.MeanRange = [2]float64{s.Mean - math.Abs(s.Mean)*margin, s.Mean + math.Abs(s.Mean)*margin}
			tp.VarianceRange = [2]float64{s.Variance * (1 - margin*0.1), s.Variance * (1 + margin*0.1)}
			tp.KurtosisRange = [2]float64{s.Kurtosis - margin*5, s.Kurtosis + margin*5}
		}

		ref.TensorProfiles[s.Name] = tp
	}

	return ref
}

func rangeFromValues(values []float64, margin float64) [2]float64 {
	if len(values) == 0 {
		return [2]float64{0, 0}
	}
	mean, std := meanAndStd(values)
	if std < 1e-10 {
		std = math.Abs(mean) * 0.1
		if std < 1e-10 {
			std = 1e-6
		}
	}
	return [2]float64{mean - margin*std, mean + margin*std}
}
