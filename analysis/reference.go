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

	// Multi-sample provenance (v2)
	SampleCount  int                `json:"sample_count,omitempty"`
	SourceHashes []string           `json:"source_hashes,omitempty"`
	Provenance   *ProfileProvenance `json:"provenance,omitempty"`
}

// ProfileProvenance records the toolchain and conversion context used
// to generate the models this profile was built from.
type ProfileProvenance struct {
	ConverterVersion string `json:"converter_version,omitempty"` // e.g., "convert_hf_to_gguf.py"
	LlamaCppCommit   string `json:"llama_cpp_commit,omitempty"`
	QuantizerFlags   string `json:"quantizer_flags,omitempty"`
	IMatrixUsed      bool   `json:"imatrix_used,omitempty"`
	CreatedAt        string `json:"created_at,omitempty"`
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

// MergeProfiles combines multiple reference profiles (from different clean samples)
// into a single profile with wider, more robust ranges.
func MergeProfiles(profiles []*ReferenceProfile, margin float64) *ReferenceProfile {
	if len(profiles) == 0 {
		return nil
	}
	base := profiles[0]

	merged := &ReferenceProfile{
		Name:           base.Name + "-merged",
		Architecture:   base.Architecture,
		Family:         base.Family,
		QuantType:      base.QuantType,
		StructureHash:  base.StructureHash,
		ParameterCount: base.ParameterCount,
		TensorProfiles: make(map[string]*TensorProfile),
		SampleCount:    len(profiles),
	}

	// Collect source hashes
	for _, p := range profiles {
		merged.SourceHashes = append(merged.SourceHashes, p.SourceHash)
	}

	// For each tensor, gather ranges from all profiles and compute the envelope
	allTensors := make(map[string]bool)
	for _, p := range profiles {
		for name := range p.TensorProfiles {
			allTensors[name] = true
		}
	}

	for name := range allTensors {
		var means, vars, kurts []float64
		for _, p := range profiles {
			tp, ok := p.TensorProfiles[name]
			if !ok {
				continue
			}
			// Use midpoints of each profile's range as samples
			means = append(means, (tp.MeanRange[0]+tp.MeanRange[1])/2)
			vars = append(vars, (tp.VarianceRange[0]+tp.VarianceRange[1])/2)
			kurts = append(kurts, (tp.KurtosisRange[0]+tp.KurtosisRange[1])/2)
		}

		if len(means) == 0 {
			continue
		}

		merged.TensorProfiles[name] = &TensorProfile{
			MeanRange:     rangeFromValues(means, margin),
			VarianceRange: rangeFromValues(vars, margin),
			KurtosisRange: rangeFromValues(kurts, margin),
		}
	}

	return merged
}

// MatchConfidence determines how well a reference profile matches a given fingerprint.
// Returns a confidence level: "high", "medium", "low", or "insufficient-data".
func MatchConfidence(ref *ReferenceProfile, fp *Fingerprint) string {
	if ref == nil {
		return "insufficient-data"
	}

	exactStructure := ref.StructureHash == fp.StructureHash
	sameQuant := ref.QuantType == fp.QuantType
	sameArch := ref.Architecture == fp.Architecture
	hasProvenance := ref.Provenance != nil
	multiSample := ref.SampleCount > 1

	switch {
	case exactStructure && sameQuant && hasProvenance && multiSample:
		return "high"
	case exactStructure && sameQuant:
		return "high"
	case sameArch && sameQuant:
		return "medium"
	case sameArch:
		return "low"
	default:
		return "insufficient-data"
	}
}
