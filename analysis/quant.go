// Package analysis: quant.go provides quant-format-aware anomaly checks
// that operate on the raw quantization block structure, not dequantized values.
package analysis

import (
	"math"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

// QuantAnomaly represents an anomaly detected in the block-level quantization structure.
type QuantAnomaly struct {
	TensorName  string  `json:"tensor_name"`
	Type        string  `json:"type"`
	Severity    string  `json:"severity"`
	Value       float64 `json:"value"`
	Threshold   float64 `json:"threshold"`
	Description string  `json:"description"`
}

// QuantReport holds the results of block-level quantization analysis.
type QuantReport struct {
	TensorsAnalyzed int             `json:"tensors_analyzed"`
	Anomalies       []QuantAnomaly  `json:"anomalies,omitempty"`
	BlockStats      []*gguf.BlockStats `json:"block_stats,omitempty"`
}

// QuantThresholds defines limits for quant-aware anomaly detection.
type QuantThresholds struct {
	MinScaleEntropy     float64 `json:"min_scale_entropy"`      // below = suspiciously uniform scales
	MaxScaleRatio       float64 `json:"max_scale_ratio"`        // max/min scale ratio
	MinCodeEntropy      float64 `json:"min_code_entropy"`       // below = suspiciously non-uniform codes
	MaxSaturationRatio  float64 `json:"max_saturation_ratio"`   // bin extreme saturation
	MaxRepeatedFraction float64 `json:"max_repeated_fraction"`  // fraction of repeated blocks
	MaxZeroScaleFrac    float64 `json:"max_zero_scale_fraction"`
}

// DefaultQuantThresholds provides sensible defaults for quant-aware checks.
var DefaultQuantThresholds = QuantThresholds{
	MinScaleEntropy:     1.0,
	MaxScaleRatio:       1e6,
	MinCodeEntropy:      2.0,
	MaxSaturationRatio:  0.5,
	MaxRepeatedFraction: 0.1,
	MaxZeroScaleFrac:    0.5,
}

// AnalyzeQuantBlocks extracts and analyzes block-level quantization statistics
// for all quantized tensors in the file.
func AnalyzeQuantBlocks(gf *gguf.File, includeStats bool) (*QuantReport, error) {
	return AnalyzeQuantBlocksWithThresholds(gf, includeStats, DefaultQuantThresholds)
}

// AnalyzeQuantBlocksWithThresholds runs quant-aware analysis with custom thresholds.
func AnalyzeQuantBlocksWithThresholds(gf *gguf.File, includeStats bool, th QuantThresholds) (*QuantReport, error) {
	report := &QuantReport{}

	for i := range gf.Tensors {
		ti := &gf.Tensors[i]
		if !isQuantizedType(ti.Type) {
			continue
		}

		data, err := gguf.ReadTensorData(gf, ti, 0)
		if err != nil {
			continue
		}

		bs := gguf.ExtractBlockStats(data, ti.Name, ti.Type)
		if bs == nil {
			continue
		}

		report.TensorsAnalyzed++
		if includeStats {
			report.BlockStats = append(report.BlockStats, bs)
		}

		// Run anomaly checks on this tensor's block stats
		report.Anomalies = append(report.Anomalies, checkQuantAnomalies(bs, &th)...)
	}

	return report, nil
}

func checkQuantAnomalies(bs *gguf.BlockStats, th *QuantThresholds) []QuantAnomaly {
	var anomalies []QuantAnomaly

	if bs.NumBlocks == 0 {
		return anomalies
	}

	// Low scale entropy: scales are suspiciously uniform (possible constant injection)
	if bs.ScaleEntropy < th.MinScaleEntropy && bs.NumBlocks > 10 {
		anomalies = append(anomalies, QuantAnomaly{
			TensorName:  bs.TensorName,
			Type:        "low_scale_entropy",
			Severity:    SeverityWarning,
			Value:       bs.ScaleEntropy,
			Threshold:   th.MinScaleEntropy,
			Description: "quantization scales have suspiciously low entropy",
		})
	}

	// Extreme scale ratio
	if math.Abs(bs.ScaleRatio) > th.MaxScaleRatio && bs.ScaleRatio != 0 {
		anomalies = append(anomalies, QuantAnomaly{
			TensorName:  bs.TensorName,
			Type:        "extreme_scale_ratio",
			Severity:    SeverityWarning,
			Value:       bs.ScaleRatio,
			Threshold:   th.MaxScaleRatio,
			Description: "max/min scale ratio is extreme, possible localized perturbation",
		})
	}

	// Low code entropy: quantized codes are suspiciously non-uniform
	if bs.CodeEntropy < th.MinCodeEntropy && bs.NumBlocks > 10 {
		anomalies = append(anomalies, QuantAnomaly{
			TensorName:  bs.TensorName,
			Type:        "low_code_entropy",
			Severity:    SeverityWarning,
			Value:       bs.CodeEntropy,
			Threshold:   th.MinCodeEntropy,
			Description: "quantized code distribution has low entropy",
		})
	}

	// High saturation: too many codes at bin extremes
	if bs.SaturationLow > th.MaxSaturationRatio {
		anomalies = append(anomalies, QuantAnomaly{
			TensorName:  bs.TensorName,
			Type:        "high_saturation_low",
			Severity:    SeverityWarning,
			Value:       bs.SaturationLow,
			Threshold:   th.MaxSaturationRatio,
			Description: "excessive codes at minimum bin extreme",
		})
	}
	if bs.SaturationHigh > th.MaxSaturationRatio {
		anomalies = append(anomalies, QuantAnomaly{
			TensorName:  bs.TensorName,
			Type:        "high_saturation_high",
			Severity:    SeverityWarning,
			Value:       bs.SaturationHigh,
			Threshold:   th.MaxSaturationRatio,
			Description: "excessive codes at maximum bin extreme",
		})
	}

	// Repeated blocks: duplicate quantization blocks
	repeatedFrac := float64(bs.RepeatedBlocks) / float64(bs.NumBlocks)
	if repeatedFrac > th.MaxRepeatedFraction {
		anomalies = append(anomalies, QuantAnomaly{
			TensorName:  bs.TensorName,
			Type:        "repeated_blocks",
			Severity:    SeverityCritical,
			Value:       repeatedFrac,
			Threshold:   th.MaxRepeatedFraction,
			Description: "high fraction of byte-identical quantization blocks",
		})
	}

	// Zero scales
	zeroFrac := float64(bs.ZeroScales) / float64(bs.NumBlocks)
	if zeroFrac > th.MaxZeroScaleFrac {
		anomalies = append(anomalies, QuantAnomaly{
			TensorName:  bs.TensorName,
			Type:        "excessive_zero_scales",
			Severity:    SeverityWarning,
			Value:       zeroFrac,
			Threshold:   th.MaxZeroScaleFrac,
			Description: "excessive blocks with near-zero scales",
		})
	}

	return anomalies
}

// isQuantizedType returns true for block-quantized types (not F32/F16/BF16).
func isQuantizedType(t gguf.GGMLType) bool {
	switch t {
	case gguf.TypeQ4_0, gguf.TypeQ4_1, gguf.TypeQ5_0, gguf.TypeQ5_1,
		gguf.TypeQ8_0, gguf.TypeQ8_1,
		gguf.TypeQ2_K, gguf.TypeQ3_K, gguf.TypeQ4_K, gguf.TypeQ5_K,
		gguf.TypeQ6_K, gguf.TypeQ8_K:
		return true
	}
	return false
}
