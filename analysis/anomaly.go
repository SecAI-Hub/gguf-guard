package analysis

import (
	"fmt"
	"math"
	"strings"
)

// Severity levels for anomalies.
const (
	SeverityCritical = "critical"
	SeverityWarning  = "warning"
	SeverityInfo     = "info"
)

// Default anomaly thresholds. Override with a ReferenceProfile for
// architecture-specific tuning.
var DefaultThresholds = Thresholds{
	MaxAbsMean:       10.0,
	MaxKurtosis:      100.0,
	MinVariance:      1e-12,
	MaxZeroFraction:  0.99,
	MaxOutlierRatio:  0.10,
	MaxNaNFraction:   0.0,
	MaxInfFraction:   0.0,
	CrossLayerMaxDev: 5.0, // max std devs from group mean
}

// Thresholds holds configurable limits for anomaly detection.
type Thresholds struct {
	MaxAbsMean       float64 `json:"max_abs_mean"`
	MaxKurtosis      float64 `json:"max_kurtosis"`
	MinVariance      float64 `json:"min_variance"`
	MaxZeroFraction  float64 `json:"max_zero_fraction"`
	MaxOutlierRatio  float64 `json:"max_outlier_ratio"`
	MaxNaNFraction   float64 `json:"max_nan_fraction"`
	MaxInfFraction   float64 `json:"max_inf_fraction"`
	CrossLayerMaxDev float64 `json:"cross_layer_max_dev"`
}

// Anomaly represents a single detected anomaly.
type Anomaly struct {
	TensorName  string  `json:"tensor_name"`
	Type        string  `json:"type"`
	Severity    string  `json:"severity"`
	Value       float64 `json:"value"`
	Threshold   float64 `json:"threshold"`
	Description string  `json:"description"`
}

// AnomalyReport is the complete output of anomaly detection.
type AnomalyReport struct {
	Score      float64   `json:"score"`
	Confidence string    `json:"confidence"`
	Anomalies  []Anomaly `json:"anomalies"`
	Summary    string    `json:"summary"`
	Layers     *LayerScores `json:"layers,omitempty"`
}

// LayerScores breaks down suspiciousness by analysis layer.
type LayerScores struct {
	TensorLocal  float64 `json:"tensor_local"`  // per-tensor moment/outlier checks
	RoleGroup    float64 `json:"role_group"`     // cross-layer consistency
	ModelGlobal  float64 `json:"model_global"`   // overall distribution of suspicion
	Reference    float64 `json:"reference"`      // deviation from reference profile
}

// DetectAnomalies runs all anomaly checks on a set of tensor statistics.
// If ref is non-nil, architecture-specific thresholds are used.
func DetectAnomalies(stats []*TensorStats, ref *ReferenceProfile) *AnomalyReport {
	th := DefaultThresholds
	if ref != nil && ref.Thresholds != nil {
		th = *ref.Thresholds
	}

	report := &AnomalyReport{
		Confidence: "medium",
	}

	if len(stats) == 0 {
		report.Summary = "no tensors analyzed"
		report.Confidence = "insufficient-data"
		return report
	}

	// Per-tensor anomaly checks
	for _, s := range stats {
		report.Anomalies = append(report.Anomalies, checkTensorAnomalies(s, &th)...)
	}

	// Cross-layer consistency check
	report.Anomalies = append(report.Anomalies, checkCrossLayerConsistency(stats, &th)...)

	// Reference profile comparison
	if ref != nil {
		report.Anomalies = append(report.Anomalies, checkAgainstReference(stats, ref)...)
		report.Confidence = "high"
	}

	// Layered scoring
	layers := &LayerScores{}
	for _, a := range report.Anomalies {
		w := severityWeight(a.Severity)
		switch {
		case strings.HasPrefix(a.Type, "cross_layer"):
			layers.RoleGroup += w
		case strings.HasPrefix(a.Type, "reference_"):
			layers.Reference += w
		default:
			layers.TensorLocal += w
		}
	}

	// Model-global check: if anomalies are concentrated in few tensors, that's more
	// suspicious than being spread evenly (targeted attack pattern)
	affectedTensors := make(map[string]int)
	for _, a := range report.Anomalies {
		affectedTensors[a.TensorName]++
	}
	if len(stats) > 0 && len(affectedTensors) > 0 {
		concentration := float64(len(report.Anomalies)) / float64(len(affectedTensors))
		if concentration > 3.0 {
			layers.ModelGlobal = 0.2 // concentrated anomalies = more suspicious
		}
	}

	// Cap layer scores
	for _, ls := range []*float64{&layers.TensorLocal, &layers.RoleGroup, &layers.ModelGlobal, &layers.Reference} {
		if *ls > 1.0 {
			*ls = 1.0
		}
	}
	report.Layers = layers

	// Composite score from all layers
	report.Score = computeLayeredScore(layers, report.Anomalies)

	// Generate summary
	critCount := countBySeverity(report.Anomalies, SeverityCritical)
	warnCount := countBySeverity(report.Anomalies, SeverityWarning)
	infoCount := countBySeverity(report.Anomalies, SeverityInfo)

	if len(report.Anomalies) == 0 {
		report.Summary = fmt.Sprintf("no anomalies detected across %d tensors", len(stats))
	} else {
		report.Summary = fmt.Sprintf(
			"%d anomalies (%d critical, %d warning, %d info) across %d tensors",
			len(report.Anomalies), critCount, warnCount, infoCount, len(stats),
		)
	}

	return report
}

func severityWeight(severity string) float64 {
	switch severity {
	case SeverityCritical:
		return 0.3
	case SeverityWarning:
		return 0.1
	case SeverityInfo:
		return 0.02
	}
	return 0
}

func computeLayeredScore(layers *LayerScores, anomalies []Anomaly) float64 {
	// Weighted combination of layer scores
	score := layers.TensorLocal*0.3 +
		layers.RoleGroup*0.25 +
		layers.ModelGlobal*0.2 +
		layers.Reference*0.25

	// Also include the raw anomaly-based score as a floor
	rawScore := computeScore(anomalies)
	if rawScore > score {
		score = rawScore
	}

	if score > 1.0 {
		score = 1.0
	}
	return score
}

func checkTensorAnomalies(s *TensorStats, th *Thresholds) []Anomaly {
	var anomalies []Anomaly

	if s.Samples == 0 {
		return anomalies
	}

	// Abnormal mean
	if math.Abs(s.Mean) > th.MaxAbsMean {
		anomalies = append(anomalies, Anomaly{
			TensorName:  s.Name,
			Type:        "abnormal_mean",
			Severity:    SeverityWarning,
			Value:       s.Mean,
			Threshold:   th.MaxAbsMean,
			Description: fmt.Sprintf("mean %.4f exceeds threshold ±%.1f", s.Mean, th.MaxAbsMean),
		})
	}

	// Extreme kurtosis (possible trojan patch — highly concentrated perturbation)
	if s.Kurtosis > th.MaxKurtosis {
		anomalies = append(anomalies, Anomaly{
			TensorName:  s.Name,
			Type:        "extreme_kurtosis",
			Severity:    SeverityCritical,
			Value:       s.Kurtosis,
			Threshold:   th.MaxKurtosis,
			Description: fmt.Sprintf("kurtosis %.2f >> %.1f, possible localized perturbation", s.Kurtosis, th.MaxKurtosis),
		})
	}

	// Near-zero variance with non-zero values (constant tensor)
	if s.Variance < th.MinVariance && s.ZeroFraction < 0.99 {
		anomalies = append(anomalies, Anomaly{
			TensorName:  s.Name,
			Type:        "constant_tensor",
			Severity:    SeverityWarning,
			Value:       s.Variance,
			Threshold:   th.MinVariance,
			Description: fmt.Sprintf("variance %.2e with non-zero values, constant/corrupted tensor", s.Variance),
		})
	}

	// Nearly all zeros
	if s.ZeroFraction > th.MaxZeroFraction {
		anomalies = append(anomalies, Anomaly{
			TensorName:  s.Name,
			Type:        "high_sparsity",
			Severity:    SeverityWarning,
			Value:       s.ZeroFraction,
			Threshold:   th.MaxZeroFraction,
			Description: fmt.Sprintf("%.1f%% zeros, possibly corrupted or zeroed tensor", s.ZeroFraction*100),
		})
	}

	// High outlier ratio
	if s.OutlierRatio > th.MaxOutlierRatio {
		anomalies = append(anomalies, Anomaly{
			TensorName:  s.Name,
			Type:        "high_outlier_ratio",
			Severity:    SeverityWarning,
			Value:       s.OutlierRatio,
			Threshold:   th.MaxOutlierRatio,
			Description: fmt.Sprintf("%.1f%% of values beyond 3σ (expected ~0.3%%)", s.OutlierRatio*100),
		})
	}

	// NaN or Inf values
	total := s.Samples + s.NaNCount + s.InfCount
	if total > 0 {
		nanFrac := float64(s.NaNCount) / float64(total)
		infFrac := float64(s.InfCount) / float64(total)
		if nanFrac > th.MaxNaNFraction {
			anomalies = append(anomalies, Anomaly{
				TensorName:  s.Name,
				Type:        "nan_values",
				Severity:    SeverityCritical,
				Value:       nanFrac,
				Threshold:   th.MaxNaNFraction,
				Description: fmt.Sprintf("%d NaN values detected", s.NaNCount),
			})
		}
		if infFrac > th.MaxInfFraction {
			anomalies = append(anomalies, Anomaly{
				TensorName:  s.Name,
				Type:        "inf_values",
				Severity:    SeverityCritical,
				Value:       infFrac,
				Threshold:   th.MaxInfFraction,
				Description: fmt.Sprintf("%d Inf values detected", s.InfCount),
			})
		}
	}

	return anomalies
}

// checkCrossLayerConsistency groups tensors by role (e.g., attn_q, ffn_down)
// and flags layers whose statistics deviate significantly from the group.
func checkCrossLayerConsistency(stats []*TensorStats, th *Thresholds) []Anomaly {
	var anomalies []Anomaly

	// Group by tensor role (strip layer number)
	groups := make(map[string][]*TensorStats)
	for _, s := range stats {
		role := extractRole(s.Name)
		if role != "" {
			groups[role] = append(groups[role], s)
		}
	}

	// For each group, check if any layer is a statistical outlier
	// Uses both standard z-score and robust (median/MAD) z-score
	for role, members := range groups {
		if len(members) < 3 {
			continue // need at least 3 layers for meaningful comparison
		}

		// Check mean consistency using both standard and robust methods
		means := make([]float64, len(members))
		for i, m := range members {
			means[i] = m.Mean
		}
		groupMean, groupStd := meanAndStd(means)
		robustMeans := ComputeRobustStats(means)

		for _, m := range members {
			// Standard z-score
			if groupStd > 1e-10 {
				dev := math.Abs(m.Mean-groupMean) / groupStd
				if dev > th.CrossLayerMaxDev {
					anomalies = append(anomalies, Anomaly{
						TensorName:  m.Name,
						Type:        "cross_layer_outlier",
						Severity:    SeverityWarning,
						Value:       dev,
						Threshold:   th.CrossLayerMaxDev,
						Description: fmt.Sprintf("mean deviates %.1fσ from %s group (%.4f vs group mean %.4f)", dev, role, m.Mean, groupMean),
					})
				}
			}

			// Robust z-score (median/MAD) — catches outliers in heavy-tailed groups
			// Skip when MAD ≈ 0: all values near-identical means no meaningful dispersion
			robustZ := math.Abs(RobustZScore(m.Mean, robustMeans.Median, robustMeans.MAD))
			if robustMeans.MAD > 1e-10 && robustZ > th.CrossLayerMaxDev {
				anomalies = append(anomalies, Anomaly{
					TensorName:  m.Name,
					Type:        "cross_layer_robust_outlier",
					Severity:    SeverityWarning,
					Value:       robustZ,
					Threshold:   th.CrossLayerMaxDev,
					Description: fmt.Sprintf("mean deviates %.1f robust-σ from %s group median", robustZ, role),
				})
			}
		}

		// Check variance consistency
		variances := make([]float64, len(members))
		for i, m := range members {
			variances[i] = m.Variance
		}
		varMean, varStd := meanAndStd(variances)
		if varStd > 1e-10 {
			for _, m := range members {
				dev := math.Abs(m.Variance-varMean) / varStd
				if dev > th.CrossLayerMaxDev {
					anomalies = append(anomalies, Anomaly{
						TensorName:  m.Name,
						Type:        "cross_layer_variance_outlier",
						Severity:    SeverityWarning,
						Value:       dev,
						Threshold:   th.CrossLayerMaxDev,
						Description: fmt.Sprintf("variance deviates %.1fσ from %s group", dev, role),
					})
				}
			}
		}
	}

	return anomalies
}

// extractRole strips the layer number to get the tensor role.
// "blk.15.attn_q.weight" -> "attn_q.weight"
// "layers.3.feed_forward.w1.weight" -> "feed_forward.w1.weight"
func extractRole(name string) string {
	parts := strings.Split(name, ".")
	if len(parts) < 3 {
		return ""
	}
	// Skip prefix and layer number
	if parts[0] == "blk" || parts[0] == "layers" {
		return strings.Join(parts[2:], ".")
	}
	return ""
}

func checkAgainstReference(stats []*TensorStats, ref *ReferenceProfile) []Anomaly {
	var anomalies []Anomaly

	for _, s := range stats {
		tp, ok := ref.TensorProfiles[s.Name]
		if !ok {
			continue
		}

		// Check mean is within reference range
		if s.Mean < tp.MeanRange[0] || s.Mean > tp.MeanRange[1] {
			anomalies = append(anomalies, Anomaly{
				TensorName:  s.Name,
				Type:        "reference_mean_deviation",
				Severity:    SeverityWarning,
				Value:       s.Mean,
				Threshold:   tp.MeanRange[1],
				Description: fmt.Sprintf("mean %.6f outside reference range [%.6f, %.6f]", s.Mean, tp.MeanRange[0], tp.MeanRange[1]),
			})
		}

		// Check variance
		if s.Variance < tp.VarianceRange[0] || s.Variance > tp.VarianceRange[1] {
			anomalies = append(anomalies, Anomaly{
				TensorName:  s.Name,
				Type:        "reference_variance_deviation",
				Severity:    SeverityWarning,
				Value:       s.Variance,
				Threshold:   tp.VarianceRange[1],
				Description: fmt.Sprintf("variance %.6e outside reference range", s.Variance),
			})
		}

		// Check kurtosis
		if s.Kurtosis < tp.KurtosisRange[0] || s.Kurtosis > tp.KurtosisRange[1] {
			anomalies = append(anomalies, Anomaly{
				TensorName:  s.Name,
				Type:        "reference_kurtosis_deviation",
				Severity:    SeverityWarning,
				Value:       s.Kurtosis,
				Threshold:   tp.KurtosisRange[1],
				Description: fmt.Sprintf("kurtosis %.2f outside reference range [%.2f, %.2f]", s.Kurtosis, tp.KurtosisRange[0], tp.KurtosisRange[1]),
			})
		}
	}

	return anomalies
}

func computeScore(anomalies []Anomaly) float64 {
	if len(anomalies) == 0 {
		return 0.0
	}
	var score float64
	for _, a := range anomalies {
		switch a.Severity {
		case SeverityCritical:
			score += 0.3
		case SeverityWarning:
			score += 0.1
		case SeverityInfo:
			score += 0.02
		}
	}
	if score > 1.0 {
		score = 1.0
	}
	return score
}

func countBySeverity(anomalies []Anomaly, severity string) int {
	count := 0
	for _, a := range anomalies {
		if a.Severity == severity {
			count++
		}
	}
	return count
}

func meanAndStd(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	n := float64(len(values))
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / n
	var m2 float64
	for _, v := range values {
		d := v - mean
		m2 += d * d
	}
	return mean, math.Sqrt(m2 / n)
}
