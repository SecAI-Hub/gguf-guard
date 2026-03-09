package analysis

import (
	"fmt"
	"math"
	"testing"
)

func TestDetectAnomaliesClean(t *testing.T) {
	stats := []*TensorStats{
		{Name: "blk.0.attn_q.weight", Samples: 100, Mean: 0.01, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.0, OutlierRatio: 0.003},
		{Name: "blk.1.attn_q.weight", Samples: 100, Mean: 0.02, Variance: 0.1, StdDev: 0.316, Kurtosis: 2.1, OutlierRatio: 0.002},
		{Name: "blk.2.attn_q.weight", Samples: 100, Mean: 0.01, Variance: 0.1, StdDev: 0.316, Kurtosis: 1.9, OutlierRatio: 0.004},
	}

	report := DetectAnomalies(stats, nil)
	if report.Score != 0.0 {
		t.Errorf("clean model score = %v, want 0.0", report.Score)
	}
	if len(report.Anomalies) != 0 {
		t.Errorf("clean model anomalies = %d, want 0", len(report.Anomalies))
	}
	if report.Confidence != "medium" {
		t.Errorf("confidence = %q, want medium", report.Confidence)
	}
}

func TestDetectAnomaliesEmpty(t *testing.T) {
	report := DetectAnomalies(nil, nil)
	if report.Confidence != "insufficient-data" {
		t.Errorf("confidence = %q, want insufficient-data", report.Confidence)
	}
}

func TestDetectAnomaliesExtremeKurtosis(t *testing.T) {
	stats := []*TensorStats{
		{Name: "trojan_layer", Samples: 100, Mean: 0.0, Variance: 0.5, StdDev: 0.707, Kurtosis: 500.0},
	}

	report := DetectAnomalies(stats, nil)
	if report.Score == 0 {
		t.Error("expected nonzero score for extreme kurtosis")
	}
	found := false
	for _, a := range report.Anomalies {
		if a.Type == "extreme_kurtosis" {
			found = true
			if a.Severity != SeverityCritical {
				t.Errorf("kurtosis severity = %q, want critical", a.Severity)
			}
		}
	}
	if !found {
		t.Error("expected extreme_kurtosis anomaly")
	}
}

func TestDetectAnomaliesHighSparsity(t *testing.T) {
	stats := []*TensorStats{
		{Name: "sparse", Samples: 100, Mean: 0.0, Variance: 0.01, StdDev: 0.1, ZeroFraction: 0.995},
	}

	report := DetectAnomalies(stats, nil)
	found := false
	for _, a := range report.Anomalies {
		if a.Type == "high_sparsity" {
			found = true
		}
	}
	if !found {
		t.Error("expected high_sparsity anomaly")
	}
}

func TestDetectAnomaliesNaN(t *testing.T) {
	stats := []*TensorStats{
		{Name: "corrupt", Samples: 90, NaNCount: 10},
	}
	report := DetectAnomalies(stats, nil)
	found := false
	for _, a := range report.Anomalies {
		if a.Type == "nan_values" {
			found = true
			if a.Severity != SeverityCritical {
				t.Errorf("nan severity = %q, want critical", a.Severity)
			}
		}
	}
	if !found {
		t.Error("expected nan_values anomaly")
	}
}

func TestDetectAnomaliesHighOutliers(t *testing.T) {
	stats := []*TensorStats{
		{Name: "outliers", Samples: 100, Mean: 0, Variance: 1, StdDev: 1, OutlierRatio: 0.15},
	}
	report := DetectAnomalies(stats, nil)
	found := false
	for _, a := range report.Anomalies {
		if a.Type == "high_outlier_ratio" {
			found = true
		}
	}
	if !found {
		t.Error("expected high_outlier_ratio anomaly")
	}
}

func TestDetectAnomaliesAbnormalMean(t *testing.T) {
	stats := []*TensorStats{
		{Name: "shifted", Samples: 100, Mean: 50.0, Variance: 1, StdDev: 1},
	}
	report := DetectAnomalies(stats, nil)
	found := false
	for _, a := range report.Anomalies {
		if a.Type == "abnormal_mean" {
			found = true
		}
	}
	if !found {
		t.Error("expected abnormal_mean anomaly")
	}
}

func TestCrossLayerConsistency(t *testing.T) {
	// 30 consistent layers + 1 extreme outlier (need N>26 so max z-score sqrt(N-1) > 5.0)
	var stats []*TensorStats
	for i := 0; i < 30; i++ {
		stats = append(stats, &TensorStats{
			Name: fmt.Sprintf("blk.%d.attn_q.weight", i), Samples: 100,
			Mean: 0.01 + float64(i%3)*0.005, Variance: 0.1,
		})
	}
	stats = append(stats, &TensorStats{
		Name: "blk.30.attn_q.weight", Samples: 100, Mean: 100.0, Variance: 0.1,
	})

	report := DetectAnomalies(stats, nil)
	found := false
	for _, a := range report.Anomalies {
		if a.Type == "cross_layer_outlier" && a.TensorName == "blk.30.attn_q.weight" {
			found = true
		}
	}
	if !found {
		t.Error("expected cross_layer_outlier for blk.30")
	}
}

func TestScoreCapping(t *testing.T) {
	// Many critical anomalies should cap at 1.0
	stats := make([]*TensorStats, 10)
	for i := range stats {
		stats[i] = &TensorStats{
			Name: "bad", Samples: 100,
			NaNCount: 50, InfCount: 50,
			Kurtosis: 999, Mean: 999, OutlierRatio: 0.5, ZeroFraction: 0.999,
		}
	}
	report := DetectAnomalies(stats, nil)
	if report.Score > 1.0 {
		t.Errorf("score = %v, should be capped at 1.0", report.Score)
	}
}

func TestDetectAnomaliesWithReference(t *testing.T) {
	stats := []*TensorStats{
		{Name: "weight", Samples: 100, Mean: 5.0, Variance: 0.1, Kurtosis: 2.0},
	}
	ref := &ReferenceProfile{
		TensorProfiles: map[string]*TensorProfile{
			"weight": {
				MeanRange:     [2]float64{-0.5, 0.5},
				VarianceRange: [2]float64{0.05, 0.2},
				KurtosisRange: [2]float64{1.0, 4.0},
			},
		},
	}

	report := DetectAnomalies(stats, ref)
	if report.Confidence != "high" {
		t.Errorf("confidence = %q, want high (with reference)", report.Confidence)
	}
	foundMeanDev := false
	for _, a := range report.Anomalies {
		if a.Type == "reference_mean_deviation" {
			foundMeanDev = true
		}
	}
	if !foundMeanDev {
		t.Error("expected reference_mean_deviation")
	}
}

func TestExtractRole(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{"blk.0.attn_q.weight", "attn_q.weight"},
		{"layers.3.feed_forward.w1.weight", "feed_forward.w1.weight"},
		{"output.weight", ""},
		{"blk.0", ""},
	}
	for _, tt := range tests {
		got := extractRole(tt.name)
		if got != tt.want {
			t.Errorf("extractRole(%q) = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestMeanAndStd(t *testing.T) {
	mean, std := meanAndStd([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	if math.Abs(mean-5.0) > 1e-6 {
		t.Errorf("mean = %v, want 5.0", mean)
	}
	if math.Abs(std-2.0) > 0.01 {
		t.Errorf("std = %v, want ~2.0", std)
	}

	mean, std = meanAndStd(nil)
	if mean != 0 || std != 0 {
		t.Errorf("empty mean/std = (%v, %v), want (0, 0)", mean, std)
	}
}
