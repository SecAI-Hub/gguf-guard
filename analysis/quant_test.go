package analysis

import (
	"testing"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

func TestAnalyzeQuantBlocksNonQuantized(t *testing.T) {
	dir := t.TempDir()
	tensors := []gguf.TensorInfo{
		{Name: "weight", NDims: 1, Dims: []uint64{4}, Type: gguf.TypeF32, ElementCount: 4},
	}
	path := createTestGGUF(t, dir, "f32.gguf", tensors, map[string]any{})
	gf, _ := gguf.Parse(path)

	report, err := AnalyzeQuantBlocks(gf, false)
	if err != nil {
		t.Fatalf("error: %v", err)
	}
	if report.TensorsAnalyzed != 0 {
		t.Errorf("analyzed = %d, want 0 (F32 is not quantized)", report.TensorsAnalyzed)
	}
}

func TestIsQuantizedType(t *testing.T) {
	if isQuantizedType(gguf.TypeF32) {
		t.Error("F32 should not be quantized")
	}
	if isQuantizedType(gguf.TypeF16) {
		t.Error("F16 should not be quantized")
	}
	if !isQuantizedType(gguf.TypeQ4_K) {
		t.Error("Q4_K should be quantized")
	}
	if !isQuantizedType(gguf.TypeQ8_0) {
		t.Error("Q8_0 should be quantized")
	}
}

func TestCheckQuantAnomaliesClean(t *testing.T) {
	bs := &gguf.BlockStats{
		TensorName:     "clean",
		QuantType:      "Q8_0",
		NumBlocks:      100,
		ScaleEntropy:   4.0,
		ScaleRatio:     10.0,
		CodeEntropy:    6.0,
		SaturationLow:  0.01,
		SaturationHigh: 0.01,
		RepeatedBlocks: 0,
		ZeroScales:     0,
	}

	anomalies := checkQuantAnomalies(bs, &DefaultQuantThresholds)
	if len(anomalies) != 0 {
		t.Errorf("expected 0 anomalies for clean block stats, got %d", len(anomalies))
	}
}

func TestCheckQuantAnomaliesRepeatedBlocks(t *testing.T) {
	bs := &gguf.BlockStats{
		TensorName:     "repeated",
		QuantType:      "Q8_0",
		NumBlocks:      100,
		ScaleEntropy:   4.0,
		CodeEntropy:    6.0,
		RepeatedBlocks: 50, // 50% repeated
	}

	anomalies := checkQuantAnomalies(bs, &DefaultQuantThresholds)
	found := false
	for _, a := range anomalies {
		if a.Type == "repeated_blocks" {
			found = true
			if a.Severity != SeverityCritical {
				t.Errorf("severity = %q, want critical", a.Severity)
			}
		}
	}
	if !found {
		t.Error("expected repeated_blocks anomaly")
	}
}

func TestCheckQuantAnomaliesLowEntropy(t *testing.T) {
	bs := &gguf.BlockStats{
		TensorName:   "low_entropy",
		QuantType:    "Q4_0",
		NumBlocks:    100,
		ScaleEntropy: 0.1, // very low
		CodeEntropy:  0.5, // very low
	}

	anomalies := checkQuantAnomalies(bs, &DefaultQuantThresholds)
	foundScale := false
	foundCode := false
	for _, a := range anomalies {
		if a.Type == "low_scale_entropy" {
			foundScale = true
		}
		if a.Type == "low_code_entropy" {
			foundCode = true
		}
	}
	if !foundScale {
		t.Error("expected low_scale_entropy")
	}
	if !foundCode {
		t.Error("expected low_code_entropy")
	}
}
