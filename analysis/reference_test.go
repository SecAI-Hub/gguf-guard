package analysis

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestProfileFromStats(t *testing.T) {
	// 4 layers with consistent stats
	stats := []*TensorStats{
		{Name: "blk.0.attn_q.weight", Samples: 100, Mean: 0.01, Variance: 0.10, Kurtosis: 2.0},
		{Name: "blk.1.attn_q.weight", Samples: 100, Mean: 0.02, Variance: 0.11, Kurtosis: 2.1},
		{Name: "blk.2.attn_q.weight", Samples: 100, Mean: 0.01, Variance: 0.10, Kurtosis: 1.9},
		{Name: "blk.3.attn_q.weight", Samples: 100, Mean: 0.02, Variance: 0.09, Kurtosis: 2.2},
	}
	fp := &Fingerprint{
		Architecture:   "llama",
		QuantType:      "Q4_K",
		StructureHash:  "abc123",
		ParameterCount: 400,
		FileHash:       "deadbeef",
	}

	ref := ProfileFromStats(stats, fp, 3.0)
	if ref.Architecture != "llama" {
		t.Errorf("arch = %q, want llama", ref.Architecture)
	}
	if ref.QuantType != "Q4_K" {
		t.Errorf("quant = %q, want Q4_K", ref.QuantType)
	}
	if len(ref.TensorProfiles) != 4 {
		t.Errorf("profiles = %d, want 4", len(ref.TensorProfiles))
	}

	// Check that the range includes all actual values
	for _, s := range stats {
		tp := ref.TensorProfiles[s.Name]
		if tp == nil {
			t.Errorf("missing profile for %q", s.Name)
			continue
		}
		if s.Mean < tp.MeanRange[0] || s.Mean > tp.MeanRange[1] {
			t.Errorf("%s mean %v outside range %v", s.Name, s.Mean, tp.MeanRange)
		}
	}
}

func TestSaveLoadReference(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "ref.json")

	ref := &ReferenceProfile{
		Name:         "test-llama-Q4_K",
		Architecture: "llama",
		QuantType:    "Q4_K",
		TensorProfiles: map[string]*TensorProfile{
			"weight": {
				MeanRange:     [2]float64{-0.1, 0.1},
				VarianceRange: [2]float64{0.05, 0.2},
				KurtosisRange: [2]float64{1.0, 4.0},
			},
		},
	}

	if err := SaveReference(path, ref); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := LoadReference(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	if loaded.Architecture != ref.Architecture {
		t.Errorf("arch = %q, want %q", loaded.Architecture, ref.Architecture)
	}
	tp := loaded.TensorProfiles["weight"]
	if tp == nil {
		t.Fatal("missing weight profile")
	}
	if tp.MeanRange[0] != -0.1 || tp.MeanRange[1] != 0.1 {
		t.Errorf("mean range = %v, want [-0.1, 0.1]", tp.MeanRange)
	}
}

func TestLoadReferenceNotFound(t *testing.T) {
	_, err := LoadReference("/nonexistent/path.json")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadReferenceInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	os.WriteFile(path, []byte("not json"), 0644)

	_, err := LoadReference(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestRangeFromValues(t *testing.T) {
	values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	r := rangeFromValues(values, 3.0)

	mean := 3.0
	// std = sqrt(2) ≈ 1.414
	expectedLo := mean - 3.0*math.Sqrt(2.0)
	expectedHi := mean + 3.0*math.Sqrt(2.0)

	if math.Abs(r[0]-expectedLo) > 0.1 {
		t.Errorf("lo = %v, want ~%v", r[0], expectedLo)
	}
	if math.Abs(r[1]-expectedHi) > 0.1 {
		t.Errorf("hi = %v, want ~%v", r[1], expectedHi)
	}
}

func TestRangeFromEmpty(t *testing.T) {
	r := rangeFromValues(nil, 3.0)
	if r[0] != 0 || r[1] != 0 {
		t.Errorf("empty range = %v, want [0, 0]", r)
	}
}
