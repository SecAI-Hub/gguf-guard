// gguf-guard: GGUF model weight integrity and anomaly scanner.
//
// Usage:
//
//	gguf-guard scan [flags] model.gguf                   Full anomaly scan
//	gguf-guard fingerprint [flags] model.gguf            Structural fingerprint
//	gguf-guard compare [flags] baseline.gguf candidate.gguf
//	gguf-guard profile [flags] model.gguf                Generate reference profile
//	gguf-guard build-reference [flags] model1.gguf [model2.gguf ...]
//	gguf-guard lineage [flags] source.gguf candidate.gguf
//	gguf-guard manifest [flags] model.gguf               Generate integrity manifest
//	gguf-guard verify-manifest [flags] model.gguf manifest.json
//	gguf-guard inspect [flags] model.gguf                Structural policy checks
//	gguf-guard info model.gguf                           Show model metadata
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/SecAI-Hub/gguf-guard/analysis"
	"github.com/SecAI-Hub/gguf-guard/gguf"
)

const version = "0.2.0"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	switch cmd {
	case "scan":
		cmdScan(os.Args[2:])
	case "fingerprint":
		cmdFingerprint(os.Args[2:])
	case "compare":
		cmdCompare(os.Args[2:])
	case "profile":
		cmdProfile(os.Args[2:])
	case "build-reference":
		cmdBuildReference(os.Args[2:])
	case "lineage":
		cmdLineage(os.Args[2:])
	case "manifest":
		cmdManifest(os.Args[2:])
	case "verify-manifest":
		cmdVerifyManifest(os.Args[2:])
	case "inspect":
		cmdInspect(os.Args[2:])
	case "info":
		cmdInfo(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Printf("gguf-guard %s\n", version)
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `gguf-guard %s — GGUF model weight integrity scanner

Usage:
  gguf-guard scan       [flags] model.gguf                Full anomaly scan
  gguf-guard fingerprint [flags] model.gguf               Structural fingerprint
  gguf-guard compare    [flags] baseline.gguf candidate.gguf
  gguf-guard profile    [flags] model.gguf                Generate reference profile
  gguf-guard build-reference [flags] model1.gguf [...]    Multi-sample reference
  gguf-guard lineage    [flags] source.gguf candidate.gguf
  gguf-guard manifest   [flags] model.gguf                Generate integrity manifest
  gguf-guard verify-manifest [flags] model.gguf manifest.json
  gguf-guard inspect    [flags] model.gguf                Structural policy checks
  gguf-guard info       model.gguf                        Show model metadata
  gguf-guard version                                      Print version

Exit codes:
  0  PASS — no significant anomalies
  1  error (parse failure, missing file, etc.)
  2  FAIL — scan score exceeds threshold

Run 'gguf-guard <command> --help' for command-specific flags.
`, version)
}

// --- scan ---

type ScanOutput struct {
	Model       string                  `json:"model"`
	Fingerprint *analysis.Fingerprint   `json:"fingerprint"`
	Policy      *analysis.PolicyReport  `json:"policy,omitempty"`
	Report      *analysis.AnomalyReport `json:"report"`
	QuantReport *analysis.QuantReport   `json:"quant_report,omitempty"`
	FamilyMatch *analysis.FamilyMatch   `json:"family_match,omitempty"`
	Stats       []*analysis.TensorStats `json:"tensor_stats,omitempty"`
	Elapsed     string                  `json:"elapsed"`
}

func cmdScan(args []string) {
	fs := flag.NewFlagSet("scan", flag.ExitOnError)
	refPath := fs.String("reference", "", "Reference profile JSON for comparison")
	outPath := fs.String("output", "", "Write JSON output to file (default: stdout)")
	maxTensors := fs.Int("max-tensors", 0, "Limit number of tensors to analyze (0=all)")
	includeStats := fs.Bool("stats", false, "Include per-tensor statistics in output")
	quiet := fs.Bool("quiet", false, "Only print pass/fail and score")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard scan [flags] model.gguf")
		os.Exit(1)
	}
	modelPath := fs.Arg(0)
	start := time.Now()

	gf, err := gguf.Parse(modelPath)
	fatal(err, "parse")

	fp, err := analysis.GenerateFingerprint(gf)
	fatal(err, "fingerprint")

	// Structural policy check (fast, runs first)
	policy := analysis.CheckStructuralPolicy(gf)

	// Family matching
	familyMatch := analysis.BestFamilyMatch(gf)

	// Dequantized statistics
	stats := analyzeTensors(gf, *maxTensors)

	// Quant-aware block analysis
	quantReport, _ := analysis.AnalyzeQuantBlocks(gf, false)

	var ref *analysis.ReferenceProfile
	if *refPath != "" {
		ref, err = analysis.LoadReference(*refPath)
		fatal(err, "load reference")
	}

	report := analysis.DetectAnomalies(stats, ref)

	// Incorporate quant anomalies into overall score
	if quantReport != nil {
		for _, qa := range quantReport.Anomalies {
			report.Anomalies = append(report.Anomalies, analysis.Anomaly{
				TensorName:  qa.TensorName,
				Type:        "quant_" + qa.Type,
				Severity:    qa.Severity,
				Value:       qa.Value,
				Threshold:   qa.Threshold,
				Description: qa.Description,
			})
		}
		// Recalculate score with quant anomalies
		if len(quantReport.Anomalies) > 0 {
			report.Score = recomputeScore(report)
		}
	}

	// Incorporate policy violations
	if !policy.Pass {
		for _, pv := range policy.Violations {
			report.Anomalies = append(report.Anomalies, analysis.Anomaly{
				Type:        "policy_" + pv.Type,
				Severity:    pv.Severity,
				Description: pv.Description,
			})
		}
		report.Score = recomputeScore(report)
	}

	if *quiet {
		result := "PASS"
		if report.Score > 0.5 {
			result = "FAIL"
		} else if report.Score > 0.2 {
			result = "WARN"
		}
		fmt.Printf("%s score=%.2f %s\n", result, report.Score, report.Summary)
		if report.Score > 0.5 {
			os.Exit(2)
		}
		return
	}

	out := ScanOutput{
		Model:       modelPath,
		Fingerprint: fp,
		Policy:      policy,
		Report:      report,
		QuantReport: quantReport,
		FamilyMatch: familyMatch,
		Elapsed:     time.Since(start).Round(time.Millisecond).String(),
	}
	if *includeStats {
		out.Stats = stats
	}

	writeJSON(out, *outPath)

	if report.Score > 0.5 {
		os.Exit(2)
	}
}

func recomputeScore(report *analysis.AnomalyReport) float64 {
	score := 0.0
	for _, a := range report.Anomalies {
		switch a.Severity {
		case "critical":
			score += 0.3
		case "warning":
			score += 0.1
		case "info":
			score += 0.02
		}
	}
	if score > 1.0 {
		score = 1.0
	}
	if score > report.Score {
		return score
	}
	return report.Score
}

// --- fingerprint ---

func cmdFingerprint(args []string) {
	fs := flag.NewFlagSet("fingerprint", flag.ExitOnError)
	outPath := fs.String("output", "", "Write JSON output to file")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard fingerprint [flags] model.gguf")
		os.Exit(1)
	}

	gf, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse")

	fp, err := analysis.GenerateFingerprint(gf)
	fatal(err, "fingerprint")

	writeJSON(fp, *outPath)
}

// --- compare ---

func cmdCompare(args []string) {
	fs := flag.NewFlagSet("compare", flag.ExitOnError)
	outPath := fs.String("output", "", "Write JSON output to file")
	maxTensors := fs.Int("max-tensors", 0, "Limit tensors to analyze (0=all)")
	fs.Parse(args)

	if fs.NArg() < 2 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard compare [flags] baseline.gguf candidate.gguf")
		os.Exit(1)
	}

	baseGF, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse baseline")
	candGF, err := gguf.Parse(fs.Arg(1))
	fatal(err, "parse candidate")

	baseFP, err := analysis.GenerateFingerprint(baseGF)
	fatal(err, "fingerprint baseline")
	candFP, err := analysis.GenerateFingerprint(candGF)
	fatal(err, "fingerprint candidate")

	baseStats := analyzeTensors(baseGF, *maxTensors)
	candStats := analyzeTensors(candGF, *maxTensors)

	result := analysis.Compare(baseStats, candStats, baseFP, candFP)
	writeJSON(result, *outPath)
}

// --- profile ---

func cmdProfile(args []string) {
	fs := flag.NewFlagSet("profile", flag.ExitOnError)
	outPath := fs.String("output", "", "Write JSON output to file (default: stdout)")
	margin := fs.Float64("margin", 3.0, "Standard deviations for acceptable range")
	maxTensors := fs.Int("max-tensors", 0, "Limit tensors (0=all)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard profile [flags] model.gguf")
		os.Exit(1)
	}

	gf, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse")

	fp, err := analysis.GenerateFingerprint(gf)
	fatal(err, "fingerprint")

	stats := analyzeTensors(gf, *maxTensors)
	ref := analysis.ProfileFromStats(stats, fp, *margin)

	writeJSON(ref, *outPath)
}

// --- build-reference ---

func cmdBuildReference(args []string) {
	fs := flag.NewFlagSet("build-reference", flag.ExitOnError)
	outPath := fs.String("output", "", "Write merged reference JSON (default: stdout)")
	margin := fs.Float64("margin", 3.0, "Standard deviations for range")
	maxTensors := fs.Int("max-tensors", 0, "Limit tensors (0=all)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard build-reference [flags] model1.gguf [model2.gguf ...]")
		os.Exit(1)
	}

	var profiles []*analysis.ReferenceProfile
	for _, path := range fs.Args() {
		gf, err := gguf.Parse(path)
		fatal(err, "parse "+path)
		fp, err := analysis.GenerateFingerprint(gf)
		fatal(err, "fingerprint "+path)
		stats := analyzeTensors(gf, *maxTensors)
		ref := analysis.ProfileFromStats(stats, fp, *margin)
		profiles = append(profiles, ref)
		fmt.Fprintf(os.Stderr, "profiled: %s (%d tensors)\n", path, len(stats))
	}

	var result *analysis.ReferenceProfile
	if len(profiles) == 1 {
		result = profiles[0]
	} else {
		result = analysis.MergeProfiles(profiles, *margin)
	}

	writeJSON(result, *outPath)
}

// --- lineage ---

func cmdLineage(args []string) {
	fs := flag.NewFlagSet("lineage", flag.ExitOnError)
	outPath := fs.String("output", "", "Write JSON output to file")
	maxTensors := fs.Int("max-tensors", 0, "Limit tensors (0=all)")
	fs.Parse(args)

	if fs.NArg() < 2 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard lineage [flags] source.gguf candidate.gguf")
		os.Exit(1)
	}

	srcGF, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse source")
	candGF, err := gguf.Parse(fs.Arg(1))
	fatal(err, "parse candidate")

	srcFP, err := analysis.GenerateFingerprint(srcGF)
	fatal(err, "fingerprint source")
	candFP, err := analysis.GenerateFingerprint(candGF)
	fatal(err, "fingerprint candidate")

	srcStats := analyzeTensors(srcGF, *maxTensors)
	candStats := analyzeTensors(candGF, *maxTensors)

	result := analysis.CompareLineage(srcStats, candStats, srcFP, candFP)
	writeJSON(result, *outPath)

	if result.Verdict == "suspicious" {
		os.Exit(2)
	}
}

// --- manifest ---

func cmdManifest(args []string) {
	fs := flag.NewFlagSet("manifest", flag.ExitOnError)
	outPath := fs.String("output", "", "Write manifest JSON (default: stdout)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard manifest [flags] model.gguf")
		os.Exit(1)
	}

	gf, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse")

	fp, err := analysis.GenerateFingerprint(gf)
	fatal(err, "fingerprint")

	m, err := analysis.GenerateManifest(gf, fp)
	fatal(err, "generate manifest")

	writeJSON(m, *outPath)
}

// --- verify-manifest ---

func cmdVerifyManifest(args []string) {
	fs := flag.NewFlagSet("verify-manifest", flag.ExitOnError)
	fs.Parse(args)

	if fs.NArg() < 2 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard verify-manifest model.gguf manifest.json")
		os.Exit(1)
	}

	gf, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse model")

	m, err := analysis.LoadManifest(fs.Arg(1))
	fatal(err, "load manifest")

	mismatches, err := analysis.VerifyManifest(gf, m)
	fatal(err, "verify")

	if len(mismatches) == 0 {
		fmt.Println("PASS: all tensors match manifest")
	} else {
		fmt.Printf("FAIL: %d mismatches\n", len(mismatches))
		for _, mm := range mismatches {
			fmt.Printf("  %s\n", mm)
		}
		os.Exit(2)
	}
}

// --- inspect ---

func cmdInspect(args []string) {
	fs := flag.NewFlagSet("inspect", flag.ExitOnError)
	outPath := fs.String("output", "", "Write JSON output to file")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard inspect [flags] model.gguf")
		os.Exit(1)
	}

	gf, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse")

	policy := analysis.CheckStructuralPolicy(gf)
	familyMatches := analysis.MatchFamily(gf)

	type InspectOutput struct {
		Policy        *analysis.PolicyReport `json:"policy"`
		FamilyMatches []analysis.FamilyMatch `json:"family_matches"`
	}

	writeJSON(InspectOutput{
		Policy:        policy,
		FamilyMatches: familyMatches,
	}, *outPath)

	if !policy.Pass {
		os.Exit(2)
	}
}

// --- info ---

type InfoOutput struct {
	Path         string         `json:"path"`
	Version      uint32         `json:"version"`
	Architecture string         `json:"architecture"`
	QuantType    string         `json:"quant_type"`
	TensorCount  int            `json:"tensor_count"`
	Parameters   uint64         `json:"parameters"`
	FileSize     int64          `json:"file_size_bytes"`
	Metadata     map[string]any `json:"metadata"`
	TypeCounts   map[string]int `json:"type_counts"`
}

func cmdInfo(args []string) {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "usage: gguf-guard info model.gguf")
		os.Exit(1)
	}

	gf, err := gguf.Parse(fs.Arg(0))
	fatal(err, "parse")

	typeCounts := make(map[string]int)
	for _, t := range gf.Tensors {
		typeCounts[t.Type.String()]++
	}

	out := InfoOutput{
		Path:         gf.Path,
		Version:      gf.Version,
		Architecture: gf.Architecture(),
		QuantType:    gf.QuantType(),
		TensorCount:  len(gf.Tensors),
		Parameters:   gf.TotalParameters(),
		FileSize:     gf.FileSize,
		Metadata:     gf.Metadata,
		TypeCounts:   typeCounts,
	}

	writeJSON(out, "")
}

// --- helpers ---

// analyzeTensors reads, dequantizes, and computes statistics for each tensor.
func analyzeTensors(gf *gguf.File, maxTensors int) []*analysis.TensorStats {
	tensors := gf.Tensors
	if maxTensors > 0 && len(tensors) > maxTensors {
		tensors = tensors[:maxTensors]
	}

	const maxSamples = 1_000_000 // cap per tensor for large models

	stats := make([]*analysis.TensorStats, 0, len(tensors))
	for i := range tensors {
		ti := &tensors[i]

		if !ti.Type.Supported() {
			stats = append(stats, &analysis.TensorStats{
				Name:         ti.Name,
				Type:         ti.Type.String(),
				Shape:        ti.Dims,
				ElementCount: ti.ElementCount,
			})
			continue
		}

		data, err := gguf.ReadTensorData(gf, ti, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warn: skipping tensor %q: %v\n", ti.Name, err)
			continue
		}

		values, err := gguf.Dequantize(data, ti.Type, maxSamples)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warn: dequant %q: %v\n", ti.Name, err)
			continue
		}
		if len(values) == 0 {
			continue
		}

		s := analysis.ComputeStats(values, ti.Name, ti.Type.String(), ti.Dims, ti.ElementCount)
		stats = append(stats, s)
	}

	return stats
}

func writeJSON(v any, outPath string) {
	data, err := json.MarshalIndent(v, "", "  ")
	fatal(err, "marshal JSON")

	if outPath != "" {
		err = os.WriteFile(outPath, append(data, '\n'), 0644)
		fatal(err, "write output file")
	} else {
		fmt.Println(string(data))
	}
}

func fatal(err error, context string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %s: %v\n", context, err)
		os.Exit(1)
	}
}
