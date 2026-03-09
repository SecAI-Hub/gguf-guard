// gguf-guard: GGUF model weight integrity and anomaly scanner.
//
// Usage:
//
//	gguf-guard scan [--reference ref.json] [--output out.json] [--max-tensors N] model.gguf
//	gguf-guard fingerprint [--output out.json] model.gguf
//	gguf-guard compare [--output out.json] baseline.gguf candidate.gguf
//	gguf-guard profile [--margin N] [--output out.json] model.gguf
//	gguf-guard info model.gguf
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

const version = "0.1.0"

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
  gguf-guard scan     [flags] model.gguf          Full anomaly scan
  gguf-guard fingerprint [flags] model.gguf       Structural fingerprint
  gguf-guard compare  [flags] baseline.gguf candidate.gguf
  gguf-guard profile  [flags] model.gguf          Generate reference profile
  gguf-guard info     model.gguf                  Show model metadata
  gguf-guard version                              Print version

Run 'gguf-guard <command> --help' for command-specific flags.
`, version)
}

// --- scan ---

type ScanOutput struct {
	Model       string                  `json:"model"`
	Fingerprint *analysis.Fingerprint   `json:"fingerprint"`
	Report      *analysis.AnomalyReport `json:"report"`
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

	stats := analyzeTensors(gf, *maxTensors)

	var ref *analysis.ReferenceProfile
	if *refPath != "" {
		ref, err = analysis.LoadReference(*refPath)
		fatal(err, "load reference")
	}

	report := analysis.DetectAnomalies(stats, ref)

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
		Report:      report,
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
			// Still record the tensor with zero stats
			stats = append(stats, &analysis.TensorStats{
				Name:         ti.Name,
				Type:         ti.Type.String(),
				Shape:        ti.Dims,
				ElementCount: ti.ElementCount,
			})
			continue
		}

		// Read raw tensor data
		data, err := gguf.ReadTensorData(gf, ti, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warn: skipping tensor %q: %v\n", ti.Name, err)
			continue
		}

		// Dequantize to float32
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
