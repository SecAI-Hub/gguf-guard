package analysis

import (
	"fmt"
	"strings"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

// PolicyViolation represents a structural issue found during policy checks.
type PolicyViolation struct {
	Type        string `json:"type"`
	Severity    string `json:"severity"`
	Description string `json:"description"`
}

// PolicyReport is the result of structural validation.
type PolicyReport struct {
	Pass       bool              `json:"pass"`
	Violations []PolicyViolation `json:"violations,omitempty"`
}

// CheckStructuralPolicy runs fast structural validation checks before deep analysis.
// These are cheap to compute and catch malformed or suspicious files early.
func CheckStructuralPolicy(gf *gguf.File) *PolicyReport {
	report := &PolicyReport{Pass: true}

	// 1. Version check
	if gf.Version != 2 && gf.Version != 3 {
		report.addViolation("unsupported_version", SeverityCritical,
			fmt.Sprintf("GGUF version %d is not supported (expected 2 or 3)", gf.Version))
	}

	// 2. Tensor offset/overlap validation
	checkTensorOverlaps(gf, report)

	// 3. Architecture metadata consistency
	checkArchConsistency(gf, report)

	// 4. Suspicious metadata
	checkMetadata(gf, report)

	// 5. Tensor shape sanity
	checkTensorShapes(gf, report)

	// 6. Unknown tensor names (for known architectures)
	arch := gf.Architecture()
	if arch != "unknown" {
		checkTensorNames(gf, arch, report)
	}

	report.Pass = len(report.Violations) == 0
	return report
}

func (r *PolicyReport) addViolation(vtype, severity, desc string) {
	r.Violations = append(r.Violations, PolicyViolation{
		Type:        vtype,
		Severity:    severity,
		Description: desc,
	})
}

func checkTensorOverlaps(gf *gguf.File, report *PolicyReport) {
	type region struct {
		name  string
		start int64
		end   int64
	}

	regions := make([]region, 0, len(gf.Tensors))
	for _, t := range gf.Tensors {
		size := t.ByteSize()
		if size == 0 {
			report.addViolation("zero_size_tensor", SeverityWarning,
				fmt.Sprintf("tensor %q has zero computed size", t.Name))
			continue
		}
		start := gf.DataOffset + int64(t.Offset)
		end := start + size
		if end > gf.FileSize {
			report.addViolation("tensor_out_of_bounds", SeverityCritical,
				fmt.Sprintf("tensor %q extends beyond file (offset %d + size %d > file size %d)",
					t.Name, start, size, gf.FileSize))
			continue
		}
		regions = append(regions, region{t.Name, start, end})
	}

	// Check for overlaps (O(n^2) but tensor count is bounded)
	for i := 0; i < len(regions); i++ {
		for j := i + 1; j < len(regions); j++ {
			if regions[i].start < regions[j].end && regions[j].start < regions[i].end {
				report.addViolation("tensor_overlap", SeverityCritical,
					fmt.Sprintf("tensors %q and %q overlap in file", regions[i].name, regions[j].name))
			}
		}
	}
}

func checkArchConsistency(gf *gguf.File, report *PolicyReport) {
	arch := gf.Architecture()
	if arch == "unknown" {
		report.addViolation("missing_architecture", SeverityWarning,
			"no general.architecture metadata key found")
		return
	}

	// Check that tensor names are consistent with declared architecture
	hasBlkPrefix := false
	hasLayersPrefix := false
	for _, t := range gf.Tensors {
		if strings.HasPrefix(t.Name, "blk.") {
			hasBlkPrefix = true
		}
		if strings.HasPrefix(t.Name, "layers.") {
			hasLayersPrefix = true
		}
	}

	// Most llama.cpp models use "blk." prefix
	if hasLayersPrefix && !hasBlkPrefix {
		report.addViolation("unusual_tensor_naming", SeverityInfo,
			"tensors use 'layers.' prefix instead of 'blk.' (uncommon for llama.cpp)")
	}
}

func checkMetadata(gf *gguf.File, report *PolicyReport) {
	// Check for duplicate-ish keys (case-insensitive)
	lowerKeys := make(map[string]string)
	for key := range gf.Metadata {
		lower := strings.ToLower(key)
		if existing, ok := lowerKeys[lower]; ok && existing != key {
			report.addViolation("suspicious_metadata_duplication", SeverityWarning,
				fmt.Sprintf("metadata keys %q and %q differ only by case", existing, key))
		}
		lowerKeys[lower] = key
	}

	// Very large metadata values
	for key, val := range gf.Metadata {
		if s, ok := val.(string); ok && len(s) > 1024*1024 {
			report.addViolation("oversized_metadata", SeverityWarning,
				fmt.Sprintf("metadata key %q has value >1MB (%d bytes)", key, len(s)))
		}
	}
}

func checkTensorShapes(gf *gguf.File, report *PolicyReport) {
	for _, t := range gf.Tensors {
		// Zero-dimension tensors
		if t.ElementCount == 0 {
			report.addViolation("zero_element_tensor", SeverityWarning,
				fmt.Sprintf("tensor %q has zero elements", t.Name))
			continue
		}

		// Extremely large single dimension
		for _, d := range t.Dims {
			if d > 1<<30 { // > 1 billion
				report.addViolation("extreme_dimension", SeverityWarning,
					fmt.Sprintf("tensor %q has dimension %d (>1B)", t.Name, d))
			}
		}

		// Scalar weights (suspicious in a model)
		if t.NDims == 0 || (t.NDims == 1 && t.Dims[0] == 1) {
			if strings.Contains(t.Name, "weight") {
				report.addViolation("scalar_weight", SeverityInfo,
					fmt.Sprintf("tensor %q is scalar but named as weight", t.Name))
			}
		}

		// Block layout consistency
		if err := gguf.ValidateBlockLayout(&t, t.ByteSize()); err != nil {
			report.addViolation("block_layout_mismatch", SeverityCritical, err.Error())
		}
	}
}

func checkTensorNames(gf *gguf.File, arch string, report *PolicyReport) {
	// Build set of known tensor name patterns for common architectures
	knownPatterns := getKnownTensorPatterns(arch)
	if len(knownPatterns) == 0 {
		return // no patterns for this architecture
	}

	for _, t := range gf.Tensors {
		matched := false
		for _, pattern := range knownPatterns {
			if matchesTensorPattern(t.Name, pattern) {
				matched = true
				break
			}
		}
		if !matched {
			report.addViolation("unknown_tensor_name", SeverityInfo,
				fmt.Sprintf("tensor %q does not match known patterns for %s", t.Name, arch))
		}
	}
}

// getKnownTensorPatterns returns tensor name suffix patterns for an architecture.
func getKnownTensorPatterns(arch string) []string {
	// Common patterns across llama-family architectures
	common := []string{
		"token_embd.weight",
		"output.weight",
		"output_norm.weight",
		"attn_q.weight",
		"attn_k.weight",
		"attn_v.weight",
		"attn_output.weight",
		"attn_norm.weight",
		"ffn_gate.weight",
		"ffn_up.weight",
		"ffn_down.weight",
		"ffn_norm.weight",
		"attn_q.bias",
		"attn_k.bias",
		"attn_v.bias",
		"attn_output.bias",
		"ffn_gate.bias",
		"ffn_up.bias",
		"ffn_down.bias",
	}

	switch arch {
	case "llama", "mistral", "qwen2", "gemma", "phi3", "phi":
		return common
	case "mixtral":
		// MoE adds expert tensors
		moe := append(common,
			"ffn_gate_inp.weight", // router
		)
		// Expert patterns: ffn_gate_exps, ffn_up_exps, ffn_down_exps
		for _, suffix := range []string{"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"} {
			moe = append(moe, suffix)
		}
		return moe
	default:
		return nil
	}
}

func matchesTensorPattern(name, pattern string) bool {
	return strings.HasSuffix(name, pattern) || name == pattern
}
