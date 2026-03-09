package analysis

import (
	"fmt"
	"strings"

	"github.com/SecAI-Hub/gguf-guard/gguf"
)

// ModelFamily describes the expected structure of a model architecture,
// including tensor naming conventions, expected shapes, and layer organization.
type ModelFamily struct {
	Name          string      `json:"name"`
	Architectures []string    `json:"architectures"` // matching arch strings
	TensorPrefix  string      `json:"tensor_prefix"` // "blk" or "layers"
	ExpectedRoles []string    `json:"expected_roles"`
	ShapeRules    []ShapeRule `json:"shape_rules,omitempty"`
}

// ShapeRule defines an expected relationship between tensor dimensions.
type ShapeRule struct {
	TensorSuffix string `json:"tensor_suffix"`
	DimCount     int    `json:"dim_count"` // expected number of dimensions
	Description  string `json:"description"`
}

// FamilyMatch describes how well a file matches a known model family.
type FamilyMatch struct {
	Family          string   `json:"family"`
	MatchScore      float64  `json:"match_score"` // 0-1
	MatchedRoles    int      `json:"matched_roles"`
	MissingRoles    int      `json:"missing_roles"`
	ExtraRoles      int      `json:"extra_roles"`
	ShapeViolations int      `json:"shape_violations"`
	Details         []string `json:"details,omitempty"`
}

// KnownFamilies contains built-in model family definitions.
var KnownFamilies = []ModelFamily{
	{
		Name:          "llama",
		Architectures: []string{"llama"},
		TensorPrefix:  "blk",
		ExpectedRoles: []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
			"attn_norm.weight",
			"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
			"ffn_norm.weight",
		},
		ShapeRules: []ShapeRule{
			{"attn_q.weight", 2, "attention query should be 2D"},
			{"attn_k.weight", 2, "attention key should be 2D"},
			{"attn_v.weight", 2, "attention value should be 2D"},
			{"ffn_gate.weight", 2, "feed-forward gate should be 2D"},
			{"ffn_norm.weight", 1, "feed-forward norm should be 1D"},
		},
	},
	{
		Name:          "mistral",
		Architectures: []string{"mistral"},
		TensorPrefix:  "blk",
		ExpectedRoles: []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
			"attn_norm.weight",
			"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
			"ffn_norm.weight",
		},
		ShapeRules: []ShapeRule{
			{"attn_q.weight", 2, "attention query should be 2D"},
			{"ffn_norm.weight", 1, "feed-forward norm should be 1D"},
		},
	},
	{
		Name:          "mixtral",
		Architectures: []string{"mixtral"},
		TensorPrefix:  "blk",
		ExpectedRoles: []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
			"attn_norm.weight",
			"ffn_gate_inp.weight",
			"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
			"ffn_norm.weight",
		},
	},
	{
		Name:          "qwen2",
		Architectures: []string{"qwen2"},
		TensorPrefix:  "blk",
		ExpectedRoles: []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
			"attn_norm.weight",
			"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
			"ffn_norm.weight",
		},
	},
	{
		Name:          "gemma",
		Architectures: []string{"gemma"},
		TensorPrefix:  "blk",
		ExpectedRoles: []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
			"attn_norm.weight",
			"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
			"ffn_norm.weight",
		},
	},
	{
		Name:          "phi",
		Architectures: []string{"phi3", "phi"},
		TensorPrefix:  "blk",
		ExpectedRoles: []string{
			"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
			"attn_norm.weight",
			"ffn_up.weight", "ffn_down.weight",
			"ffn_norm.weight",
		},
	},
}

// MatchFamily determines how well a GGUF file matches each known model family.
func MatchFamily(gf *gguf.File) []FamilyMatch {
	arch := gf.Architecture()
	var matches []FamilyMatch

	for _, fam := range KnownFamilies {
		fm := matchAgainstFamily(gf, &fam, arch)
		matches = append(matches, fm)
	}
	return matches
}

// BestFamilyMatch returns the highest-scoring family match, or nil if none score > 0.
func BestFamilyMatch(gf *gguf.File) *FamilyMatch {
	matches := MatchFamily(gf)
	var best *FamilyMatch
	for i := range matches {
		if matches[i].MatchScore > 0 && (best == nil || matches[i].MatchScore > best.MatchScore) {
			best = &matches[i]
		}
	}
	return best
}

func matchAgainstFamily(gf *gguf.File, fam *ModelFamily, arch string) FamilyMatch {
	fm := FamilyMatch{Family: fam.Name}

	// Architecture match bonus
	archMatch := false
	for _, a := range fam.Architectures {
		if strings.EqualFold(a, arch) {
			archMatch = true
			break
		}
	}
	if !archMatch && arch != "unknown" {
		fm.MatchScore = 0
		return fm
	}

	// Build set of observed roles across all layers
	observedRoles := make(map[string]bool)
	for _, t := range gf.Tensors {
		role := extractTensorRole(t.Name, fam.TensorPrefix)
		if role != "" {
			observedRoles[role] = true
		}
	}

	// Check expected roles
	for _, expectedRole := range fam.ExpectedRoles {
		if observedRoles[expectedRole] {
			fm.MatchedRoles++
		} else {
			fm.MissingRoles++
			fm.Details = append(fm.Details, fmt.Sprintf("missing expected role: %s", expectedRole))
		}
	}

	// Count extra roles (observed but not expected)
	expectedSet := make(map[string]bool)
	for _, r := range fam.ExpectedRoles {
		expectedSet[r] = true
	}
	for role := range observedRoles {
		if !expectedSet[role] && !isGlobalTensor(role) {
			fm.ExtraRoles++
		}
	}

	// Check shape rules
	for _, rule := range fam.ShapeRules {
		for _, t := range gf.Tensors {
			if strings.HasSuffix(t.Name, rule.TensorSuffix) {
				if int(t.NDims) != rule.DimCount {
					fm.ShapeViolations++
					fm.Details = append(fm.Details,
						fmt.Sprintf("shape violation: %s has %dD, expected %dD", t.Name, t.NDims, rule.DimCount))
				}
				break
			}
		}
	}

	// Compute match score
	totalExpected := float64(len(fam.ExpectedRoles))
	if totalExpected == 0 {
		fm.MatchScore = 0
		return fm
	}
	baseScore := float64(fm.MatchedRoles) / totalExpected
	if archMatch {
		baseScore = baseScore*0.7 + 0.3 // architecture match adds 30% baseline
	}
	// Penalize shape violations and extras
	penalty := float64(fm.ShapeViolations)*0.05 + float64(fm.ExtraRoles)*0.01
	fm.MatchScore = baseScore - penalty
	if fm.MatchScore < 0 {
		fm.MatchScore = 0
	}
	if fm.MatchScore > 1 {
		fm.MatchScore = 1
	}

	return fm
}

// extractTensorRole strips prefix and layer number from tensor name.
// "blk.15.attn_q.weight" -> "attn_q.weight"
func extractTensorRole(name, prefix string) string {
	parts := strings.Split(name, ".")
	if len(parts) < 3 {
		return ""
	}
	if parts[0] == prefix {
		return strings.Join(parts[2:], ".")
	}
	return ""
}

// isGlobalTensor returns true for tensors that are not per-layer.
func isGlobalTensor(role string) bool {
	globals := []string{
		"token_embd.weight", "output.weight", "output_norm.weight",
		"rope_freqs.weight",
	}
	for _, g := range globals {
		if role == g {
			return true
		}
	}
	return false
}
