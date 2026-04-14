package config

import (
	"slices"
	"strings"
)

const (
	DefaultModel                = "gpt-4o-mini-transcribe"
	DefaultResponseFormat       = "json"
	DefaultTimestampGranularity = "none"
	modeAuto                    = "auto"
	modeOff                     = "off"
	modeServerVAD               = "server_vad"
	taskTranslate               = "translate"
	taskTranscribe              = "transcribe"
)

// Config captures static configuration injected via Engram.spec.with.
type Config struct {
	Model                string    `json:"model" mapstructure:"model"`
	ResponseFormat       string    `json:"responseFormat" mapstructure:"responseFormat"`
	TimestampGranularity string    `json:"timestampGranularity" mapstructure:"timestampGranularity"`
	IncludeLogProbs      bool      `json:"includeLogProbs" mapstructure:"includeLogProbs"`
	Diarize              bool      `json:"diarize" mapstructure:"diarize"`
	Prompt               string    `json:"prompt" mapstructure:"prompt"`
	Temperature          float64   `json:"temperature" mapstructure:"temperature"`
	Language             string    `json:"language" mapstructure:"language"`
	Include              []string  `json:"include" mapstructure:"include"`
	Chunking             *Chunking `json:"chunking" mapstructure:"chunking"`
	Task                 string    `json:"task" mapstructure:"task"`
	Stream               bool      `json:"stream" mapstructure:"stream"`
	IgnoreIdentities     []string  `json:"ignoreIdentities" mapstructure:"ignoreIdentities"`
	AllowIdentities      []string  `json:"allowIdentities" mapstructure:"allowIdentities"`
}

// Chunking describes OpenAI's server-side VAD configuration.
type Chunking struct {
	Mode              string  `json:"mode" mapstructure:"mode"`
	PrefixPaddingMs   int64   `json:"prefixPaddingMs" mapstructure:"prefixPaddingMs"`
	SilenceDurationMs int64   `json:"silenceDurationMs" mapstructure:"silenceDurationMs"`
	Threshold         float64 `json:"threshold" mapstructure:"threshold"`
}

// Normalize applies defaults and clamps to sane ranges.
func Normalize(cfg Config) Config {
	cfg.Model = strings.TrimSpace(firstNonEmpty(cfg.Model, DefaultModel))
	cfg.ResponseFormat = normalizeFormat(cfg.ResponseFormat)
	cfg.TimestampGranularity = normalizeGranularity(cfg.TimestampGranularity)
	cfg.Prompt = strings.TrimSpace(cfg.Prompt)
	cfg.Temperature = clampTemperature(cfg.Temperature)
	cfg.Language = strings.TrimSpace(cfg.Language)
	if len(cfg.Include) > 0 {
		cfg.Include = NormalizeInclude(cfg.Include)
	}
	cfg.Chunking = NormalizeChunking(cfg.Chunking)
	cfg.Task = normalizeTask(cfg.Task)
	cfg.IgnoreIdentities = NormalizeIdentityFilters(cfg.IgnoreIdentities)
	cfg.AllowIdentities = NormalizeIdentityFilters(cfg.AllowIdentities)
	return cfg
}

func normalizeFormat(format string) string {
	format = strings.TrimSpace(strings.ToLower(format))
	if format == modeAuto || format == "" {
		return DefaultResponseFormat
	}
	return format
}

func normalizeGranularity(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return DefaultTimestampGranularity
	}
	return value
}

func clampTemperature(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		v = strings.TrimSpace(v)
		if v != "" {
			return v
		}
	}
	return ""
}

func NormalizeInclude(values []string) []string {
	set := make(map[string]struct{}, len(values))
	for _, v := range values {
		v = strings.TrimSpace(strings.ToLower(v))
		if v == "" {
			continue
		}
		set[v] = struct{}{}
	}
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for key := range set {
		out = append(out, key)
	}
	slices.Sort(out)
	return out
}

// NormalizeChunking ensures valid mode names and sane numeric bounds.
func NormalizeChunking(cfg *Chunking) *Chunking {
	if cfg == nil {
		return nil
	}
	mode := strings.TrimSpace(strings.ToLower(cfg.Mode))
	switch mode {
	case "", modeAuto:
		cfg.Mode = modeAuto
	case "off", "disabled":
		cfg.Mode = modeOff
	case modeServerVAD, "vad":
		cfg.Mode = modeServerVAD
	default:
		cfg.Mode = modeAuto
	}
	if cfg.PrefixPaddingMs < 0 {
		cfg.PrefixPaddingMs = 0
	}
	if cfg.SilenceDurationMs < 0 {
		cfg.SilenceDurationMs = 0
	}
	if cfg.Threshold < 0 {
		cfg.Threshold = 0
	}
	if cfg.Threshold > 1 {
		cfg.Threshold = 1
	}
	return cfg
}

func normalizeTask(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	switch value {
	case taskTranslate, "translation":
		return taskTranslate
	default:
		return taskTranscribe
	}
}

// NormalizeIdentityFilters cleans up identity filters (wildcards supported via suffix '*').
func NormalizeIdentityFilters(filters []string) []string {
	if len(filters) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(filters))
	for _, raw := range filters {
		filter := strings.ToLower(strings.TrimSpace(raw))
		if filter == "" {
			continue
		}
		set[filter] = struct{}{}
	}
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for f := range set {
		out = append(out, f)
	}
	slices.Sort(out)
	return out
}

// MatchIdentity returns true when candidate matches at least one filter.
// '*' matches everything, 'prefix*' performs prefix matching, otherwise exact compare.
func MatchIdentity(filters []string, candidate string) bool {
	if len(filters) == 0 {
		return false
	}
	candidate = strings.ToLower(strings.TrimSpace(candidate))
	for _, filter := range filters {
		switch {
		case filter == "*":
			return true
		case strings.HasSuffix(filter, "*"):
			prefix := strings.TrimSuffix(filter, "*")
			if strings.HasPrefix(candidate, prefix) {
				return true
			}
		default:
			if candidate == filter {
				return true
			}
		}
	}
	return false
}
