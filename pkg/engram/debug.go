package engram

import (
	"context"
	"log/slog"

	sdk "github.com/bubustack/bubu-sdk-go"
)

func (e *OpenAISTT) debugEnabled(ctx context.Context, logger *slog.Logger) bool {
	if sdk.DebugModeEnabled() {
		return true
	}
	if logger == nil {
		return false
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return logger.Enabled(ctx, slog.LevelDebug)
}

func (e *OpenAISTT) logSTTDebugRequest(
	ctx context.Context,
	logger *slog.Logger,
	req *STTRequest,
	audio map[string]any,
	summary map[string]any,
) {
	if !e.debugEnabled(ctx, logger) || req == nil {
		return
	}
	payload := make(map[string]any, len(summary)+2)
	for k, v := range summary {
		payload[k] = v
	}
	if audio != nil {
		payload["audio"] = audio
	}
	payload["prompt"] = req.Prompt
	payload["model"] = req.Model
	payload["requestOverrides"] = map[string]any{
		"responseFormat":       req.ResponseFormat,
		"language":             req.Language,
		"timestampGranularity": req.TimestampGranularity,
		"includeLogProbs":      req.IncludeLogProbs,
		"diarize":              req.Diarize,
		"stream":               req.Stream,
		"include":              req.Include,
		"chunking":             req.Chunking,
	}
	logger.Debug("openai stt request", slog.Any("payload", payload))
}

func (e *OpenAISTT) logSTTDebugOutput(ctx context.Context, logger *slog.Logger, output map[string]any) {
	if !e.debugEnabled(ctx, logger) || output == nil {
		return
	}
	logger.Debug("openai stt output", slog.Any("payload", output))
}

func (e *OpenAISTT) logDebug(ctx context.Context, logger *slog.Logger, msg string, attrs ...any) {
	if !e.debugEnabled(ctx, logger) || logger == nil {
		return
	}
	logger.Debug(msg, attrs...)
}
