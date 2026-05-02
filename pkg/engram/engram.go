package engram

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/bubustack/bobrapet/pkg/storage"
	sdk "github.com/bubustack/bubu-sdk-go"
	sdkengram "github.com/bubustack/bubu-sdk-go/engram"
	"github.com/bubustack/bubu-sdk-go/media"
	sttcfg "github.com/bubustack/openai-stt-engram/pkg/config"
	"github.com/bubustack/tractatus/transport"
	openai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/azure"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared/constant"
)

const (
	defaultOpenAIBaseURL      = "https://api.openai.com/v1"
	defaultAzureAPIV          = "2024-06-01"
	componentName             = "openai-stt-engram"
	modeAuto                  = "auto"
	taskTranslate             = "translate"
	taskTranscribe            = "transcribe"
	responseFormatJSON        = "json"
	responseFormatText        = "text"
	responseFormatSRT         = "srt"
	responseFormatVTT         = "vtt"
	responseFormatVerboseJSON = "verbose_json"
	audioEncodingWAV          = "wav"
	audioEncodingAAC          = "aac"
	audioEncodingFLAC         = "flac"
	audioEncodingMP3          = "mp3"
	audioEncodingMP4          = "mp4"
	audioEncodingOGG          = "ogg"
	audioEncodingWebM         = "webm"
	outputFieldText           = "text"
)

type Input = STTRequest

type audioPayload struct {
	Encoding   string                  `json:"encoding"`
	SampleRate flexibleInt             `json:"sampleRate"`
	Channels   flexibleInt             `json:"channels"`
	Data       string                  `json:"data"`
	Storage    *media.StorageReference `json:"storage,omitempty"`
}

type STTRequest struct {
	Audio                audioPayload     `json:"audio"`
	ResponseFormat       string           `json:"responseFormat"`
	Format               string           `json:"format"`
	Language             string           `json:"language"`
	Model                string           `json:"model"`
	TimestampGranularity string           `json:"timestampGranularity"`
	IncludeLogProbs      *bool            `json:"includeLogProbs"`
	Diarize              *bool            `json:"diarize"`
	Include              []string         `json:"include"`
	Prompt               string           `json:"prompt"`
	Temperature          *float64         `json:"temperature"`
	Chunking             *sttcfg.Chunking `json:"chunking"`
	Task                 string           `json:"task"`
	Stream               *bool            `json:"stream"`
}

type transcriptionStreamObserver interface {
	OnTextDelta(delta string) error
	OnTextDone(final string, logprobs []openai.TranscriptionTextDoneEventLogprob, usage map[string]any) error
}

type transcriptionStreamContext struct {
	Provider             string
	Model                string
	Language             string
	Task                 string
	ResponseFormat       string
	TimestampGranularity string
}

type transcriptionStreamContextAware interface {
	PrepareStreamContext(transcriptionStreamContext)
}

type preparedAudio struct {
	bytes      []byte
	encoding   string
	sampleRate int
	channels   int
	debug      map[string]any
}

type transcriptionSettings struct {
	model        string
	task         string
	useStreaming bool
	language     string
	prompt       string
	temp         *float64
	granularity  string
}

type streamingTranscriptState struct {
	builder   strings.Builder
	finalText string
	logprobs  []openai.TranscriptionTextDoneEventLogprob
	usage     map[string]any
}

func (r *STTRequest) resolveAudioBytes(ctx context.Context) ([]byte, error) {
	if strings.TrimSpace(r.Audio.Data) != "" {
		audioBytes, err := base64.StdEncoding.DecodeString(r.Audio.Data)
		if err != nil {
			return nil, fmt.Errorf("failed to decode audio data: %w", err)
		}
		return audioBytes, nil
	}
	if r.Audio.Storage != nil {
		sm, err := storage.SharedManager(ctx)
		if err != nil {
			return nil, fmt.Errorf("storage manager unavailable: %w", err)
		}
		data, err := media.ReadBlob(ctx, sm, r.Audio.Storage)
		if err != nil {
			return nil, fmt.Errorf("failed to load audio from storage: %w", err)
		}
		return data, nil
	}
	return nil, fmt.Errorf("audio payload required")
}

type OpenAISTT struct {
	cfg             sttcfg.Config
	secrets         *sdkengram.Secrets
	client          *openai.Client
	isAzure         bool
	azureDeployment string
}

func New() *OpenAISTT { return &OpenAISTT{} }

func (e *OpenAISTT) Init(_ context.Context, cfg sttcfg.Config, secrets *sdkengram.Secrets) error {
	cfg = sttcfg.Normalize(cfg)
	client, isAzure, deployment, err := newOpenAIClient(secrets)
	if err != nil {
		return err
	}
	e.cfg = cfg
	e.secrets = secrets
	e.client = client
	e.isAzure = isAzure
	e.azureDeployment = deployment
	return nil
}

func (e *OpenAISTT) Process(
	ctx context.Context,
	execCtx *sdkengram.ExecutionContext,
	req Input,
) (*sdkengram.Result, error) {
	logger := execCtx.Logger().With(
		"component", componentName,
		"mode", "batch",
	)
	result, err := e.transcribe(ctx, req, logger, nil)
	if err != nil {
		return nil, err
	}
	return sdkengram.NewResultFrom(result), nil
}

func (e *OpenAISTT) Stream(
	ctx context.Context,
	in <-chan sdkengram.InboundMessage,
	out chan<- sdkengram.StreamMessage,
) error {
	logger := sdk.LoggerFromContext(ctx).With(
		"component", componentName,
		"mode", "stream",
	)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case msg, ok := <-in:
			if !ok {
				return nil
			}
			hasData := hasStreamData(msg)
			if !hasData {
				if isHeartbeat(msg.Metadata) {
					e.logDebug(ctx, logger, "Ignoring heartbeat message")
					msg.Done()
					continue
				}
				logger.Warn("Ignoring empty stream message", "metadata", msg.Metadata)
				msg.Done()
				continue
			}
			if err := e.processStreamMessage(ctx, logger, msg, out); err != nil {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				logger.Warn("Failed to process STT stream packet", "error", err)
				msg.Done()
				continue
			}
		}
	}
}

func (e *OpenAISTT) processStreamMessage(
	ctx context.Context,
	baseLogger *slog.Logger,
	msg sdkengram.InboundMessage,
	out chan<- sdkengram.StreamMessage,
) error {
	logger := loggerWithParticipant(streamLogger(baseLogger, msg), msg.Metadata)
	if e.shouldSkipStreamMessage(ctx, logger, msg.Metadata) {
		msg.Done()
		return nil
	}

	req, skip, err := e.decodeStreamRequest(ctx, logger, msg)
	if err != nil {
		return err
	}
	if skip {
		msg.Done()
		return nil
	}

	observer := e.newStreamObserver(ctx, logger, msg.Metadata, out)
	result, err := e.transcribe(ctx, req, logger, observer)
	if err != nil {
		return err
	}

	addTranscriptionUserPrompt(result)
	return e.emitTranscriptionResult(ctx, logger, msg, out, result)
}

func streamLogger(baseLogger *slog.Logger, msg sdkengram.InboundMessage) *slog.Logger {
	logger := baseLogger
	if msg.MessageID != "" {
		logger = logger.With("messageID", msg.MessageID)
	}
	return logger
}

func loggerWithParticipant(logger *slog.Logger, metadata map[string]string) *slog.Logger {
	participantID := participantIdentityFromMetadata(metadata)
	if participantID == "" {
		return logger
	}
	return logger.With("participant", participantID)
}

func (e *OpenAISTT) shouldSkipStreamMessage(
	ctx context.Context,
	logger *slog.Logger,
	metadata map[string]string,
) bool {
	participantID := participantIdentityFromMetadata(metadata)
	if participantID == "" {
		if len(e.cfg.AllowIdentities) > 0 {
			logger.Warn("Dropping audio without participant identity because allowIdentities is configured")
			return true
		}
		return false
	}

	if ok, reason := e.identityAllowed(participantID); !ok {
		e.logDebug(ctx, logger, "Skipping stream message due to identity filter", "reason", reason)
		return true
	}
	return false
}

func (e *OpenAISTT) decodeStreamRequest(
	ctx context.Context,
	logger *slog.Logger,
	msg sdkengram.InboundMessage,
) (STTRequest, bool, error) {
	if msg.Audio != nil && len(msg.Audio.PCM) > 0 {
		return requestFromAudioFrame(logger, msg), false, nil
	}

	req, err := decodeJSONSTTRequest(msg)
	if err != nil {
		return STTRequest{}, false, err
	}
	if strings.TrimSpace(req.Audio.Data) == "" && req.Audio.Storage == nil {
		e.logDebug(ctx, logger, "Skipping stream message without audio payload", "metadata", msg.Metadata)
		return STTRequest{}, true, nil
	}
	return req, false, nil
}

func requestFromAudioFrame(logger *slog.Logger, msg sdkengram.InboundMessage) STTRequest {
	logger.Info("received AudioFrame",
		"bytes", len(msg.Audio.PCM),
		"sampleRate", msg.Audio.SampleRateHz,
		"channels", msg.Audio.Channels,
	)
	req := STTRequest{
		Audio: audioPayload{
			Encoding:   strings.ToLower(msg.Audio.Codec),
			SampleRate: flexibleInt{value: int(msg.Audio.SampleRateHz)},
			Channels:   flexibleInt{value: int(msg.Audio.Channels)},
			Data:       base64.StdEncoding.EncodeToString(msg.Audio.PCM),
		},
	}
	mergeStreamOptions(streamJSONBytes(msg), &req)
	return req
}

func mergeStreamOptions(raw []byte, req *STTRequest) {
	if len(raw) == 0 || req == nil {
		return
	}

	var opts STTRequest
	if err := json.Unmarshal(raw, &opts); err != nil {
		return
	}

	req.Model = opts.Model
	req.Language = opts.Language
	req.Task = opts.Task
	req.ResponseFormat = opts.ResponseFormat
	req.TimestampGranularity = opts.TimestampGranularity
	req.Stream = opts.Stream
	req.IncludeLogProbs = opts.IncludeLogProbs
	req.Diarize = opts.Diarize
	req.Include = opts.Include
	req.Prompt = opts.Prompt
	req.Temperature = opts.Temperature
	req.Chunking = opts.Chunking
}

func decodeJSONSTTRequest(msg sdkengram.InboundMessage) (STTRequest, error) {
	raw := streamJSONBytes(msg)
	if len(raw) == 0 {
		return STTRequest{}, fmt.Errorf("audio payload required (expected AudioFrame or JSON)")
	}

	var req STTRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return STTRequest{}, fmt.Errorf("failed to parse structured payload: %w", err)
	}
	return req, nil
}

func (e *OpenAISTT) newStreamObserver(
	ctx context.Context,
	logger *slog.Logger,
	metadata map[string]string,
	out chan<- sdkengram.StreamMessage,
) transcriptionStreamObserver {
	if out == nil {
		return nil
	}
	return newTranscriptFanoutObserver(
		ctx,
		logger,
		metadata,
		out,
		e.providerName(),
		e.debugEnabled(ctx, logger),
	)
}

func addTranscriptionUserPrompt(result map[string]any) {
	if text, ok := result[outputFieldText].(string); ok && strings.TrimSpace(text) != "" {
		result["userPrompt"] = text
	}
}

func buildTranscriptMetadata(
	result map[string]any,
	source map[string]string,
	provider string,
) map[string]string {
	metadata := cloneMetadata(source)
	metadata["provider"] = provider
	metadata["model"] = fmt.Sprint(result["model"])
	if strings.EqualFold(fmt.Sprint(result["task"]), taskTranslate) {
		metadata["type"] = transport.StreamTypeSpeechTranslation
		return metadata
	}
	metadata["type"] = transport.StreamTypeSpeechTranscript
	return metadata
}

func (e *OpenAISTT) emitTranscriptionResult(
	ctx context.Context,
	logger *slog.Logger,
	msg sdkengram.InboundMessage,
	out chan<- sdkengram.StreamMessage,
	result map[string]any,
) error {
	if out == nil {
		msg.Done()
		return nil
	}

	payloadBytes, err := json.Marshal(transcriptPayloadForDownstream(result))
	if err != nil {
		return fmt.Errorf("marshal stt result: %w", err)
	}
	metadata := buildTranscriptMetadata(result, msg.Metadata, e.providerName())
	streamMsg := jsonBinaryStreamMessage(metadata, payloadBytes)
	streamMsg.Inputs = append([]byte(nil), payloadBytes...)

	select {
	case out <- streamMsg:
		logger.Info("openai stt transcript sent", "model", metadata["model"])
		msg.Done()
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func transcriptPayloadForDownstream(result map[string]any) map[string]any {
	payload := make(map[string]any, len(result))
	for key, value := range result {
		switch key {
		case "model", "task", "responseFormat", "stream":
			continue
		default:
			payload[key] = value
		}
	}
	return payload
}

func streamJSONBytes(msg sdkengram.InboundMessage) []byte {
	if len(msg.Inputs) > 0 {
		return msg.Inputs
	}
	if len(msg.Payload) > 0 {
		return msg.Payload
	}
	if msg.Binary != nil && len(msg.Binary.Payload) > 0 {
		return msg.Binary.Payload
	}
	return nil
}

func hasStreamData(msg sdkengram.InboundMessage) bool {
	return len(msg.Inputs) > 0 ||
		len(msg.Payload) > 0 ||
		(msg.Binary != nil && len(msg.Binary.Payload) > 0) ||
		(msg.Audio != nil && len(msg.Audio.PCM) > 0)
}

func resolveRequestedResponseFormat(req STTRequest) string {
	return firstNonEmpty(req.ResponseFormat, req.Format)
}

func (e *OpenAISTT) transcribe(
	ctx context.Context,
	req STTRequest,
	logger *slog.Logger,
	observer transcriptionStreamObserver,
) (map[string]any, error) {
	audio, uploadFactory, err := e.prepareAudioPayload(ctx, req, logger)
	if err != nil {
		return nil, err
	}

	settings, err := e.resolveTranscriptionSettings(req, logger)
	if err != nil {
		return nil, err
	}
	if settings.task == taskTranslate {
		return e.runTranslation(
			ctx,
			uploadFactory,
			settings.model,
			settings.prompt,
			settings.temp,
			req,
			logger,
			audio.debug,
		)
	}

	params, requestSummary, responseFormat, err := e.buildTranscriptionParams(
		ctx,
		req,
		uploadFactory,
		settings,
		logger,
	)
	if err != nil {
		return nil, err
	}
	e.logSTTDebugRequest(ctx, logger, &req, audio.debug, requestSummary)

	if settings.useStreaming {
		if observerCtx, ok := observer.(transcriptionStreamContextAware); ok {
			observerCtx.PrepareStreamContext(transcriptionStreamContext{
				Provider:             e.providerName(),
				Model:                settings.model,
				Language:             settings.language,
				Task:                 settings.task,
				ResponseFormat:       string(params.ResponseFormat),
				TimestampGranularity: settings.granularity,
			})
		}
		return e.runStreamingTranscriptionWithObserver(
			ctx,
			params,
			settings.model,
			settings.language,
			settings.granularity,
			settings.task,
			logger,
			observer,
		)
	}

	return e.runBatchTranscription(ctx, params, settings, responseFormat, logger)
}

func (e *OpenAISTT) prepareAudioPayload(
	ctx context.Context,
	req STTRequest,
	logger *slog.Logger,
) (preparedAudio, func() *multipartAudioReader, error) {
	audioBytes, err := req.resolveAudioBytes(ctx)
	if err != nil {
		return preparedAudio{}, nil, err
	}

	audio := preparedAudio{
		bytes:      audioBytes,
		encoding:   strings.ToLower(strings.TrimSpace(req.Audio.Encoding)),
		sampleRate: req.Audio.SampleRate.Int(48000),
		channels:   req.Audio.Channels.Int(1),
	}

	logEncoding := audio.encoding
	if logEncoding == "" {
		logEncoding = "(default)"
	}
	logger.Info("received audio payload",
		"encoding", logEncoding,
		"bytes", len(audio.bytes),
		"sampleRate", audio.sampleRate,
		"channels", audio.channels,
	)

	if needsPCMWrapper(audio.encoding) || (audio.encoding == audioEncodingWAV && !looksLikeWAV(audio.bytes)) {
		if audio.encoding == audioEncodingWAV && !looksLikeWAV(audio.bytes) {
			logger.Warn("payload declared as wav but header missing, wrapping PCM as WAV",
				"bytes", len(audio.bytes),
			)
		}
		audio.bytes = wrapPCMAsWAV(audio.bytes, audio.sampleRate, audio.channels)
		e.logDebug(ctx, logger, "wrapped PCM payload as WAV",
			"bytes", len(audio.bytes),
			"sampleRate", audio.sampleRate,
			"channels", audio.channels,
		)
		audio.encoding = audioEncodingWAV
	}

	audio.debug = buildAudioDebugSummary(
		req.Audio,
		audio.encoding,
		audio.sampleRate,
		audio.channels,
		len(audio.bytes),
	)
	uploadFactory := func() *multipartAudioReader {
		return newMultipartAudioReader(audio.bytes, audio.encoding)
	}
	uploadPreview := uploadFactory()
	e.logDebug(ctx, logger, "prepared audio for OpenAI STT",
		"encoding", audio.encoding,
		"bytes", len(audio.bytes),
		"sampleRate", audio.sampleRate,
		"channels", audio.channels,
		"header", sniffHeader(audio.bytes),
		"uploadFilename", uploadPreview.Filename(),
		"contentType", uploadPreview.ContentType(),
	)
	return audio, uploadFactory, nil
}

func (e *OpenAISTT) resolveTranscriptionSettings(
	req STTRequest,
	logger *slog.Logger,
) (transcriptionSettings, error) {
	model, err := e.resolveModel(req.Model)
	if err != nil {
		return transcriptionSettings{}, err
	}

	settings := transcriptionSettings{
		model:        model,
		task:         resolveTask(req.Task, e.cfg.Task),
		useStreaming: e.cfg.Stream,
		language:     strings.TrimSpace(firstNonEmpty(req.Language, e.cfg.Language)),
		prompt:       firstNonEmpty(req.Prompt, e.cfg.Prompt),
		temp:         resolveTemperature(req.Temperature, e.cfg.Temperature),
		granularity:  firstNonEmpty(req.TimestampGranularity, e.cfg.TimestampGranularity),
	}
	if req.Stream != nil {
		settings.useStreaming = *req.Stream
	}
	if settings.task == taskTranslate && settings.useStreaming {
		logger.Info("translation task does not support SSE streaming; disabling stream flag")
		settings.useStreaming = false
	}
	return settings, nil
}

func (e *OpenAISTT) buildTranscriptionParams(
	ctx context.Context,
	req STTRequest,
	uploadFactory func() *multipartAudioReader,
	settings transcriptionSettings,
	logger *slog.Logger,
) (openai.AudioTranscriptionNewParams, map[string]any, openai.AudioResponseFormat, error) {
	requestedFormat := resolveRequestedResponseFormat(req)
	responseFormat, allowGranularity, err := resolveResponseFormat(
		settings.model,
		requestedFormat,
		e.cfg.ResponseFormat,
	)
	if err != nil {
		return openai.AudioTranscriptionNewParams{}, nil, "", err
	}

	params := openai.AudioTranscriptionNewParams{
		File:           uploadFactory(),
		Model:          settings.model,
		ResponseFormat: responseFormat,
	}
	if settings.language != "" {
		params.Language = openai.String(settings.language)
	}
	if settings.prompt != "" {
		params.Prompt = openai.String(settings.prompt)
	}
	if settings.temp != nil {
		params.Temperature = openai.Float(*settings.temp)
	}

	e.applyGranularity(settings, allowGranularity, responseFormat, &params, logger)

	includeLogProbs := e.resolveIncludeLogProbs(req, settings.model, logger)
	e.applyIncludeOptions(ctx, &params, includeLogProbs, req.Include, settings.model, logger)

	diarize := e.resolveDiarization(req, settings.model, logger)
	if diarize {
		params.SetExtraFields(map[string]any{"diarize": true})
	}
	e.applyChunking(ctx, &params, req.Chunking, logger)

	summary := buildTranscriptionSummary(req, settings, responseFormat, includeLogProbs, diarize)
	return params, summary, responseFormat, nil
}

func (e *OpenAISTT) applyGranularity(
	settings transcriptionSettings,
	allowGranularity bool,
	responseFormat openai.AudioResponseFormat,
	params *openai.AudioTranscriptionNewParams,
	logger *slog.Logger,
) {
	granularities := resolveGranularities(settings.granularity)
	if len(granularities) == 0 {
		return
	}
	if allowGranularity {
		params.TimestampGranularities = granularities
		return
	}
	logger.Info("timestamp granularity requires verbose_json response format",
		"model", settings.model,
		"granularities", granularities,
		"responseFormat", responseFormat,
	)
}

func (e *OpenAISTT) resolveIncludeLogProbs(
	req STTRequest,
	model string,
	logger *slog.Logger,
) bool {
	includeLogProbs := e.cfg.IncludeLogProbs
	if req.IncludeLogProbs != nil {
		includeLogProbs = *req.IncludeLogProbs
	}
	if includeLogProbs && !supportsLogProbs(model) {
		logger.Info("log probability output not supported by selected model; ignoring includeLogProbs",
			"model", model,
		)
		return false
	}
	return includeLogProbs
}

func (e *OpenAISTT) resolveDiarization(
	req STTRequest,
	model string,
	logger *slog.Logger,
) bool {
	diarize := e.cfg.Diarize
	if req.Diarize != nil {
		diarize = *req.Diarize
	}
	if diarize && !supportsDiarization(model) {
		logger.Info("speaker diarization requires gpt-4o-transcribe-diarize; ignoring diarize flag",
			"model", model,
		)
		return false
	}
	return diarize
}

func buildTranscriptionSummary(
	req STTRequest,
	settings transcriptionSettings,
	responseFormat openai.AudioResponseFormat,
	includeLogProbs bool,
	diarize bool,
) map[string]any {
	summary := map[string]any{
		"model":                settings.model,
		"language":             settings.language,
		"task":                 settings.task,
		"prompt":               settings.prompt,
		"temperature":          settings.temp,
		"responseFormat":       string(responseFormat),
		"timestampGranularity": settings.granularity,
		"useStreaming":         settings.useStreaming,
		"includeLogProbs":      includeLogProbs,
		"diarize":              diarize,
	}
	if len(req.Include) > 0 {
		summary["include"] = req.Include
	}
	if req.Chunking != nil {
		summary["chunking"] = req.Chunking
	}
	return summary
}

func (e *OpenAISTT) runBatchTranscription(
	ctx context.Context,
	params openai.AudioTranscriptionNewParams,
	settings transcriptionSettings,
	responseFormat openai.AudioResponseFormat,
	logger *slog.Logger,
) (map[string]any, error) {
	resp, err := e.client.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai stt request failed: %w", err)
	}

	logger.Info("openai stt transcript ready",
		"model", settings.model,
		"responseFormat", string(responseFormat),
		"textPreview", previewText(resp.Text, 160),
		"task", settings.task,
	)

	output := map[string]any{
		outputFieldText:        resp.Text,
		"model":                settings.model,
		"responseFormat":       string(responseFormat),
		"provider":             e.providerName(),
		"language":             settings.language,
		"timestampGranularity": settings.granularity,
		"task":                 settings.task,
		"type":                 transport.StreamTypeSpeechTranscript,
	}
	if len(resp.Logprobs) > 0 {
		output["logprobs"] = resp.Logprobs
	}
	if usage := usageToMap(resp.Usage); usage != nil {
		output["usage"] = usage
	}

	addVerboseTranscriptionFields(resp.RawJSON(), output)
	e.logSTTDebugOutput(ctx, logger, output)
	return output, nil
}

func addVerboseTranscriptionFields(raw string, output map[string]any) {
	if raw == "" {
		return
	}

	var verbose map[string]any
	if err := json.Unmarshal([]byte(raw), &verbose); err != nil {
		return
	}
	if words, ok := verbose["words"]; ok {
		output["words"] = words
	}
	if segments, ok := verbose["segments"]; ok {
		output["segments"] = segments
	}
	if speakers, ok := verbose["speakers"]; ok {
		output["speakers"] = speakers
	}
	if labels, ok := verbose["speaker_labels"]; ok {
		output["speakerLabels"] = labels
	}
	if speakerSegments, ok := verbose["speaker_segments"]; ok {
		output["speakerSegments"] = speakerSegments
	}
}

func (e *OpenAISTT) resolveModel(requested string) (string, error) {
	model := strings.TrimSpace(requested)
	if model == "" {
		model = e.cfg.Model
	}
	if e.isAzure && strings.TrimSpace(e.azureDeployment) != "" {
		model = e.azureDeployment
	}
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("model is required")
	}
	return model, nil
}

func (e *OpenAISTT) providerName() string {
	if e.isAzure {
		return "azure-openai"
	}
	return "openai"
}

func (e *OpenAISTT) applyIncludeOptions(
	ctx context.Context,
	params *openai.AudioTranscriptionNewParams,
	includeLogProbs bool,
	requestInclude []string,
	model string,
	logger *slog.Logger,
) {
	includeSet := make(map[string]struct{})
	if len(e.cfg.Include) > 0 {
		for _, inc := range sttcfg.NormalizeInclude(e.cfg.Include) {
			includeSet[inc] = struct{}{}
		}
	}
	if len(requestInclude) > 0 {
		for _, inc := range sttcfg.NormalizeInclude(requestInclude) {
			includeSet[inc] = struct{}{}
		}
	}
	if includeLogProbs {
		includeSet[string(openai.TranscriptionIncludeLogprobs)] = struct{}{}
	}
	if len(includeSet) == 0 {
		return
	}
	for inc := range includeSet {
		switch inc {
		case string(openai.TranscriptionIncludeLogprobs):
			if supportsLogProbs(model) {
				params.Include = append(params.Include, openai.TranscriptionIncludeLogprobs)
			} else {
				e.logDebug(ctx, logger, "logprobs include ignored; model unsupported", "model", model)
			}
		default:
			e.logDebug(ctx, logger, "ignoring unsupported include directive", "include", inc)
		}
	}
}

func (e *OpenAISTT) applyChunking(
	ctx context.Context,
	params *openai.AudioTranscriptionNewParams,
	override *sttcfg.Chunking,
	logger *slog.Logger,
) {
	effective := cloneChunking(override)
	if effective == nil {
		effective = cloneChunking(e.cfg.Chunking)
	}
	if effective == nil || effective.Mode == "off" {
		return
	}
	if effective.Mode == modeAuto {
		params.ChunkingStrategy = openai.AudioTranscriptionNewParamsChunkingStrategyUnion{
			OfAuto: constant.ValueOf[constant.Auto](),
		}
		e.logDebug(ctx, logger, "enabled auto chunking")
		return
	}
	config := openai.AudioTranscriptionNewParamsChunkingStrategyVadConfig{Type: "server_vad"}
	if effective.PrefixPaddingMs > 0 {
		config.PrefixPaddingMs = openai.Int(effective.PrefixPaddingMs)
	}
	if effective.SilenceDurationMs > 0 {
		config.SilenceDurationMs = openai.Int(effective.SilenceDurationMs)
	}
	if effective.Threshold > 0 {
		config.Threshold = openai.Float(effective.Threshold)
	}
	params.ChunkingStrategy = openai.AudioTranscriptionNewParamsChunkingStrategyUnion{
		OfAudioTranscriptionNewsChunkingStrategyVadConfig: &config,
	}
	e.logDebug(ctx, logger, "applied server VAD chunking",
		"prefixPaddingMs", effective.PrefixPaddingMs,
		"silenceMs", effective.SilenceDurationMs,
		"threshold", effective.Threshold,
	)
}

func cloneChunking(src *sttcfg.Chunking) *sttcfg.Chunking {
	if src == nil {
		return nil
	}
	cloned := *src
	return sttcfg.NormalizeChunking(&cloned)
}

func cloneMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return make(map[string]string)
	}
	cloned := make(map[string]string, len(in))
	for k, v := range in {
		trimmed := strings.TrimSpace(k)
		if trimmed == "" {
			continue
		}
		cloned[trimmed] = v
	}
	return cloned
}

func newOpenAIClient(secrets *sdkengram.Secrets) (*openai.Client, bool, string, error) {
	if secrets == nil {
		return nil, false, "", fmt.Errorf("secrets are required to initialize OpenAI client")
	}

	if endpoint := resolveSecret(secrets, "AZURE_ENDPOINT"); endpoint != "" {
		apiKey := resolveSecret(secrets, "AZURE_API_KEY")
		if apiKey == "" {
			return nil, false, "", fmt.Errorf("AZURE_API_KEY secret is required for Azure OpenAI")
		}
		deployment := resolveSecret(secrets, "AZURE_DEPLOYMENT")
		if deployment == "" {
			return nil, false, "", fmt.Errorf("AZURE_DEPLOYMENT secret is required for Azure OpenAI")
		}
		apiVersion := resolveSecret(secrets, "AZURE_API_VERSION")
		if apiVersion == "" {
			apiVersion = defaultAzureAPIV
		}
		httpClient := &http.Client{Timeout: 120 * time.Second}
		opts := []option.RequestOption{
			azure.WithEndpoint(endpoint, apiVersion),
			azure.WithAPIKey(apiKey),
			option.WithHTTPClient(httpClient),
		}
		client := openai.NewClient(opts...)
		return &client, true, deployment, nil
	}

	apiKey := resolveSecret(secrets, "OPENAI_API_KEY", "API_KEY")
	if apiKey == "" {
		return nil, false, "", fmt.Errorf("OPENAI_API_KEY secret is required")
	}
	baseURL := resolveSecret(secrets, "OPENAI_BASE_URL", "BASE_URL")
	if baseURL == "" {
		baseURL = defaultOpenAIBaseURL
	}
	trimmedBase := strings.TrimRight(baseURL, "/")
	httpClient := &http.Client{Timeout: 120 * time.Second}
	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
		option.WithHTTPClient(httpClient),
		option.WithBaseURL(trimmedBase),
	}
	if org := resolveSecret(secrets, "OPENAI_ORG_ID", "ORG_ID"); org != "" {
		opts = append(opts, option.WithOrganization(org))
	}
	if project := resolveSecret(secrets, "OPENAI_PROJECT_ID", "PROJECT_ID"); project != "" {
		opts = append(opts, option.WithProject(project))
	}
	client := openai.NewClient(opts...)
	return &client, false, "", nil
}

func resolveSecret(secrets *sdkengram.Secrets, keys ...string) string {
	if secrets == nil {
		return ""
	}
	for _, key := range keys {
		if strings.TrimSpace(key) == "" {
			continue
		}
		if val, ok := secrets.Get(key); ok {
			if trimmed := strings.TrimSpace(val); trimmed != "" {
				return trimmed
			}
		}
	}
	return ""
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

func resolveResponseFormat(model, requestFormat, configFormat string) (openai.AudioResponseFormat, bool, error) {
	desired := strings.TrimSpace(strings.ToLower(firstNonEmpty(requestFormat, configFormat)))
	if desired == modeAuto {
		desired = ""
	}

	isGPT4oFamily := strings.HasPrefix(strings.ToLower(model), "gpt-4o")
	gptJSONOnly := isGPT4oFamily && strings.Contains(strings.ToLower(model), "transcribe")

	if desired == "" {
		return openai.AudioResponseFormatJSON, false, nil
	}

	switch desired {
	case responseFormatJSON:
		return openai.AudioResponseFormatJSON, false, nil
	case responseFormatText:
		if gptJSONOnly {
			return "", false, fmt.Errorf("model %s only supports %s response format", model, responseFormatJSON)
		}
		return openai.AudioResponseFormatText, false, nil
	case responseFormatSRT:
		if gptJSONOnly {
			return "", false, fmt.Errorf("model %s only supports %s response format", model, responseFormatJSON)
		}
		return openai.AudioResponseFormatSRT, false, nil
	case responseFormatVerboseJSON:
		if gptJSONOnly {
			return "", false, fmt.Errorf("model %s only supports %s response format", model, responseFormatJSON)
		}
		return openai.AudioResponseFormatVerboseJSON, true, nil
	case responseFormatVTT:
		if gptJSONOnly {
			return "", false, fmt.Errorf("model %s only supports %s response format", model, responseFormatJSON)
		}
		return openai.AudioResponseFormatVTT, false, nil
	default:
		return "", false, fmt.Errorf("unsupported response format %q", desired)
	}
}

func resolveTemperature(requested *float64, defaultValue float64) *float64 {
	var value float64
	if requested != nil {
		value = *requested
	} else if defaultValue > 0 {
		value = defaultValue
	} else {
		return nil
	}
	if value < 0 {
		value = 0
	}
	if value > 1 {
		value = 1
	}
	return &value
}

func resolveGranularities(value string) []string {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" || value == "none" {
		return nil
	}

	add := func(out []string, candidate string) []string {
		for _, existing := range out {
			if existing == candidate {
				return out
			}
		}
		return append(out, candidate)
	}

	if !strings.Contains(value, ",") {
		switch value {
		case "word", "segment":
			return []string{value}
		default:
			return nil
		}
	}

	parts := strings.Split(value, ",")
	var out []string
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "word" || part == "segment" {
			out = add(out, part)
		}
	}
	return out
}

func supportsLogProbs(model string) bool {
	model = strings.ToLower(model)
	return strings.HasPrefix(model, "gpt-4o-transcribe") || strings.HasPrefix(model, "gpt-4o-mini-transcribe")
}

func supportsDiarization(model string) bool {
	return strings.EqualFold(model, "gpt-4o-transcribe-diarize")
}

func usageToMap(u openai.TranscriptionUsageUnion) map[string]any {
	switch usage := u.AsAny().(type) {
	case openai.TranscriptionUsageTokens:
		return map[string]any{
			"type":         usage.Type,
			"inputTokens":  usage.InputTokens,
			"outputTokens": usage.OutputTokens,
			"totalTokens":  usage.TotalTokens,
		}
	case openai.TranscriptionUsageDuration:
		return map[string]any{
			"type":    usage.Type,
			"seconds": usage.Seconds,
		}
	default:
		if raw := u.RawJSON(); raw != "" {
			var generic map[string]any
			if err := json.Unmarshal([]byte(raw), &generic); err == nil {
				return generic
			}
		}
	}
	return nil
}

func buildAudioDebugSummary(payload audioPayload, encoding string, sampleRate, channels, byteLen int) map[string]any {
	summary := map[string]any{
		"encoding":      encoding,
		"sampleRate":    sampleRate,
		"channels":      channels,
		"byteLength":    byteLen,
		"hasInlineData": strings.TrimSpace(payload.Data) != "",
	}
	if payload.Storage != nil {
		summary["storage"] = payload.Storage
	}
	return summary
}

func (e *OpenAISTT) runStreamingTranscriptionWithObserver(
	ctx context.Context,
	params openai.AudioTranscriptionNewParams,
	model string,
	language string,
	granularity string,
	task string,
	logger *slog.Logger,
	observer transcriptionStreamObserver,
) (map[string]any, error) {
	stream := e.client.Audio.Transcriptions.NewStreaming(ctx, params)
	if stream == nil {
		return nil, fmt.Errorf("openai streaming transcription not available")
	}
	defer func() {
		_ = stream.Close()
	}()

	state := &streamingTranscriptState{}

	for stream.Next() {
		if err := e.handleStreamingTranscriptionEvent(
			ctx,
			logger,
			stream.Current(),
			observer,
			state,
		); err != nil {
			return nil, err
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("openai streaming transcription failed: %w", err)
	}
	finalText := state.finalValue()
	responseFormat := string(params.ResponseFormat)
	if responseFormat == "" {
		responseFormat = responseFormatJSON
	}
	output := map[string]any{
		outputFieldText:        finalText,
		"model":                model,
		"provider":             e.providerName(),
		"language":             language,
		"timestampGranularity": granularity,
		"task":                 task,
		"responseFormat":       responseFormat,
		"type":                 transport.StreamTypeSpeechTranscript,
	}
	if len(state.logprobs) > 0 {
		output["logprobs"] = state.logprobs
	}
	if state.usage != nil {
		output["usage"] = state.usage
	}
	logger.Info("openai stt streaming transcript ready",
		"model", model,
		"responseFormat", responseFormat,
		"textPreview", previewText(finalText, 160),
	)
	e.logSTTDebugOutput(ctx, logger, output)
	return output, nil
}

func (e *OpenAISTT) handleStreamingTranscriptionEvent(
	ctx context.Context,
	logger *slog.Logger,
	event openai.TranscriptionStreamEventUnion,
	observer transcriptionStreamObserver,
	state *streamingTranscriptState,
) error {
	switch event.Type {
	case "transcript.text.delta":
		delta := event.AsTranscriptTextDelta()
		state.builder.WriteString(delta.Delta)
		if observer != nil {
			return observer.OnTextDelta(delta.Delta)
		}
		return nil
	case "transcript.text.done":
		done := event.AsTranscriptTextDone()
		state.finalText = done.Text
		if len(done.Logprobs) > 0 {
			state.logprobs = done.Logprobs
		}
		state.usage = streamUsageToMap(done.Usage)
		if observer == nil {
			return nil
		}
		return observer.OnTextDone(state.finalValue(), done.Logprobs, state.usage)
	default:
		e.logDebug(ctx, logger, "ignoring streaming transcription event", "type", event.Type)
		return nil
	}
}

func (s *streamingTranscriptState) finalValue() string {
	if s.finalText != "" {
		return s.finalText
	}
	return s.builder.String()
}

type transcriptFanoutObserver struct {
	ctx                  context.Context
	logger               *slog.Logger
	debug                bool
	out                  chan<- sdkengram.StreamMessage
	baseMetadata         map[string]string
	provider             string
	model                string
	language             string
	task                 string
	responseFormat       string
	timestampGranularity string
	sequence             int
}

func newTranscriptFanoutObserver(
	ctx context.Context,
	logger *slog.Logger,
	metadata map[string]string,
	out chan<- sdkengram.StreamMessage,
	provider string,
	debug bool,
) *transcriptFanoutObserver {
	cloned := cloneMetadata(metadata)
	return &transcriptFanoutObserver{
		ctx:          ctx,
		logger:       logger,
		debug:        debug,
		out:          out,
		baseMetadata: cloned,
		provider:     provider,
	}
}

func (o *transcriptFanoutObserver) PrepareStreamContext(ctx transcriptionStreamContext) {
	o.provider = ctx.Provider
	o.model = ctx.Model
	o.language = ctx.Language
	o.task = ctx.Task
	o.responseFormat = ctx.ResponseFormat
	o.timestampGranularity = ctx.TimestampGranularity
}

func (o *transcriptFanoutObserver) OnTextDelta(delta string) error {
	if o.out == nil {
		return nil
	}
	trimmed := strings.TrimSpace(delta)
	if trimmed == "" {
		return nil
	}
	o.sequence++
	payload := map[string]any{
		"delta":       delta,
		"sequence":    o.sequence,
		"language":    o.language,
		"type":        transport.StreamTypeSpeechTranscriptDelta,
		"provider":    o.provider,
		"granularity": o.timestampGranularity,
	}
	return o.emit(transport.StreamTypeSpeechTranscriptDelta, payload)
}

func (o *transcriptFanoutObserver) OnTextDone(
	final string,
	logprobs []openai.TranscriptionTextDoneEventLogprob,
	usage map[string]any,
) error {
	if o.out == nil {
		return nil
	}
	o.sequence++
	payload := map[string]any{
		outputFieldText: final,
		"sequence":      o.sequence,
		"language":      o.language,
		"type":          transport.StreamTypeSpeechTranscriptDone,
		"provider":      o.provider,
		"granularity":   o.timestampGranularity,
	}
	if len(logprobs) > 0 {
		payload["logprobs"] = logprobs
	}
	if usage != nil {
		payload["usage"] = usage
	}
	return o.emit(transport.StreamTypeSpeechTranscriptDone, payload)
}

func (o *transcriptFanoutObserver) emit(eventType string, payload map[string]any) error {
	if o.out == nil {
		return nil
	}
	metadata := cloneMetadata(o.baseMetadata)
	metadata["provider"] = o.provider
	metadata["model"] = o.model
	if o.language != "" {
		metadata["language"] = o.language
	}
	if o.task != "" {
		metadata["task"] = o.task
	}
	if o.responseFormat != "" {
		metadata["responseFormat"] = o.responseFormat
	}
	if o.timestampGranularity != "" {
		metadata["timestampGranularity"] = o.timestampGranularity
	}
	metadata["type"] = eventType
	metadata["sequence"] = strconv.Itoa(o.sequence)

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	msg := jsonBinaryStreamMessage(metadata, payloadBytes)
	select {
	case o.out <- msg:
		if o.debug && o.logger != nil {
			o.logger.Debug("emitted streaming stt event", "type", eventType, "sequence", o.sequence)
		}
		o.emitSignal(eventType, payload)
		return nil
	case <-o.ctx.Done():
		return o.ctx.Err()
	}
}

func (o *transcriptFanoutObserver) emitSignal(eventType string, payload map[string]any) {
	key := transcriptSignalKey(eventType)
	if key == "" {
		return
	}
	if err := sdk.EmitSignal(o.ctx, key, payload); err != nil && !errors.Is(err, sdk.ErrSignalsUnavailable) {
		o.logger.Warn("Failed to emit transcript signal", "signal", key, "error", err)
	}
}

func jsonBinaryStreamMessage(metadata map[string]string, payload []byte) sdkengram.StreamMessage {
	body := append([]byte(nil), payload...)
	return sdkengram.StreamMessage{
		Metadata: metadata,
		Payload:  body,
		Binary: &sdkengram.BinaryFrame{
			Payload:  body,
			MimeType: "application/json",
		},
	}
}

func transcriptSignalKey(eventType string) string {
	switch eventType {
	case transport.StreamTypeSpeechTranscriptDelta:
		return "speech.transcript.delta"
	case transport.StreamTypeSpeechTranscriptDone:
		return "speech.transcript.final"
	default:
		return ""
	}
}

func (e *OpenAISTT) runTranslation(
	ctx context.Context,
	uploadFactory func() *multipartAudioReader,
	model string,
	prompt string,
	temp *float64,
	req STTRequest,
	logger *slog.Logger,
	audioDebug map[string]any,
) (map[string]any, error) {
	params := openai.AudioTranslationNewParams{
		File:  uploadFactory(),
		Model: model,
	}
	language := strings.TrimSpace(firstNonEmpty(req.Language, e.cfg.Language))
	format, formatString, err := resolveTranslationResponseFormat(req.ResponseFormat, e.cfg.ResponseFormat)
	if err != nil {
		return nil, err
	}
	if formatString != "" {
		params.ResponseFormat = format
	}
	if prompt != "" {
		params.Prompt = openai.String(prompt)
	}
	if temp != nil {
		params.Temperature = openai.Float(*temp)
	}
	e.logSTTDebugRequest(ctx, logger, &req, audioDebug, map[string]any{
		"model":                model,
		"task":                 taskTranslate,
		"language":             language,
		"prompt":               prompt,
		"temperature":          temp,
		"responseFormat":       formatString,
		"timestampGranularity": req.TimestampGranularity,
		"useStreaming":         false,
	})
	resp, err := e.client.Audio.Translations.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("openai translation request failed: %w", err)
	}
	responseFormat := formatString
	if responseFormat == "" {
		responseFormat = responseFormatJSON
	}
	logger.Info("openai stt translation ready",
		"model", model,
		"responseFormat", responseFormat,
		"textPreview", previewText(resp.Text, 160),
	)
	output := map[string]any{
		outputFieldText:  resp.Text,
		"model":          model,
		"provider":       e.providerName(),
		"responseFormat": responseFormat,
		"language":       "en",
		"task":           taskTranslate,
		"type":           transport.StreamTypeSpeechTranslation,
	}
	if strings.EqualFold(responseFormat, responseFormatVerboseJSON) {
		if raw := resp.RawJSON(); raw != "" {
			var verbose map[string]any
			if err := json.Unmarshal([]byte(raw), &verbose); err == nil {
				if segments, ok := verbose["segments"]; ok {
					output["segments"] = segments
				}
				if words, ok := verbose["words"]; ok {
					output["words"] = words
				}
			}
		}
	}
	e.logSTTDebugOutput(ctx, logger, output)
	return output, nil
}

func streamUsageToMap(u openai.TranscriptionTextDoneEventUsage) map[string]any {
	if u.InputTokens == 0 && u.OutputTokens == 0 && u.TotalTokens == 0 {
		return nil
	}
	result := map[string]any{
		"type":         u.Type,
		"inputTokens":  u.InputTokens,
		"outputTokens": u.OutputTokens,
		"totalTokens":  u.TotalTokens,
	}
	if details := u.InputTokenDetails; details.AudioTokens != 0 || details.TextTokens != 0 {
		result["inputTokenDetails"] = map[string]any{
			"audioTokens": details.AudioTokens,
			"textTokens":  details.TextTokens,
		}
	}
	return result
}

func resolveTask(requested, fallback string) string {
	task := strings.TrimSpace(strings.ToLower(requested))
	if task == "" {
		task = strings.TrimSpace(strings.ToLower(fallback))
	}
	switch task {
	case taskTranslate, "translation":
		return taskTranslate
	default:
		return taskTranscribe
	}
}

func resolveTranslationResponseFormat(
	requested, config string,
) (openai.AudioTranslationNewParamsResponseFormat, string, error) {
	desired := strings.TrimSpace(strings.ToLower(firstNonEmpty(requested, config)))
	if desired == "" || desired == modeAuto {
		desired = responseFormatJSON
	}
	switch desired {
	case responseFormatJSON:
		return openai.AudioTranslationNewParamsResponseFormatJSON, responseFormatJSON, nil
	case responseFormatText:
		return openai.AudioTranslationNewParamsResponseFormatText, responseFormatText, nil
	case responseFormatSRT:
		return openai.AudioTranslationNewParamsResponseFormatSRT, responseFormatSRT, nil
	case responseFormatVerboseJSON:
		return openai.AudioTranslationNewParamsResponseFormatVerboseJSON, responseFormatVerboseJSON, nil
	case responseFormatVTT:
		return openai.AudioTranslationNewParamsResponseFormatVTT, responseFormatVTT, nil
	default:
		return "", "", fmt.Errorf("unsupported translation response format %q", desired)
	}
}

type flexibleInt struct {
	value int
}

func (f *flexibleInt) UnmarshalJSON(data []byte) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || string(data) == "null" {
		f.value = 0
		return nil
	}
	if data[0] == '"' {
		var raw string
		if err := json.Unmarshal(data, &raw); err != nil {
			return err
		}
		raw = strings.TrimSpace(raw)
		if raw == "" {
			f.value = 0
			return nil
		}
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			return fmt.Errorf("invalid integer value %q", raw)
		}
		f.value = parsed
		return nil
	}
	var parsed int
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}
	f.value = parsed
	return nil
}

func (f flexibleInt) Int(defaultValue int) int {
	if f.value != 0 {
		return f.value
	}
	return defaultValue
}

func needsPCMWrapper(encoding string) bool {
	switch encoding {
	case "", "pcm", "pcm16", "linear16", "raw", "s16le", "l16":
		return true
	case audioEncodingWAV,
		"wave",
		audioEncodingMP3,
		"mpga",
		"mpeg",
		audioEncodingMP4,
		"m4a",
		audioEncodingOGG,
		"oga",
		audioEncodingWebM,
		audioEncodingFLAC,
		audioEncodingAAC:
		return false
	default:
		return true
	}
}

func wrapPCMAsWAV(data []byte, sampleRate, channels int) []byte {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if channels <= 0 {
		channels = 1
	}
	blockAlign := channels * 2
	byteRate := sampleRate * blockAlign
	chunkSize := 36 + len(data)

	var buf bytes.Buffer
	buf.Grow(chunkSize + 8)
	buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(chunkSize))
	buf.WriteString("WAVEfmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(channels))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(byteRate))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(blockAlign))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(16))
	buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(len(data)))
	buf.Write(data)
	return buf.Bytes()
}

func looksLikeWAV(data []byte) bool {
	if len(data) < 12 {
		return false
	}
	return string(data[:4]) == "RIFF" && string(data[8:12]) == "WAVE"
}

func sniffHeader(data []byte) string {
	if len(data) == 0 {
		return ""
	}
	max := 8
	if len(data) < max {
		max = len(data)
	}
	return fmt.Sprintf("%x", data[:max])
}

type multipartAudioReader struct {
	*bytes.Reader
	filename    string
	contentType string
}

func (m *multipartAudioReader) Filename() string    { return m.filename }
func (m *multipartAudioReader) ContentType() string { return m.contentType }

func newMultipartAudioReader(data []byte, encoding string) *multipartAudioReader {
	ext, mimeType := resolveUploadPresentation(encoding)
	name := fmt.Sprintf("audio.%s", ext)
	return &multipartAudioReader{
		Reader:      bytes.NewReader(data),
		filename:    name,
		contentType: mimeType,
	}
}

func resolveUploadPresentation(encoding string) (string, string) {
	switch strings.ToLower(strings.TrimSpace(encoding)) {
	case audioEncodingWAV, "wave":
		return audioEncodingWAV, "audio/wav"
	case audioEncodingMP3, "mpeg", "mpga":
		return audioEncodingMP3, "audio/mpeg"
	case audioEncodingMP4, "m4a":
		return audioEncodingMP4, "audio/mp4"
	case audioEncodingOGG, "oga":
		return audioEncodingOGG, "audio/ogg"
	case audioEncodingWebM:
		return audioEncodingWebM, "audio/webm"
	case audioEncodingFLAC:
		return audioEncodingFLAC, "audio/flac"
	case audioEncodingAAC:
		return audioEncodingAAC, "audio/aac"
	default:
		return audioEncodingWAV, "audio/wav"
	}
}

func isHeartbeat(meta map[string]string) bool {
	return meta != nil && meta["bubu-heartbeat"] == "true"
}

func (e *OpenAISTT) identityAllowed(identity string) (bool, string) {
	if len(e.cfg.AllowIdentities) > 0 && !sttcfg.MatchIdentity(e.cfg.AllowIdentities, identity) {
		return false, "allowlist_miss"
	}
	if sttcfg.MatchIdentity(e.cfg.IgnoreIdentities, identity) {
		return false, "ignorelist_match"
	}
	return true, ""
}

func participantIdentityFromMetadata(meta map[string]string) string {
	if len(meta) == 0 {
		return ""
	}
	keys := []string{
		"participant.id",
		"participant.identity",
		"participant",
		"participant_identity",
		"participant-id",
		"participantId",
		"identity",
		"livekit-participant",
		"livekit.participant",
		"speaker",
		"speaker.id",
	}
	for _, key := range keys {
		if value := strings.TrimSpace(meta[key]); value != "" {
			return value
		}
	}
	for key, value := range meta {
		if strings.Contains(strings.ToLower(key), "participant") && strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func previewText(text string, limit int) string {
	text = strings.TrimSpace(text)
	if limit <= 0 || len(text) <= limit {
		return text
	}
	return text[:limit] + "…"
}
