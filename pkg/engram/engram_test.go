package engram

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	sdkengram "github.com/bubustack/bubu-sdk-go/engram"
	sttcfg "github.com/bubustack/openai-stt-engram/pkg/config"
)

func TestParticipantIdentityFromMetadata(t *testing.T) {
	t.Run("prefers explicit keys", func(t *testing.T) {
		meta := map[string]string{
			"participant.id":      "speaker-123",
			"livekit-participant": "lk-456",
		}
		if got := participantIdentityFromMetadata(meta); got != "speaker-123" {
			t.Fatalf("expected participant.id to win, got %q", got)
		}
	})

	t.Run("falls back to livekit keys", func(t *testing.T) {
		meta := map[string]string{
			"livekit-participant": "lk-identity",
		}
		if got := participantIdentityFromMetadata(meta); got != "lk-identity" {
			t.Fatalf("expected livekit identity, got %q", got)
		}
	})

	t.Run("detects generic participant fields", func(t *testing.T) {
		meta := map[string]string{
			"some-participant": "fallback-id",
		}
		if got := participantIdentityFromMetadata(meta); got != "fallback-id" {
			t.Fatalf("expected fallback participant, got %q", got)
		}
	})

	t.Run("returns empty when absent", func(t *testing.T) {
		if got := participantIdentityFromMetadata(nil); got != "" {
			t.Fatalf("expected no identity, got %q", got)
		}
	})
}

func TestIdentityAllowed(t *testing.T) {
	eng := &OpenAISTT{
		cfg: sttcfg.Config{
			AllowIdentities:  []string{"user-*"},
			IgnoreIdentities: []string{"user-playback"},
		},
	}
	if ok, _ := eng.identityAllowed("operator-1"); ok {
		t.Fatalf("expected operator-1 to be rejected by allowlist")
	}
	if ok, reason := eng.identityAllowed("user-playback"); ok || reason != "ignorelist_match" {
		t.Fatalf("expected user-playback to be blocked by ignorelist, got ok=%v reason=%q", ok, reason)
	}
	if ok, reason := eng.identityAllowed("user-alice"); !ok || reason != "" {
		t.Fatalf("expected user-alice to pass filters, ok=%v reason=%q", ok, reason)
	}
}

func TestTranscriptFanoutObserverEmitMirrorsPayloadToBinary(t *testing.T) {
	out := make(chan sdkengram.StreamMessage, 1)
	observer := &transcriptFanoutObserver{
		ctx:          context.Background(),
		baseMetadata: map[string]string{"storyRun": "sr-1"},
		out:          out,
		provider:     "openai",
		model:        "gpt-4o-transcribe",
	}

	if err := observer.emit("speech.transcript.delta.v1", map[string]any{"text": "hello"}); err != nil {
		t.Fatalf("emit returned error: %v", err)
	}

	select {
	case msg := <-out:
		if msg.Binary == nil {
			t.Fatal("expected binary payload")
		}
		if !bytes.Equal(msg.Payload, msg.Binary.Payload) {
			t.Fatalf("expected mirrored payload, payload=%q binary=%q", string(msg.Payload), string(msg.Binary.Payload))
		}
	default:
		t.Fatal("expected emitted transcript message")
	}
}

func TestTranscriptFanoutObserverPayloadOmitsInternalConfig(t *testing.T) {
	out := make(chan sdkengram.StreamMessage, 1)
	observer := &transcriptFanoutObserver{
		ctx:            context.Background(),
		baseMetadata:   map[string]string{"storyRun": "sr-1"},
		out:            out,
		provider:       "openai",
		model:          "gpt-4o-transcribe",
		task:           "transcribe",
		responseFormat: responseFormatJSON,
	}

	if err := observer.OnTextDone("hello", nil, nil); err != nil {
		t.Fatalf("OnTextDone returned error: %v", err)
	}

	select {
	case msg := <-out:
		var payload map[string]any
		if err := json.Unmarshal(msg.Payload, &payload); err != nil {
			t.Fatalf("failed to decode payload: %v", err)
		}
		for _, key := range []string{"model", "task", "responseFormat", "stream"} {
			if _, ok := payload[key]; ok {
				t.Fatalf("expected payload to omit %q, got %#v", key, payload)
			}
		}
		if msg.Metadata["model"] != "gpt-4o-transcribe" {
			t.Fatalf("expected model metadata, got %#v", msg.Metadata)
		}
	default:
		t.Fatal("expected emitted transcript message")
	}
}

func TestEmitTranscriptionResultPayloadOmitsInternalConfig(t *testing.T) {
	out := make(chan sdkengram.StreamMessage, 1)
	engine := &OpenAISTT{cfg: sttcfg.Normalize(sttcfg.Config{})}
	msg := sdkengram.NewInboundMessage(sdkengram.StreamMessage{})
	result := map[string]any{
		outputFieldText:  "hello",
		"userPrompt":     "hello",
		"provider":       "openai",
		"model":          "gpt-4o-mini-transcribe",
		"task":           "transcribe",
		"responseFormat": responseFormatJSON,
		"stream":         false,
		"type":           "speech.transcript.v1",
	}

	if err := engine.emitTranscriptionResult(context.Background(), testLogger(t), msg, out, result); err != nil {
		t.Fatalf("emitTranscriptionResult returned error: %v", err)
	}

	select {
	case emitted := <-out:
		var payload map[string]any
		if err := json.Unmarshal(emitted.Payload, &payload); err != nil {
			t.Fatalf("failed to decode payload: %v", err)
		}
		for _, key := range []string{"model", "task", "responseFormat", "stream"} {
			if _, ok := payload[key]; ok {
				t.Fatalf("expected payload to omit %q, got %#v", key, payload)
			}
		}
		if payload[outputFieldText] != "hello" || payload["userPrompt"] != "hello" {
			t.Fatalf("expected transcript text fields, got %#v", payload)
		}
		if !bytes.Equal(emitted.Payload, emitted.Binary.Payload) || !bytes.Equal(emitted.Payload, emitted.Inputs) {
			t.Fatalf("expected payload, binary, and inputs to mirror sanitized output")
		}
		if emitted.Metadata["model"] != "gpt-4o-mini-transcribe" {
			t.Fatalf("expected model metadata, got %#v", emitted.Metadata)
		}
	default:
		t.Fatal("expected emitted transcript result")
	}
}

func TestStreamJSONBytesPrefersInputsThenPayloadThenBinary(t *testing.T) {
	msg := sdkengram.NewInboundMessage(sdkengram.StreamMessage{
		Inputs:  []byte(`{"task":"transcribe"}`),
		Payload: []byte(`{"task":"translate"}`),
		Binary: &sdkengram.BinaryFrame{
			Payload:  []byte(`{"task":"binary"}`),
			MimeType: "application/json",
		},
	})

	if got := string(streamJSONBytes(msg)); got != `{"task":"transcribe"}` {
		t.Fatalf("expected inputs to win, got %q", got)
	}
}

func TestStreamJSONBytesPrefersPayloadOverBinary(t *testing.T) {
	msg := sdkengram.NewInboundMessage(sdkengram.StreamMessage{
		Payload: []byte(`{"task":"translate"}`),
		Binary: &sdkengram.BinaryFrame{
			Payload:  []byte(`{"task":"binary"}`),
			MimeType: "application/json",
		},
	})

	if got := string(streamJSONBytes(msg)); got != `{"task":"translate"}` {
		t.Fatalf("expected payload to win, got %q", got)
	}
}

func TestHasStreamDataIncludesPayloadOnlyMessages(t *testing.T) {
	msg := sdkengram.NewInboundMessage(sdkengram.StreamMessage{
		Payload: []byte(`{"task":"transcribe"}`),
	})
	if !hasStreamData(msg) {
		t.Fatal("expected payload-only message to be treated as data")
	}
}

func TestResolveRequestedResponseFormatUsesLegacyAlias(t *testing.T) {
	req := STTRequest{Format: responseFormatVerboseJSON}
	if got := resolveRequestedResponseFormat(req); got != responseFormatVerboseJSON {
		t.Fatalf("expected legacy format alias to resolve, got %q", got)
	}
	req.ResponseFormat = responseFormatText
	if got := resolveRequestedResponseFormat(req); got != responseFormatText {
		t.Fatalf("expected responseFormat to win, got %q", got)
	}
}

func TestStreamContinuesAfterMalformedPacket(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	engine := &OpenAISTT{cfg: sttcfg.Normalize(sttcfg.Config{})}
	in := make(chan sdkengram.InboundMessage, 1)
	out := make(chan sdkengram.StreamMessage, 1)

	in <- sdkengram.NewInboundMessage(sdkengram.StreamMessage{
		Payload: []byte("{invalid json"),
	})
	close(in)

	if err := engine.Stream(ctx, in, out); err != nil {
		t.Fatalf("expected malformed packet to be skipped, got err=%v", err)
	}
}

func testLogger(t *testing.T) *slog.Logger {
	t.Helper()
	return slog.Default()
}
