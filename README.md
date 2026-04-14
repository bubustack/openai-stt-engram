# 🗣️ OpenAI Speech-to-Text Engram

Streaming Engram that transcribes PCM audio using OpenAI Whisper/GPT-4o transcription APIs.

## 🌟 Highlights

- Supports batch and streaming runtime modes with the same transcription contract.
- Emits incremental and final transcript events for low-latency downstream routing.
- Supports identity allowlists/ignore lists so realtime stories can avoid self-transcription loops.
- Preserves structured transcript payloads for downstream templating and transport delivery.

## 🚀 Quick Start

```bash
make lint
go test ./...
make docker-build
```

Apply `Engram.yaml`, mount an `openai` secret with `API_KEY`, and reference the
template from your Story step.

## ⚙️ Configuration (`Engram.spec.with`)

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `model` | string | OpenAI transcription model ID to use for every request. | `gpt-4o-mini-transcribe` |
| `responseFormat` | string | Default response shape (`auto`, `json`, `text`, `srt`, `verbose_json`, `vtt`). | `auto` |
| `timestampGranularity` | string | `word`, `segment`, or `none`. | `none` |
| `includeLogProbs` | bool | Include token-level log probabilities when the model supports them. | `false` |
| `diarize` | bool | Enable speaker diarization for supported transcription models. | `false` |
| `prompt` | string | Primer text used to steer transcription output. | unset |
| `temperature` | number | Sampling temperature for transcription. | `0` |
| `language` | string | BCP-47 language hint applied to every transcription request. | `en` |
| `include` | []string | Values forwarded to OpenAI's `include` parameter (for example `logprobs`). | unset |
| `chunking` | object | Server-side VAD chunking settings (`mode`, `prefixPaddingMs`, `silenceDurationMs`, `threshold`). | unset |
| `task` | string | Default operation: `transcribe` or `translate`. | `transcribe` |
| `stream` | bool | Enable OpenAI SSE streaming mode by default. | `false` |
| `ignoreIdentities` | []string | Skip packets when the participant identity matches (supports `*`, `prefix*`). | unset |
| `allowIdentities` | []string | Optional allowlist; when provided, only matching identities are transcribed. | unset |

Use `ignoreIdentities` to prevent the engram from transcribing playback/agent participants (for
example `bubu-*` or a specific `{{ inputs.event.id }}-playback` identity). Pair it with
`allowIdentities` when the Story should only capture speech from a curated set of users.

## 🔐 Secrets

Secret `openai` must provide `API_KEY`. Optional overrides include `BASE_URL`, `ORG_ID`, and `PROJECT_ID`.

## 📥 Inputs

```json
{
  "audio": {
    "encoding": "pcm",
    "sampleRate": 48000,
    "channels": 1,
    "data": "<base64-encoded audio>"
  },
  "responseFormat": "json",
  "language": "en"
}
```

`audio` may be supplied inline via `data` or through shared-storage metadata.
`format` is still accepted as a legacy alias, but `responseFormat` is preferred.
Request payloads can also override `timestampGranularity`, `includeLogProbs`,
`diarize`, `prompt`, `temperature`, `include`, `chunking`, `task`, and `stream`.

## 📤 Outputs

```json
{
  "text": "hello world",
  "words": [...],
  "segments": [...]
}
```

`words` and `segments` are populated only when the template or request enables timestamp granularity.

## 🔄 Streaming Mode

When the engram runs in real-time mode it fans out structured messages over the transport:

| Type | Description |
|------|-------------|
| `speech.transcript.delta` | Incremental transcript text for low-latency rendering. |
| `speech.transcript.done` | Final chunk for the current stream, including usage/logprobs metadata. |
| `speech.transcript.v1` | Full transcription result (non-translation). |
| `speech.translation.v1` | Full translation result. |

Each payload includes provider/model metadata so downstream steps can decide whether to combine them with other vendors.

## 🧪 Local Development

- `make lint` – Run the shared lint and static-analysis checks.
- `go test ./...` – Run the transcription unit/integration tests.
- `make docker-build` – Build the engram image for local clusters.
- Set `BUBU_DEBUG=true` to log sanitized request summaries and returned transcripts without printing raw audio bytes.

## 🤝 Community & Support

- [Contributing](./CONTRIBUTING.md)
- [Support](./SUPPORT.md)
- [Security Policy](./SECURITY.md)
- [Code of Conduct](./CODE_OF_CONDUCT.md)
- [Discord](https://discord.gg/dysrB7D8H6)


## 📄 License

Copyright 2025 BubuStack.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
