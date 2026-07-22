# koedeck

Script voice pipeline — import markdown dialogue scripts, edit in a tabbed web UI, auto-tag with ElevenLabs v3 emotion tags via a local LLM, and batch-generate WAV audio per line.

## Quick start

```bash
# Install dependencies
uv sync

# Copy and fill in your API key
cp .env.example .env

# Run the app
uv run app
```

Opens at [localhost:8080](http://localhost:8080).

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for project management
- [ffmpeg](https://formulae.brew.sh/formula/ffmpeg) (`brew install ffmpeg`) for MP3→WAV transcoding
- [Ollama](https://ollama.ai/) (or any OpenAI-compatible local LLM) for emotion tagging
- [ElevenLabs](https://elevenlabs.io/) API key for voice generation

## Script format

Markdown dialogue scripts with speaker tags:

```
(Scene direction)

AIGIS: For you, Mitsuru-san, forty million yen.

MITSURU: FORTY MILL— ahem. Regrettably... I'll take... half a bag on credit\!

AIGIS: ...Confirming purchase. (SUDDENLY PHARMA-AD SPEED, one breath:) Please-be-advised... (normal speed, bright:) Would you also like a membership card?
```

- `SPEAKER:` lines are dialogue (sent to TTS)
- `(...)` lines are scene directions (kept for context, never spoken)
- Inline `(parentheticals)` are delivery cues — visible in editor, stripped from TTS

## Features

- **Tabbed editor** — Combined view + per-character tabs, all editing the same data
- **Autosave** — Debounced 1s, atomic writes to `project.json`
- **Stable line IDs** — Audio, cache, and take history keyed on immutable IDs
- **Re-import** — Edit your `.md` externally, re-import with fuzzy matching to preserve audio state
- **Emotion tagging** — Local LLM inserts ElevenLabs v3 `[tags]` with validation
- **Batch generation** — Async WAV generation with caching, takes, and retry
- **Per-character output** — `output/CHARACTER/001_id_slug.wav`

## Config

Generated on first import at `config.yaml`:

```yaml
tagging:
  base_url: "http://localhost:11434/v1"
  model: "qwen3:8b"
  tag_whitelist: [excited, angry, shouting, whispers, ...]
characters:
  AIGIS:
    voice_id: "your-elevenlabs-voice-id"
    hints: "Flat, professional, customer-service calm."
```

## Tests

```bash
uv run pytest
```

## Architecture

```
src/koedeck/
  models.py    — Pydantic v2 data models (Line, Project, AudioState)
  parser.py    — Markdown → Project
  exporter.py  — Project → Markdown (round-trips)
  reimport.py  — Re-import with fuzzy matching
  config.py    — config.yaml generation/loading
  app.py       — NiceGUI web frontend
```
