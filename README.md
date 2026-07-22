# koedeck

Script voice pipeline: import markdown dialogue scripts, edit in a tabbed web UI, auto-tag with ElevenLabs v3 emotion tags via a local LLM, and batch-generate WAV audio per line.

## Installation

### Quick install

```bash
curl -fsSL https://raw.githubusercontent.com/E-gonito/koedeck/main/install.sh | bash
```

This installs uv (if needed), koedeck, checks for ffmpeg and Ollama, and pulls the LLM model.

### Manual install

```bash
uv tool install git+https://github.com/E-gonito/koedeck.git
```

Then run:

```bash
koedeck
```

> Need uv? `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Developer install (from source)

```bash
git clone https://github.com/E-gonito/koedeck.git
cd koedeck
uv sync
cp .env.example .env
uv run app
```

### Prerequisites

| Dependency | Required? | Install |
|-----------|-----------|--------|
| [ElevenLabs API key](https://elevenlabs.io/) | Yes | Sign up, get key, add to .env |
| [Ollama](https://ollama.ai/) | Yes (for tagging) | Download from ollama.ai |
| LLM model | Yes (for tagging) | `ollama pull qwen3:8b` |
| [ffmpeg](https://formulae.brew.sh/formula/ffmpeg) | Optional | `brew install ffmpeg` (lower ElevenLabs tiers only) |

The built-in setup wizard checks all of these on first launch and guides you through fixing any issues.

### Ollama via Docker (alternative)

```bash
docker compose up -d
```

> Note: Docker on macOS cannot access the GPU. For best performance on Mac, install Ollama natively.

## Compatible platforms

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon) | Fully supported |
| macOS (Intel) | Works |
| Linux (x86_64) | Works |
| Linux (ARM64) | Ollama support newer |
| Windows (WSL2) | Works |
| Windows (native) | Untested |

Minimum 8GB RAM recommended (LLM model uses ~6GB).

## Tests

```bash
uv run pytest
```
