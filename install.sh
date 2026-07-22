#!/usr/bin/env bash
set -e

echo ""
echo "  koedeck installer"
echo "  ─────────────────"
echo ""

# Colors
RED="[0;31m"
GREEN="[0;32m"
YELLOW="[0;33m"
NC="[0m"

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

# 1. Check/install uv
if command -v uv &>/dev/null; then
    ok "uv already installed"
else
    echo "  Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        ok "uv installed"
    else
        fail "Failed to install uv"
        exit 1
    fi
fi

# 2. Install koedeck
echo "  Installing koedeck..."
uv tool install git+https://github.com/E-gonito/koedeck.git 2>/dev/null && ok "koedeck installed" || {
    # Already installed, try upgrade
    uv tool upgrade koedeck 2>/dev/null && ok "koedeck updated" || ok "koedeck already installed"
}

# 3. Check ffmpeg (optional)
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found"
else
    warn "ffmpeg not found (optional - only needed for lower ElevenLabs tiers)"
    echo "       Install later with: brew install ffmpeg"
fi

# 4. Check Ollama
if command -v ollama &>/dev/null; then
    ok "Ollama found"
    # Check if model is available
    if ollama list 2>/dev/null | grep -q "qwen3"; then
        ok "qwen3 model available"
    else
        echo "  Pulling qwen3:8b model (this takes a few minutes)..."
        ollama pull qwen3:8b && ok "Model pulled" || warn "Could not pull model - do it later: ollama pull qwen3:8b"
    fi
else
    warn "Ollama not found (needed for emotion tagging)"
    echo "       Install from: https://ollama.ai/"
fi

# 5. Setup .env if running from source
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    cp .env.example .env
    warn ".env created - add your ElevenLabs API key"
    echo "       Edit .env and set: ELEVENLABS_API_KEY=your-key"
fi

echo ""
echo "  ─────────────────"
echo -e "  ${GREEN}Done!${NC} Run koedeck to start:"
echo ""
echo "    koedeck"
echo ""
echo "  The setup wizard will guide you through any remaining config."
echo ""
