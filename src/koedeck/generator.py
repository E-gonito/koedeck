"""ElevenLabs TTS generation engine for koedeck.

Handles:
- Async TTS API calls via httpx
- PCM/MP3 tier fallback with ffmpeg transcode
- Per-character output folder structure
- Hash-based caching (skip unchanged lines)
- Take management (keep last 3, rename with _takeN suffix)
- Batch generation with concurrency control, retry, progress
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import httpx

from .models import AudioState, Line, Project

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
OUTPUT_DIR = Path("output")
MAX_TAKES = 3


# ---------------------------------------------------------------------------
# ffmpeg check
# ---------------------------------------------------------------------------


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which("ffmpeg") is not None


def get_ffmpeg_error_message() -> str:
    """Return a helpful error message if ffmpeg is missing."""
    return (
        "ffmpeg is required but not found on your system.\n"
        "Install it with: brew install ffmpeg"
    )


# ---------------------------------------------------------------------------
# Audio format handling
# ---------------------------------------------------------------------------


def wrap_pcm_to_wav(pcm_data: bytes, sample_rate: int = 44100, channels: int = 1, bit_depth: int = 16) -> bytes:
    """Wrap raw PCM data into a WAV container."""
    bytes_per_sample = bit_depth // 8
    data_size = len(pcm_data)
    file_size = 36 + data_size

    wav = bytearray()
    # RIFF header
    wav.extend(b"RIFF")
    wav.extend(struct.pack("<I", file_size))
    wav.extend(b"WAVE")
    # fmt chunk
    wav.extend(b"fmt ")
    wav.extend(struct.pack("<I", 16))  # chunk size
    wav.extend(struct.pack("<H", 1))  # PCM format
    wav.extend(struct.pack("<H", channels))
    wav.extend(struct.pack("<I", sample_rate))
    wav.extend(struct.pack("<I", sample_rate * channels * bytes_per_sample))  # byte rate
    wav.extend(struct.pack("<H", channels * bytes_per_sample))  # block align
    wav.extend(struct.pack("<H", bit_depth))
    # data chunk
    wav.extend(b"data")
    wav.extend(struct.pack("<I", data_size))
    wav.extend(pcm_data)

    return bytes(wav)


def transcode_mp3_to_wav(mp3_data: bytes) -> bytes:
    """Transcode MP3 data to WAV using ffmpeg."""
    if not check_ffmpeg():
        raise RuntimeError(get_ffmpeg_error_message())

    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "wav", "-acodec", "pcm_s16le", "pipe:1"],
        input=mp3_data,
        capture_output=True,
        timeout=30,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed: {proc.stderr.decode()[:200]}")

    return proc.stdout


# ---------------------------------------------------------------------------
# File naming and paths
# ---------------------------------------------------------------------------


def _slugify(text: str, max_words: int = 4) -> str:
    """Create a filesystem-safe slug from text (first ~4 words, lowercased, hyphenated)."""
    # Take first N words
    words = text.split()[:max_words]
    slug = "-".join(words).lower()
    # Remove non-alphanumeric except hyphens
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    # Collapse multiple hyphens
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:40]  # cap length


def get_output_path(line: Line) -> Path:
    """Get the canonical output path for a line's WAV file."""
    assert line.character is not None
    slug = _slugify(line.text)
    filename = f"{line.global_index:03d}_{line.line_id}_{slug}.wav"
    return OUTPUT_DIR / line.character / filename


def get_take_path(line: Line, take_number: int) -> Path:
    """Get the path for a specific take of a line."""
    assert line.character is not None
    slug = _slugify(line.text)
    filename = f"{line.global_index:03d}_{line.line_id}_{slug}_take{take_number}.wav"
    return OUTPUT_DIR / line.character / filename


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def compute_audio_hash(
    text_sent: str,
    voice_id: str,
    model_id: str,
    voice_settings: dict | None = None,
    output_format: str = "pcm_44100",
) -> str:
    """Compute the cache hash for a generation request."""
    parts = [text_sent, voice_id, model_id, output_format]
    if voice_settings:
        # Sort keys for deterministic hashing
        parts.append(str(sorted(voice_settings.items())))
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def get_text_for_tts(line: Line) -> str:
    """Get the text to send to TTS for a line.

    Uses tagged_text if available, otherwise plain text.
    Asserts that no parenthetical content is present.
    """
    text = line.tagged_text if line.tagged_text else line.text

    # Hard assertion: no parenthetical content should be in the TTS text
    # (parentheticals are stripped at parse time and tags don't add them)
    assert "(" not in text or "[" in text.split("(")[0], (
        f"Parenthetical content found in TTS text for line {line.line_id}: {text[:50]}"
    )

    return text


# ---------------------------------------------------------------------------
# Take management
# ---------------------------------------------------------------------------


def rotate_takes(line: Line) -> None:
    """Rotate the current file to a take slot, keeping last MAX_TAKES takes."""
    if line.audio.current_file is None:
        return

    current_path = Path(line.audio.current_file)
    if not current_path.exists():
        return

    # Find next take number
    existing_takes = line.audio.takes
    next_take_num = len(existing_takes) + 1

    take_path = get_take_path(line, next_take_num)
    take_path.parent.mkdir(parents=True, exist_ok=True)

    # Rename current to take
    current_path.rename(take_path)
    line.audio.takes.append(str(take_path))

    # Trim to MAX_TAKES
    while len(line.audio.takes) > MAX_TAKES:
        oldest = Path(line.audio.takes.pop(0))
        if oldest.exists():
            oldest.unlink()

    line.audio.current_file = None


# ---------------------------------------------------------------------------
# TTS API client
# ---------------------------------------------------------------------------


class GenStatus(str, Enum):
    """Status of a generation request."""
    QUEUED = "queued"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class GenResult:
    """Result of generating audio for a single line."""
    line_id: str
    status: GenStatus = GenStatus.QUEUED
    error: str | None = None
    output_path: str | None = None
    skipped: bool = False


@dataclass
class GenerationSession:
    """Tracks state for a batch generation run."""
    results: dict[str, GenResult] = field(default_factory=dict)
    cancelled: bool = False
    total: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    format_fallback_logged: bool = False  # Log fallback once, not per line

    @property
    def in_progress(self) -> bool:
        return not self.cancelled and self.completed < self.total


# Track whether we've fallen back to MP3 (persists for the session)
_use_mp3_fallback: bool = False


async def generate_single_line(
    line: Line,
    config: dict,
    api_key: str,
    session: GenerationSession,
    client: httpx.AsyncClient | None = None,
    force: bool = False,
) -> GenResult:
    """Generate TTS audio for a single line.

    Args:
        line: The dialogue line to generate.
        config: Loaded config dict.
        api_key: ElevenLabs API key.
        session: GenerationSession for tracking.
        client: Optional pre-built httpx client.
        force: If True, regenerate even if cache matches.

    Returns:
        GenResult with status and output path.
    """
    global _use_mp3_fallback

    result = GenResult(line_id=line.line_id)

    # Get voice config
    char_config = config.get("characters", {}).get(line.character or "", {})
    voice_id = char_config.get("voice_id", "")
    if not voice_id:
        result.status = GenStatus.FAILED
        result.error = f"No voice_id configured for {line.character}"
        return result

    gen_config = config.get("generation", {})
    model_id = gen_config.get("model_id", "eleven_v3")
    output_format = gen_config.get("output_format", "pcm_44100")
    fallback_format = gen_config.get("fallback_format", "mp3_44100_192")
    voice_settings = char_config.get("voice_settings")

    # Get text to send
    text_sent = get_text_for_tts(line)

    # Compute hash for caching
    effective_format = fallback_format if _use_mp3_fallback else output_format
    audio_hash = compute_audio_hash(text_sent, voice_id, model_id, voice_settings, effective_format)

    # Check cache (skip if unchanged and file exists)
    if not force and line.audio.audio_hash == audio_hash:
        if line.audio.current_file and Path(line.audio.current_file).exists():
            result.status = GenStatus.SKIPPED
            result.skipped = True
            result.output_path = line.audio.current_file
            return result

    # Rotate existing file to takes if regenerating
    if force and line.audio.current_file:
        rotate_takes(line)

    result.status = GenStatus.GENERATING

    # Build API request
    url = f"{ELEVENLABS_BASE_URL}/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    body: dict = {
        "text": text_sent,
        "model_id": model_id,
        "output_format": effective_format,
    }
    if voice_settings:
        body["voice_settings"] = voice_settings

    should_close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=60.0)

    try:
        response = await client.post(url, json=body, headers=headers)

        # Handle tier error — fall back to MP3
        if response.status_code == 422 and not _use_mp3_fallback:
            error_text = response.text
            if "format" in error_text.lower() or "tier" in error_text.lower():
                _use_mp3_fallback = True
                if not session.format_fallback_logged:
                    session.format_fallback_logged = True
                # Retry with fallback format
                body["output_format"] = fallback_format
                effective_format = fallback_format
                audio_hash = compute_audio_hash(
                    text_sent, voice_id, model_id, voice_settings, effective_format
                )
                response = await client.post(url, json=body, headers=headers)

        if response.status_code != 200:
            result.status = GenStatus.FAILED
            result.error = f"API error {response.status_code}: {response.text[:200]}"
            return result

        # Process audio data
        audio_data = response.content

        if effective_format.startswith("pcm_"):
            # Wrap raw PCM into WAV
            sample_rate = int(effective_format.split("_")[1])
            wav_data = wrap_pcm_to_wav(audio_data, sample_rate=sample_rate)
        elif effective_format.startswith("mp3_"):
            # Transcode MP3 to WAV
            wav_data = transcode_mp3_to_wav(audio_data)
        else:
            # Unknown format, try to transcode
            wav_data = transcode_mp3_to_wav(audio_data)

        # Write to output path
        output_path = get_output_path(line)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_data)

        # Update line's audio state
        line.audio.audio_hash = audio_hash
        line.audio.current_file = str(output_path)

        result.status = GenStatus.DONE
        result.output_path = str(output_path)
        return result

    except httpx.TimeoutException:
        result.status = GenStatus.FAILED
        result.error = "Request timed out"
        return result
    except Exception as e:
        result.status = GenStatus.FAILED
        result.error = str(e)
        return result
    finally:
        if should_close_client:
            await client.aclose()


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


@dataclass
class PreflightSummary:
    """Pre-flight summary before batch generation."""
    total_lines: int
    lines_to_generate: int
    lines_to_skip: int
    total_characters: int
    estimated_credits: float
    missing_voice_ids: list[str]


def compute_preflight(project: Project, config: dict) -> PreflightSummary:
    """Compute pre-flight summary for batch generation."""
    gen_config = config.get("generation", {})
    credits_per_char = gen_config.get("credits_per_character", 0.3)
    output_format = gen_config.get("output_format", "pcm_44100")
    model_id = gen_config.get("model_id", "eleven_v3")

    dialogue_lines = [
        l for l in project.lines if l.type == "dialogue" and not l.orphaned
    ]

    lines_to_generate = 0
    lines_to_skip = 0
    total_chars = 0
    missing_voice_ids: list[str] = []

    seen_missing: set[str] = set()

    for line in dialogue_lines:
        char_config = config.get("characters", {}).get(line.character or "", {})
        voice_id = char_config.get("voice_id", "")

        if not voice_id and line.character and line.character not in seen_missing:
            missing_voice_ids.append(line.character)
            seen_missing.add(line.character)

        text_sent = get_text_for_tts(line)
        voice_settings = char_config.get("voice_settings")

        # Check if cached
        effective_format = output_format  # simplified for preflight
        audio_hash = compute_audio_hash(
            text_sent, voice_id, model_id, voice_settings, effective_format
        )

        if line.audio.audio_hash == audio_hash and line.audio.current_file and Path(line.audio.current_file).exists():
            lines_to_skip += 1
        else:
            lines_to_generate += 1
            total_chars += len(text_sent)

    return PreflightSummary(
        total_lines=len(dialogue_lines),
        lines_to_generate=lines_to_generate,
        lines_to_skip=lines_to_skip,
        total_characters=total_chars,
        estimated_credits=total_chars * credits_per_char,
        missing_voice_ids=missing_voice_ids,
    )


async def generate_episode(
    project: Project,
    config: dict,
    api_key: str,
    session: GenerationSession,
    max_concurrent: int = 3,
    on_progress: Callable[[GenerationSession], None] | None = None,
) -> GenerationSession:
    """Generate audio for all dialogue lines in a project.

    Args:
        project: The project to generate.
        config: Loaded config dict.
        api_key: ElevenLabs API key.
        session: GenerationSession to track progress.
        max_concurrent: Max concurrent API requests (default 3).
        on_progress: Optional callback after each line completes.

    Returns:
        Updated GenerationSession.
    """
    global _use_mp3_fallback

    dialogue_lines = [
        l for l in project.lines if l.type == "dialogue" and not l.orphaned
    ]

    session.total = len(dialogue_lines)
    session.completed = 0
    session.skipped = 0
    session.failed = 0

    # Initialize results
    for line in dialogue_lines:
        session.results[line.line_id] = GenResult(line_id=line.line_id)

    # Semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent)

    # Shared client for connection pooling
    async with httpx.AsyncClient(timeout=60.0) as client:

        async def generate_with_retry(line: Line):
            if session.cancelled:
                session.results[line.line_id].status = GenStatus.CANCELLED
                return

            async with sem:
                if session.cancelled:
                    session.results[line.line_id].status = GenStatus.CANCELLED
                    return

                # Retry with exponential backoff on 429/5xx
                max_retries = 3
                for attempt in range(max_retries):
                    result = await generate_single_line(
                        line, config, api_key, session, client=client
                    )

                    if result.status in (GenStatus.DONE, GenStatus.SKIPPED):
                        break

                    if result.status == GenStatus.FAILED and result.error:
                        # Check if retryable (429 or 5xx)
                        if "429" in result.error or "5" == result.error[len("API error "):len("API error ") + 1:]:
                            if attempt < max_retries - 1:
                                wait = 2 ** (attempt + 1)  # 2, 4, 8 seconds
                                await asyncio.sleep(wait)
                                continue
                        break  # Non-retryable error

                session.results[line.line_id] = result

                if result.status == GenStatus.SKIPPED:
                    session.skipped += 1
                elif result.status == GenStatus.FAILED:
                    session.failed += 1

                session.completed += 1

                if on_progress:
                    on_progress(session)

        # Run all with concurrency control
        tasks = [generate_with_retry(line) for line in dialogue_lines]
        await asyncio.gather(*tasks)

    return session
