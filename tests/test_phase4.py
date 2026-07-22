"""Phase 4 tests for koedeck: generation, caching, takes, file naming."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koedeck.generator import (
    GenResult,
    GenStatus,
    GenerationSession,
    PreflightSummary,
    _slugify,
    check_ffmpeg,
    compute_audio_hash,
    compute_preflight,
    generate_single_line,
    get_output_path,
    get_text_for_tts,
    rotate_takes,
    wrap_pcm_to_wav,
)
from koedeck.models import AudioState, Line, Project
from koedeck.parser import parse_markdown


class TestSlugify:
    """Tests for filename slug generation."""

    def test_basic_slug(self):
        assert _slugify("Hello world") == "hello-world"

    def test_limits_words(self):
        slug = _slugify("one two three four five six", max_words=4)
        assert slug == "one-two-three-four"

    def test_removes_special_chars(self):
        slug = _slugify("FORTY MILL— ahem.")
        assert slug == "forty-mill-ahem"

    def test_handles_ellipsis(self):
        slug = _slugify("...Confirming purchase. Please")
        assert slug == "confirming-purchase-please"

    def test_caps_length(self):
        long_text = " ".join(["word"] * 20)
        slug = _slugify(long_text, max_words=4)
        assert len(slug) <= 40


class TestOutputPath:
    """Tests for output file path generation."""

    def test_correct_format(self):
        line = Line(
            type="dialogue", character="AIGIS",
            raw_text="AIGIS: Hello world", text="Hello world",
            global_index=3, line_id="k3x9f2",
        )
        path = get_output_path(line)
        assert path == Path("output/AIGIS/003_k3x9f2_hello-world.wav")

    def test_zero_padded_index(self):
        line = Line(
            type="dialogue", character="MITSURU",
            raw_text="", text="Test line",
            global_index=42, line_id="abc123",
        )
        path = get_output_path(line)
        assert "042_abc123" in str(path)

    def test_character_folder(self):
        line = Line(
            type="dialogue", character="YUKARI",
            raw_text="", text="Hello",
            global_index=0, line_id="xyz789",
        )
        path = get_output_path(line)
        assert path.parent == Path("output/YUKARI")


class TestAudioHash:
    """Tests for cache hash computation."""

    def test_same_inputs_same_hash(self):
        h1 = compute_audio_hash("Hello", "v1", "eleven_v3")
        h2 = compute_audio_hash("Hello", "v1", "eleven_v3")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = compute_audio_hash("Hello", "v1", "eleven_v3")
        h2 = compute_audio_hash("World", "v1", "eleven_v3")
        assert h1 != h2

    def test_different_voice_different_hash(self):
        h1 = compute_audio_hash("Hello", "v1", "eleven_v3")
        h2 = compute_audio_hash("Hello", "v2", "eleven_v3")
        assert h1 != h2

    def test_different_format_different_hash(self):
        h1 = compute_audio_hash("Hello", "v1", "eleven_v3", output_format="pcm_44100")
        h2 = compute_audio_hash("Hello", "v1", "eleven_v3", output_format="mp3_44100_192")
        assert h1 != h2

    def test_voice_settings_affect_hash(self):
        h1 = compute_audio_hash("Hello", "v1", "eleven_v3", voice_settings={"stability": 0.5})
        h2 = compute_audio_hash("Hello", "v1", "eleven_v3", voice_settings={"stability": 0.8})
        assert h1 != h2


class TestGetTextForTTS:
    """Tests for TTS text extraction."""

    def test_uses_tagged_text_if_available(self):
        line = Line(
            type="dialogue", character="AIGIS",
            raw_text="", text="Hello world",
            tagged_text="[excited]Hello world",
        )
        assert get_text_for_tts(line) == "[excited]Hello world"

    def test_falls_back_to_plain_text(self):
        line = Line(
            type="dialogue", character="AIGIS",
            raw_text="", text="Hello world",
        )
        assert get_text_for_tts(line) == "Hello world"

    def test_no_parenthetical_content_in_tts(self):
        """Text sent to TTS never contains ( ) content from parentheticals."""
        script = "AIGIS: Hello (whispered:) world\n"
        project = parse_markdown(script, "test.md")
        line = project.lines[0]
        tts_text = get_text_for_tts(line)
        assert "whispered" not in tts_text
        assert "(" not in tts_text


class TestWrapPCM:
    """Tests for PCM to WAV wrapping."""

    def test_valid_wav_header(self):
        pcm = b"\x00" * 1000
        wav = wrap_pcm_to_wav(pcm)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[12:16] == b"fmt "
        assert wav[36:40] == b"data"

    def test_correct_data_size(self):
        pcm = b"\x00" * 512
        wav = wrap_pcm_to_wav(pcm)
        import struct
        data_size = struct.unpack_from("<I", wav, 40)[0]
        assert data_size == 512

    def test_pcm_data_preserved(self):
        pcm = b"\x01\x02\x03\x04" * 100
        wav = wrap_pcm_to_wav(pcm)
        assert wav[44:] == pcm


class TestRotateTakes:
    """Tests for take rotation."""

    def test_rotates_current_to_take(self, tmp_path: Path):
        # Create a fake current file
        char_dir = tmp_path / "output" / "AIGIS"
        char_dir.mkdir(parents=True)
        current = char_dir / "001_abc123_hello.wav"
        current.write_bytes(b"fake wav data")

        line = Line(
            type="dialogue", character="AIGIS",
            raw_text="", text="Hello", line_id="abc123",
            audio=AudioState(current_file=str(current)),
        )

        with patch("koedeck.generator.get_take_path") as mock_take:
            take_path = char_dir / "001_abc123_hello_take1.wav"
            mock_take.return_value = take_path
            rotate_takes(line)

        assert line.audio.current_file is None
        assert len(line.audio.takes) == 1
        assert take_path.exists()
        assert not current.exists()

    def test_limits_to_max_takes(self, tmp_path: Path):
        char_dir = tmp_path / "output" / "AIGIS"
        char_dir.mkdir(parents=True)

        # Pre-existing takes
        takes = []
        for i in range(3):
            take = char_dir / f"take{i+1}.wav"
            take.write_bytes(b"data")
            takes.append(str(take))

        current = char_dir / "current.wav"
        current.write_bytes(b"new data")

        line = Line(
            type="dialogue", character="AIGIS",
            raw_text="", text="Hello", line_id="abc123",
            audio=AudioState(current_file=str(current), takes=takes),
        )

        with patch("koedeck.generator.get_take_path") as mock_take:
            new_take = char_dir / "take4.wav"
            mock_take.return_value = new_take
            rotate_takes(line)

        # Should have 3 takes (oldest removed, new one added)
        assert len(line.audio.takes) == 3
        # Oldest take file should be deleted
        assert not Path(takes[0]).exists()


class TestCacheSkip:
    """Tests for cache-based skip logic."""

    @pytest.mark.asyncio
    async def test_skips_unchanged_line(self, tmp_path: Path):
        """Unchanged line with existing file is skipped."""
        char_dir = tmp_path / "output" / "AIGIS"
        char_dir.mkdir(parents=True)

        line = Line(
            type="dialogue", character="AIGIS",
            raw_text="", text="Hello world", line_id="abc123",
            global_index=0,
        )

        config = {
            "generation": {"model_id": "eleven_v3", "output_format": "pcm_44100"},
            "characters": {"AIGIS": {"voice_id": "voice123"}},
        }

        # Pre-compute hash and create fake file
        text_sent = get_text_for_tts(line)
        expected_hash = compute_audio_hash(text_sent, "voice123", "eleven_v3", output_format="pcm_44100")
        fake_file = char_dir / "000_abc123_hello-world.wav"
        fake_file.write_bytes(b"existing wav")

        line.audio.audio_hash = expected_hash
        line.audio.current_file = str(fake_file)

        session = GenerationSession()
        result = await generate_single_line(
            line, config, "test-key", session
        )

        assert result.status == GenStatus.SKIPPED
        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_regenerates_changed_line(self):
        """Changed line (different hash) triggers regeneration."""
        line = Line(
            type="dialogue", character="AIGIS",
            raw_text="", text="New text now", line_id="abc123",
            global_index=0,
            audio=AudioState(audio_hash="old_hash_value"),
        )

        config = {
            "generation": {"model_id": "eleven_v3", "output_format": "pcm_44100"},
            "characters": {"AIGIS": {"voice_id": "voice123"}},
        }

        session = GenerationSession()

        # Mock the HTTP call
        with patch("koedeck.generator.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"\x00" * 1000  # fake PCM
            mock_client.post = AsyncMock(return_value=mock_response)

            result = await generate_single_line(
                line, config, "test-key", session, client=mock_client
            )

        # Should attempt to generate (may succeed or fail based on file write)
        assert result.status in (GenStatus.DONE, GenStatus.GENERATING)


class TestPreflight:
    """Tests for pre-flight summary."""

    def test_counts_lines_correctly(self):
        script = "AIGIS: Hello.\n\nMITSURU: World.\n\n(Direction)\n"
        project = parse_markdown(script, "test.md")
        config = {
            "generation": {"model_id": "eleven_v3", "output_format": "pcm_44100", "credits_per_character": 0.3},
            "characters": {
                "AIGIS": {"voice_id": "v1"},
                "MITSURU": {"voice_id": "v2"},
            },
        }

        preflight = compute_preflight(project, config)
        assert preflight.total_lines == 2
        assert preflight.lines_to_generate == 2
        assert preflight.lines_to_skip == 0
        assert preflight.missing_voice_ids == []

    def test_detects_missing_voice_ids(self):
        script = "AIGIS: Hello.\n\nMITSURU: World.\n"
        project = parse_markdown(script, "test.md")
        config = {
            "generation": {"model_id": "eleven_v3", "output_format": "pcm_44100", "credits_per_character": 0.3},
            "characters": {
                "AIGIS": {"voice_id": "v1"},
                "MITSURU": {"voice_id": ""},  # missing!
            },
        }

        preflight = compute_preflight(project, config)
        assert "MITSURU" in preflight.missing_voice_ids

    def test_counts_cached_lines(self, tmp_path: Path):
        script = "AIGIS: Hello.\n"
        project = parse_markdown(script, "test.md")
        config = {
            "generation": {"model_id": "eleven_v3", "output_format": "pcm_44100", "credits_per_character": 0.3},
            "characters": {"AIGIS": {"voice_id": "v1"}},
        }

        # Pre-cache the line
        line = project.lines[0]
        text_sent = get_text_for_tts(line)
        h = compute_audio_hash(text_sent, "v1", "eleven_v3", output_format="pcm_44100")
        fake_file = tmp_path / "cached.wav"
        fake_file.write_bytes(b"data")
        line.audio.audio_hash = h
        line.audio.current_file = str(fake_file)

        preflight = compute_preflight(project, config)
        assert preflight.lines_to_skip == 1
        assert preflight.lines_to_generate == 0

    def test_estimates_credits(self):
        script = "AIGIS: Hello world this is a test.\n"
        project = parse_markdown(script, "test.md")
        config = {
            "generation": {"model_id": "eleven_v3", "output_format": "pcm_44100", "credits_per_character": 0.5},
            "characters": {"AIGIS": {"voice_id": "v1"}},
        }

        preflight = compute_preflight(project, config)
        expected_chars = len("Hello world this is a test.")
        assert preflight.total_characters == expected_chars
        assert preflight.estimated_credits == expected_chars * 0.5
