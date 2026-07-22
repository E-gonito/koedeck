"""Phase 3 tests for koedeck: tag validation, whitelist, retry, batch."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koedeck.config import DEFAULT_TAG_WHITELIST
from koedeck.models import Line, Project
from koedeck.parser import parse_markdown
from koedeck.tagger import (
    TaggingSession,
    TagResult,
    TagStatus,
    build_tagging_prompt,
    strip_tags,
    tag_episode,
    tag_single_line,
    validate_tagged_output,
)


WHITELIST = DEFAULT_TAG_WHITELIST


class TestStripTags:
    """Tests for tag stripping."""

    def test_strips_single_tag(self):
        assert strip_tags("[excited]Hello world") == "Hello world"

    def test_strips_multiple_tags(self):
        assert strip_tags("[excited]Hello [angry]world") == "Hello world"

    def test_strips_mid_line_tags(self):
        text = "Hello [whispers]world, [excited]how are you?"
        assert strip_tags(text) == "Hello world, how are you?"

    def test_no_tags_unchanged(self):
        text = "Hello world, no tags here!"
        assert strip_tags(text) == text

    def test_strips_adjacent_tags(self):
        assert strip_tags("[excited][angry]Hello") == "Hello"

    def test_handles_empty_string(self):
        assert strip_tags("") == ""


class TestValidation:
    """Tests for the hard validation invariant."""

    def test_valid_tagged_output(self):
        original = "FORTY MILL— ahem. Regrettably..."
        tagged = "[shouting]FORTY MILL—[nervous] ahem. Regrettably..."
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert valid
        assert err == ""

    def test_rejects_altered_text(self):
        original = "FORTY MILL— ahem."
        tagged = "[shouting]FORTY MILLION— ahem."  # added letters
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid
        assert "mismatch" in err.lower() or "Text" in err

    def test_rejects_removed_text(self):
        original = "Hello world, how are you?"
        tagged = "[excited]Hello world"  # removed text
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid

    def test_rejects_reordered_text(self):
        original = "Hello world"
        tagged = "[excited]world Hello"
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid

    def test_rejects_added_punctuation(self):
        original = "Hello world"
        tagged = "[excited]Hello world!"
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid

    def test_rejects_off_whitelist_tag(self):
        original = "Hello world"
        tagged = "[happy]Hello world"  # 'happy' not in whitelist
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid
        assert "whitelist" in err.lower()

    def test_rejects_multiple_off_whitelist_tags(self):
        original = "Hello world"
        tagged = "[happy][joyful]Hello world"
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid
        assert "happy" in err
        assert "joyful" in err

    def test_accepts_all_whitelist_tags(self):
        """Every tag in the whitelist should be accepted."""
        original = "Hello"
        for tag in WHITELIST:
            tagged = f"[{tag}]Hello"
            valid, err = validate_tagged_output(original, tagged, WHITELIST)
            assert valid, f"Tag [{tag}] should be valid but got: {err}"

    def test_byte_identical_requirement(self):
        """Output must be BYTE-identical, not just similar."""
        original = "Hello  world"  # double space
        tagged = "[excited]Hello world"  # single space — should fail!
        valid, err = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid

    def test_preserves_em_dash(self):
        original = "FORTY MILL— ahem"
        tagged = "[shouting]FORTY MILL— ahem"
        valid, _ = validate_tagged_output(original, tagged, WHITELIST)
        assert valid

    def test_preserves_ellipsis(self):
        original = "Regrettably... I'll take..."
        tagged = "[nervous]Regrettably... I'll take..."
        valid, _ = validate_tagged_output(original, tagged, WHITELIST)
        assert valid

    def test_empty_tagged_output_fails(self):
        original = "Hello world"
        tagged = ""
        valid, _ = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid

    def test_tags_only_output_fails(self):
        original = "Hello world"
        tagged = "[excited][angry]"
        valid, _ = validate_tagged_output(original, tagged, WHITELIST)
        assert not valid


class TestPromptBuilder:
    """Tests for prompt construction."""

    def test_builds_messages_with_context(self):
        line = Line(
            type="dialogue",
            character="AIGIS",
            raw_text="AIGIS: Hello world",
            text="Hello world",
        )
        preceding = [
            Line(type="direction", raw_text="(Scene)", text="(Scene)"),
            Line(
                type="dialogue",
                character="MITSURU",
                raw_text="MITSURU: Good morning",
                text="Good morning",
            ),
        ]
        messages = build_tagging_prompt(line, "Flat robotic voice", preceding, WHITELIST)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # System prompt contains whitelist and hints
        assert "Flat robotic voice" in messages[0]["content"]
        assert "excited" in messages[0]["content"]

        # User message contains context and the line
        assert "MITSURU: Good morning" in messages[1]["content"]
        assert "(Scene)" in messages[1]["content"]
        assert "Hello world" in messages[1]["content"]

    def test_builds_messages_without_context(self):
        line = Line(
            type="dialogue",
            character="AIGIS",
            raw_text="AIGIS: Hello",
            text="Hello",
        )
        messages = build_tagging_prompt(line, "Hints", [], WHITELIST)
        assert len(messages) == 2
        assert "PRECEDING CONTEXT" not in messages[1]["content"]
        assert "Hello" in messages[1]["content"]

    def test_whitelist_in_system_prompt(self):
        line = Line(type="dialogue", character="X", raw_text="X: Hi", text="Hi")
        messages = build_tagging_prompt(line, "", [], ["angry", "shouting"])
        assert "angry" in messages[0]["content"]
        assert "shouting" in messages[0]["content"]


class TestTagSingleLine:
    """Tests for single-line tagging with mocked LLM."""

    @pytest.fixture
    def config(self):
        return {
            "tagging": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3:8b",
                "tag_whitelist": WHITELIST,
            },
            "characters": {
                "AIGIS": {"voice_id": "", "hints": "Flat robotic voice."},
            },
        }

    @pytest.fixture
    def line(self):
        return Line(
            type="dialogue",
            character="AIGIS",
            raw_text="AIGIS: Hello world",
            text="Hello world",
        )

    @pytest.mark.asyncio
    async def test_success_on_first_try(self, config, line):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[excited]Hello world"
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await tag_single_line(line, [], config, client=mock_client)

        assert result.status == TagStatus.SUCCESS
        assert result.tagged_text == "[excited]Hello world"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_retries_on_validation_failure(self, config, line):
        mock_client = AsyncMock()

        # First attempt: invalid (adds extra text)
        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "[excited]Hello world!"

        # Second attempt: valid
        good_response = MagicMock()
        good_response.choices = [MagicMock()]
        good_response.choices[0].message.content = "[excited]Hello world"

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[bad_response, good_response]
        )

        result = await tag_single_line(line, [], config, client=mock_client)

        assert result.status == TagStatus.SUCCESS
        assert result.tagged_text == "[excited]Hello world"
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_fails_after_3_retries(self, config, line):
        mock_client = AsyncMock()

        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "[excited]Hello WORLD"  # always invalid

        mock_client.chat.completions.create = AsyncMock(return_value=bad_response)

        result = await tag_single_line(line, [], config, client=mock_client)

        assert result.status == TagStatus.FAILED
        assert result.attempts == 3
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_handles_llm_exception(self, config, line):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        result = await tag_single_line(line, [], config, client=mock_client)

        assert result.status == TagStatus.FAILED
        assert "Connection refused" in result.error

    @pytest.mark.asyncio
    async def test_strips_quotes_from_output(self, config, line):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '"[excited]Hello world"'
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await tag_single_line(line, [], config, client=mock_client)

        assert result.status == TagStatus.SUCCESS
        assert result.tagged_text == "[excited]Hello world"


class TestBatchTagging:
    """Tests for batch tagging with concurrency control."""

    @pytest.mark.asyncio
    async def test_batch_tags_all_dialogue_lines(self):
        script = "AIGIS: Confirming purchase.\n\nMITSURU: Forty million yen.\n\n(Direction)\n"
        project = parse_markdown(script, "test.md")
        config = {
            "tagging": {
                "base_url": "http://localhost:11434/v1",
                "model": "test",
                "tag_whitelist": WHITELIST,
            },
            "characters": {
                "AIGIS": {"hints": ""},
                "MITSURU": {"hints": ""},
            },
        }
        session = TaggingSession()

        with patch("koedeck.tagger.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            async def fake_create(**kwargs):
                msgs = kwargs.get("messages", [])
                user_msg = msgs[-1]["content"] if msgs else ""
                # Use TAG THIS LINE to identify the actual line being tagged
                if "TAG THIS LINE" in user_msg and "Confirming purchase." in user_msg.split("TAG THIS LINE")[1]:
                    resp = MagicMock()
                    resp.choices = [MagicMock()]
                    resp.choices[0].message.content = "[excited]Confirming purchase."
                    return resp
                else:
                    resp = MagicMock()
                    resp.choices = [MagicMock()]
                    resp.choices[0].message.content = "[angry]Forty million yen."
                    return resp

            mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

            result_session = await tag_episode(
                project, config, session, max_concurrent=2
            )

        assert result_session.total == 2
        assert result_session.completed == 2

        successes = [
            r for r in result_session.results.values() if r.status == TagStatus.SUCCESS
        ]
        assert len(successes) == 2

    @pytest.mark.asyncio
    async def test_batch_respects_cancellation(self):
        script = "AIGIS: One.\n\nAIGIS: Two.\n\nAIGIS: Three.\n"
        project = parse_markdown(script, "test.md")
        config = {
            "tagging": {
                "base_url": "http://localhost:11434/v1",
                "model": "test",
                "tag_whitelist": WHITELIST,
            },
            "characters": {"AIGIS": {"hints": ""}},
        }
        session = TaggingSession()

        call_count = 0

        with patch("koedeck.tagger.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            async def slow_create(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # After first call, cancel the session
                    session.cancelled = True
                await asyncio.sleep(0.01)
                resp = MagicMock()
                resp.choices = [MagicMock()]
                resp.choices[0].message.content = "[excited]One."
                return resp

            mock_client.chat.completions.create = AsyncMock(side_effect=slow_create)

            await tag_episode(project, config, session, max_concurrent=1)

        # Not all lines should have been processed
        cancelled = [
            r
            for r in session.results.values()
            if r.status == TagStatus.CANCELLED
        ]
        assert len(cancelled) > 0

    @pytest.mark.asyncio
    async def test_concurrency_limited(self):
        """Verify that max 2 lines run concurrently."""
        script = "\n\n".join([f"AIGIS: Line {i}." for i in range(5)]) + "\n"
        project = parse_markdown(script, "test.md")
        config = {
            "tagging": {
                "base_url": "http://localhost:11434/v1",
                "model": "test",
                "tag_whitelist": WHITELIST,
            },
            "characters": {"AIGIS": {"hints": ""}},
        }
        session = TaggingSession()

        max_concurrent_seen = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        with patch("koedeck.tagger.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            async def tracked_create(**kwargs):
                nonlocal max_concurrent_seen, current_concurrent
                async with lock:
                    current_concurrent += 1
                    max_concurrent_seen = max(max_concurrent_seen, current_concurrent)

                await asyncio.sleep(0.05)

                async with lock:
                    current_concurrent -= 1

                # Return valid tagged text
                msgs = kwargs.get("messages", [])
                user_msg = msgs[-1]["content"] if msgs else ""
                # Extract the original text from the prompt
                for i in range(5):
                    if f"Line {i}." in user_msg:
                        resp = MagicMock()
                        resp.choices = [MagicMock()]
                        resp.choices[0].message.content = f"[excited]Line {i}."
                        return resp

                resp = MagicMock()
                resp.choices = [MagicMock()]
                resp.choices[0].message.content = "[excited]Line 0."
                return resp

            mock_client.chat.completions.create = AsyncMock(side_effect=tracked_create)

            await tag_episode(project, config, session, max_concurrent=2)

        assert max_concurrent_seen <= 2
