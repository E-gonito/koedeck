"""Emotion tagging engine for koedeck.

Uses a local LLM (OpenAI-compatible API) to insert ElevenLabs v3 emotion tags
into dialogue lines. Includes strict validation: stripping tags from output
must produce byte-identical text to the input.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

from openai import AsyncOpenAI

from .config import load_config
from .models import Line, Project

# Regex to match [tag] patterns (ElevenLabs v3 style)
TAG_RE = re.compile(r"\[([^\]]+)\]")


class TagStatus(str, Enum):
    """Status of a tagging attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TagResult:
    """Result of tagging a single line."""

    line_id: str
    original_text: str
    tagged_text: str | None = None
    status: TagStatus = TagStatus.PENDING
    error: str | None = None
    attempts: int = 0


@dataclass
class TaggingSession:
    """Tracks state for a batch tagging run."""

    results: dict[str, TagResult] = field(default_factory=dict)
    cancelled: bool = False
    total: int = 0
    completed: int = 0

    @property
    def in_progress(self) -> bool:
        return not self.cancelled and self.completed < self.total


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def strip_tags(text: str) -> str:
    """Remove all [tag] markers from text."""
    return TAG_RE.sub("", text)


def validate_tagged_output(
    original_text: str, tagged_text: str, whitelist: list[str]
) -> tuple[bool, str]:
    """Validate that tagged output meets the hard invariant.

    Rules:
    1. Stripping all [...] tags from tagged_text must produce text
       byte-identical to original_text.
    2. Every tag used must be in the whitelist.

    Returns:
        (is_valid, error_message) — error_message is empty string if valid.
    """
    # Rule 1: Byte-identical after stripping
    stripped = strip_tags(tagged_text)
    if stripped != original_text:
        # Find the first difference for a useful error message
        for i, (a, b) in enumerate(zip(original_text, stripped)):
            if a != b:
                context_start = max(0, i - 10)
                return False, (
                    f"Text mismatch at position {i}: "
                    f"expected ...{repr(original_text[context_start:i+10])}... "
                    f"got ...{repr(stripped[context_start:i+10])}..."
                )
        if len(original_text) != len(stripped):
            return False, (
                f"Length mismatch: original={len(original_text)}, "
                f"stripped={len(stripped)}"
            )
        return False, "Text is not byte-identical after stripping tags"

    # Rule 2: All tags must be in whitelist
    tags_used = TAG_RE.findall(tagged_text)
    invalid_tags = [t for t in tags_used if t not in whitelist]
    if invalid_tags:
        return False, f"Tags not in whitelist: {invalid_tags}"

    return True, ""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_tagging_prompt(
    line: Line,
    character_hints: str,
    preceding_lines: list[Line],
    tag_whitelist: list[str],
) -> list[dict[str, str]]:
    """Build the chat messages for the tagging LLM call.

    Args:
        line: The dialogue line to tag.
        character_hints: Delivery hints for the character from config.
        preceding_lines: Up to 2 preceding lines for context.
        tag_whitelist: Allowed emotion tags.

    Returns:
        List of message dicts for the OpenAI chat API.
    """
    whitelist_str = ", ".join(tag_whitelist)

    system_prompt = f"""\
You are a voice direction assistant. Your job is to insert ElevenLabs v3 audio emotion tags into dialogue text.

RULES (non-negotiable):
1. Insert tags in square brackets at appropriate points in the text. Tags may appear at the start, middle, or end of the line.
2. You MUST use ONLY tags from this whitelist: [{whitelist_str}]
3. You MUST NOT change, add, remove, or reorder any words, punctuation, or whitespace in the original text.
4. Your output must be the EXACT original text with ONLY [tag] insertions added. Nothing else.
5. If unsure, use fewer tags rather than risk changing the text.
6. Return ONLY the tagged text. No explanations, no markdown, no quotes around it.

CHARACTER VOICE: {character_hints}"""

    # Build context from preceding lines
    context_lines = []
    for pl in preceding_lines:
        if pl.type == "direction":
            context_lines.append(f"({pl.text})")
        elif pl.character:
            context_lines.append(f"{pl.character}: {pl.text}")

    user_content = ""
    if context_lines:
        user_content += "PRECEDING CONTEXT:\n"
        user_content += "\n".join(context_lines)
        user_content += "\n\n"

    user_content += f"TAG THIS LINE ({line.character}):\n{line.text}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_retry_prompt(
    messages: list[dict[str, str]],
    previous_output: str,
    error: str,
) -> list[dict[str, str]]:
    """Extend the conversation with feedback about the validation failure."""
    return messages + [
        {"role": "assistant", "content": previous_output},
        {
            "role": "user",
            "content": (
                f"VALIDATION FAILED: {error}\n\n"
                "Please try again. Remember:\n"
                "- Output the EXACT original text with ONLY [tag] insertions.\n"
                "- Do NOT change any words, punctuation, or spacing.\n"
                "- Use ONLY tags from the allowed whitelist.\n"
                "- Return ONLY the tagged text, nothing else."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# LLM call + retry loop
# ---------------------------------------------------------------------------


async def tag_single_line(
    line: Line,
    preceding_lines: list[Line],
    config: dict,
    client: AsyncOpenAI | None = None,
) -> TagResult:
    """Tag a single dialogue line with emotion tags via the LLM.

    Retries up to 3 times on validation failure.

    Args:
        line: The line to tag.
        preceding_lines: Up to 2 lines before this one for context.
        config: The loaded config dict.
        client: Optional pre-built AsyncOpenAI client.

    Returns:
        TagResult with success or failure info.
    """
    result = TagResult(line_id=line.line_id, original_text=line.text)

    tagging_config = config.get("tagging", {})
    base_url = tagging_config.get("base_url", "http://localhost:11434/v1")
    model = tagging_config.get("model", "qwen3:8b")
    whitelist = tagging_config.get("tag_whitelist", [])

    # Get character hints
    char_config = config.get("characters", {}).get(line.character or "", {})
    hints = char_config.get("hints", "No specific hints.")

    if client is None:
        client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    # Build initial prompt
    messages = build_tagging_prompt(line, hints, preceding_lines, whitelist)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        result.attempts = attempt
        result.status = TagStatus.IN_PROGRESS

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=len(line.text) * 3,  # generous but bounded
            )

            tagged_output = response.choices[0].message.content
            if tagged_output is None:
                result.error = "LLM returned empty response"
                continue

            # Clean up: strip any leading/trailing whitespace or quotes
            tagged_output = tagged_output.strip()
            if tagged_output.startswith('"') and tagged_output.endswith('"'):
                tagged_output = tagged_output[1:-1]
            if tagged_output.startswith("'") and tagged_output.endswith("'"):
                tagged_output = tagged_output[1:-1]

            # Validate
            is_valid, error = validate_tagged_output(
                line.text, tagged_output, whitelist
            )

            if is_valid:
                result.tagged_text = tagged_output
                result.status = TagStatus.SUCCESS
                result.error = None
                return result

            # Validation failed — build retry prompt
            result.error = error
            messages = build_retry_prompt(messages, tagged_output, error)

        except Exception as e:
            result.error = f"LLM call failed: {str(e)}"

    # All retries exhausted
    result.status = TagStatus.FAILED
    return result


# ---------------------------------------------------------------------------
# Batch tagging
# ---------------------------------------------------------------------------


async def tag_episode(
    project: Project,
    config: dict,
    session: TaggingSession,
    max_concurrent: int = 2,
    on_progress: callable | None = None,
) -> TaggingSession:
    """Tag all dialogue lines in a project with concurrency control.

    Args:
        project: The project to tag.
        config: Loaded config dict.
        session: TaggingSession to track progress.
        max_concurrent: Max concurrent LLM calls (default 2 for local models).
        on_progress: Optional callback(session) called after each line completes.

    Returns:
        Updated TaggingSession with results.
    """
    # Get dialogue lines that need tagging
    dialogue_lines = [
        l for l in project.lines
        if l.type == "dialogue" and not l.orphaned
    ]

    session.total = len(dialogue_lines)
    session.completed = 0

    # Initialize results
    for line in dialogue_lines:
        session.results[line.line_id] = TagResult(
            line_id=line.line_id,
            original_text=line.text,
        )

    # Build a client once
    tagging_config = config.get("tagging", {})
    base_url = tagging_config.get("base_url", "http://localhost:11434/v1")
    client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    # Semaphore for concurrency control
    sem = asyncio.Semaphore(max_concurrent)

    async def tag_with_semaphore(line: Line, preceding: list[Line]):
        if session.cancelled:
            session.results[line.line_id].status = TagStatus.CANCELLED
            return

        async with sem:
            if session.cancelled:
                session.results[line.line_id].status = TagStatus.CANCELLED
                return

            result = await tag_single_line(line, preceding, config, client)
            session.results[line.line_id] = result
            session.completed += 1

            if on_progress:
                on_progress(session)

    # Build tasks with preceding context
    tasks = []
    all_lines = [l for l in project.lines if not l.orphaned]

    for line in dialogue_lines:
        # Find up to 2 preceding lines
        line_idx = next(
            (i for i, l in enumerate(all_lines) if l.line_id == line.line_id), 0
        )
        preceding = all_lines[max(0, line_idx - 2) : line_idx]
        tasks.append(tag_with_semaphore(line, preceding))

    # Run all with concurrency control
    await asyncio.gather(*tasks)

    return session
