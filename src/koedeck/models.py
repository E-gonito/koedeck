"""Pydantic v2 data models for the koedeck project."""

from __future__ import annotations

import secrets
import string
from typing import Literal

from pydantic import BaseModel, Field


def _generate_line_id() -> str:
    """Generate a short random 6-character base36 ID."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(6))


class Parenthetical(BaseModel):
    """An inline parenthetical (delivery/visual cue) within a dialogue line."""

    text: str  # The full parenthetical text including parens, e.g. "(SUDDENLY PHARMA-AD SPEED, one breath:)"
    offset: int  # Character offset in the *clean* (stripped) text where this parenthetical was located


class AudioState(BaseModel):
    """Tracks audio generation state for a single line."""

    audio_hash: str | None = None  # sha256(text_sent + voice_id + model_id + voice_settings + output_format)
    current_file: str | None = None  # path to current canonical WAV
    takes: list[str] = Field(default_factory=list)  # paths to previous takes (most recent last)


class Line(BaseModel):
    """A single line in the script — either dialogue or a scene direction."""

    line_id: str = Field(default_factory=_generate_line_id)
    global_index: int = 0  # display/order only; recomputed, NOT part of identity
    type: Literal["dialogue", "direction"]
    character: str | None = None  # None for directions
    raw_text: str  # Original text as it appeared in the markdown (with escapes, with parentheticals)
    text: str  # Clean spoken text: parentheticals stripped, unescaped
    parentheticals: list[Parenthetical] = Field(default_factory=list)
    tagged_text: str | None = None  # text with [v3 tags] inserted; None until tagging
    audio: AudioState = Field(default_factory=AudioState)
    orphaned: bool = False  # True if line was not matched during re-import


class Project(BaseModel):
    """Top-level project model, serialized to project.json."""

    source_path: str  # Path to the original .md file
    characters: list[str] = Field(default_factory=list)  # Ordered by first appearance
    lines: list[Line] = Field(default_factory=list)

    def reindex(self) -> None:
        """Recompute global_index for all non-orphaned lines in order."""
        idx = 0
        for line in self.lines:
            if not line.orphaned:
                line.global_index = idx
                idx += 1

    def derive_characters(self) -> None:
        """Rebuild the characters list from lines, ordered by first appearance."""
        seen: set[str] = set()
        chars: list[str] = []
        for line in self.lines:
            if line.type == "dialogue" and line.character and line.character not in seen:
                seen.add(line.character)
                chars.append(line.character)
        self.characters = chars
