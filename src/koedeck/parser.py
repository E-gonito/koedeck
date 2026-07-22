"""Markdown script parser for koedeck.

Parses dialogue scripts into structured Line objects with stable IDs.

Parsing rules:
1. A dialogue line starts with SPEAKERNAME: (all-caps, colon) at line start.
   Everything until the next blank line belongs to that line.
2. A line starting with ( containing no speaker tag is a scene direction.
3. Inline parentheticals are extracted and stored with offsets.
4. Markdown escapes (\\! → !, \\+ → +, etc.) are unescaped.
5. Em-dashes, ellipses, ALL-CAPS, and punctuation are preserved exactly.
"""

from __future__ import annotations

import re

from .models import Line, Parenthetical, Project

# Matches a speaker tag at the start of a line: one or more ALL-CAPS words followed by a colon
_SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 '-]+?):\s*(.*)$", re.DOTALL)

# Matches inline parentheticals: text enclosed in ( )
_PAREN_RE = re.compile(r"\([^)]*\)")

# Matches markdown escape sequences: backslash followed by punctuation
_ESCAPE_RE = re.compile(r"\\([^\w\s])")


def _unescape_markdown(text: str) -> str:
    """Remove markdown escape backslashes: \\! → !, \\+ → +, etc."""
    return _ESCAPE_RE.sub(r"\1", text)


def _extract_parentheticals(text: str) -> tuple[str, list[Parenthetical]]:
    """Extract inline parentheticals from dialogue text.

    Returns:
        (clean_text, parentheticals) where clean_text has parentheticals removed
        and each Parenthetical records its offset in the clean text.
    """
    parentheticals: list[Parenthetical] = []
    clean_parts: list[str] = []
    last_end = 0
    clean_offset = 0

    for match in _PAREN_RE.finditer(text):
        # Add text before this parenthetical
        before = text[last_end : match.start()]
        clean_parts.append(before)
        clean_offset += len(before)

        # Record the parenthetical at the current clean offset
        parentheticals.append(
            Parenthetical(text=match.group(0), offset=clean_offset)
        )

        last_end = match.end()

    # Add remaining text after last parenthetical
    clean_parts.append(text[last_end:])

    clean_text = "".join(clean_parts)
    # Collapse multiple spaces that may result from removing parentheticals
    clean_text = re.sub(r"  +", " ", clean_text).strip()

    return clean_text, parentheticals


def _is_direction(line: str) -> bool:
    """Check if a line is a scene direction (starts with '(' and has no speaker tag)."""
    stripped = line.strip()
    return stripped.startswith("(") and not _SPEAKER_RE.match(stripped)


def parse_markdown(text: str, source_path: str = "") -> Project:
    """Parse a markdown dialogue script into a Project.

    Args:
        text: The full markdown text of the script.
        source_path: Path to the source .md file.

    Returns:
        A Project with all lines parsed, indexed, and characters derived.
    """
    lines_out: list[Line] = []
    raw_lines = text.split("\n")

    i = 0
    while i < len(raw_lines):
        raw_line = raw_lines[i]
        stripped = raw_line.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # Check for scene direction
        if _is_direction(stripped):
            # A direction may span multiple lines until blank line
            direction_parts = [stripped]
            i += 1
            while i < len(raw_lines) and raw_lines[i].strip():
                direction_parts.append(raw_lines[i].strip())
                i += 1

            raw_text = " ".join(direction_parts)
            lines_out.append(
                Line(
                    type="direction",
                    character=None,
                    raw_text=raw_text,
                    text=_unescape_markdown(raw_text),
                )
            )
            continue

        # Check for dialogue line
        speaker_match = _SPEAKER_RE.match(stripped)
        if speaker_match:
            character = speaker_match.group(1).strip()
            first_line_text = speaker_match.group(2)

            # Collect continuation lines until blank line
            dialogue_parts = [first_line_text]
            i += 1
            while i < len(raw_lines) and raw_lines[i].strip():
                # If next line is a new speaker or direction, stop
                next_stripped = raw_lines[i].strip()
                if _SPEAKER_RE.match(next_stripped) or _is_direction(next_stripped):
                    break
                dialogue_parts.append(next_stripped)
                i += 1

            raw_text_body = " ".join(dialogue_parts)
            # Store the full raw line with speaker tag for export round-trip
            raw_with_speaker = f"{character}: {raw_text_body}"

            # Unescape markdown
            unescaped = _unescape_markdown(raw_text_body)

            # Extract parentheticals from unescaped text
            clean_text, parentheticals = _extract_parentheticals(unescaped)

            lines_out.append(
                Line(
                    type="dialogue",
                    character=character,
                    raw_text=raw_with_speaker,
                    text=clean_text,
                    parentheticals=parentheticals,
                )
            )
            continue

        # Any other non-blank line that doesn't match — treat as direction
        other_parts = [stripped]
        i += 1
        while i < len(raw_lines) and raw_lines[i].strip():
            other_parts.append(raw_lines[i].strip())
            i += 1
        raw_text = " ".join(other_parts)
        lines_out.append(
            Line(
                type="direction",
                character=None,
                raw_text=raw_text,
                text=_unescape_markdown(raw_text),
            )
        )

    # Build project
    project = Project(source_path=source_path, lines=lines_out)
    project.reindex()
    project.derive_characters()
    return project
