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
        and each Parenthetical records its offset in the *final* clean text
        (after space normalization).

    The offset represents the position in the clean text where the parenthetical
    should be re-inserted during export.
    """
    parentheticals: list[Parenthetical] = []
    clean_parts: list[str] = []
    last_end = 0

    for match in _PAREN_RE.finditer(text):
        # Add text before this parenthetical
        before = text[last_end : match.start()]
        clean_parts.append(before)
        last_end = match.end()

        # Mark where this parenthetical goes: use a sentinel we'll find later
        # We use a unique marker that won't appear in real text
        marker = f"\x00PAREN{len(parentheticals)}\x00"
        clean_parts.append(marker)

        parentheticals.append(
            Parenthetical(text=match.group(0), offset=0)  # offset computed below
        )

    # Add remaining text after last parenthetical
    clean_parts.append(text[last_end:])

    # Join and normalize spaces
    raw_joined = "".join(clean_parts)

    # Now compute final offsets by finding markers in the normalized text
    # First normalize spaces (but preserve markers)
    normalized = re.sub(r"  +", " ", raw_joined).strip()

    # Extract offsets from marker positions, then remove markers
    for i, paren in enumerate(parentheticals):
        marker = f"\x00PAREN{i}\x00"
        pos = normalized.find(marker)
        # The offset in the final clean text (without markers before this one)
        # We need to account for all markers before this position
        offset_adjustment = 0
        for j in range(i):
            prev_marker = f"\x00PAREN{j}\x00"
            offset_adjustment += len(prev_marker)
        paren.offset = pos - offset_adjustment

    # Remove all markers to get the actual clean text
    clean_text = normalized
    for i in range(len(parentheticals)):
        marker = f"\x00PAREN{i}\x00"
        clean_text = clean_text.replace(marker, "")

    # Final space cleanup after marker removal
    clean_text = re.sub(r"  +", " ", clean_text).strip()

    # Adjust offsets for any space changes from marker removal
    # Recompute by building text incrementally
    # Actually let's just recompute correctly from scratch using a simpler approach
    return _extract_parentheticals_v2(text)


def _extract_parentheticals_v2(text: str) -> tuple[str, list[Parenthetical]]:
    """Simpler parenthetical extraction with correct offset computation.

    Strategy: build the clean text character by character, tracking positions.
    When we encounter a parenthetical, record the current clean-text position.
    """
    parentheticals: list[Parenthetical] = []

    # First, find all parenthetical spans
    paren_spans: list[tuple[int, int, str]] = []
    for match in _PAREN_RE.finditer(text):
        paren_spans.append((match.start(), match.end(), match.group(0)))

    if not paren_spans:
        clean = re.sub(r"  +", " ", text).strip()
        return clean, []

    # Build clean text by removing parenthetical spans
    clean_parts: list[str] = []
    last_end = 0
    offsets: list[int] = []

    for start, end, paren_text in paren_spans:
        before = text[last_end:start]
        clean_parts.append(before)
        # Record the current length as the offset for this parenthetical
        current_clean_len = sum(len(p) for p in clean_parts)
        offsets.append(current_clean_len)
        last_end = end

    clean_parts.append(text[last_end:])
    raw_clean = "".join(clean_parts)

    # Normalize spaces
    clean_text = re.sub(r"  +", " ", raw_clean).strip()

    # Adjust offsets for space normalization and stripping
    # Find how much the start was stripped
    lstrip_amount = len(raw_clean) - len(raw_clean.lstrip())

    for i, (start, end, paren_text) in enumerate(paren_spans):
        # Adjust offset for leading strip
        adjusted = offsets[i] - lstrip_amount

        # Adjust for space collapsing: count how many extra spaces were removed
        # before this offset in the raw_clean
        prefix = raw_clean[lstrip_amount : offsets[i]]
        collapsed_prefix = re.sub(r"  +", " ", prefix)
        adjusted = len(collapsed_prefix)

        parentheticals.append(Parenthetical(text=paren_text, offset=adjusted))

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
