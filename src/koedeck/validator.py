"""Script validation and cleanup for koedeck.

Analyzes uploaded markdown scripts for issues before processing:
- Missing or malformed speaker tags
- Unbalanced parentheses
- Lines that look like dialogue but lack proper tags
- Excessive or unnecessary escaping
- Trailing whitespace, inconsistent line endings
- Empty or meaningless lines
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ScriptIssue:
    """A single issue found in the script."""
    line_number: int
    severity: IssueSeverity
    message: str
    original_text: str
    suggested_fix: str | None = None
    auto_fixable: bool = False


@dataclass
class ValidationResult:
    """Result of validating a script."""
    issues: list[ScriptIssue] = field(default_factory=list)
    cleaned_text: str = ""
    stats: dict = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)


_SPEAKER_RE = re.compile(r"^([A-Z][A-Z0-9 '-]+?):\s*(.*)", re.DOTALL)
_LOOKS_LIKE_DIALOGUE_RE = re.compile(r"^([A-Za-z][A-Za-z ]+):\s+")


def validate_script(text: str) -> ValidationResult:
    """Validate a markdown script and find issues.

    Returns a ValidationResult with all issues found and a cleaned version.
    """
    result = ValidationResult()
    lines = text.split("\n")
    cleaned_lines: list[str] = []

    dialogue_count = 0
    direction_count = 0
    characters_found: set[str] = set()

    for i, line in enumerate(lines, start=1):
        original = line

        # --- Check: trailing whitespace ---
        if line != line.rstrip():
            result.issues.append(ScriptIssue(
                line_number=i, severity=IssueSeverity.INFO,
                message="Trailing whitespace",
                original_text=repr(line[-10:]),
                suggested_fix=line.rstrip(),
                auto_fixable=True,
            ))
            line = line.rstrip()

        # --- Check: Windows line endings (\r) ---
        if "\r" in line:
            result.issues.append(ScriptIssue(
                line_number=i, severity=IssueSeverity.INFO,
                message="Windows line ending (\r\n)",
                original_text=repr(line),
                suggested_fix=line.replace("\r", ""),
                auto_fixable=True,
            ))
            line = line.replace("\r", "")

        # Skip blank lines
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # --- Check: unbalanced parentheses ---
        open_count = stripped.count("(")
        close_count = stripped.count(")")
        if open_count != close_count:
            result.issues.append(ScriptIssue(
                line_number=i, severity=IssueSeverity.WARNING,
                message=f"Unbalanced parentheses: {open_count} open, {close_count} close",
                original_text=stripped[:60],
            ))

        # --- Check: is this a dialogue line? ---
        speaker_match = _SPEAKER_RE.match(stripped)
        if speaker_match:
            dialogue_count += 1
            characters_found.add(speaker_match.group(1).strip())
            cleaned_lines.append(line)
            continue

        # --- Check: direction line ---
        if stripped.startswith("("):
            direction_count += 1
            cleaned_lines.append(line)
            continue

        # --- Check: looks like dialogue but missing proper speaker tag ---
        looks_like = _LOOKS_LIKE_DIALOGUE_RE.match(stripped)
        if looks_like:
            possible_speaker = looks_like.group(1)
            # Only flag if the "speaker" part isn't all-caps (proper tag would be)
            if possible_speaker != possible_speaker.upper():
                suggested = possible_speaker.upper() + stripped[len(possible_speaker):]
                result.issues.append(ScriptIssue(
                    line_number=i, severity=IssueSeverity.WARNING,
                    message=f"Possible dialogue without proper speaker tag (not ALL-CAPS)",
                    original_text=stripped[:60],
                    suggested_fix=suggested,
                    auto_fixable=True,
                ))

        # --- Check: excessive escaping ---
        escape_count = stripped.count("\\")
        if escape_count > 3:
            result.issues.append(ScriptIssue(
                line_number=i, severity=IssueSeverity.INFO,
                message=f"Heavy escaping ({escape_count} backslashes) - may be unnecessary",
                original_text=stripped[:60],
            ))

        cleaned_lines.append(line)

    # Build cleaned text
    result.cleaned_text = "\n".join(cleaned_lines)

    # Remove excessive blank lines (more than 1 consecutive)
    result.cleaned_text = re.sub(r"\n{3,}", "\n\n", result.cleaned_text)
    result.cleaned_text = result.cleaned_text.strip() + "\n"

    # Stats
    result.stats = {
        "total_lines": len(lines),
        "dialogue_lines": dialogue_count,
        "direction_lines": direction_count,
        "characters": sorted(characters_found),
        "character_count": len(characters_found),
    }

    # --- Global checks ---
    if dialogue_count == 0:
        result.issues.insert(0, ScriptIssue(
            line_number=0, severity=IssueSeverity.ERROR,
            message="No dialogue lines found. Expected SPEAKER: format.",
            original_text="",
        ))

    return result


def apply_auto_fixes(text: str) -> str:
    """Apply all auto-fixable corrections to a script.

    - Normalize line endings
    - Remove trailing whitespace
    - Collapse excessive blank lines
    - Uppercase speaker tags for lines that look like dialogue
    """
    lines = text.split("\n")
    fixed: list[str] = []

    for line in lines:
        # Normalize line endings
        line = line.replace("\r", "")
        # Remove trailing whitespace
        line = line.rstrip()

        # Fix lowercase speaker tags
        stripped = line.strip()
        looks_like = _LOOKS_LIKE_DIALOGUE_RE.match(stripped)
        if looks_like and not _SPEAKER_RE.match(stripped) and not stripped.startswith("("):
            possible_speaker = looks_like.group(1)
            if possible_speaker != possible_speaker.upper() and len(possible_speaker.split()) <= 3:
                line = possible_speaker.upper() + stripped[len(possible_speaker):]

        fixed.append(line)

    result = "\n".join(fixed)
    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip() + "\n"
