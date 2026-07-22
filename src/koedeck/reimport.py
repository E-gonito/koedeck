"""Re-import logic for koedeck.

When a markdown file is re-imported into an existing project:
1. Match incoming lines to existing lines by exact text match first.
2. Then fuzzy match (difflib ratio >= 0.85, same character) for remaining.
3. Matched lines KEEP their existing line_id and audio state.
4. Unmatched incoming lines get new IDs.
5. Unmatched existing lines are marked orphaned (not deleted).
"""

from __future__ import annotations

import difflib

from .models import Line, Project
from .parser import parse_markdown

# Minimum fuzzy match ratio to consider a line "the same"
FUZZY_THRESHOLD = 0.85


def reimport_markdown(text: str, existing_project: Project) -> Project:
    """Re-import a markdown script into an existing project.

    Preserves line_ids and audio state for matched lines. Marks unmatched
    existing lines as orphaned. Assigns new IDs to unmatched incoming lines.

    Args:
        text: The new markdown text to import.
        existing_project: The current project with existing lines.

    Returns:
        Updated project with matched lines preserved and new lines added.
    """
    # Parse the incoming markdown to get new lines
    incoming_project = parse_markdown(text, existing_project.source_path)
    incoming_lines = incoming_project.lines

    # Build lookup structures from existing lines (non-orphaned only)
    existing_lines = [l for l in existing_project.lines if not l.orphaned]

    # Track which existing lines have been matched
    matched_existing: set[str] = set()  # line_ids of matched existing lines
    matched_incoming: set[int] = set()  # indices of matched incoming lines

    # Result lines in incoming order
    result_lines: list[Line] = []

    # --- Pass 1: Exact text match (same character + same text) ---
    # Build index: (character, text) -> list of existing lines with that combo
    exact_index: dict[tuple[str | None, str], list[Line]] = {}
    for eline in existing_lines:
        key = (eline.character, eline.text)
        exact_index.setdefault(key, []).append(eline)

    for i, iline in enumerate(incoming_lines):
        key = (iline.character, iline.text)
        if key in exact_index:
            # Find first unmatched existing line with this exact text
            for eline in exact_index[key]:
                if eline.line_id not in matched_existing:
                    # Match found — preserve the existing line's identity
                    result_lines.append(_merge_line(iline, eline))
                    matched_existing.add(eline.line_id)
                    matched_incoming.add(i)
                    break
            else:
                result_lines.append(iline)  # placeholder, may get fuzzy-matched
        else:
            result_lines.append(iline)  # placeholder

    # --- Pass 2: Fuzzy match for remaining unmatched lines ---
    unmatched_existing = [l for l in existing_lines if l.line_id not in matched_existing]

    for i, iline in enumerate(incoming_lines):
        if i in matched_incoming:
            continue

        best_match: Line | None = None
        best_ratio: float = 0.0

        for eline in unmatched_existing:
            if eline.line_id in matched_existing:
                continue

            # Only match same type and character
            if iline.type != eline.type:
                continue
            if iline.character != eline.character:
                continue

            ratio = difflib.SequenceMatcher(
                None, iline.text, eline.text
            ).ratio()

            if ratio >= FUZZY_THRESHOLD and ratio > best_ratio:
                best_ratio = ratio
                best_match = eline

        if best_match is not None:
            result_lines[i] = _merge_line(iline, best_match)
            matched_existing.add(best_match.line_id)
            matched_incoming.add(i)

    # --- Mark unmatched existing lines as orphaned ---
    orphaned_lines: list[Line] = []
    seen_orphan_ids: set[str] = set()

    for eline in existing_lines:
        if eline.line_id not in matched_existing:
            eline.orphaned = True
            orphaned_lines.append(eline)
            seen_orphan_ids.add(eline.line_id)

    # Also carry over any previously-orphaned lines (avoid duplicates)
    for eline in existing_project.lines:
        if eline.orphaned and eline.line_id not in seen_orphan_ids:
            orphaned_lines.append(eline)
            seen_orphan_ids.add(eline.line_id)

    # Build final project
    # Active lines come first (in incoming order), then orphaned at the end
    all_lines = result_lines + orphaned_lines

    project = Project(
        source_path=existing_project.source_path,
        lines=all_lines,
    )
    project.reindex()
    project.derive_characters()
    return project


def _merge_line(incoming: Line, existing: Line) -> Line:
    """Merge an incoming line with an existing matched line.

    Keeps the existing line_id and audio state, but updates text/parentheticals
    from the incoming version.
    """
    return Line(
        line_id=existing.line_id,
        global_index=incoming.global_index,
        type=incoming.type,
        character=incoming.character,
        raw_text=incoming.raw_text,
        text=incoming.text,
        parentheticals=incoming.parentheticals,
        tagged_text=existing.tagged_text if incoming.text == existing.text else None,
        audio=existing.audio,
        orphaned=False,
    )
