"""Markdown script exporter for koedeck.

Reconstructs the original markdown format from a Project:
- Re-inserts parentheticals at their stored offsets
- Re-escapes punctuation that markdown would interpret
- Produces speaker tags for dialogue, bare text for directions
- Separates lines with blank lines
"""

from __future__ import annotations

from .models import Line, Parenthetical, Project


def _re_escape_outside_parens(text: str) -> str:
    """Re-escape markdown punctuation only outside of parenthetical sections.

    Only escapes ! (the primary char escaped in dialogue markdown scripts).
    Content inside parentheses is left verbatim.
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "(":
            # Find matching close paren — copy verbatim
            depth = 1
            j = i + 1
            while j < len(text) and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            result.append(text[i:j])
            i = j
        elif text[i] == "!":
            result.append("\\!")
            i += 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def _reinsert_parentheticals(text: str, parentheticals: list[Parenthetical]) -> str:
    """Re-insert parentheticals into clean text at their stored offsets.

    Each parenthetical's offset is the character position in the clean text
    where the parenthetical originally appeared. We insert right-to-left
    to preserve earlier offsets.
    """
    if not parentheticals:
        return text

    # Sort by offset descending so insertions don't shift earlier offsets
    sorted_parens = sorted(parentheticals, key=lambda p: p.offset, reverse=True)

    result = text
    for paren in sorted_parens:
        offset = min(paren.offset, len(result))
        before = result[:offset]
        after = result[offset:]

        # Determine spacing for natural reinsertion
        sep_before = ""
        sep_after = ""

        if before and not before.endswith(" "):
            sep_before = " "
        if after and not after.startswith(" "):
            sep_after = " "

        result = before + sep_before + paren.text + sep_after + after

    return result


def export_line(line: Line) -> str:
    """Export a single Line back to its markdown representation."""
    if line.type == "direction":
        return line.raw_text

    # For dialogue: reconstruct from clean text + parentheticals
    # Re-insert parentheticals first (into unescaped text)
    full_text = _reinsert_parentheticals(line.text, line.parentheticals)

    # Re-escape markdown punctuation only outside parentheticals
    full_text = _re_escape_outside_parens(full_text)

    return f"{line.character}: {full_text}"


def export_project(project: Project) -> str:
    """Export a full Project back to markdown format.

    Produces the original script format with blank lines between entries.
    Only exports non-orphaned lines in global_index order.
    """
    active_lines = [line for line in project.lines if not line.orphaned]
    active_lines.sort(key=lambda line: line.global_index)

    parts: list[str] = []
    for line in active_lines:
        parts.append(export_line(line))

    return "\n\n".join(parts) + "\n"
