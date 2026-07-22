"""koedeck — NiceGUI web frontend for the script voice pipeline."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from nicegui import app, ui

from .config import generate_config
from .exporter import export_project
from .models import Project
from .parser import parse_markdown
from .reimport import reimport_markdown

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

PROJECT_FILE = Path("project.json")

_project: Project | None = None
_save_timer: asyncio.TimerHandle | None = None
_save_indicator_timer: asyncio.TimerHandle | None = None


def _get_project() -> Project | None:
    return _project


def _set_project(p: Project) -> None:
    global _project
    _project = p


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _load_project() -> Project | None:
    """Load project.json if it exists."""
    if PROJECT_FILE.exists():
        data = json.loads(PROJECT_FILE.read_text())
        return Project.model_validate(data)
    return None


def _save_project_sync() -> None:
    """Atomically write project.json (write tmp, then rename)."""
    proj = _get_project()
    if proj is None:
        return
    data = proj.model_dump_json(indent=2)
    tmp = PROJECT_FILE.with_suffix(".tmp")
    tmp.write_text(data)
    tmp.rename(PROJECT_FILE)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def _schedule_autosave(save_label: ui.label) -> None:
    """Debounced autosave: resets timer on each call, fires 1s after last."""
    global _save_timer, _save_indicator_timer
    loop = asyncio.get_event_loop()

    # Cancel previous timer
    if _save_timer is not None:
        _save_timer.cancel()

    def do_save():
        global _save_indicator_timer
        _save_project_sync()
        save_label.text = "✓ saved"
        save_label.classes(replace="text-green-500 text-xs transition-opacity opacity-100")

        # Fade the indicator after 2s
        if _save_indicator_timer is not None:
            _save_indicator_timer.cancel()
        _save_indicator_timer = loop.call_later(
            2.0,
            lambda: (
                save_label.classes(replace="text-green-500 text-xs transition-opacity opacity-0"),
            ),
        )

    _save_timer = loop.call_later(1.0, do_save)


def _is_line_stale(line) -> bool:
    """Check if a line's audio is stale (text changed since generation)."""
    if line.audio.audio_hash is None:
        return False
    # If audio exists but hash was computed from different text, it's stale
    # We can't recompute the full hash here (missing voice_id etc.) but we can
    # check if the file exists and the text field on the line differs from
    # what was used. For now, we mark stale if audio_hash exists but text changed.
    # The actual hash comparison happens at generation time.
    # Simple heuristic: if there's audio but tagged_text/text may have changed
    return line.audio.current_file is not None and line.audio.audio_hash is not None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@ui.page("/")
async def index_page():
    """Main page: either show import or editor."""
    proj = _get_project()
    if proj is None:
        proj = _load_project()
        if proj is not None:
            _set_project(proj)

    if _get_project() is not None:
        _build_editor_page()
    else:
        _build_import_page()


@ui.page("/import")
async def import_page():
    """Import/re-import screen."""
    _build_import_page()


def _build_import_page():
    """Build the import screen UI."""
    ui.dark_mode(True)

    with ui.column().classes("w-full max-w-2xl mx-auto p-8 gap-6"):
        ui.label("koedeck").classes("text-3xl font-bold")
        ui.label("Import a markdown dialogue script to get started.").classes("text-gray-400")

        with ui.card().classes("w-full p-6"):
            ui.label("Import script").classes("text-lg font-semibold mb-2")
            ui.label(
                "Select a .md file with dialogue in SPEAKER: format."
            ).classes("text-sm text-gray-400 mb-4")

            upload = ui.upload(
                label="Choose .md file",
                auto_upload=True,
                on_upload=_handle_upload,
            ).props('accept=".md"').classes("w-full")

        # If there's an existing project, show option to re-import
        existing = _load_project()
        if existing is not None:
            with ui.card().classes("w-full p-6"):
                ui.label("Existing project found").classes("text-lg font-semibold mb-2")
                ui.label(
                    f"Source: {existing.source_path} • {len(existing.lines)} lines • "
                    f"{len(existing.characters)} characters"
                ).classes("text-sm text-gray-400 mb-4")
                ui.button("Open existing project", on_click=lambda: ui.navigate.to("/editor")).props(
                    "outline"
                )


async def _handle_upload(e):
    """Handle markdown file upload."""
    content = e.content.read().decode("utf-8")
    filename = e.name

    existing = _get_project() or _load_project()

    if existing is not None:
        # Re-import into existing project
        proj = reimport_markdown(content, existing)
        ui.notify(
            f"Re-imported {filename}: {len([l for l in proj.lines if not l.orphaned])} active lines, "
            f"{len([l for l in proj.lines if l.orphaned])} orphaned",
            type="positive",
        )
    else:
        # Fresh import
        proj = parse_markdown(content, filename)
        ui.notify(
            f"Imported {filename}: {len(proj.lines)} lines, {len(proj.characters)} characters",
            type="positive",
        )

    # Generate config skeleton
    generate_config(proj)

    # Save project
    _set_project(proj)
    _save_project_sync()

    # Navigate to editor
    await asyncio.sleep(0.5)
    ui.navigate.to("/editor")


@ui.page("/editor")
async def editor_page():
    """Editor page with tabbed interface."""
    proj = _get_project()
    if proj is None:
        proj = _load_project()
        if proj is not None:
            _set_project(proj)
        else:
            ui.navigate.to("/")
            return

    _build_editor_page()


def _build_editor_page():
    """Build the tabbed editor UI."""
    proj = _get_project()
    if proj is None:
        return

    ui.dark_mode(True)

    # Top bar
    with ui.header().classes("bg-gray-900 items-center justify-between px-6"):
        with ui.row().classes("items-center gap-4"):
            ui.label("koedeck").classes("text-xl font-bold")
            ui.label(f"— {proj.source_path}").classes("text-gray-400 text-sm")

        save_label = ui.label("").classes("text-green-500 text-xs transition-opacity opacity-0")

        with ui.row().classes("items-center gap-2"):
            ui.button("Export .md", icon="download", on_click=lambda: _handle_export()).props(
                "flat dense"
            )
            ui.button("Re-import", icon="upload", on_click=lambda: ui.navigate.to("/import")).props(
                "flat dense"
            )

    # Tabs
    with ui.tabs().classes("w-full") as tabs:
        combined_tab = ui.tab("Combined")
        character_tabs = {}
        for char in proj.characters:
            character_tabs[char] = ui.tab(char)

    with ui.tab_panels(tabs, value=combined_tab).classes("w-full flex-grow"):
        # Combined panel
        with ui.tab_panel(combined_tab).classes("p-4"):
            _build_line_list(proj, filter_char=None, save_label=save_label)

        # Per-character panels
        for char, tab in character_tabs.items():
            with ui.tab_panel(tab).classes("p-4"):
                _build_line_list(proj, filter_char=char, save_label=save_label)


def _build_line_list(proj: Project, filter_char: str | None, save_label: ui.label):
    """Build the scrollable list of editable lines."""
    lines = [l for l in proj.lines if not l.orphaned]
    if filter_char is not None:
        lines = [l for l in lines if l.character == filter_char]

    with ui.scroll_area().classes("w-full").style("height: calc(100vh - 140px)"):
        with ui.column().classes("w-full gap-2 max-w-4xl mx-auto"):
            for line in lines:
                _build_line_card(line, save_label, show_index=(filter_char is not None))

    # Show orphaned lines at bottom if in combined view
    if filter_char is None:
        orphaned = [l for l in proj.lines if l.orphaned]
        if orphaned:
            with ui.expansion("Orphaned lines", icon="warning").classes(
                "w-full max-w-4xl mx-auto mt-4 text-amber-400"
            ):
                for line in orphaned:
                    _build_orphan_card(line, save_label)


def _build_line_card(line, save_label: ui.label, show_index: bool = False):
    """Build an editable card for a single line."""
    is_stale = _is_line_stale(line)

    card_classes = "w-full p-3"
    if is_stale:
        card_classes += " border-l-4 border-amber-500"

    with ui.card().classes(card_classes):
        if line.type == "direction":
            # Directions: dimmed, italic, not editable (they're stage directions)
            with ui.row().classes("items-center gap-2 w-full"):
                if show_index:
                    ui.label(f"#{line.global_index}").classes("text-xs text-gray-600 font-mono")
                ui.label(line.text).classes("italic text-gray-500 w-full")
        else:
            # Dialogue: editable
            with ui.row().classes("items-center gap-2 w-full"):
                if show_index:
                    ui.label(f"#{line.global_index}").classes(
                        "text-xs text-gray-600 font-mono min-w-[2rem]"
                    )

                # Character badge
                ui.badge(line.character).classes("text-xs")

                # Stale indicator
                if is_stale:
                    ui.icon("warning", size="xs").classes("text-amber-500").tooltip(
                        "Audio is stale — text changed since last generation"
                    )

                # Audio indicator
                if line.audio.current_file:
                    ui.icon("volume_up", size="xs").classes("text-green-600")

            # Editable text area with parentheticals shown inline
            display_text = _build_display_text(line)

            textarea = ui.textarea(
                value=display_text,
            ).classes("w-full mt-1").props('autogrow dense outlined')

            # On change: parse editable text back, update line, trigger autosave
            textarea.on(
                "update:model-value",
                lambda e, ln=line: _handle_text_change(e, ln, save_label),
            )


def _build_orphan_card(line, save_label: ui.label):
    """Build a card for an orphaned line with resolve options."""
    with ui.card().classes("w-full p-3 bg-amber-900/20 border border-amber-700"):
        with ui.row().classes("items-center gap-2 w-full"):
            if line.character:
                ui.badge(line.character).classes("text-xs")
            ui.label(line.text).classes("text-gray-400 flex-grow")

            # Resolve buttons
            ui.button(
                icon="delete",
                on_click=lambda ln=line: _delete_orphan(ln, save_label),
            ).props("flat dense round").tooltip("Delete permanently")


def _build_display_text(line) -> str:
    """Build display text with parentheticals re-inserted for editing."""
    if not line.parentheticals:
        return line.text

    # Re-insert parentheticals at their offsets for display
    result = line.text
    sorted_parens = sorted(line.parentheticals, key=lambda p: p.offset, reverse=True)
    for paren in sorted_parens:
        offset = min(paren.offset, len(result))
        before = result[:offset]
        after = result[offset:]
        sep_before = " " if before and not before.endswith(" ") else ""
        sep_after = " " if after and not after.startswith(" ") else ""
        result = before + sep_before + paren.text + sep_after + after

    return result


def _handle_text_change(e, line, save_label: ui.label):
    """Handle text edit: re-parse parentheticals, update line, trigger autosave."""
    import re

    new_text = e.args if isinstance(e.args, str) else str(e.args)

    # Extract parentheticals from the edited text
    from .parser import _PAREN_RE, _extract_parentheticals_v2

    clean_text, parentheticals = _extract_parentheticals_v2(new_text)

    # Update line (ID stays the same!)
    line.text = clean_text
    line.parentheticals = parentheticals
    line.raw_text = f"{line.character}: {new_text}" if line.character else new_text

    # If text changed and there was tagged_text, invalidate it
    # (tagged_text was for the old text)
    if line.tagged_text is not None:
        # Check if stripping tags from tagged_text still matches
        import re as _re

        stripped = _re.sub(r"\[.*?\]", "", line.tagged_text)
        if stripped != clean_text:
            line.tagged_text = None

    # Trigger debounced autosave
    _schedule_autosave(save_label)


def _delete_orphan(line, save_label: ui.label):
    """Remove an orphaned line from the project."""
    proj = _get_project()
    if proj is None:
        return
    proj.lines = [l for l in proj.lines if l.line_id != line.line_id]
    _schedule_autosave(save_label)
    ui.notify("Orphaned line removed", type="info")
    # Refresh the page
    ui.navigate.to("/editor")


async def _handle_export():
    """Export project to markdown and offer download."""
    proj = _get_project()
    if proj is None:
        ui.notify("No project loaded", type="warning")
        return

    content = export_project(proj)
    # Determine filename
    source = Path(proj.source_path)
    export_name = source.stem + "_exported" + source.suffix if source.suffix else source.name + "_exported.md"

    # Write to temp file and trigger download
    tmp = Path(tempfile.mkdtemp()) / export_name
    tmp.write_text(content)
    ui.download(str(tmp), filename=export_name)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    """Launch the koedeck NiceGUI app."""
    ui.run(
        title="koedeck",
        host="127.0.0.1",
        port=8080,
        reload=False,
        show=True,
    )
