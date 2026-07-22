"""koedeck — NiceGUI web frontend for the script voice pipeline."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from nicegui import app, ui

from .config import generate_config, load_config
from .exporter import export_project
from .models import Line, Project
from .parser import parse_markdown
from .reimport import reimport_markdown
from .tagger import (
    TaggingSession,
    TagResult,
    TagStatus,
    tag_episode,
    tag_single_line,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

PROJECT_FILE = Path("project.json")

_project: Project | None = None
_save_timer: asyncio.TimerHandle | None = None
_save_indicator_timer: asyncio.TimerHandle | None = None
_tagging_session: TaggingSession | None = None


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
# UI Helpers
# ---------------------------------------------------------------------------


def _schedule_autosave(save_label: ui.label) -> None:
    """Debounced autosave: resets timer on each call, fires 1s after last."""
    global _save_timer, _save_indicator_timer
    loop = asyncio.get_event_loop()

    if _save_timer is not None:
        _save_timer.cancel()

    def do_save():
        global _save_indicator_timer
        _save_project_sync()
        save_label.text = "✓ saved"
        save_label.classes(replace="text-green-500 text-xs transition-opacity opacity-100")

        if _save_indicator_timer is not None:
            _save_indicator_timer.cancel()
        _save_indicator_timer = loop.call_later(
            2.0,
            lambda: save_label.classes(
                replace="text-green-500 text-xs transition-opacity opacity-0"
            ),
        )

    _save_timer = loop.call_later(1.0, do_save)


def _is_line_stale(line: Line) -> bool:
    """Check if a line's audio is stale (text changed since generation)."""
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
        ui.navigate.to("/editor")
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
            ui.label("Select a .md file with dialogue in SPEAKER: format.").classes(
                "text-sm text-gray-400 mb-4"
            )
            ui.upload(
                label="Choose .md file",
                auto_upload=True,
                on_upload=_handle_upload,
            ).props('accept=".md"').classes("w-full")

        existing = _load_project()
        if existing is not None:
            with ui.card().classes("w-full p-6"):
                ui.label("Existing project found").classes("text-lg font-semibold mb-2")
                ui.label(
                    f"Source: {existing.source_path} • {len(existing.lines)} lines • "
                    f"{len(existing.characters)} characters"
                ).classes("text-sm text-gray-400 mb-4")
                ui.button(
                    "Open existing project", on_click=lambda: ui.navigate.to("/editor")
                ).props("outline")


async def _handle_upload(e):
    """Handle markdown file upload."""
    content = e.content.read().decode("utf-8")
    filename = e.name
    existing = _get_project() or _load_project()

    if existing is not None:
        proj = reimport_markdown(content, existing)
        ui.notify(
            f"Re-imported {filename}: "
            f"{len([l for l in proj.lines if not l.orphaned])} active, "
            f"{len([l for l in proj.lines if l.orphaned])} orphaned",
            type="positive",
        )
    else:
        proj = parse_markdown(content, filename)
        ui.notify(
            f"Imported {filename}: {len(proj.lines)} lines, {len(proj.characters)} characters",
            type="positive",
        )

    generate_config(proj)
    _set_project(proj)
    _save_project_sync()

    await asyncio.sleep(0.5)
    ui.navigate.to("/editor")


# ---------------------------------------------------------------------------
# Editor page
# ---------------------------------------------------------------------------


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
            ui.button("Tag episode", icon="auto_awesome", on_click=_handle_tag_episode).props(
                "dense color=purple"
            )
            ui.button("Export .md", icon="download", on_click=_handle_export).props("flat dense")
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
        with ui.tab_panel(combined_tab).classes("p-4"):
            _build_line_list(proj, filter_char=None, save_label=save_label)
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

    if filter_char is None:
        orphaned = [l for l in proj.lines if l.orphaned]
        if orphaned:
            with ui.expansion("Orphaned lines", icon="warning").classes(
                "w-full max-w-4xl mx-auto mt-4 text-amber-400"
            ):
                for line in orphaned:
                    _build_orphan_card(line, save_label)


def _build_line_card(line: Line, save_label: ui.label, show_index: bool = False):
    """Build an editable card for a single line."""
    is_stale = _is_line_stale(line)
    card_classes = "w-full p-3"
    if is_stale:
        card_classes += " border-l-4 border-amber-500"

    with ui.card().classes(card_classes):
        if line.type == "direction":
            with ui.row().classes("items-center gap-2 w-full"):
                if show_index:
                    ui.label(f"#{line.global_index}").classes("text-xs text-gray-600 font-mono")
                ui.label(line.text).classes("italic text-gray-500 w-full")
        else:
            # Header row
            with ui.row().classes("items-center gap-2 w-full"):
                if show_index:
                    ui.label(f"#{line.global_index}").classes(
                        "text-xs text-gray-600 font-mono min-w-[2rem]"
                    )
                ui.badge(line.character).classes("text-xs")

                if is_stale:
                    ui.icon("warning", size="xs").classes("text-amber-500").tooltip(
                        "Audio stale — text changed since generation"
                    )
                if line.audio.current_file:
                    ui.icon("volume_up", size="xs").classes("text-green-600")
                if line.tagged_text:
                    ui.icon("label", size="xs").classes("text-purple-400").tooltip(
                        "Tagged"
                    )

                # Spacer
                ui.element("div").classes("flex-grow")

                # Per-line tag button
                ui.button(
                    icon="auto_awesome",
                    on_click=lambda ln=line: _handle_tag_line(ln),
                ).props("flat dense round size=sm color=purple").tooltip("Tag this line")

            # Editable text area
            display_text = _build_display_text(line)
            textarea = ui.textarea(value=display_text).classes("w-full mt-1").props(
                "autogrow dense outlined"
            )
            textarea.on(
                "update:model-value",
                lambda e, ln=line: _handle_text_change(e, ln, save_label),
            )

            # Show tagged text if present
            if line.tagged_text:
                with ui.row().classes("items-center gap-1 mt-1"):
                    ui.icon("label", size="xs").classes("text-purple-400")
                    ui.label(line.tagged_text).classes(
                        "text-xs text-purple-300 font-mono bg-purple-900/30 px-2 py-1 rounded"
                    )


def _build_orphan_card(line: Line, save_label: ui.label):
    """Build a card for an orphaned line."""
    with ui.card().classes("w-full p-3 bg-amber-900/20 border border-amber-700"):
        with ui.row().classes("items-center gap-2 w-full"):
            if line.character:
                ui.badge(line.character).classes("text-xs")
            ui.label(line.text).classes("text-gray-400 flex-grow")
            ui.button(
                icon="delete",
                on_click=lambda ln=line: _delete_orphan(ln, save_label),
            ).props("flat dense round").tooltip("Delete permanently")


def _build_display_text(line: Line) -> str:
    """Build display text with parentheticals re-inserted for editing."""
    if not line.parentheticals:
        return line.text
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


def _handle_text_change(e, line: Line, save_label: ui.label):
    """Handle text edit: re-parse parentheticals, update line, trigger autosave."""
    import re

    new_text = e.args if isinstance(e.args, str) else str(e.args)
    from .parser import _extract_parentheticals_v2

    clean_text, parentheticals = _extract_parentheticals_v2(new_text)

    line.text = clean_text
    line.parentheticals = parentheticals
    line.raw_text = f"{line.character}: {new_text}" if line.character else new_text

    # Invalidate tagged_text if text changed
    if line.tagged_text is not None:
        from .tagger import strip_tags

        if strip_tags(line.tagged_text) != clean_text:
            line.tagged_text = None

    _schedule_autosave(save_label)


def _delete_orphan(line: Line, save_label: ui.label):
    """Remove an orphaned line from the project."""
    proj = _get_project()
    if proj is None:
        return
    proj.lines = [l for l in proj.lines if l.line_id != line.line_id]
    _schedule_autosave(save_label)
    ui.notify("Orphaned line removed", type="info")
    ui.navigate.to("/editor")


async def _handle_export():
    """Export project to markdown and offer download."""
    proj = _get_project()
    if proj is None:
        ui.notify("No project loaded", type="warning")
        return
    content = export_project(proj)
    source = Path(proj.source_path)
    export_name = (
        source.stem + "_exported" + source.suffix
        if source.suffix
        else source.name + "_exported.md"
    )
    tmp = Path(tempfile.mkdtemp()) / export_name
    tmp.write_text(content)
    ui.download(str(tmp), filename=export_name)


# ---------------------------------------------------------------------------
# Tagging UI
# ---------------------------------------------------------------------------


async def _handle_tag_episode():
    """Start batch tagging all dialogue lines, then show diff review."""
    global _tagging_session
    proj = _get_project()
    if proj is None:
        ui.notify("No project loaded", type="warning")
        return

    try:
        config = load_config()
    except FileNotFoundError:
        ui.notify("config.yaml not found — import a script first", type="negative")
        return

    session = TaggingSession()
    _tagging_session = session

    # Show progress dialog
    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("w-96 p-6"):
        ui.label("Tagging episode...").classes("text-lg font-semibold")
        progress_label = ui.label("Starting...").classes("text-sm text-gray-400")
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")

        cancel_btn = ui.button(
            "Cancel", on_click=lambda: _cancel_tagging(session, dialog)
        ).props("flat color=red")

    dialog.open()

    def on_progress(s: TaggingSession):
        if s.total > 0:
            progress_bar.value = s.completed / s.total
            progress_label.text = f"{s.completed}/{s.total} lines processed"

    # Run tagging
    await tag_episode(proj, config, session, max_concurrent=2, on_progress=on_progress)

    dialog.close()

    if session.cancelled:
        ui.notify("Tagging cancelled", type="warning")
        return

    # Show diff review
    successful = [r for r in session.results.values() if r.status == TagStatus.SUCCESS]
    failed = [r for r in session.results.values() if r.status == TagStatus.FAILED]

    if not successful and not failed:
        ui.notify("No lines to tag", type="info")
        return

    if failed:
        ui.notify(
            f"{len(failed)} line(s) failed tagging — shown as 'needs manual tags'",
            type="warning",
        )

    # Navigate to diff review page
    ui.navigate.to("/review")


async def _handle_tag_line(line: Line):
    """Tag a single line and show inline diff review."""
    proj = _get_project()
    if proj is None:
        return

    try:
        config = load_config()
    except FileNotFoundError:
        ui.notify("config.yaml not found", type="negative")
        return

    # Get preceding lines for context
    all_lines = [l for l in proj.lines if not l.orphaned]
    line_idx = next((i for i, l in enumerate(all_lines) if l.line_id == line.line_id), 0)
    preceding = all_lines[max(0, line_idx - 2) : line_idx]

    # Show a small loading indicator
    ui.notify("Tagging line...", type="info", timeout=1500)

    result = await tag_single_line(line, preceding, config)

    if result.status == TagStatus.SUCCESS and result.tagged_text:
        # Show inline accept/reject dialog
        _show_single_line_review(line, result)
    else:
        ui.notify(
            f"Tagging failed: {result.error or 'unknown error'}",
            type="negative",
            timeout=5000,
        )


def _show_single_line_review(line: Line, result: TagResult):
    """Show a dialog to accept/reject a single line's tagging result."""
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-full max-w-2xl p-6"):
        ui.label("Review tagged line").classes("text-lg font-semibold mb-2")

        # Original
        with ui.row().classes("items-start gap-2 w-full"):
            ui.label("Original:").classes("text-sm text-gray-400 min-w-[5rem]")
            ui.label(result.original_text).classes("text-sm font-mono")

        # Tagged
        with ui.row().classes("items-start gap-2 w-full mt-2"):
            ui.label("Tagged:").classes("text-sm text-purple-400 min-w-[5rem]")
            ui.label(result.tagged_text).classes("text-sm font-mono text-purple-300")

        # Buttons
        with ui.row().classes("gap-2 mt-4 justify-end"):
            ui.button("Reject", on_click=dialog.close).props("flat color=red")

            def accept():
                line.tagged_text = result.tagged_text
                _save_project_sync()
                dialog.close()
                ui.notify("Tag accepted", type="positive")
                ui.navigate.to("/editor")

            ui.button("Accept", on_click=accept).props("color=purple")

    dialog.open()


def _cancel_tagging(session: TaggingSession, dialog):
    """Cancel an in-progress tagging session."""
    session.cancelled = True
    dialog.close()


# ---------------------------------------------------------------------------
# Diff review page
# ---------------------------------------------------------------------------


@ui.page("/review")
async def review_page():
    """Diff review page: shows all tagging results for acceptance."""
    global _tagging_session
    proj = _get_project()
    session = _tagging_session

    if proj is None or session is None:
        ui.navigate.to("/editor")
        return

    ui.dark_mode(True)

    # Collect results with their lines
    successful_results: list[tuple[Line, TagResult]] = []
    failed_results: list[tuple[Line, TagResult]] = []

    line_map = {l.line_id: l for l in proj.lines}
    for result in session.results.values():
        line = line_map.get(result.line_id)
        if line is None:
            continue
        if result.status == TagStatus.SUCCESS:
            successful_results.append((line, result))
        elif result.status == TagStatus.FAILED:
            failed_results.append((line, result))

    # Sort by global_index
    successful_results.sort(key=lambda x: x[0].global_index)
    failed_results.sort(key=lambda x: x[0].global_index)

    # Track acceptance state
    accepted: dict[str, bool] = {}  # line_id -> accepted?
    edited_text: dict[str, str] = {}  # line_id -> edited tagged text

    with ui.header().classes("bg-gray-900 items-center justify-between px-6"):
        with ui.row().classes("items-center gap-4"):
            ui.label("koedeck").classes("text-xl font-bold")
            ui.label("— Diff review").classes("text-gray-400 text-sm")

        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Accept all",
                icon="done_all",
                on_click=lambda: _accept_all(proj, successful_results, dialog_ref),
            ).props("dense color=purple")
            ui.button(
                "Back to editor",
                icon="arrow_back",
                on_click=lambda: ui.navigate.to("/editor"),
            ).props("flat dense")

    # Placeholder for dialog ref
    dialog_ref = {"done": False}

    with ui.scroll_area().classes("w-full").style("height: calc(100vh - 80px)"):
        with ui.column().classes("w-full gap-3 max-w-4xl mx-auto p-4"):
            # Summary
            ui.label(
                f"{len(successful_results)} tagged successfully • "
                f"{len(failed_results)} failed"
            ).classes("text-sm text-gray-400 mb-2")

            # Successful results with Accept/Edit/Reject
            for line, result in successful_results:
                _build_review_card(line, result, accepted, edited_text, proj)

            # Failed results
            if failed_results:
                ui.separator().classes("my-4")
                ui.label("Failed (needs manual tags)").classes(
                    "text-lg font-semibold text-amber-400"
                )
                for line, result in failed_results:
                    _build_failed_card(line, result)


def _build_review_card(
    line: Line,
    result: TagResult,
    accepted: dict,
    edited_text: dict,
    proj: Project,
):
    """Build a diff review card for a single successful tagging result."""
    card = ui.card().classes("w-full p-4")
    with card:
        # Header
        with ui.row().classes("items-center gap-2 mb-2"):
            ui.badge(line.character).classes("text-xs")
            ui.label(f"#{line.global_index}").classes("text-xs text-gray-600 font-mono")

        # Original text
        with ui.row().classes("items-start gap-2 w-full"):
            ui.label("Original:").classes("text-xs text-gray-500 min-w-[5rem]")
            ui.label(result.original_text).classes("text-sm font-mono text-gray-300")

        # Tagged text (editable)
        with ui.row().classes("items-start gap-2 w-full mt-1"):
            ui.label("Tagged:").classes("text-xs text-purple-400 min-w-[5rem]")
            tag_input = ui.input(value=result.tagged_text or "").classes(
                "flex-grow font-mono text-sm"
            ).props("dense outlined")
            tag_input.on(
                "update:model-value",
                lambda e, lid=line.line_id: edited_text.update({lid: e.args}),
            )

        # Action buttons
        with ui.row().classes("gap-2 mt-2 justify-end"):
            reject_btn = ui.button("Reject", icon="close").props("flat dense color=red")
            accept_btn = ui.button("Accept", icon="check").props("dense color=purple")

            # Status indicator
            status_label = ui.label("").classes("text-xs self-center")

            def do_accept(
                ln=line,
                res=result,
                lbl=status_label,
                crd=card,
                lid=line.line_id,
            ):
                # Use edited text if available, otherwise original result
                final_text = edited_text.get(lid, res.tagged_text)
                ln.tagged_text = final_text
                accepted[lid] = True
                lbl.text = "✓ accepted"
                lbl.classes(replace="text-xs text-green-400 self-center")
                crd.classes(replace="w-full p-4 border-l-4 border-green-600")
                _save_project_sync()

            def do_reject(
                ln=line,
                lbl=status_label,
                crd=card,
                lid=line.line_id,
            ):
                accepted[lid] = False
                lbl.text = "✗ rejected"
                lbl.classes(replace="text-xs text-red-400 self-center")
                crd.classes(replace="w-full p-4 border-l-4 border-red-600 opacity-50")

            accept_btn.on_click(do_accept)
            reject_btn.on_click(do_reject)


def _build_failed_card(line: Line, result: TagResult):
    """Build a card for a line that failed tagging."""
    with ui.card().classes("w-full p-4 border-l-4 border-amber-500"):
        with ui.row().classes("items-center gap-2 mb-2"):
            ui.badge(line.character).classes("text-xs")
            ui.label(f"#{line.global_index}").classes("text-xs text-gray-600 font-mono")
            ui.icon("warning", size="xs").classes("text-amber-400")
            ui.label("needs manual tags").classes("text-xs text-amber-400")

        ui.label(result.original_text).classes("text-sm font-mono text-gray-300")

        if result.error:
            ui.label(f"Error: {result.error}").classes("text-xs text-red-400 mt-1")

        ui.label(f"Attempts: {result.attempts}/3").classes("text-xs text-gray-500")


def _accept_all(
    proj: Project,
    results: list[tuple[Line, TagResult]],
    dialog_ref: dict,
):
    """Accept all successful tagging results."""
    count = 0
    for line, result in results:
        if result.tagged_text:
            line.tagged_text = result.tagged_text
            count += 1

    _save_project_sync()
    ui.notify(f"Accepted {count} tagged lines", type="positive")
    ui.navigate.to("/editor")


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
