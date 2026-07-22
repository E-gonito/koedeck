"""koedeck — NiceGUI web frontend for the script voice pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
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
from .generator import (
    GenerationSession,
    GenResult,
    GenStatus,
    PreflightSummary,
    check_ffmpeg,
    compute_preflight,
    generate_episode,
    generate_single_line,
    get_ffmpeg_error_message,
)
from .preferences import get_theme, set_theme, is_dark
from .setup import run_all_checks, is_first_launch, CheckStatus
from .validator import validate_script, apply_auto_fixes

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

PROJECT_FILE = Path("project.json")

_project: Project | None = None
_save_timer: asyncio.TimerHandle | None = None
_save_indicator_timer: asyncio.TimerHandle | None = None
_tagging_session: TaggingSession | None = None
_generation_session: GenerationSession | None = None
_pending_script: str | None = None  # Raw script text awaiting cleanup


def _get_project() -> Project | None:
    return _project


def _set_project(p: Project) -> None:
    global _project
    _project = p


def _get_api_key() -> str | None:
    """Load ElevenLabs API key from .env."""
    load_dotenv()
    return os.environ.get("ELEVENLABS_API_KEY")


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
    """Main page: route to setup, import, or editor."""
    # First launch: go to setup wizard
    if is_first_launch():
        ui.navigate.to("/setup")
        return

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
    ui.dark_mode(is_dark())

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
    content = await e.file.text()
    filename = e.file.name
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
    # For fresh imports, go through cleanup screen first
    if existing is None:
        global _pending_script
        _pending_script = content
        ui.navigate.to("/cleanup")
    else:
        ui.navigate.to("/editor")


# ---------------------------------------------------------------------------
# Setup wizard page
# ---------------------------------------------------------------------------


@ui.page("/setup")
async def setup_page():
    """Onboarding wizard that validates all dependencies."""
    ui.dark_mode(is_dark())

    with ui.column().classes("w-full max-w-2xl mx-auto p-8 gap-6"):
        ui.label("koedeck").classes("text-3xl font-bold")
        ui.label("Setup wizard").classes("text-xl text-gray-400")
        ui.label(
            "Let's make sure everything is configured before you start."
        ).classes("text-gray-500")

        # Check results container
        checks_container = ui.column().classes("w-full gap-3")

        with checks_container:
            with ui.card().classes("w-full p-4"):
                ui.label("Running checks...").classes("text-gray-400")
                ui.spinner()

        # Run checks
        status = await run_all_checks()

        # Clear and show results
        checks_container.clear()

        with checks_container:
            for check in status.checks:
                _build_check_card(check)

            ui.separator().classes("my-4")

            if status.all_passed:
                with ui.row().classes("items-center gap-2"):
                    ui.icon("check_circle", size="md").classes("text-green-500")
                    ui.label("All checks passed! Ready to go.").classes("text-green-400 text-lg")

                ui.button(
                    "Continue to import", icon="arrow_forward",
                    on_click=lambda: ui.navigate.to("/import"),
                ).props("color=green").classes("mt-4")
            else:
                failures = status.critical_failures
                with ui.row().classes("items-center gap-2"):
                    ui.icon("warning", size="md").classes("text-amber-400")
                    ui.label(f"{len(failures)} issue(s) need attention").classes("text-amber-400 text-lg")

                with ui.row().classes("gap-2 mt-4"):
                    ui.button(
                        "Re-run checks", icon="refresh",
                        on_click=lambda: ui.navigate.to("/setup"),
                    ).props("outline")
                    ui.button(
                        "Continue anyway", icon="arrow_forward",
                        on_click=lambda: ui.navigate.to("/import"),
                    ).props("flat")


def _build_check_card(check):
    """Build a card showing a single check result."""
    color_map = {
        CheckStatus.PASS: ("check_circle", "text-green-500", "bg-green-900/20"),
        CheckStatus.FAIL: ("cancel", "text-red-500", "bg-red-900/20"),
        CheckStatus.WARN: ("warning", "text-amber-400", "bg-amber-900/20"),
        CheckStatus.SKIP: ("remove_circle", "text-gray-500", "bg-gray-800"),
    }
    icon_name, icon_color, bg_color = color_map.get(
        check.status, ("help", "text-gray-500", "bg-gray-800")
    )

    with ui.card().classes(f"w-full p-3 {bg_color}"):
        with ui.row().classes("items-center gap-3 w-full"):
            ui.icon(icon_name, size="sm").classes(icon_color)
            with ui.column().classes("flex-grow gap-0"):
                ui.label(check.name).classes("font-medium text-sm")
                ui.label(check.message).classes("text-xs text-gray-400")
            if check.fix_hint:
                ui.label(check.fix_hint).classes(
                    "text-xs font-mono bg-gray-800 px-2 py-1 rounded text-gray-300"
                )


# ---------------------------------------------------------------------------
# Script cleanup page
# ---------------------------------------------------------------------------


@ui.page("/cleanup")
async def cleanup_page():
    """Script validation and cleanup screen shown after import."""
    global _pending_script
    proj = _get_project()
    if proj is None or _pending_script is None:
        ui.navigate.to("/editor")
        return

    script_text = _pending_script
    validation = validate_script(script_text)

    ui.dark_mode(is_dark())

    with ui.column().classes("w-full max-w-3xl mx-auto p-8 gap-6"):
        ui.label("koedeck").classes("text-3xl font-bold")
        ui.label("Script cleanup").classes("text-xl text-gray-400")

        # Stats summary
        with ui.card().classes("w-full p-4"):
            with ui.row().classes("gap-6"):
                with ui.column().classes("gap-0"):
                    ui.label(str(validation.stats.get("dialogue_lines", 0))).classes("text-2xl font-bold text-green-400")
                    ui.label("dialogue lines").classes("text-xs text-gray-500")
                with ui.column().classes("gap-0"):
                    ui.label(str(validation.stats.get("direction_lines", 0))).classes("text-2xl font-bold text-blue-400")
                    ui.label("directions").classes("text-xs text-gray-500")
                with ui.column().classes("gap-0"):
                    ui.label(str(validation.stats.get("character_count", 0))).classes("text-2xl font-bold text-purple-400")
                    ui.label("characters").classes("text-xs text-gray-500")
                with ui.column().classes("gap-0"):
                    chars = validation.stats.get("characters", [])
                    ui.label(", ".join(chars[:5])).classes("text-sm text-gray-300")
                    if len(chars) > 5:
                        ui.label(f"...+{len(chars)-5} more").classes("text-xs text-gray-500")

        # Issues
        if validation.issues:
            with ui.card().classes("w-full p-4"):
                with ui.row().classes("items-center justify-between mb-3"):
                    ui.label(f"{len(validation.issues)} issue(s) found").classes("text-lg font-semibold")

                    auto_fixable = [i for i in validation.issues if i.auto_fixable]
                    if auto_fixable:
                        async def do_auto_fix():
                            global _pending_script
                            fixed = apply_auto_fixes(script_text)
                            _pending_script = fixed
                            # Re-parse with fixed text
                            new_proj = parse_markdown(fixed, proj.source_path)
                            _set_project(new_proj)
                            _save_project_sync()
                            ui.notify(f"Applied {len(auto_fixable)} auto-fix(es)", type="positive")
                            ui.navigate.to("/cleanup")

                        ui.button(
                            f"Auto-fix {len(auto_fixable)} issue(s)", icon="auto_fix_high",
                            on_click=do_auto_fix,
                        ).props("dense color=blue")

                with ui.scroll_area().classes("max-h-80"):
                    for issue in validation.issues:
                        _build_issue_row(issue)
        else:
            with ui.card().classes("w-full p-4 bg-green-900/20"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("check_circle", size="sm").classes("text-green-500")
                    ui.label("Script looks clean! No issues found.").classes("text-green-400")

        # Action buttons
        with ui.row().classes("gap-3 mt-2"):
            ui.button(
                "Continue to editor", icon="arrow_forward",
                on_click=lambda: _finish_cleanup(),
            ).props("color=green")
            if validation.has_errors:
                ui.label("(errors present — you may want to fix them first)").classes(
                    "text-xs text-amber-400 self-center"
                )


def _build_issue_row(issue):
    """Build a row for a single script issue."""
    severity_colors = {
        "error": ("cancel", "text-red-500"),
        "warning": ("warning", "text-amber-400"),
        "info": ("info", "text-blue-400"),
    }
    icon_name, icon_color = severity_colors.get(issue.severity.value, ("help", "text-gray-500"))

    with ui.row().classes("items-start gap-2 py-2 border-b border-gray-800 w-full"):
        ui.icon(icon_name, size="xs").classes(icon_color)
        with ui.column().classes("flex-grow gap-0"):
            with ui.row().classes("items-center gap-2"):
                if issue.line_number > 0:
                    ui.label(f"L{issue.line_number}").classes("text-xs font-mono text-gray-600")
                ui.label(issue.message).classes("text-sm")
            if issue.original_text:
                ui.label(issue.original_text).classes("text-xs font-mono text-gray-500")
            if issue.suggested_fix:
                with ui.row().classes("items-center gap-1 mt-1"):
                    ui.icon("arrow_forward", size="xs").classes("text-green-600")
                    ui.label(issue.suggested_fix[:80]).classes("text-xs font-mono text-green-400")


def _finish_cleanup():
    """Finish cleanup and proceed to editor."""
    global _pending_script
    _pending_script = None
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

    ui.dark_mode(is_dark())

    # Top bar
    with ui.header().classes("items-center justify-between px-6"):
        with ui.row().classes("items-center gap-4"):
            ui.label("koedeck").classes("text-xl font-bold")
            ui.label(f"— {proj.source_path}").classes("text-sm opacity-60")

        save_label = ui.label("").classes("text-green-500 text-xs transition-opacity opacity-0")

        with ui.row().classes("items-center gap-2"):
            ui.button("Generate", icon="mic", on_click=_handle_generate_episode).props(
                "dense color=green"
            )
            ui.button("Tag episode", icon="auto_awesome", on_click=_handle_tag_episode).props(
                "dense color=purple"
            )
            ui.button("Export .md", icon="download", on_click=_handle_export).props("flat dense")
            ui.button("Re-import", icon="upload", on_click=lambda: ui.navigate.to("/import")).props(
                "flat dense"
            )

            # Theme toggle
            def toggle_theme():
                new_theme = "light" if is_dark() else "dark"
                set_theme(new_theme)
                ui.navigate.to("/editor")

            theme_icon = "dark_mode" if not is_dark() else "light_mode"
            ui.button(icon=theme_icon, on_click=toggle_theme).props("flat dense round").tooltip(
                "Toggle light/dark mode"
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

                # Per-line regenerate button
                ui.button(
                    icon="replay",
                    on_click=lambda ln=line: _handle_regenerate_line(ln),
                ).props("flat dense round size=sm color=green").tooltip("Regenerate audio")

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

            # Audio player if file exists
            if line.audio.current_file and Path(line.audio.current_file).exists():
                ui.audio(line.audio.current_file).classes("w-full mt-1").props("dense")


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

    ui.dark_mode(is_dark())

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

    with ui.header().classes("items-center justify-between px-6"):
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
# Generation UI
# ---------------------------------------------------------------------------


async def _handle_generate_episode():
    """Show pre-flight summary, then run batch generation."""
    global _generation_session
    proj = _get_project()
    if proj is None:
        ui.notify("No project loaded", type="warning")
        return

    api_key = _get_api_key()
    if not api_key or api_key == "your-key-here":
        ui.notify("Set ELEVENLABS_API_KEY in .env first", type="negative")
        return

    if not check_ffmpeg():
        ui.notify(get_ffmpeg_error_message(), type="negative", timeout=8000)
        return

    try:
        config = load_config()
    except FileNotFoundError:
        ui.notify("config.yaml not found — import a script first", type="negative")
        return

    # Compute pre-flight summary
    preflight = compute_preflight(proj, config)

    # Show pre-flight dialog
    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[28rem] p-6"):
        ui.label("Generate episode").classes("text-lg font-semibold mb-3")

        if preflight.missing_voice_ids:
            with ui.row().classes("items-center gap-2 mb-2"):
                ui.icon("warning").classes("text-amber-400")
                ui.label(
                    f"Missing voice_id for: {', '.join(preflight.missing_voice_ids)}"
                ).classes("text-sm text-amber-400")

        with ui.column().classes("gap-1 mb-4"):
            ui.label(f"Total dialogue lines: {preflight.total_lines}").classes("text-sm")
            ui.label(f"Lines to generate: {preflight.lines_to_generate}").classes(
                "text-sm text-green-400"
            )
            ui.label(f"Lines to skip (cached): {preflight.lines_to_skip}").classes(
                "text-sm text-gray-400"
            )
            ui.label(f"Total characters: {preflight.total_characters:,}").classes("text-sm")
            ui.label(
                f"Estimated cost: ~{preflight.estimated_credits:,.0f} credits"
            ).classes("text-sm text-blue-400")

        with ui.row().classes("gap-2 justify-end"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            can_generate = preflight.lines_to_generate > 0 and not preflight.missing_voice_ids

            async def start_gen():
                dialog.close()
                await _run_generation(proj, config, api_key)

            ui.button(
                "Generate", on_click=start_gen
            ).props(f"color=green {'disabled' if not can_generate else ''}")

    dialog.open()


async def _run_generation(proj: Project, config: dict, api_key: str):
    """Run the batch generation with progress UI."""
    global _generation_session

    session = GenerationSession()
    _generation_session = session

    # Progress dialog
    dialog = ui.dialog().props("persistent")
    with dialog, ui.card().classes("w-96 p-6"):
        ui.label("Generating audio...").classes("text-lg font-semibold")
        progress_label = ui.label("Starting...").classes("text-sm text-gray-400")
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
        stats_label = ui.label("").classes("text-xs text-gray-500 mt-1")
        ui.button(
            "Cancel", on_click=lambda: _cancel_generation(session, dialog)
        ).props("flat color=red")

    dialog.open()

    def on_progress(s: GenerationSession):
        if s.total > 0:
            progress_bar.value = s.completed / s.total
            progress_label.text = f"{s.completed}/{s.total} lines"
            stats_label.text = (
                f"Done: {s.completed - s.skipped - s.failed} • "
                f"Skipped: {s.skipped} • Failed: {s.failed}"
            )

    gen_config = config.get("generation", {})
    max_concurrent = gen_config.get("max_concurrent_requests", 3)

    await generate_episode(
        proj, config, api_key, session,
        max_concurrent=max_concurrent,
        on_progress=on_progress,
    )

    dialog.close()

    # Save project with updated audio state
    _save_project_sync()

    if session.cancelled:
        ui.notify("Generation cancelled", type="warning")
    else:
        done = session.completed - session.skipped - session.failed
        msg = f"Done: {done} generated, {session.skipped} cached, {session.failed} failed"
        ui.notify(msg, type="positive" if session.failed == 0 else "warning", timeout=6000)

        # If there are failures, offer retry
        if session.failed > 0:
            _show_retry_dialog(proj, config, api_key, session)

    ui.navigate.to("/editor")


def _show_retry_dialog(proj: Project, config: dict, api_key: str, session: GenerationSession):
    """Show a dialog listing failed lines with a retry button."""
    failed = [r for r in session.results.values() if r.status == GenStatus.FAILED]
    if not failed:
        return

    dialog = ui.dialog()
    with dialog, ui.card().classes("w-[30rem] p-6"):
        ui.label(f"{len(failed)} line(s) failed").classes("text-lg font-semibold text-red-400")

        with ui.scroll_area().classes("max-h-60 w-full"):
            for result in failed:
                with ui.row().classes("items-center gap-2 py-1"):
                    ui.label(result.line_id).classes("text-xs font-mono text-gray-500")
                    ui.label(result.error or "Unknown error").classes("text-xs text-red-300")

        with ui.row().classes("gap-2 justify-end mt-4"):
            ui.button("Dismiss", on_click=dialog.close).props("flat")

            async def retry():
                dialog.close()
                await _retry_failed(proj, config, api_key, failed)

            ui.button("Retry failed", on_click=retry).props("color=green")

    dialog.open()


async def _retry_failed(
    proj: Project, config: dict, api_key: str, failed_results: list[GenResult]
):
    """Retry only the failed lines."""
    line_map = {l.line_id: l for l in proj.lines}
    session = GenerationSession()
    session.total = len(failed_results)

    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        for result in failed_results:
            line = line_map.get(result.line_id)
            if line is None:
                continue
            new_result = await generate_single_line(
                line, config, api_key, session, client=client
            )
            session.results[line.line_id] = new_result
            session.completed += 1

    _save_project_sync()
    done = sum(1 for r in session.results.values() if r.status == GenStatus.DONE)
    ui.notify(f"Retry complete: {done}/{len(failed_results)} succeeded", type="info")
    ui.navigate.to("/editor")


async def _handle_regenerate_line(line: Line):
    """Regenerate audio for a single line (force, creating a new take)."""
    api_key = _get_api_key()
    if not api_key or api_key == "your-key-here":
        ui.notify("Set ELEVENLABS_API_KEY in .env first", type="negative")
        return

    if not check_ffmpeg():
        ui.notify(get_ffmpeg_error_message(), type="negative")
        return

    try:
        config = load_config()
    except FileNotFoundError:
        ui.notify("config.yaml not found", type="negative")
        return

    ui.notify("Regenerating...", type="info", timeout=2000)

    session = GenerationSession()
    result = await generate_single_line(
        line, config, api_key, session, force=True
    )

    _save_project_sync()

    if result.status == GenStatus.DONE:
        ui.notify("Audio regenerated", type="positive")
    else:
        ui.notify(f"Failed: {result.error}", type="negative", timeout=5000)

    ui.navigate.to("/editor")


def _cancel_generation(session: GenerationSession, dialog):
    """Cancel an in-progress generation."""
    session.cancelled = True
    dialog.close()


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
