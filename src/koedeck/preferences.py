"""User preferences for koedeck (persisted to preferences.json)."""

from __future__ import annotations

import json
from pathlib import Path

PREFS_FILE = Path("preferences.json")

_defaults = {
    "theme": "light",  # "light" or "dark"
}


def _load() -> dict:
    if PREFS_FILE.exists():
        try:
            return {**_defaults, **json.loads(PREFS_FILE.read_text())}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_defaults)


def _save(prefs: dict) -> None:
    PREFS_FILE.write_text(json.dumps(prefs, indent=2))


def get_theme() -> str:
    return _load().get("theme", "light")


def set_theme(theme: str) -> None:
    prefs = _load()
    prefs["theme"] = theme
    _save(prefs)


def is_dark() -> bool:
    return get_theme() == "dark"
