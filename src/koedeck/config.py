"""Config skeleton generator for koedeck.

Generates a config.yaml with:
- Tagging settings (LLM base_url, model, tag whitelist)
- Per-character entries with empty voice_id and placeholder hints
- Generation settings (ElevenLabs model, concurrency, etc.)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import Project

# Default tag whitelist for ElevenLabs v3
DEFAULT_TAG_WHITELIST = [
    "excited",
    "angry",
    "shouting",
    "whispers",
    "sighs",
    "laughs",
    "nervous",
    "sarcastic",
    "curious",
    "crying",
    "mischievously",
    "deadpan",
    "slowly",
    "cheerfully",
    "annoyed",
    "frustrated",
]

DEFAULT_CONFIG = {
    "tagging": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:8b",
        "tag_whitelist": DEFAULT_TAG_WHITELIST,
    },
    "generation": {
        "model_id": "eleven_v3",
        "output_format": "pcm_44100",
        "fallback_format": "mp3_44100_192",
        "max_concurrent_requests": 3,
        "credits_per_character": 0.3,
    },
    "characters": {},
}


def generate_config(project: Project, config_path: Path | str = "config.yaml") -> Path:
    """Generate a config.yaml skeleton from the project's characters.

    If the config file already exists, only adds missing characters without
    overwriting existing entries.

    Args:
        project: The parsed project with characters derived.
        config_path: Path to write the config file.

    Returns:
        Path to the written config file.
    """
    config_path = Path(config_path)

    # Load existing config if present
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = dict(DEFAULT_CONFIG)

    # Ensure characters section exists
    if "characters" not in config:
        config["characters"] = {}

    # Ensure tagging section exists with defaults
    if "tagging" not in config:
        config["tagging"] = dict(DEFAULT_CONFIG["tagging"])

    # Ensure generation section exists
    if "generation" not in config:
        config["generation"] = dict(DEFAULT_CONFIG["generation"])

    # Add missing characters
    for character in project.characters:
        if character not in config["characters"]:
            config["characters"][character] = {
                "voice_id": "",
                "hints": f"Describe {character}'s vocal style and when to use emotion tags.",
            }

    # Write config with nice formatting
    with open(config_path, "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )

    return config_path


def load_config(config_path: Path | str = "config.yaml") -> dict:
    """Load and return the config.yaml as a dict.

    Args:
        config_path: Path to the config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f) or {}
