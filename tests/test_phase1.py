"""Phase 1 tests for koedeck: parser, exporter, re-import, stable IDs."""

from pathlib import Path

from koedeck.config import generate_config, load_config
from koedeck.exporter import export_project
from koedeck.models import Line, Project, _generate_line_id
from koedeck.parser import parse_markdown
from koedeck.reimport import reimport_markdown

# The spec example script
SPEC_EXAMPLE = """\
(Continues directly from Episode 8 — argument already running in the background)

AIGIS: For you, Mitsuru-san, forty million yen.

MITSURU: FORTY MILL— ahem. Regrettably... I'll take... half a bag on credit\\!

AIGIS: ...Confirming purchase. (SUDDENLY PHARMA-AD SPEED, one breath:) Please-be-advised... (normal speed, bright:) Would you also like a membership card?

(Visual gag: Aigis becomes the card terminal.)

(Cut to hospital)
"""


class TestParser:
    """Tests for the markdown parser."""

    def test_parses_spec_example(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        assert len(project.lines) == 6
        assert project.characters == ["AIGIS", "MITSURU"]

    def test_identifies_directions(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        directions = [l for l in project.lines if l.type == "direction"]
        assert len(directions) == 3
        assert all(d.character is None for d in directions)

    def test_identifies_dialogue(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        dialogue = [l for l in project.lines if l.type == "dialogue"]
        assert len(dialogue) == 3
        assert dialogue[0].character == "AIGIS"
        assert dialogue[1].character == "MITSURU"
        assert dialogue[2].character == "AIGIS"

    def test_unescapes_markdown(self):
        """\\! → !, \\+ → +, \\? → ?"""
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        mitsuru = [l for l in project.lines if l.character == "MITSURU"][0]
        assert mitsuru.text.endswith("credit!")
        assert "\\" not in mitsuru.text

    def test_preserves_em_dash_and_ellipsis(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        mitsuru = [l for l in project.lines if l.character == "MITSURU"][0]
        assert "—" in mitsuru.text  # em-dash
        assert "..." in mitsuru.text  # ellipsis

    def test_preserves_all_caps(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        mitsuru = [l for l in project.lines if l.character == "MITSURU"][0]
        assert "FORTY MILL" in mitsuru.text

    def test_assigns_unique_line_ids(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        ids = [l.line_id for l in project.lines]
        assert len(ids) == len(set(ids)), "Line IDs must be unique"

    def test_line_id_format(self):
        """IDs should be 6-char base36."""
        lid = _generate_line_id()
        assert len(lid) == 6
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789" for c in lid)

    def test_global_indices_sequential(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        indices = [l.global_index for l in project.lines]
        assert indices == list(range(len(project.lines)))

    def test_characters_ordered_by_first_appearance(self):
        script = "MITSURU: Hello.\n\nAIGIS: Hi.\n\nMITSURU: How are you?\n"
        project = parse_markdown(script, "test.md")
        assert project.characters == ["MITSURU", "AIGIS"]

    def test_multiword_speaker_name(self):
        """Speaker names can have spaces, e.g. 'OLD MAN:'."""
        script = "OLD MAN: Back in my day...\n"
        project = parse_markdown(script, "test.md")
        assert project.lines[0].character == "OLD MAN"


class TestParenthetical:
    """Tests for parenthetical extraction and stripping."""

    def test_parentheticals_stripped_from_text(self):
        """text sent to TTS never contains ( ) content from parentheticals."""
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        aigis_tagged = [
            l for l in project.lines if l.character == "AIGIS" and l.parentheticals
        ][0]
        assert "(" not in aigis_tagged.text
        assert ")" not in aigis_tagged.text
        assert "SUDDENLY" not in aigis_tagged.text
        assert "normal speed" not in aigis_tagged.text

    def test_parentheticals_stored_with_offsets(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        aigis_tagged = [
            l for l in project.lines if l.character == "AIGIS" and l.parentheticals
        ][0]
        assert len(aigis_tagged.parentheticals) == 2
        p1, p2 = aigis_tagged.parentheticals
        assert "SUDDENLY PHARMA-AD SPEED" in p1.text
        assert "normal speed, bright" in p2.text
        assert p1.offset < p2.offset

    def test_no_parentheticals_for_simple_line(self):
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        aigis_simple = project.lines[1]  # "For you, Mitsuru-san..."
        assert aigis_simple.parentheticals == []

    def test_direction_parens_are_not_stripped(self):
        """Directions keep their parentheses as-is — they're the whole content."""
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        direction = project.lines[0]
        assert direction.text.startswith("(")
        assert direction.text.endswith(")")

    def test_tts_text_never_contains_paren_content(self):
        """Comprehensive: for ALL dialogue lines, text must not contain paren content."""
        script = """\
AIGIS: Hello (whispered:) world!

MITSURU: (laughing:) This is great (sarcastically:) really great.

YUKARI: No parens here at all.
"""
        project = parse_markdown(script, "test.md")
        for line in project.lines:
            if line.type == "dialogue" and line.parentheticals:
                for paren in line.parentheticals:
                    # The inner text (without outer parens) should not appear in text
                    inner = paren.text[1:-1]  # strip ( )
                    assert inner not in line.text, (
                        f"Parenthetical content '{inner}' found in TTS text: '{line.text}'"
                    )


class TestRoundTrip:
    """Tests for parser → export round-trip."""

    def test_spec_example_round_trip(self):
        """Import → export produces semantically identical markdown."""
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        exported = export_project(project)
        assert exported.strip() == SPEC_EXAMPLE.strip()

    def test_simple_dialogue_round_trip(self):
        script = "AIGIS: For you, Mitsuru-san, forty million yen.\n"
        project = parse_markdown(script, "test.md")
        exported = export_project(project)
        assert exported.strip() == script.strip()

    def test_escaped_chars_round_trip(self):
        script = "MITSURU: What\\! No way\\!\n"
        project = parse_markdown(script, "test.md")
        exported = export_project(project)
        assert exported.strip() == script.strip()

    def test_direction_round_trip(self):
        script = "(Cut to hospital)\n"
        project = parse_markdown(script, "test.md")
        exported = export_project(project)
        assert exported.strip() == script.strip()

    def test_mixed_content_round_trip(self):
        script = """\
(Opening scene)

AIGIS: Welcome.

(Pause)

MITSURU: Thank you.
"""
        project = parse_markdown(script, "test.md")
        exported = export_project(project)
        assert exported.strip() == script.strip()


class TestStableIDs:
    """Tests for stable line ID preservation."""

    def test_editing_text_does_not_change_id(self):
        """Editing a line's text must never change its ID."""
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        line = project.lines[1]  # AIGIS first dialogue
        original_id = line.line_id

        # Simulate an edit
        line.text = "Completely different text now."
        assert line.line_id == original_id

    def test_inserting_line_preserves_other_ids(self):
        """Inserting a line must never renumber other lines' IDs."""
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        original_ids = {l.line_id for l in project.lines}

        # Insert a new line
        new_line = Line(
            type="dialogue",
            character="YUKARI",
            raw_text="YUKARI: Hey everyone!",
            text="Hey everyone!",
        )
        project.lines.insert(2, new_line)
        project.reindex()

        # All original IDs still exist
        current_ids = {l.line_id for l in project.lines}
        assert original_ids.issubset(current_ids)

        # New line has a different ID
        assert new_line.line_id not in original_ids

    def test_reimport_preserves_ids_exact_match(self):
        """Re-importing identical text preserves all line IDs."""
        project = parse_markdown(SPEC_EXAMPLE, "test.md")
        original_ids = [l.line_id for l in project.lines]

        # Re-import the same text
        new_project = reimport_markdown(SPEC_EXAMPLE, project)
        new_ids = [l.line_id for l in new_project.lines if not l.orphaned]

        assert new_ids == original_ids

    def test_reimport_preserves_ids_fuzzy_match(self):
        """Re-importing with minor text changes preserves IDs via fuzzy match."""
        original = "AIGIS: For you, Mitsuru-san, forty million yen.\n"
        project = parse_markdown(original, "test.md")
        original_id = project.lines[0].line_id

        # Slightly modified (fuzzy should match: same character, >0.85 ratio)
        modified = "AIGIS: For you, Mitsuru-san, forty million yen today.\n"
        new_project = reimport_markdown(modified, project)

        matched = [l for l in new_project.lines if not l.orphaned]
        assert matched[0].line_id == original_id

    def test_reimport_marks_orphans(self):
        """Lines removed from the script become orphaned."""
        original = "AIGIS: Line one.\n\nAIGIS: Line two.\n"
        project = parse_markdown(original, "test.md")
        id_line_two = project.lines[1].line_id

        # Only keep line one
        modified = "AIGIS: Line one.\n"
        new_project = reimport_markdown(modified, project)

        orphaned = [l for l in new_project.lines if l.orphaned]
        assert len(orphaned) == 1
        assert orphaned[0].line_id == id_line_two

    def test_reimport_new_lines_get_new_ids(self):
        """Entirely new lines get fresh IDs."""
        original = "AIGIS: Hello.\n"
        project = parse_markdown(original, "test.md")
        original_id = project.lines[0].line_id

        modified = "AIGIS: Hello.\n\nMITSURU: Goodbye.\n"
        new_project = reimport_markdown(modified, project)

        active = [l for l in new_project.lines if not l.orphaned]
        assert active[0].line_id == original_id
        assert active[1].line_id != original_id

    def test_reimport_preserves_audio_state(self):
        """Audio state is preserved for matched lines."""
        original = "AIGIS: Hello world.\n"
        project = parse_markdown(original, "test.md")

        # Simulate existing audio
        project.lines[0].audio.audio_hash = "abc123"
        project.lines[0].audio.current_file = "output/AIGIS/001_xxx_hello-world.wav"

        # Re-import same text
        new_project = reimport_markdown(original, project)
        line = [l for l in new_project.lines if not l.orphaned][0]
        assert line.audio.audio_hash == "abc123"
        assert line.audio.current_file == "output/AIGIS/001_xxx_hello-world.wav"


class TestConfig:
    """Tests for config.yaml generation."""

    def test_generates_config_for_all_characters(self, tmp_path: Path):
        script = "AIGIS: Hi.\n\nMITSURU: Hello.\n\nYUKARI: Hey.\n"
        project = parse_markdown(script, "test.md")

        config_path = tmp_path / "config.yaml"
        generate_config(project, config_path)

        config = load_config(config_path)
        assert "AIGIS" in config["characters"]
        assert "MITSURU" in config["characters"]
        assert "YUKARI" in config["characters"]

    def test_config_has_tagging_section(self, tmp_path: Path):
        project = parse_markdown("AIGIS: Hi.\n", "test.md")
        config_path = tmp_path / "config.yaml"
        generate_config(project, config_path)

        config = load_config(config_path)
        assert config["tagging"]["base_url"] == "http://localhost:11434/v1"
        assert config["tagging"]["model"] == "qwen3:8b"
        assert "excited" in config["tagging"]["tag_whitelist"]

    def test_config_does_not_overwrite_existing(self, tmp_path: Path):
        """Re-running generate_config doesn't overwrite manually set values."""
        project = parse_markdown("AIGIS: Hi.\n\nMITSURU: Hey.\n", "test.md")
        config_path = tmp_path / "config.yaml"

        # First generation
        generate_config(project, config_path)

        # Manually modify
        config = load_config(config_path)
        config["characters"]["AIGIS"]["voice_id"] = "my-voice-123"
        config["characters"]["AIGIS"]["hints"] = "Custom hint"
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Re-generate (add new character)
        project2 = parse_markdown("AIGIS: Hi.\n\nMITSURU: Hey.\n\nYUKARI: Hello.\n", "test.md")
        generate_config(project2, config_path)

        config2 = load_config(config_path)
        # Existing entry preserved
        assert config2["characters"]["AIGIS"]["voice_id"] == "my-voice-123"
        assert config2["characters"]["AIGIS"]["hints"] == "Custom hint"
        # New entry added
        assert "YUKARI" in config2["characters"]

    def test_character_entry_has_voice_id_and_hints(self, tmp_path: Path):
        project = parse_markdown("AIGIS: Hi.\n", "test.md")
        config_path = tmp_path / "config.yaml"
        generate_config(project, config_path)

        config = load_config(config_path)
        assert "voice_id" in config["characters"]["AIGIS"]
        assert "hints" in config["characters"]["AIGIS"]
