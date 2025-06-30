"""UI tests for Triple-Head Reward Labeling page using Streamlit AppTest."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

# Local imports require adding app/ to sys.path like other UI tests
import sys
import os as _os
sys.path.insert(0, _os.path.join(Path(__file__).resolve().parent, "..", "..", "app"))


def _build_test_script(pref_dir: str) -> str:
    """Return a Streamlit test script string for AppTest."""
    conversation = {
        "conversation_id": "test_conv_001",
        "utterances": [
            {"speaker": "A", "text": "Hello there."},
            {"speaker": "B", "text": "General Kenobi."},
        ],
        "control_tokens": ["friendly"],
        "memory": {
            "embedding": [0.0] * 8,
            "importance": 0.5,
            "surprise": 0.1,
            "valence": 0.0,
            "persistence": 0.6,
        },
    }

    # Embed JSON inline to avoid file I/O inside the test script
    conv_json = json.dumps(conversation)

    return f"""
import os, json, streamlit as st
os.environ['PREFERENCE_DATA_DIR'] = '{pref_dir}'

from pages.triple_head_reward_labeling import page_triple_head_reward_labeling

# Provide conversation directly
st.session_state.conversation = json.loads('''{conv_json}''')
# Enable auto-save so we don't need to click the button
st.session_state.auto_save_preferences = True

page_triple_head_reward_labeling()
"""


class TestTripleHeadRewardLabelingUI:
    """Verify page renders and persists preferences."""

    def test_page_renders_and_saves_preferences(self):
        with tempfile.TemporaryDirectory() as tmp_pref_dir:
            test_script = _build_test_script(tmp_pref_dir)
            at = AppTest.from_string(test_script).run()

            # Page should load without errors
            assert not at.exception

            # Expect preference files to have been created and populated
            gen_file = Path(tmp_pref_dir) / "generation_preferences.jsonl"
            ctrl_file = Path(tmp_pref_dir) / "control_preferences.jsonl"
            mem_file = Path(tmp_pref_dir) / "memory_preferences.jsonl"
            for p in (gen_file, ctrl_file, mem_file):
                assert p.exists(), f"Missing preference file: {p}"
                assert p.read_text().strip(), f"{p.name} should not be empty"

            # Validate basic JSON structure in generation preferences
            sample = json.loads(gen_file.read_text().splitlines()[0])
            expected_keys = {"conversation_id", "content_quality", "creativity", "factual_accuracy", "coordination", "timestamp"}
            assert expected_keys.issubset(sample.keys()), "Preference schema mismatch" 