"""
🎛️ Triple-Head Reward Labeling Interface

Allows human evaluators to rate outputs from the generation, control, and memory heads
in a single view.  Ratings are written to head-specific JSONL files so that downstream
training pipelines (DPO / reward modelling) can consume them directly.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import streamlit as st

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------

PREF_DIR = Path(os.environ.get("PREFERENCE_DATA_DIR", "preference_data"))
PREF_DIR.mkdir(parents=True, exist_ok=True)

GEN_FILE = PREF_DIR / "generation_preferences.jsonl"
CTRL_FILE = PREF_DIR / "control_preferences.jsonl"
MEM_FILE = PREF_DIR / "memory_preferences.jsonl"
for _f in (GEN_FILE, CTRL_FILE, MEM_FILE):
    _f.touch(exist_ok=True)

# ----------------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------------

def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a dictionary to a JSONL file (UTF-8)."""
    with path.open("a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")


def _save_preferences(conv_id: str, ratings: Dict[str, Dict[str, Any]]) -> None:
    """Persist per-head ratings using the agreed filenames."""
    ts = datetime.utcnow().isoformat()

    # Generation head
    _append_jsonl(GEN_FILE, {
        "conversation_id": conv_id,
        "timestamp": ts,
        **ratings["generation"],
    })

    # Control head
    _append_jsonl(CTRL_FILE, {
        "conversation_id": conv_id,
        "timestamp": ts,
        **ratings["control"],
    })

    # Memory head
    _append_jsonl(MEM_FILE, {
        "conversation_id": conv_id,
        "timestamp": ts,
        **ratings["memory"],
    })


# ----------------------------------------------------------------------------------
# Page Body
# ----------------------------------------------------------------------------------

def page_triple_head_reward_labeling():  # noqa: N802 – Streamlit naming convention
    """Render the triple-head reward labeling page."""

    st.title("🧑‍🔬 Triple-Head Reward Labeling")

    # The conversation to be rated should be provided in session_state for tests / caller code.
    conversation: Dict[str, Any] | None = st.session_state.get("conversation")

    if conversation is None:
        st.info("Load a conversation JSON dict into st.session_state['conversation'] to begin.")
        uploaded = st.file_uploader("Or upload a single-turn conversation JSON file", type=["json"])
        if uploaded is None:
            st.stop()
        try:
            conversation = json.load(uploaded)
            st.session_state["conversation"] = conversation
        except Exception as exc:  # pragma: no cover – guardrail
            st.error(f"Failed to parse JSON: {exc}")
            st.stop()

    conv_id = conversation.get("conversation_id", "manual_conv")

    # Preview section
    with st.expander("Conversation Preview", expanded=True):
        st.json(conversation, expanded=False)

    # ------------------------------------------------------------------
    # Generation head ratings
    # ------------------------------------------------------------------
    st.header("📝 Generation Head – Content Quality")
    gen_quality = st.slider("Content quality & coherence (1-10)", 1, 10, 5, key="gen_quality")
    gen_creativity = st.slider("Creativity & engagement (1-10)", 1, 10, 5, key="gen_creativity")
    gen_factual = st.slider("Factual accuracy & consistency (1-10)", 1, 10, 5, key="gen_factual")

    # ------------------------------------------------------------------
    # Control head ratings
    # ------------------------------------------------------------------
    st.header("🎭 Control Head – Emotional Appropriateness")
    ctrl_emotion = st.slider("Emotional appropriateness (1-10)", 1, 10, 5, key="ctrl_emotion")
    ctrl_personality = st.slider("Personality consistency (1-10)", 1, 10, 5, key="ctrl_personality")
    ctrl_mood = st.slider("Mood matching / transitions (1-10)", 1, 10, 5, key="ctrl_mood")

    # ------------------------------------------------------------------
    # Memory head ratings
    # ------------------------------------------------------------------
    st.header("🧬 Memory Head – Memory Formation")
    mem_accuracy = st.slider("Memory accuracy (1-10)", 1, 10, 5, key="mem_accuracy")
    mem_consistency = st.slider("Memory consistency (1-10)", 1, 10, 5, key="mem_consistency")
    mem_quality = st.slider("Memory formation quality (1-10)", 1, 10, 5, key="mem_quality")

    # ------------------------------------------------------------------
    # Cross-head coordination rating
    # ------------------------------------------------------------------
    st.subheader("🤝 Cross-Head Coordination")
    coordination = st.slider("Overall harmony across heads (1-10)", 1, 10, 5, key="coordination")

    # ------------------------------------------------------------------
    # Persist ratings – either via button or automatic mode for tests
    # ------------------------------------------------------------------
    auto_save = bool(st.session_state.get("auto_save_preferences", False))
    if st.button("💾 Save Preferences") or auto_save:
        _save_preferences(
            conv_id,
            ratings={
                "generation": {
                    "content_quality": gen_quality,
                    "creativity": gen_creativity,
                    "factual_accuracy": gen_factual,
                    "coordination": coordination,
                },
                "control": {
                    "emotional_appropriateness": ctrl_emotion,
                    "personality_consistency": ctrl_personality,
                    "mood_matching": ctrl_mood,
                    "coordination": coordination,
                },
                "memory": {
                    "memory_accuracy": mem_accuracy,
                    "memory_consistency": mem_consistency,
                    "formation_quality": mem_quality,
                    "coordination": coordination,
                },
            },
        )
        st.success("Preferences saved – thank you for the feedback! ✅")


# Expose entry-point for Streamlit navigation discovery
if __name__ == "__main__":  # pragma: no cover
    page_triple_head_reward_labeling() 