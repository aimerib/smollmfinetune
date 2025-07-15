#!/usr/bin/env python3
"""
Triple-Head Self-Play Harness

This script generates conversational snippets between two Narrative-LLM instances (or lightweight
mock models) and logs each model head separately:

• generation_log.jsonl – free-text responses
• control_log.jsonl    – emitted control tokens
• memory_log.jsonl     – generated memory vectors + metadata

Designed for Ring 4-7 preference data collection.

Usage (full model):
    python scripts/run_triple_head_self_play.py --output-dir selfplay_out --conversations 3 --turns 4

CI / Test quick-run:
    python scripts/run_triple_head_self_play.py --mock-mode --output-dir /tmp/out

The mock-mode path avoids heavyweight model imports so unit tests complete in <2 s on CPU-only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------------------
# Lightweight Mock Model (used when --mock-mode is passed)
# --------------------------------------------------------------------------------------

class _MockTripleHeadModel:
    """Deterministic stub that mimics Narrative-LLM triple-head outputs."""

    def __init__(self, name: str):
        self.name = name
        self._rng = random.Random(name)  # deterministic per instance

    def generate(self, prompt: str, turn_idx: int) -> Dict[str, object]:
        # Very small, deterministic vocabulary for generation text
        responses = [
            "Of course – that makes perfect sense.",
            "I hadn't thought of it that way before!",
            "Let's explore that idea together.",
            "Could you clarify what you mean?",
            "Absolutely, I'm on board with this plan.",
        ]
        text = responses[(self._rng.randint(0, 1000) + turn_idx) % len(responses)]

        # Pick some fake control tokens
        control_token_pool = ["happy", "curious", "thoughtful", "surprised"]
        control_tokens = [
            tok for tok in control_token_pool if self._rng.random() < 0.25
        ] or [control_token_pool[turn_idx % len(control_token_pool)]]

        # Pseudo-memory vector (small for disk, but keeps interface)
        memory_embedding = [round(self._rng.random(), 4) for _ in range(8)]  # 8-dim mini vec
        memory_meta = {
            "importance": round(self._rng.random(), 3),
            "surprise": round(self._rng.random(), 3),
            "valence": round(self._rng.uniform(-1, 1), 3),
            "persistence": round(self._rng.random(), 3),
        }
        return {
            "generated_text": text,
            "control_tokens": control_tokens,
            "memory": {"embedding": memory_embedding, **memory_meta},
        }


# --------------------------------------------------------------------------------------
# Core generation and logging logic
# --------------------------------------------------------------------------------------

def _write_jsonl(path: Path, obj: Dict[str, object]) -> None:
    """Append JSON line (UTF-8) atomically."""
    with path.open("a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")


def _run_conversation_pair(
    model_a,
    model_b,
    conversation_id: str,
    turns: int,
    gen_log: Path,
    ctrl_log: Path,
    mem_log: Path,
) -> None:
    """Generate a back-and-forth conversation between two models and log each head."""
    prompt_a, prompt_b = "Hello!", "Hi there!"
    for turn in range(turns):
        # Model A → Model B
        response_a = model_a.generate(prompt_b, turn)
        _log_turn(gen_log, ctrl_log, mem_log, conversation_id, turn * 2, "A", response_a)

        # Model B → Model A
        response_b = model_b.generate(prompt_a, turn)
        _log_turn(gen_log, ctrl_log, mem_log, conversation_id, turn * 2 + 1, "B", response_b)

        # Update prompts for next cycle
        prompt_a, prompt_b = response_a["generated_text"], response_b["generated_text"]


def _log_turn(
    gen_log: Path,
    ctrl_log: Path,
    mem_log: Path,
    conversation_id: str,
    turn_index: int,
    speaker: str,
    head_outputs: Dict[str, object],
) -> None:
    ts = datetime.utcnow().isoformat()

    _write_jsonl(
        gen_log,
        {
            "conversation_id": conversation_id,
            "turn": turn_index,
            "speaker": speaker,
            "text": head_outputs["generated_text"],
            "timestamp": ts,
        },
    )

    _write_jsonl(
        ctrl_log,
        {
            "conversation_id": conversation_id,
            "turn": turn_index,
            "speaker": speaker,
            "control_tokens": head_outputs["control_tokens"],
            "timestamp": ts,
        },
    )

    _write_jsonl(
        mem_log,
        {
            "conversation_id": conversation_id,
            "turn": turn_index,
            "speaker": speaker,
            "memory": head_outputs["memory"],
            "timestamp": ts,
        },
    )


# --------------------------------------------------------------------------------------
# CLI entry-point
# --------------------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Triple-Head Self-Play Harness (R4-7)")
    parser.add_argument("--output-dir", type=str, default="self_play_output", help="Directory to store JSONL logs")
    parser.add_argument("--conversations", type=int, default=3, help="Number of independent conversations to generate")
    parser.add_argument("--turns", type=int, default=4, help="Turns per conversation (each model speaks once per turn)")
    parser.add_argument("--mock-mode", action="store_true", help="Use lightweight mock models (runs fast, no GPU)")

    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare log files (overwrite existing)
    gen_log = out_dir / "generation_log.jsonl"
    ctrl_log = out_dir / "control_log.jsonl"
    mem_log = out_dir / "memory_log.jsonl"
    for p in (gen_log, ctrl_log, mem_log):
        if p.exists():
            p.unlink()
        p.touch()

    # Instantiate models
    if args.mock_mode:
        model_a = _MockTripleHeadModel("ModelA")
        model_b = _MockTripleHeadModel("ModelB")
    else:
        try:
            from backend.app.narrative_engine.model import create_narrative_model  # heavy import
        except Exception as exc:
            print("❌ Failed to import Narrative-LLM. Try --mock-mode for quick tests.", file=sys.stderr)
            raise exc
        model_a = create_narrative_model()
        model_b = create_narrative_model()

    # Run conversations
    for idx in range(args.conversations):
        conv_id = f"conv_{idx:04d}"
        _run_conversation_pair(
            model_a,
            model_b,
            conv_id,
            args.turns,
            gen_log,
            ctrl_log,
            mem_log,
        )

    print(
        f"✅ Completed {args.conversations} conversation(s) • logs written to {out_dir.resolve()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main() 