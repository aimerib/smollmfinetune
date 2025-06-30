"""tests/scripts/test_run_triple_head_self_play.py
Unit tests for the Triple-Head Self-Play Harness (scripts/run_triple_head_self_play.py).
These tests run in mock-mode so they finish quickly and do not require GPU resources.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts/run_triple_head_self_play.py")


class TestTripleHeadSelfPlayHarness:
    """Basic harness validation in mock-mode."""

    def test_script_is_importable(self):
        """The script module should import without side-effects."""
        import importlib.util  # noqa: WPS433 – stdlib use only

        spec = importlib.util.spec_from_file_location("run_th_self_play", SCRIPT_PATH)
        assert spec is not None, "Spec should be created"
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        assert hasattr(module, "main"), "Module should expose main()"

    def test_mock_mode_generates_head_logs(self):
        """Running the script in mock-mode should create 3 JSONL log files with content."""
        assert SCRIPT_PATH.exists(), f"Self-play script not found at {SCRIPT_PATH}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT_PATH),
                    "--mock-mode",
                    "--output-dir",
                    tmp_dir,
                    "--conversations",
                    "2",
                    "--turns",
                    "1",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            # Should exit successfully
            assert result.returncode == 0, result.stderr

            out_path = Path(tmp_dir)
            gen_log = out_path / "generation_log.jsonl"
            ctrl_log = out_path / "control_log.jsonl"
            mem_log = out_path / "memory_log.jsonl"

            # All logs should exist and be non-empty
            for p in (gen_log, ctrl_log, mem_log):
                assert p.exists(), f"Missing log file: {p}"
                content = p.read_text().strip().splitlines()
                assert len(content) > 0, f"{p.name} should contain at least one line"

            # Basic schema check on one sample line
            sample = json.loads((gen_log.read_text().splitlines())[0])
            expected_keys = {"conversation_id", "turn", "speaker", "text", "timestamp"}
            assert expected_keys.issubset(sample.keys()), "Generation log schema mismatch" 
