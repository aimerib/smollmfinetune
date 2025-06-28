---
# R3-7  Training Telemetry SDK
Status: **Todo**
Ring: R3
Created: 2025-06-27
---

## Goal
Provide a lightweight, reusable Python package (`telemetry_sdk`) that every training/evaluation script can import to auto-log configs, git commit, hyper-params, metrics, and artifact hashes to both WandB and the local SQLite DB introduced in R3-1.

## Context
Once experiments ramp up (Mini-MoE, Tiny-Llama spikes, etc.) we need iron-clad reproducibility. A single import line should guarantee that every run is tracked uniformly so comparisons are trustworthy.

## Acceptance Criteria
- [ ] **Package**: `app/utils/telemetry_sdk/__init__.py` exposing `init(run_name: str, cfg: dict)` and `log(metrics: dict)` APIs.
- [ ] **Auto-Capture**: On `init`, captures `git rev-parse HEAD`, `pip freeze`, and any env variables in `.env` starting with `RUN_`.
- [ ] **Backends**: Pluggable WandB + SQLite writers; failures fallback to local CSV.
- [ ] **CLI**: `python -m telemetry_sdk.report run_id` prints human-readable summary.
- [ ] **Unit Tests**: `tests/unit/test_telemetry_sdk.py` achieving 95 % branch coverage.
- [ ] **Migration**: Refactor existing scripts (`run_sft.py`, `run_mini_moe_sft.py`) to import and use SDK.

## Implementation Notes
```text
• SQLite table `runs(id TEXT PRIMARY KEY, started_at, git_sha, config_json, notes)` plus `metrics(run_id, step, key, value)`.
• Use `atexit` to flush metric buffer.
• Provide decorator `@telemetry_sdk.capture_cfg` to auto-log dataclass configs.
```

## Checklist / Steps
1. Scaffold package & write minimal `init`/`log` stubs.
2. Implement SQLite writer + unit tests.
3. Implement WandB writer; respect `WANDB_DISABLED` flag.
4. Add CLI reporter.
5. Update SFT scripts to adopt SDK.

## References
Interfaces with R3-1 SQLite backend and feeds dashboards in R3-4 Observability. 