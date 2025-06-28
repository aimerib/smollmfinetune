---
# R4β-1  Dataset Versioning & Lineage
Status: **Todo**
Ring: R4β
Created: 2025-06-27
---

## Goal
Guarantee reproducible experiments by tracking every dataset used in training/evaluation via content hashes, metadata manifests, and lightweight tooling integrated with git/DVC.

## Context
As we scale to multiple spikes and weekend runs, silent dataset drift will break comparability. A tiny helper CLI and manifest spec will lock each dataset snapshot to a git commit and telemetry run.

## Acceptance Criteria
- [ ] **Manifest Spec**: `dataset.lock` JSON format containing `name`, `hash_sha256`, `num_samples`, `schema_version`, `source_url`, `created_at`.
- [ ] **CLI Tool**: `scripts/dataset_register.py path --name` computes hash & appends/updates lock file.
- [ ] **CI Gate**: GitHub Action fails PR if any dataset directory changed without updated `dataset.lock`.
- [ ] **Telemetry Integration**: telemetry_sdk auto-attaches dataset manifest entry to every run.
- [ ] **Unit Tests**: `tests/unit/test_dataset_versioning.py` verifying hash consistency, lock update behaviour, CI check logic.

## Implementation Notes
```text
• Use Python `hashlib.sha256` streaming over file(s).
• For multi-file datasets, hash the tar/zip archive or deterministic concat of per-file hashes.
• Store manifests under `datasets/<name>/dataset.lock` checked into git.
• Example CI step: `python scripts/check_dataset_lock.py` run in workflow.
```

## Checklist / Steps
1. Design manifest schema & update docs.
2. Implement register CLI + tests.
3. Implement CI check script and workflow.
4. Update telemetry_sdk.

## References
Feeds R3-9 Telemetry, consumed by all R4 training scripts. 