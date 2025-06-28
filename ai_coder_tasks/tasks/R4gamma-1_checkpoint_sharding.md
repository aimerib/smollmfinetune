---
# R4γ-1  Distributed Checkpoint Sharding
Status: **Todo**
Ring: R4γ
Created: 2025-06-27
---

## Goal
Enable cheap, long-running weekend training by saving Narrative-LLM checkpoints as sharded `.safetensors` across an S3-compatible object store, allowing interruptions and resume on spot instances.

## Context
Pretraining 500 M+ models on spot GPUs risks eviction. Sharded checkpoints with manifest files let us resume seamlessly and reduce single-object size limits.

## Acceptance Criteria
- [ ] **Shard Writer**: Utility `utils/checkpointing/shard_save.py` that splits model + optimizer state into N ≤4 GB shards; writes `checkpoint.json` manifest.
- [ ] **Shard Loader**: Complement `shard_load.py` that reassembles contiguous `state_dict` on resume.
- [ ] **S3 Backend**: Support local dir + any `AWS_ENDPOINT_URL` compatible store (MinIO, Wasabi).
- [ ] **Trainer Hook**: Training scripts (`run_sft.py`, `run_dpo.py`, etc.) accept `--shard-size-gb` and use shard saver every `save_steps`.
- [ ] **Resume Logic**: If `--resume-from` points to manifest, loader fetches only missing shards.
- [ ] **Unit Tests** `tests/unit/test_checkpoint_sharding.py` writing dummy tensors -> shards -> reload -> equality.
- [ ] **Documentation** `docs/checkpoint_sharding.md` with env var examples and cost analysis.

## Implementation Notes
```text
• Use PyTorch `torch.save(chunk, f)` with `_use_new_zipfile_serialization=False` to keep CPU RAM low.
• Manifest JSON: {"shards": ["pytorch_model-00001-of-00010.safetensors", ...], "total_size": ...}
• Implement multipart upload retries + progress bar with boto3.
```

## Checklist / Steps
1. Implement shard save/load utils + tests.
2. Integrate into SFT script.
3. Verify resume on local workstation.
4. Push shards to S3; test download + resume.

## References
Feeds R4γ-2 spot orchestrator; prerequisite for multi-weekend training strategy. 