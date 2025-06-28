---
# R1-12  Initial Documentation
Status: **Todo**
Ring: R1
Created: 2025-06-27
---

## Goal
Enable users to navigate the platform. Content creators will be the first users and primary consumers of documentation. We should try to mimic React's documentation more than a vibe-coded project, so no emojis, raw information. Remember, we are offering a novel experience. We should document everything the platform is able to do up to this point, along with paint the broader future vision.

## Acceptance Criteria

### 1. Clear and comprehensive documentation
- [ ] Covers what each part of the platform does
- [ ] Easy to follow - friendly tone and high information density
- [ ] Easy to read in the repo, but offers actual documentation site
- [ ] Focus on the end-user experience first. Deployment docs will come towards the end of the projects.

## Implementation Notes
- *UI/UX* - This documentation should dazzle
- Only semi-related, but we don't have a better step. Does it make sense to keep the `notebooks/` folder anymore? If not, let's remove them. If so, we need to audit the existing ones (now severely out of date), and assess what types of new notebooks should we keep and why.


## References
- Every card in `completed/` as it will give you a good idea about the platform at a high level
- Any code you deem necessary

## Implementation Snippet
```python
# utils/rlhf_trainer.py
from trl import GRPOTrainer, PPOTrainer, GRPOConfig, PPOConfig

def run_rlhf(model_path: str, pref_ds, algo="grpo", **cfg):
    if algo == "grpo":
        trainer_cls, cfg_cls = GRPOTrainer, GRPOConfig
    else:
        trainer_cls, cfg_cls = PPOTrainer, PPOConfig
    training_args = cfg_cls(output_dir=cfg.get("output_dir", "rlhf_out"), **cfg)
    trainer = trainer_cls(model=model_path, train_dataset=pref_ds, args=training_args,
                          reward_funcs=lambda outs, **k: outs["reward"])
    trainer.train()
    return trainer.save_model()
```

## Docs & References
* TRL GRPO docs – <https://huggingface.co/docs/trl/main/en/grpo_trainer>
* PPO docs – <https://huggingface.co/docs/trl/main/en/ppo_trainer>
* Example code inspiration – <https://github.com/e-p-armstrong/augmentoolkit/.../do_grpo_rl_with_a_prompt>

## Pipeline Detail
```text
1. SFT (DoRA) → adapter_sft.safetensors
2. Aggregate preference NDJSON → HF Dataset => pref_ds
3. run_rlhf(algo=selected_algo)
4. Save adapter_grpo.safetensors (or adapter_ppo)
5. Update runtime packet export
```

## Implementation Notes
For GRPO
* Memory-saving mode set `beta=0` (no reference model) unless creator ticks "Strict KL".  
* Generate k=4 completions per prompt inside trainer to exploit group-relative scoring.

For PPO
* Use TRL `PPOTrainer`; regulator hyperparams: `kl_penalty=0.1`, `batch_size=16`.  
* Reward fn: simple +1 if model output == creator-selected, else -1; can upgrade later.

## Recommended Default Hyper-parameters (GRPO)
| Param | Value | Rationale |
|-------|-------|-----------|
| learning_rate | `5e-6` | conservative; RLHF tends to diverge with >1e-5 |
| adam_beta1 / 2 | `0.9` / `0.99` | same as TRL examples |
| weight_decay | `0.1` | small regularisation |
| warmup_ratio | `0.1` | 10 % warm-up |
| per_device_train_batch_size | `1` | memory-friendly on RunPod 24 GB |
| gradient_accumulation_steps | `1` | creators may raise in UI |
| num_generations | `6` | gives 6 completions → good group diversity |
| max_steps | `500` | first experiments; expose in advanced drawer |
| beta (KL) | `0.0` | rely on group-relative objective; slider in advanced |

### Code Reference
Snippets adapted from HF docs / Unsloth fast loader:
```python
from trl import GRPOConfig
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    num_generations=6,
    max_prompt_length=4000,
    max_completion_length=2500,
    max_steps=500,
    beta=0.0,
    report_to="wandb",
)
```
`setup_model` in UnsLoTH style can be used when available; otherwise fall back to regular `AutoModelForCausalLM` + PEFT.

## UI Integration
• Training Config page: toggle "Enable PPO if pairs ≥ threshold".  
• Training Dashboard: new progress section after SFT completes.
