---
# R1-11  Preference Fine-Tuning Pipeline (GRPO/PPO)
Status: **Todo**
Ring: R1
Created: 2025-06-18
---

## Goal
Enable an optional **GRPO (Group-Relative Policy Optimisation)** or **PPO** fine-tuning phase after the main supervised (SFT) run. This allows the model to learn from creator preferences, aligning it more closely with the desired character voice and behavior.

## Acceptance Criteria

### 1. Preference Data Collection
- [ ] The preference data is collected via the **multi-turn dataset generation UI** (`R1-6`), where creators select their preferred response from multiple `assistant_options`, as well as from `world_lore` preferences.
- [ ] Each selection is logged to a `preference_logs.ndjson` file within the character's data directory.
- [ ] Log entry schema: `{ "prompt": str, "chosen": str, "rejected": List[str] }`.

### 2. RLHF Trainer Implementation
- [ ] Create `utils/rlhf_trainer.py` containing a `run_rlhf` function that can instantiate either a `GRPOTrainer` or `PPOTrainer` from the TRL library.
- [ ] Create a script `scripts/aggregate_preferences.py` that reads the `preference_logs.ndjson` files and creates a Hugging Face `Dataset` suitable for the TRL trainers.

### 3. UI Integration (`page_training_config` in `app.py`)
- [ ] On the training page, add a section: "**🧠 Reinforcement Learning Fine-Tuning**".
- [ ] This section becomes active only when a character has a sufficient number of preference pairs (e.g., >100, configurable via `.env`).
- [ ] UI elements:
      - `st.checkbox("Enable RL Fine-Tuning")`
      - `st.selectbox("Algorithm", ["GRPO", "PPO"])` (defaults to GRPO).
- [ ] The training dashboard (`page_training_dashboard`) will be updated to show a new stage for RLHF, displaying the reward curve and KL divergence from WandB.

### 4. Pipeline Integration
- [ ] The main training pipeline in `TrainingManager` will be updated to optionally run the `rlhf_trainer` after the SFT phase completes.
- [ ] The process will save two adapter versions: `adapter_sft.safetensors` and the final `adapter_rlhf.safetensors`.

## Implementation Notes
- **Default Algorithm**: GRPO is preferred as it's often more sample-efficient and stable for this type of preference data.
- **Hyperparameters**: The default hyperparameters provided in the original card are an excellent starting point. They should be configurable in an "Advanced Settings" drawer in the UI.
- **Memory**: For GRPO, `beta=0` is a smart default to avoid needing a reference model, saving VRAM.

## Example `run_rlhf` Snippet
```python
# utils/rlhf_trainer.py
from trl import GRPOTrainer, PPOTrainer, GRPOConfig, PPOConfig

def run_rlhf(model_path: str, pref_dataset, algo="grpo", **kwargs):
    TrainerClass = GRPOTrainer if algo == "grpo" else PPOTrainer
    ConfigClass = GRPOConfig if algo == "grpo" else PPOConfig
    
    training_args = ConfigClass(output_dir="rlhf_output", **kwargs)
    
    trainer = TrainerClass(
        model=model_path,
        args=training_args,
        train_dataset=pref_dataset
    )
    trainer.train()
    trainer.save_model()
```

## References
- This card is the direct consumer of the preference data generated as part of `R1-6`.
- It builds upon the SFT training pipeline established in R0.
- TRL Docs: GRPOTrainer, PPOTrainer.

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
