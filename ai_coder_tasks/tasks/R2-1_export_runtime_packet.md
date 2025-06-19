---
# R2-1  Export Character Runtime Packet
Status: **Todo**
Ring: R2
Created: 2025-06-18
---

## Goal
Implement a function in `TrainingManager` that bundles all of a character's assets into a self-contained "Runtime Packet" or "Cartridge," ready for use in a game engine or other runtime environment.

## Acceptance Criteria

### 1. Function Implementation
- [ ] Add the function `export_runtime_packet(character_name: str)` to the `TrainingManager` class in `app/utils/training.py`.
- [ ] This function will create a directory at `runtime_packets/<character_name>`.

### 2. Packet Contents
- [ ] The exported packet **must** contain the following files:
      - `adapter.safetensors`: The final trained LoRA/DoRA adapter (this should be the RLHF-tuned version if it exists, otherwise the SFT version).
      - `character_core.json`: The character's core data, including Big-Five personality traits.
      - `world_lore.json`: The lore file for the character's world.
      - `tokens.json`: The control token definitions for the world.
      - `runtime_config.json`: A configuration file for the runtime.

### 3. Runtime Config Schema
- [ ] The `runtime_config.json` should contain, at a minimum:
      ```json
      {
        "base_model": "name-of-base-model-e.g-meta-llama/Llama-2-7b-chat-hf",
        "adapter_path": "adapter.safetensors",
        "tokenizer_path": ".cache/tokenizers/name-of-base-model-patched",
        "character_file": "character_core.json",
        "world_file": "world_lore.json",
        "tokens_file": "tokens.json"
      }
      ```

### 4. UI Integration
- [ ] On the **Model Management** page (`page_model_management` in `app.py`), add an "Export" button next to each trained model.
- [ ] Clicking the button calls `export_runtime_packet` and provides a download link or path to the user.

## References
- This is the primary deliverable for Ring 2 and fulfills the "Cartridge" part of the project vision.
- It consumes the outputs of `R1-2` (character core), `R1-1` (world lore), `R1-10` (tokens), and `R1-11` (adapter). 