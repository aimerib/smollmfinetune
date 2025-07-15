# Narrative Engine Architecture

This document describes the high-level architecture of the custom NarrativeLLM, its data flow, and its integration into the broader AI Coder platform. It serves as the canonical reference for developers.

## 1. Vision

The Narrative Engine is a bespoke Large Language Model designed specifically for driving interactive, story-rich conversational experiences. Unlike general-purpose chatbots, it is architected from the ground up to handle:

- **Dual Outputs**: Seamlessly generating both natural language prose and structured, deterministic game engine actions (tool use).
- **Dynamic Personas**: Hot-swapping character personalities and narrative genres via lightweight adapters.
- **Stateful Memory**: Incorporating long-term memories into its reasoning process via cross-attention.
- **Session-Awareness**: Maintaining conversational context across long interactions without prompt bloat.

## 2. System Context & Data Flow

### Training Data Flow:
```
AI Coder Devkit (Worlds, Characters, UI)
│
└───> Raw Assets (character_core.json, world_lore.json, etc.)
     │
     └───> scripts/convert_to_narrative_format.py
          │
          └───> Unified Dataset (JSONL files)
               │
               └───> narrative_engine/data_pipeline.py (Tokenization & Masking)
                    │
                    └───> Batched Tensors for Training
```

### Inference Data Flow:
```
User Input ↔ Orchestrator API
               │
               ├──> 1. Retrieve Memories (MemoryAPIClient)
               │
               ├──> 2. Assemble Prompt & Context
               │
               └──> NarrativeLLM.generate(input_ids, session_id, memory_states)
                    │
                    ├──> Loads active Persona Adapters (from Cartridges)
                    │
                    └──> Returns Stream of Tokens
                         │
                         └───> Orchestrator decodes...
                               ├──> Prose Text → User
                               └───> Action JSON → Game Engine
```

## 3. Core Components

- **NarrativeLLM (model.py)**: The core torch.nn.Module. It is composed of embedding layers, a transformer backbone, a cross-attention memory module, and dual output heads.
- **NarrativeLLMConfig (config.py)**: A single dataclass holding all hyperparameters, ensuring reproducibility.
- **DatasetSample (data_schema.py)**: Pydantic models defining the strict schema for all training data.
- **TripleHeadLoss (loss.py)**: A custom loss function that routes gradients to the correct output head (Generation, Control, or Memory) based on data tagging.
- **DualHeadLoss (loss.py)**: Legacy dual-head loss function maintained for backward compatibility.
- **MemoryAPIClient (memory_client.py)**: A client to interact with an external vector database for long-term memory.

## 4. Training Strategy

The model is trained in three successive phases:

1. **Supervised Fine-Tuning (SFT)**: The model learns the basic conversational and action-generation format from a curated dataset.
2. **Reward Modeling (RM)**: A separate model is trained on human preference data to learn what makes a "good" response.
3. **Direct Preference Optimization (DPO)**: The SFT model is further tuned using the preference data to maximize the implicit reward, aligning it with human expectations. 