---
sidebar_position: 1
---

# Character Creation Devkit

A comprehensive platform for creating, training, and deploying AI characters with persistent personalities and memory.

## Core Concepts

The platform is built on these key principles:

- **Devkit + Cartridge Architecture**: A creative suite for designing characters and worlds that exports self-contained "cartridges" for runtime use
- **Structured Authoring**: Define character psychology using the Big Five personality model, goals, relationships, and memories
- **Emergent Narrative**: Characters with consistent internal motivations enable unique story interactions
- **Triple-Head Architecture**: Model generates text, control tokens for UI/emotional state, and memory vectors for persistence

## Getting Started

Follow these guides in order:

1. **[Setup Guide](./setup-guide.md)**: Install and configure the platform
2. **[Getting Started](./getting-started.md)**: Create and train your first character
3. **[Training Guide](./training-guide.md)**: Deep dive into SFT and RLHF training pipelines
4. **[Client Guide](./client-guide.md)**: Use the production client for character interactions

## Documentation Structure

- **Getting Started**: Setup and tutorial guides
- **Core Documentation**: Architecture, concepts, and features
- **Advanced**: Specialized topics for developers

## Platform Components

### Architecture Overview

The platform consists of three main components:

**The Devkit** - Creative tools for character development:
- Conversational character builder
- Interactive personality editor
- AI-powered dataset generation
- Real-time training dashboard

**The Runtime** - Production inference and client:
- High-performance inference engine
- React-based chat client
- Real-time emotion visualization
- Memory formation tracking

### Key Features

**Character Creation**
- AI-guided character discovery through natural conversation
- Interactive Big Five personality trait visualization
- World integration for character consistency
- Real-time character synthesis and analysis

**Training Pipeline**
- Supervised Fine-Tuning (SFT) for initial character voice training
- Reinforcement Learning (RLHF) for preference-based alignment
- Live training dashboards with pause/resume controls
- Personality alignment and lore adherence metrics

**Production Inference**
- Optimized for real-time character interactions
- Hot-swappable adapters for character switching
- Session persistence across conversations
- Scalable architecture for multiple concurrent users

## Quick Setup

```bash
# Launch everything
./launch-client.sh

# Or run components separately:
cd app && ./startup.sh                    # Devkit
python scripts/run_inference_server.py   # Inference Server  
cd client && npm start                    # React Client
```

## Complete Workflow

```mermaid
graph LR
    A[Setup Platform] --> B[Create World]
    B --> C[Design Character]
    C --> D[Generate Dataset]
    D --> E[Train Model]
    E --> F[Test Character]
    F --> G[Deploy Client]
    G --> H[User Interaction]
```

## Platform Architecture

1. **Character Devkit** (Streamlit) - Creative tools for character development
2. **Inference Engine** (FastAPI) - Optimized model serving
3. **React Client** - User-facing chat interface

## Current Status

- ✅ Ring 1: Devkit Complete
- ✅ Ring 2: Runtime Packets  
- ✅ Ring 3: Multi-User Platform
- ✅ Ring 4: Production Inference
- 🔄 Ring 5: Experience Polish

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **API Documentation**: Technical integration details

[Get Started](./setup-guide) | [Core Concepts](./core-concepts) 