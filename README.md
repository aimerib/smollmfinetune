# 🎭 Character Creation Devkit

[![Evaluation CI](https://github.com/aimerib/smollmfinetune/actions/workflows/eval_ci.yml/badge.svg)](https://github.com/aimerib/smollmfinetune/actions/workflows/eval_ci.yml)

Transform AI character creation from simple chatbots into believable, persistent digital actors using our "Devkit + Cartridge" approach.

## 🚀 Quick Start

The **Character Creation Devkit** is a comprehensive platform for creating, training, and deploying AI characters with psychological depth and narrative consistency.

### Core Concept: Devkit + Cartridge

- **🛠️ Devkit** (This Application): Creative suite where writers define character psychology, world lore, and train character voices
- **🎮 Cartridge** (Runtime Packet): Self-contained deployment packages ready for game engines and interactive experiences

## 📊 Quality Assurance

Our CI pipeline automatically validates every change with:

- ✅ **JSON Correctness**: Ensures structured output capabilities (≥70%)
- 🧠 **Personality Alignment**: Validates Big Five trait consistency (≥0.5)
- 🔄 **Regression Testing**: Catches quality degradation before merge
- 📈 **Performance Monitoring**: Tracks training and evaluation metrics

## 🏗️ Project Structure

This repository contains multiple interconnected components:

```
├── app/                    # Main Streamlit application (Character Creation Devkit)
├── scripts/               # Training and evaluation scripts
├── services/              # Microservices (judge service, etc.)
├── narrative_engine/      # Custom model architecture research
├── ai_coder_tasks/        # Development roadmap and task tracking
└── tests/                 # Comprehensive test suite
```

## 🧪 Development Approach

We follow **Test-Driven Development (TDD)** with a **Concentric Ring** delivery model:

- **Ring 0**: ✅ End-to-end pipeline (upload → generate → train → test)
- **Ring 1**: ✅ Structured authoring with Big Five personality traits  
- **Ring 2**: ✅ Runtime packet export system
- **Ring 3**: ✅ Multi-user platform with async training
- **Ring 4**: 🔄 Narrative engine with triple-head architecture (Generation + Control + Memory)
- **Ring 5**: 📋 Advanced features (real-time training, proactive agents)

## 🔧 Getting Started

### For Character Creators
```bash
cd app
pip install -r requirements.txt
./run-local.sh
```

### For Developers
```bash
# Run the full test suite
python -m pytest tests/ -v

# Run CI evaluation locally
python scripts/run_sft_ci.py --quick-mode
python scripts/run_basic_evaluation.py --checkpoint-path output --quick-mode
```

## 📚 Documentation

- **[App README](app/README.md)**: Detailed usage guide for the Character Creation Devkit
- **[Development Guide](ai_coder_tasks/how-to-use-tasks.md)**: TDD workflow and task system
- **[Project Overview](ai_coder_tasks/overview.md)**: Vision, architecture, and roadmap

## 🤝 Contributing

We welcome contributions! Our development process:

1. **Follow TDD**: Write tests first, then implement features
2. **Check CI**: All PRs must pass the evaluation harness
3. **Ring-based Development**: Complete features within their ring before moving to the next
4. **Comprehensive Testing**: Unit, integration, and UI tests required

## 📈 Recent Achievements

- ✅ **Comprehensive Telemetry SDK**: Track all experiments with reproducible results
- ✅ **Multi-User Platform**: Full authentication, database backend, async training
- ✅ **Evaluation Harness**: Automated quality gates with personality alignment metrics
- ✅ **Production Ready**: Docker deployment, health monitoring, error tracking

## 🎯 Vision

We're building toward an **anecdote factory** where players don't just follow plots—they create unique, personal stories through interactions with characters that have consistent internal lives and motivations.

---

**Ready to create digital personas that feel genuinely alive? Let's build the future of interactive storytelling! 🎭✨** 