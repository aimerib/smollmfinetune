# A Guide to the Task-Based TDD Project

Welcome to the Character Creation Devkit project! This document outlines the development process we follow. Our methodology is designed for clarity, predictability, and quality, ensuring we build a robust and maintainable AI character creation platform.

## The Concentric Ring Model

The project is divided into "Concentric Rings," moving from the most critical core functionality outwards to advanced features and polish.

### Ring Overview

*   **Ring 0: Green Baseline** - End-to-end functionality: upload character card → generate dataset → train model → test character. Basic but complete pipeline.
*   **Ring 1: Devkit 1.0** - Structured world and character authoring with Big Five personality traits, world integration, and comprehensive UI tools.
*   **Ring 2: Runtime Packet** - Export trained characters as self-contained "cartridges" with prompt factory for external deployment.
*   **Ring 3: Multi-User Platform** - Database backend, async training jobs, user management, and production infrastructure.
*   **Ring 4: Narrative Engine** - Custom model architecture with dual-head outputs, external memory, and advanced training techniques.
*   **Ring 5: Advanced Features** - Director's Chair real-time training, proactive agents, living interfaces, and digital ecology systems.

You should complete tasks sequentially within each ring. This ensures dependencies are met before beginning new work. At the end of every task, before marking it as completed, ensure that the feature is fully integrated with the wider platform. At this point perform a full code search to ensure you know how to fully integrate the feature. After integration, run a full test suite `python -m pytest tests/ -v` and if the tests come green, mark the task as completed, move it to the completed folder, and make a commit with all files added.

## The TDD Workflow (Red-Green-Refactor)

Every task in this project follows Test-Driven Development (TDD). For each feature, you will:

1.  **RED:** Read the "TDD Instructions" for the current task. Write failing tests in the appropriate test files that precisely verify the acceptance criteria. For Streamlit UI components, use `streamlit.testing.v1.AppTest`; for business logic, use `pytest` unit tests. The test should fail because the implementation doesn't exist yet.

2.  **GREEN:** Write the **absolute minimum** amount of code required to make the failing test pass. Don't add extra features or refactor yet. Focus solely on achieving a green test result.

3.  **REFACTOR:** With passing tests as a safety net, clean up your implementation. Improve structure, remove duplication, and enhance readability without changing external behavior. Rerun tests to ensure they still pass.

## Technology Stack & Testing

### Core Technologies
- **Backend**: Python with Streamlit for UI
- **AI/ML**: Transformers, PEFT (LoRA/DoRA), TRL for RLHF
- **Data**: Pandas, HuggingFace Datasets, SQLite/PostgreSQL
- **Testing**: pytest for unit tests, streamlit.testing for UI tests

### Testing Patterns
- **Unit Tests**: `tests/test_*.py` for business logic in `app/utils/`
- **UI Tests**: `tests/ui/test_*.py` for Streamlit pages using `AppTest`
- **Integration Tests**: `tests/integration/test_*.py` for end-to-end workflows
- **Test Structure**: Mirror the app structure (e.g., `tests/test_character_manager.py` for `app/utils/character/character.py`)

### Multi-Layer TDD Approach
We follow a **three-circle TDD methodology**:

- **🔴 Inner Circle (Core Logic)**: Pure business logic, data models, algorithms
  - Write unit tests for `utils/` modules first
  - Focus on core functionality without UI dependencies  
  - Fast feedback loop (~seconds)

- **🟡 Middle Circle (Integration)**: Component integration, workflows, data flow
  - Integration tests for manager classes working together
  - Test complete workflows (e.g., character creation → training → testing)
  - Medium feedback loop (~10-30 seconds)

- **🟢 Outer Circle (UI)**: Streamlit interface, user interactions, visual components
  - Use `streamlit.testing.v1.AppTest` for UI tests
  - Test user workflows: button clicks, form submissions, page navigation
  - Simulate complete user journeys end-to-end
  - Slower feedback loop (~30-60 seconds)

## Task File Structure

Each task is defined in its own markdown file (e.g., `R1-5_personality_radar_chart.md`). The structure includes:

*   **Goal:** High-level description of what this task achieves.
*   **Acceptance Criteria:** Clear, verifiable checklist. The task is "done" only when all criteria are met.
*   **Implementation Notes:** Technical guidance and architectural decisions.
*   **TDD Instructions:** Specific guidance on what tests to write for the RED phase.

## Character Creation Devkit Specifics

### Project Vision
Transform AI character creation from simple chatbots into believable, persistent digital actors using the "Devkit + Cartridge" approach:
- **Devkit**: This creative suite where writers define character psychology and world lore
- **Cartridge**: Self-contained runtime packets ready for deployment

### Key Quality Standards
- **Comprehensive Test Coverage**: All new components require tests
- **TDD Mandatory**: Write tests first for all new features
- **Separation of Concerns**: Clear boundaries between UI, business logic, and data layers
- **Character Psychology Focus**: Everything serves creating psychologically consistent characters

### Example TDD Cycle

For a character personality feature:

1. **RED**: Write test expecting `PersonalityEditor` component to render Big Five sliders
2. **GREEN**: Create minimal component that renders 5 sliders (no logic yet)
3. **REFACTOR**: Add proper trait names, validation, and radar chart visualization

## How to Proceed

1.  Start with the task provided in context
2.  Follow the TDD cycle for each acceptance criterion
3.  Run tests frequently: `pytest` for unit tests, `streamlit run` + manual testing for UI
4.  Integrate the feature with the rest of the codebase
5.  Run full test suite
6.  Once complete and all criteria met, mark the task as done
7.  Move the task file to `ai_coder_tasks/tasks/completed/` with a completion summary

## File Organization

```
app/
├── utils/           # Business logic (unit tested)
├── pages/           # Streamlit pages (UI tested)  
├── components/      # Reusable UI components
└── requirements.txt (and requirements-runpod.txt) - opportunity to reduce duplication by splitting the runpod deps into its own file. Best practices?

tests/
├── test_*.py        # Unit tests for utils/
├── ui/             # UI tests for pages/
└── integration/    # End-to-end workflow tests
```

By following this structured, test-driven process, we build the Character Creation Devkit methodically with high quality and confidence in our code.