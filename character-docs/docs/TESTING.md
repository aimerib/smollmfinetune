# Testing Guide 🧪

## Overview

This project uses a sophisticated testing infrastructure designed to separate fast development cycles from comprehensive model validation. Our testing strategy ensures rapid feedback during development while maintaining rigorous quality standards for model releases.

## Test Categories & Markers

### Fast Tests ⚡ (REQUIRED)
- **Purpose**: Development and CI feedback
- **Execution Time**: Seconds to minutes
- **Coverage**: Business logic, UI components, data handling
- **Markers**: No special markers (default)
- **Required**: ✅ Before task completion, ✅ Before commits, ✅ CI/CD

```python
def test_character_creation():
    """Fast test - no LLM calls, mocked dependencies"""
    pass
```

### Slow Tests 🐌 (Model Validation)
- **Purpose**: Model inference, evaluation, integration testing
- **Execution Time**: Minutes to hours
- **Coverage**: LLM calls, model training, performance evaluation
- **Markers**: `@pytest.mark.slow`, `@pytest.mark.llm`, `@pytest.mark.evaluation`
- **Required**: Only for model version validation

```python
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.evaluation
def test_model_personality_consistency():
    """Slow test - actual LLM calls for model validation"""
    pass
```

## Test Script Usage

### 🚀 Development Workflow

```bash
# Start any task - ensure clean baseline
./scripts/run_tests.sh fast

# During development - fail-fast mode
./scripts/run_tests.sh dev

# Before any commit - REQUIRED
./scripts/run_tests.sh pre-commit

# Complete any task - REQUIRED
./scripts/run_tests.sh fast
```

### 📊 Model Validation

```bash
# Test model performance
./scripts/run_tests.sh evaluation-slow

# Full slow test suite
./scripts/run_tests.sh slow

# Fast evaluation tests only
./scripts/run_tests.sh evaluation-fast
```

### 🤖 CI/CD

```bash
# Platform CI (fast tests + coverage)
./scripts/run_tests.sh ci

# Complete test suite (includes slow tests)
./scripts/run_tests.sh all
```

## Testing Rules & Requirements

### ✅ Mandatory Rules

1. **NO TASK COMPLETION** without passing fast tests
2. **NO COMMITS** with failing fast tests  
3. **ALL NEW TESTS** must have appropriate markers
4. **LLM/MODEL TESTS** must use `@pytest.mark.slow @pytest.mark.llm`
5. **EVALUATION TESTS** must use `@pytest.mark.evaluation`

### 🎯 When to Use Each Test Type

#### Fast Tests
- Pure business logic
- Data validation and transformation
- Component interfaces and APIs
- UI interactions (with mocked dependencies)
- Configuration and setup logic

#### Slow Tests
- Model inference calls (OpenAI, local models)
- LLM judge evaluations
- Training pipeline validation
- End-to-end character testing
- Performance benchmarks

## CI/CD Integration

### Platform CI (Automatic)
- **Trigger**: Push to main/develop, PRs to main
- **Tests**: Fast tests only
- **Purpose**: Gate merging, ensure platform stability
- **Workflow**: `.github/workflows/ci.yml`

### Model Validation (Manual/Scheduled)
- **Trigger**: Manual dispatch, weekly schedule
- **Tests**: Slow/evaluation tests
- **Purpose**: Model performance validation
- **Workflow**: `.github/workflows/model-validation.yml`

## Task Completion Checklist

Before marking any task as complete:

- [ ] Feature implemented and integrated
- [ ] Fast tests written for new components
- [ ] Slow/LLM tests properly marked if applicable
- [ ] `./scripts/run_tests.sh fast` passes completely
- [ ] No failing tests remaining
- [ ] Changes committed with passing pre-commit tests

## Troubleshooting

### Test Failures
- **Fast test failures**: MUST be fixed before proceeding
- **Unknown test purpose**: Ask for clarification before modifying
- **Can't fix reasonably**: Request permission to delete test
- **Slow test failures**: Investigate but don't block development

### Performance
- Tests run in parallel using `pytest-xdist` (`-n auto`)
- Fast tests typically complete in < 2 minutes
- Slow tests may take hours depending on model calls

### Environment Requirements
- Python 3.9+ (CI tests on 3.9, 3.10, 3.11)
- Required packages: `pytest`, `pytest-xdist`, `pytest-cov`
- Optional: `OPENAI_API_KEY` for LLM tests

## Examples

### Writing a Fast Test
```python
def test_character_manager_create():
    """Test character creation logic without LLM calls"""
    manager = CharacterManager()
    character = manager.create_character(
        name="Test Character",
        personality={"openness": 0.7}
    )
    assert character.name == "Test Character"
    assert character.personality["openness"] == 0.7
```

### Writing a Slow Test
```python
@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.evaluation
def test_character_voice_consistency():
    """Test that character maintains consistent voice across conversations"""
    character = load_trained_character("alice")
    responses = []
    
    for prompt in test_prompts:
        response = character.generate(prompt)  # Actual LLM call
        responses.append(response)
    
    consistency_score = evaluate_voice_consistency(responses)
    assert consistency_score > 0.8
```

## Best Practices

1. **Start with fast tests** - Write business logic tests first
2. **Mock external dependencies** - Keep fast tests truly fast
3. **Use descriptive test names** - Explain what is being tested
4. **Group related tests** - Use test classes for organization
5. **Test edge cases** - Don't just test the happy path
6. **Keep tests isolated** - No dependencies between tests
7. **Mark appropriately** - Use correct pytest markers

---

*This testing infrastructure ensures we can develop rapidly while maintaining high quality standards for our character creation platform.* 