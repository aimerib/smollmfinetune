#!/bin/bash

# Test runner script for different development scenarios
# Usage: ./scripts/run_tests.sh [fast|slow|all|integration|evaluation|ci]

set -e

echo "🧪 SmolLM Fine-tune Test Runner"
echo "================================"

case ${1:-help} in
  fast)
    echo "🚀 Running fast tests only (excludes slow/LLM tests)..."
    python -m pytest -v -n auto -m "not slow and not llm" tests/
    ;;
  
  slow)
    echo "⏳ Running slow tests only (LLM and integration tests)..."
    python -m pytest -v -n auto -m "slow or llm" tests/
    ;;
  
  evaluation)
    echo "📊 Running evaluation harness tests only..."
    python -m pytest -v -n auto -m "evaluation" tests/
    ;;
  
  evaluation-fast)
    echo "📊⚡ Running fast evaluation tests only..."
    python -m pytest -v -n auto -m "evaluation and not slow and not llm" tests/
    ;;
  
  evaluation-slow)
    echo "📊⏳ Running slow evaluation tests only..."
    python -m pytest -v -n auto -m "evaluation and (slow or llm)" tests/
    ;;
  
  integration)
    echo "🔗 Running integration tests only..."
    python -m pytest -v -n auto -m "integration" tests/
    ;;
  
  ui)
    echo "🖥️ Running UI tests only..."
    python -m pytest -v -n auto -m "ui" tests/
    ;;
  
  all)
    echo "🌍 Running ALL tests (including slow/LLM tests)..."
    python -m pytest -v -n auto tests/
    ;;
  
  ci)
    echo "🤖 Running CI test suite (all tests with coverage)..."
    python -m pytest -v -n auto --cov=app --cov=narrative_engine --cov-report=html --cov-report=term tests/
    ;;
  
  dev)
    echo "👨‍💻 Running development test suite (fast tests only, fail fast)..."
    python -m pytest -v -x -n auto -m "not slow and not llm and not integration" tests/
    ;;
  
  pre-commit)
    echo "✅ Running pre-commit test suite (fast + some slow tests)..."
    echo "   - Fast tests first..."
    python -m pytest -v -n auto -m "not slow and not llm" tests/
    echo "   - Critical slow tests..."
    python -m pytest -v -n auto -m "slow and evaluation" -k "test_voice_consistency_single_character or test_emotional_arc_tracking" tests/
    ;;
  
  help|*)
    echo "Available test modes:"
    echo ""
    echo "Development modes:"
    echo "  fast           - Fast tests only (unit tests, no LLM calls)"
    echo "  dev            - Development mode (fast tests, fail on first error)"
    echo "  pre-commit     - Pre-commit checks (fast + sample slow tests)"
    echo ""
    echo "Evaluation modes:"
    echo "  evaluation     - All evaluation tests"
    echo "  evaluation-fast - Fast evaluation tests only"
    echo "  evaluation-slow - Slow evaluation tests only"
    echo ""
    echo "Integration modes:"
    echo "  slow           - Slow/LLM tests only"
    echo "  integration    - Integration tests only"
    echo "  ui             - UI tests only"
    echo ""
    echo "Complete modes:"
    echo "  all            - All tests"
    echo "  ci             - CI mode (all tests with coverage)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_tests.sh fast              # Development testing"
    echo "  ./scripts/run_tests.sh pre-commit        # Before git commit"
    echo "  ./scripts/run_tests.sh evaluation-slow   # Test LLM evaluators"
    echo "  ./scripts/run_tests.sh ci                # Full CI suite"
    ;;
esac

echo ""
echo "✨ Test run complete!" 