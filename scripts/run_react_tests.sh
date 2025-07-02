#!/bin/bash

# Run React tests for the Character Creation Platform
# Usage: ./scripts/run_react_tests.sh [mode]
# Modes: 
#   - watch (default): Run tests in watch mode
#   - ci: Run tests once with coverage
#   - coverage: Run tests with coverage report

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the mode from command line argument
MODE=${1:-watch}

echo -e "${BLUE}🧪 Running React tests in ${MODE} mode...${NC}"

# Change to client directory
cd client

case $MODE in
    "watch")
        echo -e "${GREEN}Starting tests in watch mode...${NC}"
        npm test
        ;;
    
    "ci")
        echo -e "${GREEN}Running tests in CI mode...${NC}"
        CI=true npm test -- --coverage --watchAll=false
        ;;
    
    "coverage")
        echo -e "${GREEN}Running tests with coverage report...${NC}"
        npm test -- --coverage --watchAll=false
        ;;
    
    "update")
        echo -e "${YELLOW}Updating test snapshots...${NC}"
        npm test -- --updateSnapshot
        ;;
    
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        echo "Available modes: watch (default), ci, coverage, update"
        exit 1
        ;;
esac

# Return to original directory
cd ..

echo -e "${GREEN}✨ React test run complete!${NC}" 