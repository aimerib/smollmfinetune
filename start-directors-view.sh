#!/bin/bash

# Start Director's View - FastAPI Backend and React Frontend

echo "🎬 Starting Director's View..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to cleanup on exit
cleanup() {
    echo -e "\n${RED}Shutting down Director's View...${NC}"
    
    # Kill background processes
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    if [ ! -z "$CLIENT_PID" ]; then
        kill $CLIENT_PID 2>/dev/null
    fi
    
    # Kill any remaining processes on the ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:3001 | xargs kill -9 2>/dev/null
    
    exit 0
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Start API Backend
echo -e "${BLUE}Starting FastAPI Backend...${NC}"
cd api
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Installing API dependencies..."
pip install -r requirements.txt >/dev/null 2>&1

# Start FastAPI in background
uvicorn app.main:app --reload --port 8000 &
API_PID=$!
echo -e "${GREEN}✓ API started on http://localhost:8000${NC}"

# Wait for API to be ready
echo "Waiting for API to be ready..."
while ! curl -s http://localhost:8000/api/metrics/health >/dev/null; do
    sleep 1
done
echo -e "${GREEN}✓ API is ready${NC}"

# Start simulation activity
echo -e "${BLUE}Starting world simulation...${NC}"
python -c "
import asyncio
import sys
sys.path.append('.')
from app.services.state_service import state_service

async def start_simulation():
    await state_service.initialize()
    # Start world activity simulation
    asyncio.create_task(state_service.simulate_world_activity())
    # Keep running
    await asyncio.Event().wait()

asyncio.run(start_simulation())
" &
SIMULATION_PID=$!

cd ..

# Start React Frontend
echo -e "${BLUE}Starting React Frontend...${NC}"
cd client

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing React dependencies..."
    npm install
fi

# Start React app on port 3001
PORT=3001 npm start &
CLIENT_PID=$!

echo -e "${GREEN}✓ React app starting on http://localhost:3001${NC}"

# Display status
echo -e "\n${GREEN}==================================${NC}"
echo -e "${GREEN}Director's View is starting up!${NC}"
echo -e "${GREEN}==================================${NC}"
echo -e "API Backend: ${BLUE}http://localhost:8000${NC}"
echo -e "API Docs: ${BLUE}http://localhost:8000/docs${NC}"
echo -e "React Frontend: ${BLUE}http://localhost:3001${NC}"
echo -e "\nPress ${RED}Ctrl+C${NC} to stop all services"
echo -e "${GREEN}==================================${NC}\n"

# Keep script running
wait 