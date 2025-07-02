#!/bin/bash

# Character Creation Devkit - Development Startup Script

echo "🚀 Starting Character Creation Devkit..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if Redis is running (for local development without Docker)
if [[ "$1" == "local" ]]; then
    echo -e "${YELLOW}Running in local mode...${NC}"
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        echo -e "${RED}❌ Redis is not installed. Please install Redis first.${NC}"
        exit 1
    fi
    
    # Start Redis if not running
    if ! pgrep -x "redis-server" > /dev/null; then
        echo -e "${YELLOW}Starting Redis...${NC}"
        redis-server --daemonize yes
    fi
    
    # Create Python virtual environment for backend
    if [ ! -d "backend/venv" ]; then
        echo -e "${YELLOW}Creating Python virtual environment...${NC}"
        cd backend
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        cd ..
    fi
    
    # Install frontend dependencies
    if [ ! -d "client/node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        cd client
        npm install
        cd ..
    fi
    
    echo -e "${GREEN}✅ Starting services locally...${NC}"
    
    # Start backend in background
    echo -e "${YELLOW}Starting FastAPI backend...${NC}"
    cd backend
    source venv/bin/activate
    uvicorn app.main:app --reload --port 8000 &
    BACKEND_PID=$!
    cd ..
    
    # Start Celery workers
    echo -e "${YELLOW}Starting Celery workers...${NC}"
    cd backend
    source venv/bin/activate
    celery -A app.celery_app worker -Q dataset,training,export -l info &
    CELERY_PID=$!
    cd ..
    
    # Start Flower
    echo -e "${YELLOW}Starting Flower (Celery monitoring)...${NC}"
    cd backend
    source venv/bin/activate
    celery -A app.celery_app flower &
    FLOWER_PID=$!
    cd ..
    
    # Start frontend
    echo -e "${YELLOW}Starting React frontend...${NC}"
    cd client
    npm start &
    FRONTEND_PID=$!
    cd ..
    
    echo -e "${GREEN}✅ All services started!${NC}"
    echo -e "${GREEN}Frontend: http://localhost:3000${NC}"
    echo -e "${GREEN}Backend API: http://localhost:8000${NC}"
    echo -e "${GREEN}API Docs: http://localhost:8000/api/docs${NC}"
    echo -e "${GREEN}Flower (Celery): http://localhost:5555${NC}"
    
    # Function to cleanup on exit
    cleanup() {
        echo -e "${YELLOW}Shutting down services...${NC}"
        kill $BACKEND_PID $CELERY_PID $FLOWER_PID $FRONTEND_PID 2>/dev/null
        exit 0
    }
    
    # Set trap to cleanup on Ctrl+C
    trap cleanup INT
    
    # Wait for Ctrl+C
    echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
    wait
    
else
    # Docker mode (default)
    echo -e "${YELLOW}Running with Docker Compose...${NC}"
    
    # Build images if needed
    if [[ "$1" == "build" ]] || [[ "$2" == "build" ]]; then
        echo -e "${YELLOW}Building Docker images...${NC}"
        docker-compose -f docker-compose.dev.yml build
    fi
    
    # Start services
    echo -e "${GREEN}Starting Docker services...${NC}"
    docker-compose -f docker-compose.dev.yml up -d
    
    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 5
    
    # Check service health
    if docker-compose -f docker-compose.dev.yml ps | grep -q "Up"; then
        echo -e "${GREEN}✅ All services started successfully!${NC}"
        echo ""
        echo -e "${GREEN}🎉 Character Creation Devkit is ready!${NC}"
        echo ""
        echo -e "${GREEN}Access points:${NC}"
        echo -e "${GREEN}  Frontend: http://localhost:3000${NC}"
        echo -e "${GREEN}  Backend API: http://localhost:8000${NC}"
        echo -e "${GREEN}  API Docs: http://localhost:8000/api/docs${NC}"
        echo -e "${GREEN}  Flower (Celery): http://localhost:5555${NC}"
        echo ""
        echo -e "${YELLOW}To view logs: docker-compose -f docker-compose.dev.yml logs -f${NC}"
        echo -e "${YELLOW}To stop: docker-compose -f docker-compose.dev.yml down${NC}"
    else
        echo -e "${RED}❌ Some services failed to start. Check logs with:${NC}"
        echo -e "${YELLOW}docker-compose -f docker-compose.dev.yml logs${NC}"
        exit 1
    fi
fi 