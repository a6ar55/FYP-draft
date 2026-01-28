#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================"
echo "StockAI - Starting Application"
echo "========================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    pkill -f "node server.js" || true
    pkill -f "npm start" || true
    exit
}

trap cleanup SIGINT SIGTERM

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Python virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install Python dependencies if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -q -r requirements.txt
fi

# Install backend dependencies
if [ ! -d "backend-node/node_modules" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    cd backend-node
    npm install --silent
    cd ..
fi

# Install frontend dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd frontend
    npm install --silent
    cd ..
fi

# Start backend
echo -e "${GREEN}Starting backend on http://localhost:5001...${NC}"
cd backend-node
node server.js > /tmp/stockai-backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend
sleep 3
if ps -p $BACKEND_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}✗ Backend failed to start${NC}"
    exit 1
fi

# Start frontend
echo -e "${GREEN}Starting frontend on http://localhost:3000...${NC}"
cd frontend
BROWSER=none npm start > /tmp/stockai-frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

sleep 5
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ StockAI is running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Frontend:${NC} http://localhost:3000"
echo -e "${GREEN}Backend:${NC}  http://localhost:5001"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep running
while true; do
    if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo -e "${RED}Backend died unexpectedly${NC}"
        cleanup
    fi
    if ! ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${RED}Frontend died unexpectedly${NC}"
        cleanup
    fi
    sleep 5
done
