#!/bin/bash

echo "Stopping StockAI..."

# Stop Node.js backend
pkill -f "node server.js" && echo "✓ Backend stopped"

# Stop React frontend
pkill -f "npm start" && echo "✓ Frontend stopped"

# Clean up ports
lsof -ti:5001 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "✓ All services stopped"
