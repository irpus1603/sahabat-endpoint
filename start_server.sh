#!/bin/bash

# Sahabat-9B API Server Startup Script
# This script starts the FastAPI server with proper configuration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Sahabat-9B API Server...${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Using default configuration.${NC}"
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}You can copy .env.example to .env and customize it.${NC}"
    fi
fi

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo -e "${GREEN}Loading environment variables from .env...${NC}"
    export $(grep -v '^#' .env | xargs)
fi

# Set default values if not set
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-9000}
export WORKERS=${WORKERS:-1}
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export DEVICE=${DEVICE:-"cpu"}

echo -e "${GREEN}Configuration:${NC}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"
echo -e "  Workers: ${WORKERS}"
echo -e "  Log Level: ${LOG_LEVEL}"
echo -e "  Device: ${DEVICE}"
echo ""

# Check if required packages are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}Error: FastAPI not installed. Please run: pip install -r requirements.txt${NC}"
    exit 1
fi

if ! python -c "import uvicorn" 2>/dev/null; then
    echo -e "${RED}Error: Uvicorn not installed. Please run: pip install -r requirements.txt${NC}"
    exit 1
fi

# Start the server
echo -e "${GREEN}Starting server on http://${HOST}:${PORT}${NC}"
echo -e "${YELLOW}Press CTRL+C to stop the server${NC}"
echo ""

# Use exec to replace the shell with uvicorn process
exec python -m uvicorn main:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --log-level "${LOG_LEVEL}"
