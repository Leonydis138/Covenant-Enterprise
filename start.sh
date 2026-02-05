#!/bin/bash

# COVENANT.AI Startup Script
# This script helps you quickly start the COVENANT.AI system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                     COVENANT.AI                               ║"
echo "║   Constitutional Alignment Framework for Autonomous AI        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi
print_info "Python version OK: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_info "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
pip install -e . > /dev/null 2>&1

print_info "Dependencies installed successfully"

# Check for .env file
if [ ! -f ".env" ]; then
    print_warn ".env file not found. Creating from .env.example..."
    cp .env.example .env
    print_info "Please edit .env file with your configuration"
fi

# Parse command line arguments
MODE=${1:-api}

case $MODE in
    api)
        print_info "Starting API server..."
        python -m covenant.api.main
        ;;
    
    example)
        print_info "Running basic example..."
        python examples/basic_usage.py
        ;;
    
    test)
        print_info "Running tests..."
        pytest tests/ -v
        ;;
    
    docker)
        print_info "Starting with Docker..."
        docker-compose -f docker/docker-compose.yml up --build
        ;;
    
    shell)
        print_info "Starting Python shell with COVENANT.AI loaded..."
        python -i -c "
from covenant.core.constitutional_engine import *
from covenant.agents.swarm_orchestrator import *
print('COVENANT.AI modules loaded. Ready to use!')
"
        ;;
    
    *)
        print_error "Unknown mode: $MODE"
        echo ""
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  api       - Start API server (default)"
        echo "  example   - Run basic example"
        echo "  test      - Run test suite"
        echo "  docker    - Start with Docker Compose"
        echo "  shell     - Start interactive Python shell"
        exit 1
        ;;
esac
