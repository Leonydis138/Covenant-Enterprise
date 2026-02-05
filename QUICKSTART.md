# COVENANT.AI - Quick Start Guide

Get up and running with COVENANT.AI in 5 minutes!

## Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) Docker for containerized deployment

## Installation

### Method 1: From Source (Recommended for Development)

```bash
# Extract the archive
tar -xzf covenant-ai-complete.tar.gz
cd covenant-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Method 2: Using Docker

```bash
# Extract the archive
tar -xzf covenant-ai-complete.tar.gz
cd covenant-ai

# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build
```

## First Steps

### 1. Run the Basic Example

```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Run the basic example
python examples/basic_usage.py
```

You should see output showing:
- Constitutional engine initialization
- Constraint definition
- Action evaluation
- Metrics and statistics

### 2. Start the API Server

```bash
# Using the start script (recommended)
./start.sh api

# Or manually
python -m covenant.api.main
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 3. Test an API Call

```bash
# In a new terminal
curl -X POST "http://localhost:8000/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "mission": "help_user",
    "confidence": 0.95,
    "evidence": 0.9,
    "harm": 0.0,
    "resource_ratio": 0.5,
    "auditable": true,
    "data_provenance": true
  }'
```

## Basic Usage in Code

```python
import asyncio
from covenant.core.constitutional_engine import (
    ConstitutionalEngine,
    Constraint,
    ConstraintType,
    Action,
)

async def main():
    # 1. Create engine
    engine = ConstitutionalEngine()
    
    # 2. Add constraints
    engine.add_constraint(Constraint(
        id="safety_first",
        type=ConstraintType.SAFETY,
        description="Never cause harm",
        formal_spec="∀action: harm(action) = 0",
        weight=2.0,
        is_hard=True,  # Must be satisfied
    ))
    
    # 3. Create an action
    action = Action(
        id="send_message_001",
        agent_id="chatbot",
        action_type="send_message",
        parameters={
            "message": "Hello! How can I help?",
            "recipient": "user_123"
        }
    )
    
    # 4. Evaluate the action
    result = await engine.evaluate_action(action)
    
    # 5. Check the result
    if result.is_allowed:
        print(f"✓ Action allowed (score: {result.score:.2f})")
    else:
        print(f"✗ Action blocked")
        print(f"  Violations: {result.violations}")
        print(f"  Suggestions: {result.suggestions}")

asyncio.run(main())
```

## Common Tasks

### Add a New Constraint

```python
from covenant.core.constitutional_engine import Constraint, ConstraintType

my_constraint = Constraint(
    id="privacy_protection",
    type=ConstraintType.PRIVACY,
    description="Protect user privacy",
    formal_spec="∀data: is_personal(data) → is_encrypted(data)",
    weight=1.5,
    is_hard=True
)

engine.add_constraint(my_constraint)
```

### Evaluate Multiple Actions

```python
actions = [
    Action(id=f"action_{i}", agent_id="bot", 
           action_type="test", parameters={"value": i})
    for i in range(10)
]

results = []
for action in actions:
    result = await engine.evaluate_action(action)
    results.append(result)

# Get metrics
metrics = engine.get_metrics()
print(f"Evaluated {metrics['total_evaluations']} actions")
print(f"Allowed: {metrics['allowed_actions']}")
print(f"Blocked: {metrics['blocked_actions']}")
```

### Run Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=covenant

# Run integration tests
python tests/integration_test.py
```

## Project Structure Overview

```
covenant-ai/
├── src/covenant/           # Main source code
│   ├── core/              # Constitutional engine
│   ├── agents/            # Multi-agent coordination
│   ├── api/               # REST API
│   ├── blockchain/        # Audit trail
│   └── commandments/      # Ten commandments
├── tests/                 # Test suite
├── examples/              # Usage examples
├── docker/                # Docker files
├── docs/                  # Documentation
├── requirements.txt       # Dependencies
└── setup.py              # Package setup
```

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Copy example
cp .env.example .env

# Edit as needed
nano .env
```

Key settings:
- `API_PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)
- `USE_REAL_QUANTUM`: Enable real quantum backends

## Next Steps

1. **Read the Documentation**
   - [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Comprehensive project overview
   - [README.md](README.md) - Full README with examples
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines

2. **Explore Examples**
   - `examples/basic_usage.py` - Basic constitutional engine
   - `examples/advanced_usage.py` - Advanced features
   - `examples/multi_agent_example.py` - Multi-agent systems

3. **Customize**
   - Add your own constraints
   - Integrate with your LLM
   - Deploy to production

4. **Contribute**
   - Report bugs
   - Suggest features
   - Submit pull requests

## Troubleshooting

### Import Errors

```bash
# Make sure package is installed
pip install -e .

# Check Python path
python -c "import covenant; print(covenant.__version__)"
```

### API Won't Start

```bash
# Check if port is in use
lsof -i :8000

# Try different port
uvicorn covenant.api.main:app --port 8001
```

### Tests Failing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests with verbose output
pytest -v

# Run single test
pytest tests/test_commandment_01.py -v
```

## Getting Help

- **Documentation**: https://docs.covenant-ai.org
- **Issues**: https://github.com/covenant-ai/covenant-ai/issues
- **Discord**: https://discord.gg/covenant-ai
- **Email**: team@covenant-ai.org

## What's Next?

Now that you have COVENANT.AI running:

1. Explore the `/docs` API at http://localhost:8000/docs
2. Try different constraint types
3. Build a multi-agent system
4. Integrate with your application
5. Deploy to production

Happy coding! 🚀

---

**Remember**: Autonomy without law fails. We make law executable.
