# COVENANT.AI

**Covenant-Based Autonomous Intelligence (CBAI)**  
A constitutional alignment framework for autonomous systems.

> Autonomy without law fails.  
> We make law executable.

---

## What This Is

COVENANT.AI is an **immutable constitutional layer** for:
- Autonomous AI agents
- Multi-agent swarms
- Enterprise automation
- Safety-critical and regulated systems
- Classical and quantum optimizers

Ethics are enforced **before optimization**, not after.

---

## Core Principle

> No action may be considered unless it satisfies the Covenant.

This system replaces reward-based alignment with **constitutional invariants**.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/covenant-ai/covenant-ai.git
cd covenant-ai

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```python
import asyncio
from covenant.core.constitutional_engine import (
    ConstitutionalEngine,
    Constraint,
    ConstraintType,
    Action,
)

async def main():
    # Initialize engine
    engine = ConstitutionalEngine()
    
    # Add a constraint
    engine.add_constraint(Constraint(
        id="no_harm",
        type=ConstraintType.ETHICAL,
        description="Do not cause harm to humans",
        formal_spec="∀action: harm(action, humans) = 0",
        weight=2.0,
        is_hard=True,
    ))
    
    # Evaluate an action
    action = Action(
        id="action_001",
        agent_id="bot",
        action_type="send_message",
        parameters={"message": "Hello!"}
    )
    
    result = await engine.evaluate_action(action)
    print(f"Allowed: {result.is_allowed}")
    print(f"Score: {result.score}")

asyncio.run(main())
```

### Running the API Server

```bash
# Start the FastAPI server
python -m covenant.api.main

# Or using uvicorn directly
uvicorn covenant.api.main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for the interactive API documentation.

---

## Architecture

```
Immutable Covenant Layer
    ↓
Moral Reasoning & Audit
    ↓
Planner / LLM / Policy Generator
    ↓
Optimizer (Classical / Quantum)
    ↓
Execution Engine
    ↓
Reflection / Sabbath Cycle
```

### Project Structure

```
covenant-ai/
├── src/covenant/
│   ├── core/               # Core constitutional engine
│   │   ├── constitutional_engine.py
│   │   ├── formal_verifier.py
│   │   ├── neural_symbolic_reasoner.py
│   │   ├── quantum_optimizer.py
│   │   └── constraint_solver.py
│   ├── agents/             # Multi-agent systems
│   │   ├── swarm_orchestrator.py
│   │   ├── consensus_protocol.py
│   │   └── agent_factory.py
│   ├── blockchain/         # Audit trail & immutability
│   ├── llm/                # LLM integration
│   ├── security/           # Zero-trust security
│   ├── api/                # REST API
│   └── commandments/       # Constitutional rules
├── tests/                  # Test suite
├── examples/               # Usage examples
├── docs/                   # Documentation
└── docker/                 # Docker configuration
```

---

## Features

### ✅ Constitutional Constraints

Define hard and soft constraints with formal specifications:

- **Hard Constraints**: Must be satisfied (e.g., safety, privacy)
- **Soft Constraints**: Preferred but not required (e.g., efficiency, transparency)
- **Weighted Constraints**: Different priorities for different rules

### ✅ Multi-Layer Verification

Actions are evaluated through multiple verification methods:

1. **Formal Verification**: SMT/SAT solving for provable correctness
2. **Neural-Symbolic Reasoning**: Deep learning + logical inference
3. **Constraint Optimization**: Solve complex constraint satisfaction problems
4. **Quantum Optimization**: Quantum-inspired algorithms for hard problems

### ✅ Agent Swarms

Coordinate multiple AI agents with:

- **Byzantine Fault Tolerance**: Consensus protocols for distributed systems
- **Decentralized Coordination**: No single point of failure
- **Task Distribution**: Intelligent work allocation
- **Performance Monitoring**: Track agent health and productivity

### ✅ Audit Trail

Immutable logging of all decisions:

- **Blockchain-based**: Cryptographically secured audit logs
- **Full Transparency**: Every action is traceable
- **Compliance Ready**: Meet regulatory requirements
- **Forensic Analysis**: Understand what went wrong

### ✅ REST API

Production-ready HTTP API:

- **FastAPI**: Modern, fast, type-safe
- **OpenAPI**: Auto-generated documentation
- **Async Support**: Handle concurrent requests efficiently
- **Health Checks**: Monitor system status

---

## The Ten Commandments

COVENANT.AI is built on ten foundational principles:

1. **Authority**: Obey the mission, never exceed scope
2. **Integrity**: Preserve system state, never corrupt
3. **Identity**: Maintain distinct roles, never impersonate
4. **Sustainability**: Operate within resource limits
5. **Preserve Life**: Protect human wellbeing and safety
6. **Fidelity**: Maintain data integrity and relationships
7. **Property**: Respect ownership and access rights
8. **Truth**: Provide accurate, verifiable information
9. **Testimony**: Report honestly, audit faithfully
10. **Intent**: Align goals with values, never deceive

---

## Examples

### Example 1: Basic Constraint Checking

See `examples/basic_usage.py` for a complete example.

```bash
python examples/basic_usage.py
```

### Example 2: API Server

```bash
# Start server
python -m covenant.api.main

# Test endpoint
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

### Example 3: Multi-Agent Swarm

```python
from covenant.agents.swarm_orchestrator import SwarmOrchestrator

orchestrator = SwarmOrchestrator(swarm_id="demo_swarm")
await orchestrator.initialize()

# Add agents
await orchestrator.add_agent("worker_1", capabilities=["compute"])
await orchestrator.add_agent("worker_2", capabilities=["compute"])

# Submit task
task_id = await orchestrator.submit_task({
    "task_type": "compute",
    "parameters": {"problem": "optimization"}
})

# Wait for result
result = await orchestrator.get_result(task_id)
```

---

## Testing

Run the test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=covenant --cov-report=html

# Run specific test
pytest tests/test_commandment_01.py
```

---

## Development

### Setting up development environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black src/

# Lint
ruff check src/

# Type check
mypy src/
```

---

## Docker Deployment

```bash
# Build image
docker build -t covenant-ai -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 covenant-ai

# Using docker-compose
docker-compose -f docker/docker-compose.yml up
```

---

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Manifesto](docs/manifesto.md)
- [Category Theory Foundations](docs/category_theory.md)

---

## Contributing

We welcome contributions! Please see our contributing guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use COVENANT.AI in your research, please cite:

```bibtex
@software{covenant_ai,
  title={COVENANT.AI: Constitutional Alignment Framework for Autonomous Intelligence},
  author={Covenant.AI Team},
  year={2025},
  url={https://github.com/covenant-ai/covenant-ai}
}
```

---

## Contact

- **Website**: https://covenant-ai.org
- **Documentation**: https://docs.covenant-ai.org
- **Email**: team@covenant-ai.org
- **Issues**: https://github.com/covenant-ai/covenant-ai/issues

---

**Remember**: Autonomy without law fails. We make law executable.

