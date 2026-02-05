# Contributing to COVENANT.AI

Thank you for your interest in contributing to COVENANT.AI! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:
- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Python version and environment details
- Relevant logs or error messages

### Suggesting Enhancements

We welcome feature suggestions! Please:
- Check if the feature has already been suggested
- Provide a clear use case
- Explain why this would be valuable
- Consider implementation details if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/covenant-ai.git
   cd covenant-ai
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -e ".[dev]"
   pre-commit install
   ```

4. **Make your changes**
   - Write clear, documented code
   - Follow our code style (see below)
   - Add tests for new functionality
   - Update documentation as needed

5. **Run tests**
   ```bash
   pytest
   black src/
   ruff check src/
   mypy src/
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding tests
   - `refactor:` code refactoring
   - `chore:` maintenance tasks

7. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Development Guidelines

### Code Style

- Use **Black** for formatting (line length: 100)
- Use **Ruff** for linting
- Use **MyPy** for type checking
- Follow PEP 8 conventions
- Write docstrings for all public functions/classes

Example:
```python
from typing import Optional

def evaluate_action(
    action_id: str,
    parameters: dict,
    context: Optional[dict] = None
) -> bool:
    """
    Evaluate an action against constitutional constraints.
    
    Args:
        action_id: Unique identifier for the action
        parameters: Action parameters
        context: Optional context information
        
    Returns:
        True if action is allowed, False otherwise
        
    Raises:
        ValueError: If action_id is invalid
    """
    # Implementation here
    pass
```

### Testing

- Write tests for all new code
- Aim for >80% code coverage
- Use pytest fixtures for common setups
- Include both unit and integration tests

Example test:
```python
import pytest
from covenant.core.constitutional_engine import ConstitutionalEngine

@pytest.fixture
def engine():
    return ConstitutionalEngine()

def test_engine_initialization(engine):
    assert engine is not None
    assert len(engine.constraints) == 0

async def test_evaluate_action(engine):
    # Test implementation
    pass
```

### Documentation

- Update README.md if adding features
- Add docstrings to all public APIs
- Include usage examples
- Update CHANGELOG.md

### Commit Messages

Good commit messages:
```
feat: add quantum optimization support
fix: resolve race condition in swarm coordinator
docs: update installation instructions
test: add tests for consensus protocol
```

Bad commit messages:
```
update
fix bug
changes
WIP
```

## Architecture Guidelines

When adding new features:

1. **Follow the layered architecture**
   - Core layer: Essential business logic
   - Service layer: Orchestration and coordination
   - API layer: External interfaces

2. **Maintain separation of concerns**
   - Each module should have a single responsibility
   - Avoid circular dependencies
   - Use dependency injection

3. **Ensure immutability where appropriate**
   - Constitutional constraints should be immutable
   - Use frozen dataclasses for value objects

4. **Design for testability**
   - Write testable code
   - Avoid global state
   - Use interfaces/protocols

## Review Process

1. **Automated checks** must pass:
   - All tests pass
   - Code coverage meets threshold
   - Linting passes
   - Type checking passes

2. **Code review** by maintainers:
   - Code quality and style
   - Design and architecture
   - Documentation completeness
   - Test coverage

3. **Feedback and iteration**:
   - Address review comments
   - Update PR as needed
   - Discuss design decisions

## Getting Help

- **Discord**: Join our community server
- **GitHub Discussions**: For questions and ideas
- **Email**: team@covenant-ai.org
- **Documentation**: https://docs.covenant-ai.org

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to the contributors' channel

Thank you for making COVENANT.AI better! 🙏
