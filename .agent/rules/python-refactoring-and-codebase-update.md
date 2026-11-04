---
trigger: always_on
---

# Python Refactoring Rules/Guidance

You're Senior Python Backend Architect working on the OpenAI client harness.
When doing codebase refactoring/updates, make sure to always follow:

## Core Principles

1. **Follow Python Best Practices and Design Patterns**
   - Use appropriate design patterns (Factory, Singleton, Observer, etc.)
   - Follow SOLID principles
   - Implement proper separation of concerns
   - Use composition over inheritance when appropriate

2. **Follow Ruff Rules**
   - Configure and adhere to Ruff linting rules
   - Use `ruff check` and `ruff format` for code quality
   - Line length is 100; `E501` and `D1` (missing docstrings) are intentionally ignored
   - `unfixable = ["B"]`: bugbear violations must be fixed manually
   - Fix all linting warnings before committing

3. **Full Type Hinting**
   - Provide complete type annotations for all functions and variables
   - Use `pyrefly check` for type validation (this repo does not use mypy)
   - Type hints should be comprehensive and accurate

4. **Optional Parameters - Use Pipe Operator**
   - Use `|` operator for optional types instead of `Optional` from typing
   - Example: `str | None` instead of `Optional[str]`
   - This follows modern Python 3.14+ syntax used by this project

5. **Explicit Imports Only**
   - Never import entire modules with `import module`
   - Always use explicit imports: `from module import class, function`
   - Exception: when importing module for type checking only with `TYPE_CHECKING`

6. **Use pathlib for Path Operations**
   - Replace all `os.path` operations with `pathlib` / `anyio.Path`
   - Use `Path` objects for file system operations
   - Leverage pathlib's object-oriented interface

7. **Optimize for Performance, Security, Maintainability, Scalability**
   - Write efficient code with proper algorithms
   - Implement security best practices (input validation, prompt-injection sanitization)
   - Ensure code is maintainable and well-documented
   - Use proper error handling with try/except blocks

8. **Use uv for Dependencies**
   - Install packages with `uv add package_name`
   - Use `uv` for all package management operations
   - Keep dependencies in `pyproject.toml`

9. **Use uv Workflow for Commands**
   - Execute commands through `uv run` when appropriate
   - Use `uv` scripts for common development tasks
   - Leverage uv's virtual environment management

10. **Testing with pytest**
    - Tests live in `tests/` and import via `openai_client...` (`pythonpath = src`)
    - Use `pytest-asyncio` for async services
    - Mock external HTTP providers with `AsyncMock` / `patch`
    - Centralize shared fixtures in `tests/conftest.py`
    - Follow pytest best practices for test organization

11. **Latest Dependencies**
    - Always use the latest stable versions of dependencies
    - Keep dependencies updated regularly

12. **Code Documentation**
    - Use docstrings for public functions and classes
    - Follow Google or NumPy docstring format
    - Include type information in docstrings

13. **Error Handling**
    - Use specific exception types instead of bare `except:`
    - Map provider errors through `raise_ai_error` / the `AgentError` hierarchy
    - Implement proper logging for error tracking
    - Create custom exception classes when needed

14. **Testing Requirements**
    - Write comprehensive unit tests with pytest
    - Use parametrized tests for multiple scenarios
    - Mock external dependencies appropriately
    - Exercise the production behavior (completion, streaming, search, prompts)

15. **Security Best Practices**
    - Validate all user inputs
    - Use environment variables for sensitive data
    - Sanitize prompt inputs (`sanitize_for_prompt`) and check model output
      for system-prompt leakage (`_detect_output_injection`)
    - Follow OWASP security guidelines

16. **Performance Optimization**
    - Use appropriate data structures
    - Implement caching when beneficial
    - Use async/await for I/O operations

17. **Code Organization**
    - Follow the existing structure: `client/`, `configs/`, `decorators/`,
      `errors/`, `schemas/ai/`, `services/`, `services/prompts/`,
      `services/templates/`, `typings/`, `utils/`
    - Separate concerns into different modules
    - Use `__init__.py` files properly
    - Mirror production BaliRoam OpenAI behavior when updating core systems

18. **Version Control**
    - Write meaningful commit messages
    - Use conventional commit format
    - Include tests in all commits

19. **Configuration Management**
    - Use environment-based configuration via `pydantic-settings` (`.env`, `.env.configs`)
    - Separate config from code
    - Use `pyproject.toml` for project configuration

20. **Logging**
    - Use the standard-library `logging` module
    - Implement proper log levels
    - Avoid logging secrets or full prompt payloads

## Development Workflow

1. **Setup**: `uv sync`
2. **Development**: Follow all coding rules during implementation
3. **Testing**: `uv run pytest`
4. **Linting**: `ruff check src tests` and `ruff format src tests`
5. **Type Checking**: `pyrefly check`
6. **Documentation**: Maintain up-to-date docstrings
7. **Security**: Regular security audits and updates
8. **Lint and type check problem fixes**: follow
   [ruff and pyrefly fixes rules](ruff-pyrefly-fixes.md)

## Tools and Commands

```bash
# Install dependencies
uv sync
uv add package_name

# Run tests
uv run pytest

# Lint code
ruff check src tests --fix
ruff format src tests

# Type checking
pyrefly check

# Run application
uv run openai-client
```

Remember: Quality code is maintainable, testable, and follows established best practices!