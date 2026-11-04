# Ruff And Pyrefly Check Fixes

Rules for fixing specific problems reported by `ruff check` and `pyrefly check`
in this repository.

Project settings that matter while fixing:

- Ruff line length: `100`; `E501` is ignored.
- Ruff ignores `D1` (missing docstrings); `unfixable = ["B"]` (fix bugbear manually).
- Type checker: `pyrefly`, not mypy.
- Python: `>=3.14`; Ruff `target-version = "py313"`, so syntax accepted only in
  3.14 (e.g. unparenthesized multi-exception `except A, B:`) still fails Ruff.

## Ruff Problems

- **`Boolean-typed positional argument in function definition`:**

  - Use `*,` before boolean parameters, e.g.:

      ```python
      async def ask(self, search_context: str = "", *, stream: bool = False) -> ChatResponse: ...
      ```

- **`Too many arguments in function definition`:**

  - Use the `@dataclass(frozen=True)` approach to group related arguments.

- **`Avoid specifying long messages outside the exception class`**

  - Define a string variable and pass it to the exception class, e.g.:

    ```python
    details = f"Invalid email service task: {task}"
    raise ValueError(details)
    ```

- **`Dynamically typed expressions (typing.Any) are disallowed`**

  - First try to define a correct type hint; otherwise use `object`; use
    `# noqa` only when it is a genuine stub limitation.

- **Import ordering (`I001`) / trailing comma (`COM812`) / missing newline (`W292`)**

  - Prefer automatic fixes:

    ```bash
    ruff check src tests --fix
    ruff format src tests
    ```

- **`invalid-syntax: Multiple exception types must be parenthesized on Python 3.13`**

  - Even though the project runs on 3.14, Ruff targets 3.13. Parenthesize:

    ```python
    except (TypeError, ValueError):
        ...
    ```

- **`F541` f-string without placeholders**

  - Remove the `f` prefix.

## Pyrefly Problems

- **Method return unions are not narrowed (`AsyncGenerator | str | None`)**

  - Add `@overload` for the boolean switch instead of casting at every call site:

    ```python
    @overload
    async def generate(self, configs: dict[str, object], *, stream: Literal[False] = False) -> str | None: ...

    @overload
    async def generate(self, configs: dict[str, object], *, stream: Literal[True]) -> AsyncGenerator[object]: ...

    async def generate(
        self,
        configs: dict[str, object],
        *,
        stream: bool = False,
    ) -> AsyncGenerator[object] | str | None: ...
    ```

- **A decorator instance is "not callable" (`not-callable`)**

  - Give the decorator class a `__call__` method rather than wrapping the
    decorator in a factory, e.g. `class CircuitBreaker: def __call__(self, func): ...`.

- **Complex inheritance / overrides**

  - Consider `@overload` and `@override` methods. Example error:
    `Returned type bytes | str | None is not assignable to declared return type str | None`
    (`Pyrefly(bad-return)`).

- **Widened container values**

  - Prefer `typing.cast` over suppressing the error when the runtime value is
    guaranteed by construction (e.g. casting a wrapped stream to `Stream`).

- **Proper fixes first**

  - Prioritize real fixes over `# type: ignore` / `# pyrefly: ignore` unless it
    is a stub limitation that cannot be expressed.

## Verification

```bash
ruff check src tests
ruff format --check src tests
pyrefly check
uv run pytest
```

All four must pass before the work is considered complete.