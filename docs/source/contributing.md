# Contributing

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/dioscuri-tda/pyBallMapper.git
cd pyBallMapper
uv sync
uv run --group dev pre-commit install   # enable git hooks
```

Never use `pip install` for local development — always `uv sync`.

## Pull requests

Open pull requests against the **`develop`** branch, not `main`. The `main` branch is reserved for releases.

## CI pipeline

Every push and PR runs three stages in order. All must pass before merging.

| Stage | Command | What it checks |
|---|---|---|
| Lint | `uv run pre-commit run --all-files` | Ruff lint + format |
| Typecheck | `uv run mypy pyballmapper/` | Static types (package only) |
| Test | `uv run pytest` | Unit tests |

## Running tests

```bash
uv run pytest                            # all tests
uv run pytest -x                         # stop on first failure
uv run pytest tests/test_ballmapper.py   # single file
uv run pytest -k "TestBallMapperConstruction"   # single test class
```

Fixtures live in `tests/conftest.py` and are shared across all test files.

## Code quality

### Ruff (lint + format)

Configured in `pyproject.toml`: line-length 88, target `py313`, rules E/F/W/I (ignores E203/E501/E731).

```bash
uv run pre-commit run --all-files   # run on entire tree
```

Pre-commit runs `ruff` (with `--fix`) and `ruff-format` automatically on every commit.

### Mypy (type checking)

```bash
uv run mypy pyballmapper/
```

Only the package source is checked — tests and docs are excluded. `ignore_missing_imports = true` for untyped third-party libraries.

## Pre-commit hooks

Hooks are defined in `.pre-commit-config.yaml`:

- `ruff` — lint and auto-fix
- `ruff-format` — format
- `mypy` — typecheck on `pyballmapper/` only

Run them on demand: `uv run pre-commit run --all-files`

## Packaging

Built with setuptools. Version is read dynamically from `pyballmapper/__init__.py`.

```bash
uv build
```

Dependencies are declared in `pyproject.toml` under `[project] dependencies`. Dev-only deps (`pre-commit`, `ruff`, `mypy`, `pytest`) go under `[dependency-groups] dev`.
