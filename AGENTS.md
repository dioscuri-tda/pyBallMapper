# pyBallMapper — AGENTS.md

## Setup & commands
- Managed with **uv**. Run `uv sync` to install all deps.
- CI order: lint → typecheck → test.
- Lint: `uv run pre-commit run --all-files` (ruff lint + ruff-format)
- Typecheck: `uv run mypy pyballmapper/` — only the package, **not** tests/docs
- Test: `uv run pytest` or `uv run pytest tests/test_<file>.py -k "TestClass"`

## Package structure
- `pyballmapper/ballmapper.py` — core `BallMapper` class + landmark selection functions
- `pyballmapper/mobm.py` — `MapperonBallMapper` (DBSCAN-based refinement)
- `pyballmapper/plotting.py` — Bokeh interactive viz, pie charts, KeplerMapper export
- `pyballmapper/__init__.py` exports `BallMapper` and `__version__` only
- Tests in `tests/` (pytest, fixtures in `conftest.py`)
- Notebooks in `notebooks/`

## Landmark methods
- `"greedy"` (default) — first-uncovered in given order
- `"nearest"` — deterministic, picks uncovered point nearest any ball
- `"adaptive"` — shrinks radius per ball to cap `max_size` points

## Toolchain quirks
- Ruff line-length: 88, target `py313`, rules E/F/W/I (ignores E203/E501/E731)
- Mypy: `ignore_missing_imports = true`
- Requires Python >= 3.13
- Bokeh `graph_GUI` for interactive HTML; `kmapper_visualize` for KeplerMapper output

## What NOT to do
- Do NOT run mypy on tests/ or docs/ — CI only checks `pyballmapper/`
- Do NOT use pip directly for dev setup — use `uv sync`
