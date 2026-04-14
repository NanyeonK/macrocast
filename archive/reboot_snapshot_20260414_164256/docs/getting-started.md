# Installing macrocast

This page follows the same practical order most users need:
- confirm environment
- install the package
- install docs extras if needed
- obtain source when working from the repository
- use editable mode for active development

## Python support

`macrocast` currently targets the project environment defined in the repository and is expected to be used through `uv`.

## Install with uv

If you are working from source in the repository:

```bash
uv sync
```

This is the recommended install path for current package work.

## Install docs extras

If you also want to build or serve the docs:

```bash
uv sync --extra docs
```

## Quick start

After installing:

```python
from macrocast import macrocast_single_run

out = macrocast_single_run()
```

This starts the guided single-run flow and writes a YAML recipe as choices are made.

## Obtain the source

Repository:

```bash
git clone https://github.com/NanyeonK/macrocast.git
cd macrocast
```

Then install with:

```bash
uv sync
```

## Installation from source

If you already have the repository and want the project environment only:

```bash
uv sync
```

If you need docs support too:

```bash
uv sync --extra docs
```

## Editable development workflow

For package development, work from the repository checkout and keep using the local `uv` environment rather than trying to install detached wheels manually.

Typical workflow:

```bash
uv sync --extra docs
uv run pytest -q
uv run mkdocs build --strict
```

## Optional docs preview

To preview the docs locally or on a server:

```bash
uv run mkdocs serve -a 0.0.0.0:8000
```

## What to read next

After installation:
1. `User Guide`
2. `User Guide > Stage Map`
3. `User Guide > Stage 0`
