# Gaussian Elimination

A numerical method for solving systems of linear equations of the form A·x = b, implemented with partial pivoting for improved numerical stability.

![GAUSSIAN ELIMINATION](https://github.com/user-attachments/assets/52320c8e-16c0-4b53-8972-f1b8f3e56a2d)

## Overview

Gaussian elimination is one of the most fundamental algorithms in linear algebra — and by extension, in machine learning and data science. At its core, machine learning often reduces to solving or optimizing systems of equations, and understanding how matrices encode and solve those systems is essential groundwork.

This implementation walks through:

- **Forward elimination** — reducing the matrix to upper triangular form
- **Partial pivoting** — swapping rows to avoid division by small (or zero) pivots, improving stability
- **Back substitution** — solving for unknowns from the bottom up

Whether you're learning numerical methods, brushing up on linear algebra, or tracing the mathematical roots of how models like linear regression work under the hood, this is a great place to start.

This project solves system A.x = b with n-equations with n-variables. It won't be able to conclude for infinitely many solution.

## Usage

```python
from gaussian import Gaussian

ge = Gaussian(["x + y = 3", "x - y = -1"])
ge.solve()
print(ge.solution)  # {'x': 1.0, 'y': 2.0}
```

Equations are plain strings (`'2x + 3y = 8'`); an implicit coefficient of 1 is supported (e.g. `'x - y = 0'`).

## Project Setup

Prerequisites:

- Python 3.12 or newer
- `uv` (fast Python package installer and resolver)

### Install `uv` (if not already installed)

```bash
# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

### Sync up the project

```bash
# Sync dependencies and create virtual environment
uv sync
```

No need to activate the virtual environment — `uv run` picks it up automatically.

### Run

```bash
uv run src/gaussian.py
```

### Alternative: Interactive demo

There is also a Jupyter notebook demo:

```bash
uv run jupyter lab
```

## Demo

Visit `demo.ipynb` and you will find different scenarios for which `Gaussian` is tested.

If you can think of other test cases, then you can add and play around.

## Development

Dev tooling (`ruff`, `mypy`, `pytest`, `pre-commit`) is managed as a `uv` dependency group.

```bash
# Install dev dependencies
uv sync

# Install git hooks (lint + type check on commit, tests on push)
uv run pre-commit install --install-hooks \
  --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

```bash
uv run ruff check . --fix   # lint
uv run ruff format .        # format
uv run mypy                  # type check
uv run pytest                # run tests in tests/
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) — the `commit-msg` hook enforces this.
