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

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
```

### Run

```bash
uv run gaussian.py
```

### Alternative: Interactive demo

There is also a Jupyter notebook demo:

```bash
uv run jupyter lab demo.ipynb
```

## Demo

Visit `demo.ipynb` and you will find different scenarios for which `Gaussian` is tested.

If you can find other test cases, then you can add and play around.
