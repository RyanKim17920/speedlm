# Contributing

Use Python 3.12 and install the project with its development tools:

```bash
python -m pip install -e ".[dev]"
```

Run the same checks as [CI](.github/workflows/ci.yml):

```bash
python -m pytest -m "not e2e"
ruff check .
mypy
```

`tests/e2e` requires H100 GPU infrastructure, a live vLLM installation, and a
SLURM allocation. It is not part of the normal local or CI test run.
