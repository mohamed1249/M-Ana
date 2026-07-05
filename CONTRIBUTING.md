# Contributing to M-Ana

Thank you for helping improve M-Ana.

## Development setup

Create an isolated environment, then install the package and development tools:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,test]"
```

Install only the optional feature groups needed for your change. For example:

```bash
python -m pip install -e ".[dev,rag,big]"
```

## Before opening a pull request

Run the core checks from the repository root:

```bash
ruff check MAna tests
python -m unittest discover -s tests -v
python -m build
twine check dist/*
```

Changes to public behavior should include tests and a short changelog entry. Keep
optional dependencies behind lazy imports with an actionable installation error.
Never commit API keys, database credentials, local datasets, or generated build
artifacts.

## Compatibility

M-Ana supports Python 3.9 and newer. Public APIs follow semantic versioning.
Deprecate public names before removal unless security or correctness makes a
transition period unsafe.
