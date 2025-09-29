# Contributing

- Create a Conda env with `environment.yml`
- Install pre-commit and enable hooks: `pip install pre-commit && pre-commit install`
- Run linters/tests before pushing: `ruff check . && black --check . && pytest -q`
