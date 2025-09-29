.PHONY: lint test format

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

test:
	pytest -q
