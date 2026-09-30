.PHONY: setup lint format test

setup:  ## Install the locked environment and the git hooks
	uv sync
	uv run pre-commit install

lint:  ## Check style without modifying files (used by CI)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format the code
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest
