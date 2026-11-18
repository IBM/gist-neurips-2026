.PHONY: install
install: ## Create the environment with all extras and dev tools
	@uv sync --all-extras

.PHONY: check
check: ## Lock consistency, lint, format and type checks
	@uv lock --locked
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run mypy

.PHONY: test
test: ## Run the tests
	@uv run python -m pytest

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
