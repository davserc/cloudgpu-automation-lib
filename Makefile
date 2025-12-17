.PHONY: install dev lint format check clean help search list balance

# Default Python
PYTHON := ./venv/bin/python3
PIP := ./venv/bin/pip

help:
	@echo "Vast.ai GPU CLI - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install dependencies"
	@echo "  make dev         Install dev dependencies (pre-commit, ruff)"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint        Run linter (ruff check)"
	@echo "  make format      Format code (ruff format)"
	@echo "  make check       Run all pre-commit hooks"
	@echo ""
	@echo "GPU commands:"
	@echo "  make search      Search for available GPUs"
	@echo "  make list        List your instances"
	@echo "  make balance     Show account balance"
	@echo "  make billing     Show billing history"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       Remove cache files"

# Setup
install:
	python3 -m venv venv
	$(PIP) install -r requirements.txt

dev: install
	$(PIP) install pre-commit ruff
	./venv/bin/pre-commit install

# Code quality
lint:
	./venv/bin/ruff check .

format:
	./venv/bin/ruff format .
	./venv/bin/ruff check --fix .

check:
	./venv/bin/pre-commit run --all-files

# GPU commands (shortcuts)
search:
	$(PYTHON) cli.py search

list:
	$(PYTHON) cli.py list

balance:
	$(PYTHON) cli.py balance

billing:
	$(PYTHON) cli.py billing

# Utilities
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
