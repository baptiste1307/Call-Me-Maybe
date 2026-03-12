DEPENDENCIES = mypy flake8 pytest

install:
	python3 -m pip install $(DEPENDENCIES)

run:
	uv run python3 -m src

debug:
	uv run python3 -m pdb -m src

test:
	PYTHONPATH=src pytest -q

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} \;
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .venv
	rm -rf data/output
	find . -type d -name "*.egg-info" -exec rm -rf {} \;

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict

.PHONY: install run debug clean lint lint-strict