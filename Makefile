.PHONY: format test lint mypy

format:
	poetry run isort q test scripts
	poetry run black q test scripts

test:
	poetry run pytest ./test

lint:
	poetry run flake8 q test scripts

type-check:
	poetry run pyright

check: test lint type-check