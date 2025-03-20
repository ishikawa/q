.PHONY: format test lint mypy

format:
	poetry run isort q test scripts
	poetry run black q test scripts

test:
	poetry run pytest ./test

lint:
	poetry run flake8 q test scripts

mypy:
	poetry run mypy q test scripts

check: test lint mypy