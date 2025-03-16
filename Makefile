.PHONY: format

format:
	poetry run isort q test scripts
	poetry run black q test scripts