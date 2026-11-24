.PHONY: install dev test lint serve demo docker clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest --cov=expforge --cov-report=term-missing

lint:
	ruff check expforge tests examples

demo:
	expforge demo

serve:
	expforge server --port 8000

docker:
	docker compose up --build

clean:
	rm -rf .expforge .pytest_cache .coverage build *.egg-info
