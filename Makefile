.PHONY: install setup test lint check bench demo clean health serve

install:
	pip install -e ".[dev]"

setup:
	molecule setup

test:
	pytest -v

lint:
	ruff check src tests

check: lint test
	molecule ingest AGENTS.md
	molecule eval eval_queries.json
	molecule eval eval_queries_adversarial.json

bench:
	python scripts/benchmark_5k.py

demo:
	python demo.py

clean:
	rm -rf .molecule/ __pycache__ src/**/__pycache__ .pytest_cache dist *.egg-info

health:
	molecule health

serve:
	molecule serve
