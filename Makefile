.PHONY: install setup test lint check verify bench demo clean health serve

install:
	pip install -e ".[dev]"

setup:
	molecule setup

test:
	pytest -q

lint:
	ruff check .

check: lint test

verify:
	python verify_efficiency.py

bench:
	python benchmarks/run_benchmarks.py

demo:
	python demo.py

clean:
	rm -rf .molecule/ .molecule_demo_tmp/ .pytest_cache .ruff_cache dist build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

health:
	molecule doctor

serve:
	molecule serve
