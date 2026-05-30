.PHONY: help install pull fit figures test lint format clean

help:
	@echo "Targets:"
	@echo "  install   install Python deps via uv (or pip)"
	@echo "  pull      download NHANES 2003-2010 files into data/raw/"
	@echo "  fit       run single-pathogen catalytic model (smoke test)"
	@echo "  figures   render publication figures"
	@echo "  test      run pytest suite"
	@echo "  lint      run ruff + mypy"
	@echo "  format    auto-format with ruff"
	@echo "  clean     remove caches and build artifacts"

install:
	uv sync || pip install -e ".[dev]"

pull:
	python -m analysis.data_pull.run

pull-cycle-c:
	python -m analysis.data_pull.run --cycles C

fit:
	@echo "[stub] model fit not yet implemented"

figures:
	@echo "[stub] figure generation not yet implemented"

test:
	pytest -ra

lint:
	ruff check .
	mypy analysis tests || true

format:
	ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
