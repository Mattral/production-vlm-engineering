.PHONY: setup setup-gpu run-example benchmark test lint docs-serve docker-build docker-build-gpu clean

UV := uv

setup:
	$(UV) venv .venv
	$(UV) pip install -e ".[dev,cli,onnx,serving,demo]"
	$(UV) run pre-commit install

setup-gpu:
	$(UV) venv .venv
	$(UV) pip install -e ".[dev,cli,ml,onnx,serving,demo]"
	$(UV) run pre-commit install

run-example:
	@if [ -z "$(NAME)" ]; then echo "Usage: make run-example NAME=vlm_chart_finetune"; exit 1; fi
	$(UV) run python -m production_vlm.cli run-example $(NAME)

benchmark:
	$(UV) run python -m production_vlm.cli benchmark $(NAME)

test:
	$(UV) run pytest --cov=src/production_vlm --cov-report=term-missing

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

docs-serve:
	$(UV) run mkdocs serve -a 0.0.0.0:8000

docker-build:
	docker build -f docker/Dockerfile -t production-vlm-engineering:latest .

docker-build-gpu:
	docker build -f docker/Dockerfile.gpu -t production-vlm-engineering:gpu .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache outputs/*
