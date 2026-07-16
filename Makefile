.PHONY: install dev test lint demo train serve infra docker clean

install:
	pip install -e .

dev:
	pip install -e ".[serve,dev]"

test:
	pytest --cov=driftguard --cov-report=term-missing

lint:
	ruff check driftguard scripts

demo:
	python -m driftguard demo

train:
	python scripts/train_baseline.py

serve:
	driftguard serve

infra:
	docker compose up --build

docker:
	docker build -t driftguard:latest .

clean:
	rm -rf .pytest_cache build dist *.egg-info mlruns_local
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
