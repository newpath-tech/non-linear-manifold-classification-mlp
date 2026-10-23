.PHONY: install test lint train benchmark clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m flake8 src tests --max-line-length=120
	$(PYTHON) -m black --check src tests

train:
	$(PYTHON) train.py --config configs/default_config.yaml

benchmark:
	$(PYTHON) benchmark.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
