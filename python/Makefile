PYTHON := python3

install:
$(PYTHON) -m pip install -e .

lint:
flake8 src/geolift

test:
pytest

.PHONY: install lint test
