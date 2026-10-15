PY ?= python
CARDS ?= data/sample/cards.jsonl
GOLD ?= data/sample/gold.jsonl

.PHONY: setup test predict eval all

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

predict:
	$(PY) -m attrx predict --cards $(CARDS) --out results/predictions.csv

eval:
	$(PY) -m attrx eval --pred results/predictions.csv --gold $(GOLD) --report reports/metrics.md

all: test predict eval
