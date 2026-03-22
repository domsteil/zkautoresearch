PYTHON ?= python
STATESET_STARK_DIR ?= ../stateset-stark

.PHONY: doctor install-ves-stark test test-integrity demo-seeded

doctor:
	$(PYTHON) -m stark_autoresearch.environment

install-ves-stark:
	$(PYTHON) scripts/install_ves_stark.py --stateset-stark-dir $(STATESET_STARK_DIR)

test:
	$(PYTHON) -m pytest -q

test-integrity:
	$(PYTHON) test_stark_integrity.py

demo-seeded:
	$(PYTHON) run_network.py --agents 2 --experiments 5 --seed 42
