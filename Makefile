PYTHON ?= python
STATESET_STARK_DIR ?= ../icommerce-app/stateset-stark

.PHONY: doctor install-ves-stark test test-core test-autoresearch test-integrity demo-seeded smoke-auto

doctor:
	$(PYTHON) -m stark_autoresearch.environment

install-ves-stark:
	$(PYTHON) scripts/install_ves_stark.py --stateset-stark-dir $(STATESET_STARK_DIR)

test: test-core test-autoresearch

test-core:
	$(PYTHON) -m pytest -q test_environment.py test_network_behavior.py test_stark_integrity.py

test-autoresearch:
	$(PYTHON) -m pytest -q test_experiment_runtime.py test_prepare_selection_metric.py test_subprocess_auto_research.py test_auto_research_cli.py

test-integrity:
	$(PYTHON) test_stark_integrity.py

demo-seeded:
	$(PYTHON) run_network.py --agents 2 --experiments 5 --seed 42

smoke-auto:
	$(PYTHON) auto_research.py --runtime-profile smoke --max-experiments 2 --time-budget 60 --output-dir ./auto_research_results_smoke
