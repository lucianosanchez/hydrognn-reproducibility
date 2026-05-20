# Convenience wrappers around scripts/run_paper_experiments.sh and the
# paper compilation. Run `make help` for a full list of targets.
#
# These targets assume:
#   * The active Python interpreter (see PYTHON variable below).
#   * The repository root is the current working directory.
#   * `pip install -e .` has been run so that `import seq2seq_runoff`
#     works from any subdirectory.

PYTHON  ?= python
SHELL   := /usr/bin/env bash

# ----------------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------------
.PHONY: help
help:
	@echo "Available targets:"
	@echo ""
	@echo "  Reproducibility:"
	@echo "    make install        Install the package in editable mode + deps"
	@echo "    make smoke          Import the package and print a sanity line"
	@echo "    make all            Reproduce every paper experiment (~4-5 h CPU)"
	@echo "    make data           Regenerate synthetic datasets if missing"
	@echo "    make train          UA-HydroGNN on synth-N16, synth-N64, Ebro"
	@echo "    make robustness     Ebro multi-seed {0, 7, 123}"
	@echo "    make l_alpha        Quantile sensitivity sweep"
	@echo "    make w4             Variance decomposition"
	@echo "    make ensembles      Maximin-Savage agreement audit"
	@echo "    make phase22        HydroGNN Phase 2.2 topology recovery"
	@echo "    make ebro_informed  Ebro with river-length prior"
	@echo "    make physicalize    Post-hoc topology cleanup"
	@echo "    make phase22_acyclic Acyclic candidate graph grid"
	@echo "    make grid           Optional 8-config remediation grid (~7-9 h)"
	@echo "    make grid_confirm   Optional 200-epoch confirmation (~2-3 h)"
	@echo ""
	@echo "  Housekeeping:"
	@echo "    make clean          Remove __pycache__/*.pyc and .DS_Store"
	@echo "    make tree           Print the top-level layout"

# ----------------------------------------------------------------------------
# Reproducibility wrappers
# ----------------------------------------------------------------------------
.PHONY: install smoke all data train robustness l_alpha w4 ensembles phase22 \
        ebro_informed physicalize phase22_acyclic grid grid_confirm

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) -c "import seq2seq_runoff, synth_simulator; print('imports OK')"

all data train robustness l_alpha w4 ensembles phase22 ebro_informed \
physicalize phase22_acyclic:
	PYTHON=$(PYTHON) bash scripts/run_paper_experiments.sh $@

grid:
	PYTHON=$(PYTHON) bash scripts/run_remediation_grid.sh
	$(PYTHON) scripts/summarize_remediation_grid.py \
	    --grid-root outputs/grid \
	    --output outputs/grid/grid_summary.csv

grid_confirm:
	PYTHON=$(PYTHON) bash scripts/run_grid_confirm_200ep.sh

# ----------------------------------------------------------------------------
# Housekeeping
# ----------------------------------------------------------------------------
.PHONY: clean tree

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc"      -type f -delete 2>/dev/null || true
	find . -name ".DS_Store"  -type f -delete 2>/dev/null || true

tree:
	@echo "hydrognn/" && \
	ls -F | sed -e 's|^|  |' && \
	echo "" && \
	du -sh */ 2>/dev/null | sort -h
