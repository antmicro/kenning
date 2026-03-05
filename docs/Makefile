# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

source/generated/AnomalyDetectionDatasetInfo.md:
	mkdir -p source/generated
	PYTHONPATH="${PYTHONPATH}:$$(realpath ..)" python3 -m kenning info AnomalyDetectionDataset -md --docstrings > $@

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: source/generated/AnomalyDetectionDatasetInfo.md Makefile
# Normally when Kenning HTML report is generated, the pages are in a subfolder matching the report name, while zephyr_traces_report.html trace file (which is linked in them) is
# placed alongside the .md report file. Therefore in the reports path to the trace file is given as ../zephyr_traces_report.html. In the docs folder structure is different, so
# we need to adjust by changing the path.
	@if [ -f "source/generated/sample-zephyr-tracing.md" ]; then sed -i 's|../zephyr_traces_report.html|zephyr_traces_report.html|g' source/generated/sample-zephyr-tracing.md; fi
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
