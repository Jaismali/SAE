# Concept Selection - Month 1 (Historical)

This folder contains Month 1's original concept-selection pipeline work.

As of Month 2, the following files were consolidated into a single
canonical location and REMOVED from this folder to eliminate drift risk:
  - contract.py
  - monosemanticity_filter.py
  - deduplication.py

These now live ONLY in ../concept_selection_v2/, which is the active,
canonical location going forward. It also contains everything built
in Month 2: layer_selection.py, real_backend.py, activation_statistics.py,
structural_token_filter.py, concept_filter_pipeline.py, manifest_freeze.py.

Original copies remain accessible via git history if ever needed.
