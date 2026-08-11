# Analysis Directory

This directory contains:

- `data/` stores input data and output files for the analysis.
- `scripts/` contains Python scripts and environment files.
- `notebooks/` contains analysis notebooks used to reproduce the results in the manuscript.

## Notebook Guide

### Manuscript reproduction notebooks

- `notebooks/ablation_experiment.ipynb`: Runs model ablation studies, including disabling flow matching and testing randomized graph structure variants.
- `notebooks/baseline.ipynb`: Runs baseline methods, including a linear model across datasets and an additive model analysis for Norman combo perturbations.
- `notebooks/cellflow.ipynb`: Reproduces the CellFlow workflow, including ESM2 embedding preparation and downstream training or prediction steps.
- `notebooks/data_preprocessing.ipynb`: Preprocesses raw perturbation datasets.
- `notebooks/evaluation.ipynb`: Computes evaluation metrics for model predictions and baseline comparisons across datasets and splits.
- `notebooks/gears.ipynb`: Reproduces the GEARS workflow.
- `notebooks/gfm.ipynb`: Runs the main GFM training, prediction, and metric computation workflow used in the manuscript experiments.
- `notebooks/graph_connectivity.ipynb`: Inspect the correlation of graph connectivity with GFM predictive performance.
- `notebooks/replogle_k562_gwps.ipynb`: Train GFM with Replogle K562 gwps dataset, predict the entire perturbation space, and shortlist therapeutic targets associated with HSP90.
- `notebooks/sample_size_test.ipynb`: Creates reduced-training-size splits to measure how performance changes with smaller training sets.
- `notebooks/scgpt.ipynb`: Reproduces the scGPT workflow.