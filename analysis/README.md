# Analysis Directory

This directory contains two kinds of materials:

- `quickstart.ipynb` is a minimal end-to-end example of running GFM on a small dataset.
- `notebooks/` contains analysis notebooks used to reproduce the results in the manuscript.

## Notebook Guide

### Top-level notebook

- `quickstart.ipynb`: Minimal GFM usage example that runs training, prediction, and a small visualization workflow on a reduced Replogle K562 dataset.

### Manuscript reproduction notebooks

- `notebooks/ablation_experiment.ipynb`: Runs model ablation studies, including disabling flow matching and testing randomized graph structure variants.
- `notebooks/baseline.ipynb`: Runs baseline methods, including a linear model across datasets and an additive model analysis for Norman combo perturbations.
- `notebooks/cellflow.ipynb`: Reproduces the CellFlow baseline workflow, including ESM2 embedding preparation and downstream training or prediction steps.
- `notebooks/data_preprocessing.ipynb`: Preprocesses raw perturbation datasets.
- `notebooks/evaluation.ipynb`: Computes evaluation metrics for model predictions and baseline comparisons across datasets and splits.
- `notebooks/gears.ipynb`: Reproduces the GEARS baseline workflow.
- `notebooks/gfm.ipynb`: Runs the main GFM training, prediction, and metric computation workflow used in the manuscript experiments.
- `notebooks/graph_connectivity.ipynb`: Inspect the correlation of graph connectivity with GFM predictive performance.
- `notebooks/replogle_k562_gwps.ipynb`: Train GFM with Replogle K562 gwps dataset, predict the entire perturbation space, and shortlist therapeutic targets associated with HSP90.
- `notebooks/sample_size_test.ipynb`: Creates reduced-training-size splits to measure how performance changes with smaller training sets.
- `notebooks/scgpt.ipynb`: Reproduces the scGPT baseline workflow.

## Related directories

- `data/`: Input datasets, processed AnnData files, model outputs, and plotting artifacts used by the notebooks.
- `scripts/`: Python scripts and environment files invoked from the notebooks.