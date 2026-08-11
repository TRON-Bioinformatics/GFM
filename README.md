# GFM

Graph-guided Flow Matching (GFM) for single-cell perturbation prediction. This package trains a latent flow matching model conditioned on the knowledge graph-derived perturbation embeddings to generate perturbed single-cell gene expression profiles.

## Requirements

- Python 3.10+
- PyTorch-compatible environment (CPU or CUDA)

## Installation

Recommended (`uv`):

```bash
uv sync
```

Then run project commands with `uv run`.

Alternative (editable install with pip):

```bash
pip install -e .
```

Core dependencies are defined in `pyproject.toml`.

## Quick Usage
Use the steps below to run training on the Replogle K562 dataset.

### 1. Prepare required data files

Prepare the input single-cell adata and the corresponding data split file. Run `get_example_data.py` to obtain a smaller version of Replogle K562 adate and the data split file. Make sure that `analysis/data/` contains `1-s2.0-S0092867422005979-mmc2.xlsx` for preprocessing, and `analysis/data/graph_data` exists for GFM training.

```bash
cd analysis/

save_dir="./data"

uv run python ./scripts/get_example_data.py "${save_dir}"
```

### 2. Run training, prediction and evaluation

Train GFM using the GO, PPI, and the perturbation graphs.

```bash
adata_path="./data/preprocessed_replogle_k562_small.h5ad"
split_dict_path="./data/replogle_k562_small_split_dict.pkl"
graph_dir="./data/graph_data/"
output_dir="./data/output/"

uv run python ./scripts/gfm_run_all.py \
    --adata-path "${adata_path}" \
    --split-dict-path "${split_dict_path}" \
    --output-dir "${output_dir}" \
    --graph-type "go+pert+ppi" \
    --graph-dir "${graph_dir}"
```

The `--output-dir` should contain the trained GFM modules (including `vae.pt` and `gfm.pt`), the predicted single-cell gene expression of the held-out perturbations `adata_pred_gfm.h5ad`, and the metric evaluation results `results_test_gfm.csv`.

## Reproducibility

To reproduce the analysis in the GFM manuscript, please check out the notebooks in `analysis/notebooks/`.

