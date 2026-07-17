# GFM

Graph Flow Matching (GFM) for single-cell perturbation prediction.

This package trains a latent-space flow model conditioned on perturbations to generate perturbed single-cell gene expression profiles.

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

Prepare the input `Anndata` and the data split file. Place the graph data under the graph directory (here we use `analysis/data/` as an example):

- `pert_graph_edge_index_20.pt`
- `pert_graph_edge_list_20.csv`
- `pert_graph_edge_weight_20.pt`
- `pert_graph_pert_names_graph_20.npy`

### 2. Run training, prediction and evaluation

```bash
cd analysis/

adata_path="./data/preprocessed_replogle_k562.h5ad"
split_dict_path="./data/replogle_k562_single_1_0.75.pkl"
graph_dir="./data"

uv run python ./gfm_run_all.py \
    --adata-path "${adata_path}" \
    --split-dict-path "${split_dict_path}" \
    --output-dir "./output/" \
    --graph-type "go+pert+ppi" \
    --graph-dir "${graph_dir}"
```

This command trains GFM using the `go+pert+ppi` graph configuration and writes outputs to `--output-dir`.

