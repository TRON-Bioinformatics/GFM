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

Place these files under `analysis/data/` (matching the paths below):

- `gene2go_all.pkl`
- `essential_all_data_pert_genes.pkl`
- `go_essential_all/go_essential_all.csv`
- `9606.protein.links.v12.0_with_gene_names.csv`
- `preprocessed_replogle_2022_k562_gwps.h5ad`
- `preprocessed_replogle_k562.h5ad`
- `replogle_k562_single_1_0.75.pkl`

### 2. Run training

```bash
cd analysis/

adata_path="./data/preprocessed_replogle_k562.h5ad"
pert_adata_path="./data/preprocessed_replogle_2022_k562_gwps.h5ad"
split_dict_path="./data/replogle_k562_single_1_0.75.pkl"

uv run python ./gfm_run_all.py \
    --adata-path "${adata_path}" \
    --split-dict-path "${split_dict_path}" \
    --working-dir "./output/" \
    --graph-type "go+pert+ppi" \
    --pert-adata-path "${pert_adata_path}"
```

This command trains GFM using the `go+pert+ppi` graph configuration and writes outputs to `--working-dir`.

