import sys

import numpy as np
import scanpy as sc
import torch

from gfm.helpers import SplitHandler, compute_metrics

adata_path = sys.argv[1]
adata_pred_path = sys.argv[2]
split_dict_path = sys.argv[3]
results_path = sys.argv[4]

print("Starting evaluation...")
adata = sc.read_h5ad(adata_path, backed="r")
adata_pred = sc.read_h5ad(adata_pred_path, backed="r")
split_handler = SplitHandler(split_dict_path=split_dict_path)
split_handler.add_split_to_adata(adata)

if split_handler.split_df is None:
    raise ValueError("SplitHandler did not load split_df.")
pred_df = split_handler.split_df[split_handler.split_df["split"] == "test"]

np.random.seed(42)
torch.manual_seed(42)
_ = compute_metrics(
    adata,
    adata_pred,
    pred_df,
    control_type="pert_train",
    covariate_columns=split_handler.covariate_columns,
    top_n=20,
    n_jobs=32,
    save_path=results_path,
)
print(f"Metrics saved to {results_path}")
