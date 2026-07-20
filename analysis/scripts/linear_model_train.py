#!/usr/bin/env python3
import os
import sys

import scanpy as sc
import torch

from gfm.helpers import SplitHandler, compute_metrics, train_and_predict_linear_model

device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
print(device)

adata_path = sys.argv[1]
split_dict_path = sys.argv[2]
working_dir = sys.argv[3]
seed = sys.argv[4]
try:
    is_drug = sys.argv[5].lower() == "true"
except IndexError:
    is_drug = False

if not os.path.exists(working_dir):
    os.makedirs(working_dir, exist_ok=True)

adata = sc.read_h5ad(adata_path, backed="r")
split_handler = SplitHandler(split_dict_path=split_dict_path)
split_handler.add_split_to_adata(adata)

adata_pred_lm = train_and_predict_linear_model(adata, drug_graph=is_drug, device=device)
adata_pred_lm.write_h5ad(os.path.join(working_dir, f"adata_pred_{seed}.h5ad"))

pred_df = split_handler.split_df[split_handler.split_df["split"] == "test"]
results_path = os.path.join(working_dir, f"results_test_{seed}.csv")
df = compute_metrics(
    adata,
    adata_pred_lm,
    pred_df,
    control_type="pert_train",
    covariate_columns=split_handler.covariate_columns,
    top_n=20,
    n_jobs=32,
    is_drug=is_drug,
    save_path=None,
)
df["w2d"] = None
df["mmd"] = None
df["e_distance"] = None
df.to_csv(results_path, index=False)
