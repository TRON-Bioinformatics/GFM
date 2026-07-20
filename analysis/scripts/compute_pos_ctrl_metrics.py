import sys

import scanpy as sc

from gfm.helpers import SplitHandler, compute_pos_ctrl_metrics

adata_path = sys.argv[1]
split_dict_path = sys.argv[2]
results_path = sys.argv[3]

print("Starting evaluation...")
adata = sc.read_h5ad(adata_path, backed="r")
split_handler = SplitHandler(split_dict_path=split_dict_path)
split_handler.add_split_to_adata(adata)

pred_df = split_handler.split_df[split_handler.split_df["split"] == "test"]
_ = compute_pos_ctrl_metrics(
    adata,
    pred_df,
    control_type="pert_train",
    covariate_columns=split_handler.covariate_columns,
    top_n=20,
    n_jobs=32,
    save_path=results_path,
)
print(f"Metrics saved to {results_path}")
