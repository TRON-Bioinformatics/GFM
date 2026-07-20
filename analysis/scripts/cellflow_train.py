#!/usr/bin/env python3
import functools
import json
import os
import pickle
import sys
import time

import anndata as ad
import cellflow
import numpy as np
import pandas as pd
import scanpy as sc
from cellflow.model import CellFlow
from cellflow.preprocessing import (
    centered_pca,
    project_pca,
    reconstruct_pca,
)
from cellflow.utils import match_linear

adata_dir_esm = sys.argv[1]
split_dir = sys.argv[2]
seed = int(sys.argv[3])
working_dir = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(adata_dir_esm)
if not os.path.exists(working_dir):
    os.makedirs(working_dir, exist_ok=True)

adata = sc.read_h5ad(adata_dir_esm)

train_start = time.time()

split_dict = pickle.load(open(split_dir, "rb"))

adata_train = adata[adata.obs["condition"].isin(split_dict["train"])].copy()

all_conditions = list(split_dict["val"]) + ["ctrl"]
adata_val = adata[adata.obs["condition"].isin(all_conditions)].copy()

centered_pca(adata_train, method="scanpy", keep_centered_data=False, n_comps=100)
project_pca(adata_val, ref_adata=adata_train)


cf = CellFlow(adata_train, solver="otfm")
cf.prepare_data(
    sample_rep="X_pca",
    control_key="is_control",
    perturbation_covariates={"genetic_perturbation": ["gene_ensembl"]},
    perturbation_covariate_reps={"genetic_perturbation": "esm_embeddings"},
    max_combination_length=1,
)
# subsample for validation
adatas_train_subsampled = []
for cond in adata_train.obs["gene_ensembl"].unique():
    adata_tmp = adata_train[adata_train.obs["gene_ensembl"] == cond]
    adatas_train_subsampled.append(
        sc.pp.subsample(adata_tmp, n_obs=min(adata_tmp.n_obs, 100), copy=True)
    )

adata_train_for_validation = ad.concat(adatas_train_subsampled)

adatas_val_subsampled = []
for cond in adata_val.obs["gene_ensembl"].unique():
    adata_tmp = adata_val[adata_val.obs["gene_ensembl"] == cond]
    adatas_val_subsampled.append(
        sc.pp.subsample(adata_tmp, n_obs=min(adata_tmp.n_obs, 100), copy=True)
    )

adata_val_for_validation = ad.concat(adatas_val_subsampled)
adata_train_for_validation.uns = adata_train.uns.copy()
adata_val_for_validation.uns = adata_val.uns.copy()

cf.prepare_validation_data(
    adata_train_for_validation,
    name="train",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)

cf.prepare_validation_data(
    adata_val_for_validation,
    name="val",
    n_conditions_on_log_iteration=None,
    n_conditions_on_train_end=None,
)
layers_before_pool = {
    "genetic_perturbation": {"layer_type": "mlp", "dims": [512, 512], "dropout_rate": 0.0}
}

layers_after_pool = {
    "layer_type": "mlp",
    "dims": [1024, 1024],
    "dropout_rate": 0.0,
}
match_fn = functools.partial(match_linear, epsilon=0.5, tau_a=1.0, tau_b=1.0)
cf.prepare_model(
    condition_mode="deterministic",
    regularization=0.0,
    pooling="attention_token",
    layers_before_pool=layers_before_pool,
    layers_after_pool=layers_after_pool,
    condition_embedding_dim=256,
    cond_output_dropout=0.9,
    hidden_dims=[2048, 2048, 2048],
    decoder_dims=[4096, 4096, 4096],
    probability_path={"constant_noise": 0.5},
    match_fn=match_fn,
)
metrics_callback = cellflow.training.Metrics(metrics=["mmd", "e_distance"])
callbacks = [metrics_callback]
cf.train(num_iterations=300_000, batch_size=2048, callbacks=callbacks, valid_freq=60_000)

train_time = time.time() - train_start
print(f"Training time: {train_time:.2f}s")

cf.save(dir_path=working_dir, file_prefix=f"model_{seed}", overwrite=True)

cf = cellflow.model.CellFlow.load(filename=f"{working_dir}/model_{seed}_CellFlow.pkl")

all_conditions = list(split_dict["test"]) + ["ctrl"]
adata_test = adata[adata.obs["condition"].isin(all_conditions)].copy()
project_pca(adata_test, ref_adata=adata_train)

cond_trim = []
for cond in adata_test.obs["condition"]:
    condition_parts = cond.split("+")
    new_l = []
    for gene in condition_parts:
        if gene != "ctrl":
            new_l.append(gene)
    new_cond = "+".join(new_l)
    cond_trim.append(new_cond)
adata_test.obs["condition"] = cond_trim

adata_ctrl_for_prediction = adata_test[(adata_test.obs["is_control"].to_numpy())].copy()
n = 100  # specify the number of cells to select
adata_ctrl_for_prediction = adata_ctrl_for_prediction[
    np.random.choice(adata_ctrl_for_prediction.obs_names, n, replace=False)
].copy()
covariate_data = (
    adata_test[~adata_test.obs["is_control"].to_numpy()]
    .obs.drop_duplicates(subset=["condition"])
    .copy()
)

infer_start = time.time()
preds = cf.predict(
    adata=adata_ctrl_for_prediction,
    sample_rep="X_pca",
    condition_id_key="condition",
    covariate_data=covariate_data,
)
adata_preds = []
for cond, array in preds.items():
    obs_data = pd.DataFrame({"condition": [cond] * array.shape[0]})
    adata_pred = ad.AnnData(X=np.empty((len(array), adata_train.n_vars)), obs=obs_data)
    adata_pred.obsm["X_pca"] = np.squeeze(array)
    adata_preds.append(adata_pred)

adata_preds = ad.concat(adata_preds)
adata_preds.var_names = adata_train.var_names

reconstruct_pca(adata_preds, use_rep="X_pca", ref_adata=adata_train)
adata_preds.X = adata_preds.layers["X_recon"]

infer_time = time.time() - infer_start
print(f"Inference time: {infer_time:.2f}s")

adata_preds.write_h5ad(f"{working_dir}/adata_pred_{seed}.h5ad")
print("Saved predicted adata")

timing = {"train_time_s": train_time, "inference_time_s": infer_time}
with open(os.path.join(working_dir, f"timing_{seed}.json"), "w") as f:
    json.dump(timing, f, indent=2)
print(f"Timing saved to {os.path.join(working_dir, f'timing_{seed}.json')}")
