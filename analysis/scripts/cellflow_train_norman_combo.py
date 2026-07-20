import functools
import os
import pickle
import sys

import anndata as ad
import cellflow
import numpy as np
import pandas as pd
import scanpy as sc
from cellflow.model import CellFlow
from cellflow.preprocessing import centered_pca, get_esm_embedding, project_pca, reconstruct_pca
from cellflow.utils import match_linear

adata_path = sys.argv[1]
split_path = sys.argv[2]
seed = sys.argv[3]
work_dir = os.path.dirname(adata_path)


def get_esm_embedding_adata(adata, adata_path_esm):
    get_esm_embedding(
        adata,
        gene_key=["gene_ensembl_1", "gene_ensembl_2"],
        gene_emb_key="esm_embeddings",
        null_value="control",
        esm_model_name="esm2_t36_3B_UR50D",
        use_cuda=True,
        cache_dir="../data/cellflow/esm_cache",
    )

    for key in adata.uns["esm_embeddings"].keys():
        adata.uns["esm_embeddings"][key] = adata.uns["esm_embeddings"][key].cpu().detach().numpy()

    # Create control embedding
    adata.uns["esm_embeddings"]["control"] = np.zeros_like(
        next(iter(adata.uns["esm_embeddings"].values()))
    )

    if "esm_embeddings_metadata" in adata.uns:
        meta = adata.uns["esm_embeddings_metadata"]
        for col in meta.columns:
            if meta[col].dtype == object:
                meta[col] = meta[col].fillna("").astype(str)
            elif meta[col].dtype == bool:
                meta[col] = meta[col].astype(int)
            else:
                meta[col] = meta[col].fillna(-1)
        adata.uns["esm_embeddings_metadata"] = meta
    adata.write_h5ad(adata_path_esm)


adata_path_esm = "_with_esm.".join(adata_path.split("."))
if os.path.exists(adata_path_esm):
    adata = sc.read_h5ad(adata_path_esm)
else:
    adata = sc.read_h5ad(adata_path)
    get_esm_embedding_adata(adata, adata_path_esm)

split_dict = pickle.load(open(split_path, "rb"))


adata_train = adata[adata.obs["condition"].isin(split_dict["train"])].copy()
all_conditions = list(split_dict["val"]) + ["ctrl"]
adata_val = adata[adata.obs["condition"].isin(all_conditions)].copy()

centered_pca(adata_train, method="scanpy", keep_centered_data=False, n_comps=100)
project_pca(adata_val, ref_adata=adata_train)

cf = CellFlow(adata_train, solver="otfm")

cf.prepare_data(
    sample_rep="X_pca",
    control_key="is_control",
    perturbation_covariates={"genetic_perturbation": ["gene_ensembl_1", "gene_ensembl_2"]},
    perturbation_covariate_reps={"genetic_perturbation": "esm_embeddings"},
)
# subset for validation
adatas_train_subsampled = []
for cond in adata_train.obs["condition"].unique():
    adata_tmp = adata_train[adata_train.obs["condition"] == cond]
    adatas_train_subsampled.append(
        sc.pp.subsample(adata_tmp, n_obs=min(adata_tmp.n_obs, 100), copy=True)
    )

adata_train_for_validation = ad.concat(adatas_train_subsampled)

adatas_val_subsampled = []
for cond in adata_val.obs["condition"].unique():
    adata_tmp = adata_val[adata_val.obs["condition"] == cond]
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

# metrics_callback = cellflow.training.Metrics(metrics=["mmd", "e_distance"])
# callbacks = [metrics_callback]
# cf.train(
#         num_iterations=300_000,
#         batch_size=2048,
#         callbacks=callbacks,
#         valid_freq=60_000
#         )
# cf.save(
#     dir_path=work_dir,
#     file_prefix=f"model_{seed}",
#     overwrite=True
# )

# Prediction
cf = cellflow.model.CellFlow.load(filename=f"{work_dir}/model_{seed}_CellFlow.pkl")

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

preds = cf.predict(
    adata=adata_ctrl_for_prediction,
    sample_rep="X_pca",
    condition_id_key="condition",
    covariate_data=covariate_data,
)

adata_preds = []
for gene_names, array in preds.items():
    cond = (
        covariate_data["condition"][
            (covariate_data["gene_ensembl_1"] == gene_names[0])
            & (covariate_data["gene_ensembl_2"] == gene_names[1])
        ]
        .unique()
        .tolist()[0]
    )
    obs_data = pd.DataFrame({"condition": [cond] * array.shape[0]})
    adata_pred = ad.AnnData(X=np.empty((len(array), adata_train.n_vars)), obs=obs_data)
    adata_pred.obsm["X_pca"] = np.squeeze(array)
    adata_preds.append(adata_pred)

adata_preds = ad.concat(adata_preds)
adata_preds.var_names = adata_train.var_names

reconstruct_pca(adata_preds, use_rep="X_pca", ref_adata=adata_train)
adata_preds.X = adata_preds.layers["X_recon"]
adata_preds.write_h5ad(f"{work_dir}/adata_pred_{seed}.h5ad")
