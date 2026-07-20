import json
import os
import sys
import time

import numpy as np
import scanpy as sc
import torch
from gears import GEARS, PertData

pert_data_dir = sys.argv[1]
seed = sys.argv[2]
split_path = sys.argv[3]
save_dir = sys.argv[4]
model_save_dir = os.path.join(save_dir, f"model_{seed}")
proj_name = f"gears_{seed}"

gpu = torch.cuda.current_device()
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(device)

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

pert_data = PertData(data_path=os.path.dirname(pert_data_dir))
train_start = time.time()
pert_data.load(data_path=pert_data_dir)

# only single perturbation
pert_data.prepare_split(split_dict_path=split_path, split="custom", seed=seed)
# make sure that the conditions not in the graph are filtered out of the training set
pert_data.set2conditions["train"] = [
    cond
    for cond in pert_data.set2conditions["train"]
    if cond in pert_data.adata.obs["condition"].unique()
]
pert_data.set2conditions["val"] = [
    cond
    for cond in pert_data.set2conditions["val"]
    if cond in pert_data.adata.obs["condition"].unique()
]
pert_data.set2conditions["test"] = [
    cond
    for cond in pert_data.set2conditions["test"]
    if cond in pert_data.adata.obs["condition"].unique()
]
pert_data.get_dataloader(batch_size=32, test_batch_size=128)


gears_model = GEARS(
    pert_data, device=device, weight_bias_track=False, proj_name=proj_name, exp_name=proj_name
)
gears_model.model_initialize()
gears_model.train()

train_time = time.time() - train_start
print(f"Training time: {train_time:.2f}s")

gears_model.save_model(model_save_dir)


def predict(model, pert_list, pool_size=100):
    """
    Predict the transcriptome given a list of genes/gene combinations being
    perturbed

    Parameters
    ----------
    pert_list: list
        list of genes/gene combiantions to be perturbed

    Returns
    -------
    results_pred: dict
        dictionary of predicted transcriptome
    results_logvar: dict
        dictionary of uncertainty score

    """
    ## given a list of single/combo genes, return the transcriptome
    ## if uncertainty mode is on, also return uncertainty score.

    from gears.utils import create_cell_graph_dataset_for_prediction
    from torch_geometric.loader import DataLoader

    model.ctrl_adata = model.adata[model.adata.obs["condition"] == "ctrl"]
    pert_list_filtered = []
    for perturbation_genes in pert_list:
        pert = []
        for gene in perturbation_genes:
            if gene in gears_model.pert_list:
                pert.append(gene)
        if pert:
            pert_list_filtered.append(pert)

    model.best_model = model.best_model.to(model.device)
    model.best_model.eval()

    preds_list = []
    conds_list = []
    for pert in pert_list_filtered:
        cg = create_cell_graph_dataset_for_prediction(
            pert, model.ctrl_adata, model.pert_list, model.device
        )
        loader = DataLoader(cg, pool_size, shuffle=False)
        batch = next(iter(loader))
        batch.to(model.device)

        with torch.no_grad():
            preds = model.best_model(batch)

            preds_list.append(preds.cpu().numpy())
            pert_name = "+".join(pert)
            conds_list.extend([pert_name] * preds.shape[0])

    adata_pred = sc.AnnData(np.vstack(preds_list), obs={"condition": conds_list})
    adata_pred.var_names = model.adata.var["gene_name"].tolist()

    return adata_pred


conds = pert_data.set2conditions["test"]
split_conds = [x.split("+") for x in conds]
split_conds = [list(filter(lambda y: y != "ctrl", x)) for x in split_conds]

infer_start = time.time()
adata_pred = predict(gears_model, split_conds)
infer_time = time.time() - infer_start
print(f"Inference time: {infer_time:.2f}s")

timing = {"train_time_s": train_time, "inference_time_s": infer_time}
with open(os.path.join(save_dir, f"timing_{seed}.json"), "w") as f:
    json.dump(timing, f, indent=2)
print(f"Timing saved to {os.path.join(save_dir, f'timing_{seed}.json')}")

adata_pred.write(f"{save_dir}/adata_pred_{seed}.h5ad")
