import pickle
import sys
import warnings
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

save_dir = sys.argv[1]

sc.settings.datasetdir = save_dir

DATASET_FILENAME = "replogle_2022_k562_essential.h5ad"
DATASET_URL = "https://exampledata.scverse.org/pertpy/replogle_2022_k562_essential.h5ad"


def load_replogle_essential_adata(save_dir):
    dataset_path = Path(save_dir) / DATASET_FILENAME

    if dataset_path.exists():
        return sc.read_h5ad(dataset_path)

    try:
        return pt.data.replogle_2022_k562_essential()
    except FileNotFoundError as exc:
        if Path(exc.filename or "").name != f"{DATASET_FILENAME}.lock":
            raise

        if not dataset_path.exists():
            urlretrieve(DATASET_URL, dataset_path)

        return sc.read_h5ad(dataset_path)


adata = load_replogle_essential_adata(save_dir)
adata

supp2_ess_k562 = pd.read_excel(
    f"{save_dir}/1-s2.0-S0092867422005979-mmc2.xlsx", sheet_name="TabB_K562_day6_summary_stat"
)


def get_strong_perts(supp):
    filtered = supp[supp["Number of DEGs (anderson-darling)"] > 50]
    filtered = filtered[filtered["percent knockdown"] <= -0.3]
    filtered = filtered[filtered["number of cells (filtered)"] > 25]
    strong_perts = filtered["genetic perturbation"].values
    strong_perts = [s.split("_")[1] for s in strong_perts]
    return strong_perts


strong_perts_ess_k562 = get_strong_perts(supp2_ess_k562)

strong_perts_ess_k562 = strong_perts_ess_k562 + ["non-targeting"]
pert_filter_k562 = adata[adata.obs["gene"].isin(strong_perts_ess_k562)]

n_perts = pert_filter_k562.obs["perturbation"].nunique()
print(f"n perts in k562: {n_perts}")


def filter_cells_by_pert_effect(adata, k=10):
    perc_underk = []
    subset_idxs = []
    ctrl_adata = adata[adata.obs["gene"] == "non-targeting"]

    for g in adata.obs["gene"].unique():
        subset = adata[adata.obs["gene"] == g]

        if g == "non-targeting":
            subset_idxs.append(subset.obs.index.values)
            continue

        try:
            gene_loc = np.where(adata.var_names == g)[0][0]
            thresh = np.percentile(ctrl_adata.X[:, gene_loc], k)
            perc_underk.append(sum(subset.X[:, gene_loc] > thresh))

            subset_idxs.append(subset.obs.index[subset.X[:, gene_loc] <= thresh].values)
        except IndexError:
            subset_idxs.append(subset.obs.index.values)

    subset_idxs = [item for sublist in subset_idxs for item in sublist]
    filtered_adata = adata[subset_idxs, :]

    return perc_underk, filtered_adata


perc_underk_ess_k562, adata = filter_cells_by_pert_effect(pert_filter_k562)


def reformat_adata(adata, cell_type="K562"):
    adata.obs = adata.obs.rename(columns={"gene": "condition"})
    adata.obs["condition"] = [c + "+ctrl" for c in adata.obs["condition"]]
    adata.obs["cell_type"] = cell_type
    adata.obs = adata.obs.loc[:, ["condition", "cell_type"]]

    mapper = {k: k for k in adata.obs["condition"].unique()}
    mapper["non-targeting+ctrl"] = "ctrl"
    adata.obs["condition"] = adata.obs["condition"].map(mapper)

    adata.var["gene_name"] = adata.var.index

    return adata


cell_type = "K562"
print("Reformatting adata...")
adata = reformat_adata(adata, cell_type=cell_type)
# make a small version of the dataset for testing
sel_ind = [0, 1, 2, 100, 300, 600, 900, 1000, 1090, 1092]
sel_cond = adata.obs["condition"].value_counts().index[sel_ind].tolist()
adata_small = adata[adata.obs["condition"].isin(sel_cond)].copy()

# subset to 1000 ctrl cells
ctrl_idx = (
    adata_small.obs.loc[adata_small.obs["condition"] == "ctrl"]
    .sample(n=min(1000, (adata_small.obs["condition"] == "ctrl").sum()), random_state=42)
    .index
)

adata_small = adata_small[
    (adata_small.obs["condition"] != "ctrl") | adata_small.obs.index.isin(ctrl_idx)
].copy()

conditions = adata_small.obs["condition"].value_counts().index.tolist()
non_ctrl_conditions = [cond for cond in conditions if cond != "ctrl"]
split_dict = {
    "train": ["ctrl", *non_ctrl_conditions[:-4]],
    "val": [non_ctrl_conditions[-4]],
    "test": non_ctrl_conditions[-3:],
}
print("Dataset split dictionary:")
print(split_dict)


def preprocess_adata(adata):
    adata.X = csr_matrix(adata.X)

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # keep top 5000 hvgs and perturbed genes
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=False)

    conditions = [(c.split("+")[0], c.split("+")[1]) for c in adata.obs["condition"] if "+" in c]
    conditions = [item for sublist in conditions for item in sublist]
    genes_to_keep = np.unique(conditions)

    adata.var["highly_variable"] = adata.var["highly_variable"] + adata.var_names.isin(
        genes_to_keep
    )
    adata = adata[:, adata.var["highly_variable"]]

    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    return adata


def rank_genes_groups_list(
    adata,
    groupby="condition",
    reference="ctrl",
    method="wilcoxon",
    key_added="rank_genes_groups_list",
):
    adata_copy = adata.copy()

    sc.tl.rank_genes_groups(
        adata_copy, groupby=groupby, reference=reference, method=method, rankby_abs=True
    )

    gene_dict = {}
    de_genes = pd.DataFrame(adata_copy.uns["rank_genes_groups"]["names"])
    for cond in de_genes.columns:
        gene_dict[cond] = de_genes[cond].tolist()

    adata.uns[key_added] = gene_dict


dataset_name = "replogle_k562"

print("Preprocessing adata...")
adata_small = preprocess_adata(adata_small)
rank_genes_groups_list(adata_small, groupby="condition", reference="ctrl", method="wilcoxon")

adata_small.write_h5ad(f"{save_dir}/preprocessed_{dataset_name}_small.h5ad")
with open(f"{save_dir}/replogle_k562_small_split_dict.pkl", "wb") as handle:
    pickle.dump(split_dict, handle)
