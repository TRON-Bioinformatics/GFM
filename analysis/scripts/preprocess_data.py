import sys

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

adata_path = sys.argv[1]
cell_type = sys.argv[2]
dataset_name = sys.argv[3]


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


print("Reading adata...")
adata = sc.read_h5ad(adata_path)
print("Reformatting adata...")
adata = reformat_adata(adata, cell_type=cell_type)
print("Preprocessing adata...")
adata = preprocess_adata(adata)
rank_genes_groups_list(adata, groupby="condition", reference="ctrl", method="wilcoxon")

adata.write_h5ad(f"./data/preprocessed_{dataset_name}.h5ad")
