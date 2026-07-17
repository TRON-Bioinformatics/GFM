import gzip
import os
import pickle
import shutil
import tarfile
import warnings

import mygene
import numpy as np
import ot
import pandas as pd
import requests
import scanpy as sc
import torch
import torch.nn as nn
import umap
from scipy import stats
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from gfm.vae_training_utils import make_vae_dataloader

GEARS_GENE2GO_ALL_URL = "https://dataverse.harvard.edu/api/access/datafile/6153417"
GEARS_ESSENTIAL_GENES_URL = "https://dataverse.harvard.edu/api/access/datafile/6934320"
GEARS_GO_ESSENTIAL_ALL_TAR_URL = "https://dataverse.harvard.edu/api/access/datafile/6934319"


def _download_file_if_missing(url, save_path):
    """Download a file only if it is not already present on disk."""
    if os.path.exists(save_path):
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading missing artifact: {os.path.basename(save_path)}")

    # Dataverse can reject default clients; emulate GEARS' requests-based behavior
    # and keep a fallback URL variant.
    candidate_urls = [url]
    if "?" not in url:
        candidate_urls.append(f"{url}?format=original")

    headers = {
        "User-Agent": "GEARS-compatible-downloader/1.0",
        "Accept": "*/*",
    }
    last_error = None

    for candidate_url in candidate_urls:
        try:
            response = requests.get(candidate_url, stream=True, headers=headers, timeout=120)
            response.raise_for_status()
            with open(save_path, "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
            return
        except requests.HTTPError as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Failed to download artifact to {save_path}. Last error: {last_error}")


def _download_and_extract_tar(url, tar_save_path, extract_dir):
    """Download a .tar.gz archive and extract it into extract_dir."""
    _download_file_if_missing(url, tar_save_path)
    print(f"Extracting archive: {os.path.basename(tar_save_path)}")
    with tarfile.open(tar_save_path) as tar:
        tar.extractall(path=extract_dir)


def _ensure_go_graph_artifacts(graph_dir):
    """Ensure GEARS GO graph artifacts exist locally, downloading from source if needed."""
    gene2go_path = os.path.join(graph_dir, "gene2go_all.pkl")
    essential_path = os.path.join(graph_dir, "essential_all_data_pert_genes.pkl")
    go_csv_path = os.path.join(graph_dir, "go_essential_all", "go_essential_all.csv")

    _download_file_if_missing(GEARS_GENE2GO_ALL_URL, gene2go_path)
    _download_file_if_missing(GEARS_ESSENTIAL_GENES_URL, essential_path)

    if not os.path.exists(go_csv_path):
        os.makedirs(graph_dir, exist_ok=True)
        tar_save_path = os.path.join(graph_dir, "go_essential_all.tar.gz")
        _download_and_extract_tar(GEARS_GO_ESSENTIAL_ALL_TAR_URL, tar_save_path, graph_dir)

    if not os.path.exists(go_csv_path):
        raise FileNotFoundError(
            f"Missing required GO artifact after download/extraction: {go_csv_path}"
        )


class SplitHandler:
    def __init__(self, split_dict_path=None, split_dict=None, split_df_path=None, split_df=None):
        # Check that exactly one parameter is provided
        provided_params = sum(
            [
                split_dict_path is not None,
                split_dict is not None,
                split_df_path is not None,
                split_df is not None,
            ]
        )

        if provided_params == 0:
            raise ValueError(
                "Please provide exactly one of: split_dict_path, split_dict, split_df_path, or split_df."
            )
        elif provided_params > 1:
            raise ValueError(
                "Please provide only one of: split_dict_path, split_dict, split_df_path, or split_df."
            )

        self.split_dict_path = split_dict_path
        self.split_dict = split_dict
        self.split_df_path = split_df_path
        self.split_df = split_df

        # Load from paths if provided
        if split_dict_path is not None:
            if os.path.exists(split_dict_path):
                print(f"Loading split_dict from {split_dict_path}")
                self.split_dict = pickle.load(open(split_dict_path, "rb"))
            else:
                raise FileNotFoundError(f"Split file not found: {split_dict_path}")

        if self.split_dict is not None:
            if "train" not in self.split_dict:
                raise ValueError("split_dict must contain at least 'train' key.")

            # Create split_df from split_dict
            rows = []
            for split_name, conditions in self.split_dict.items():
                for condition in conditions:
                    rows.append({"condition": condition, "split": split_name})

            self.split_df = pd.DataFrame(rows)

        if split_df_path is not None:
            if os.path.exists(split_df_path):
                print(f"Loading split_df from {split_df_path}")
                if split_df_path.endswith(".csv"):
                    self.split_df = pd.read_csv(split_df_path)
                elif split_df_path.endswith(".tsv"):
                    self.split_df = pd.read_csv(split_df_path, sep="\t")
                elif split_df_path.endswith(".pkl") or split_df_path.endswith(".pickle"):
                    self.split_df = pd.read_pickle(split_df_path)
                else:
                    raise ValueError("Unsupported file format. Use .csv, .tsv, .pkl, or .pickle")
            else:
                raise FileNotFoundError(f"Split file not found: {split_df_path}")

        # Validate and set covariate_columns from split_df
        if self.split_df is not None:
            if "split" not in self.split_df.columns:
                raise ValueError("split_df must contain a 'split' column.")
            if "condition" not in self.split_df.columns:
                raise ValueError("split_df must contain a 'condition' column.")

            # Covariate columns are all columns except 'split'
            self.covariate_columns = [col for col in self.split_df.columns if col != "split"]

            # Store the split names
            self.keys = self.split_df["split"].unique().tolist()
        else:
            self.covariate_columns = []
            self.keys = []

    def add_split_to_adata(self, adata):
        """Add split assignments to adata.obs based on covariate columns.

        Parameters
        ----------
        adata : AnnData
            Annotated data object to add split column to

        Raises
        ------
        ValueError
            If split_df is None or if required covariate columns are missing from adata.obs
        """
        if self.split_df is None:
            raise ValueError("Split data not found. Please provide a valid split parameter.")

        # Validate that all covariate columns exist in adata.obs
        missing_cols = [col for col in self.covariate_columns if col not in adata.obs.columns]
        if missing_cols:
            raise ValueError(
                f"Covariate columns {missing_cols} not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        # Use pandas merge to add split column
        # Create a temporary DataFrame from adata.obs with an index column to preserve order
        obs_df = pd.DataFrame(adata.obs[self.covariate_columns])
        obs_df["_temp_index"] = range(len(obs_df))

        # Ensure both DataFrames have the same dtypes for merge columns
        for col in self.covariate_columns:
            obs_df[col] = obs_df[col].astype(str)
            self.split_df[col] = self.split_df[col].astype(str)

        # Merge with split_df on covariate columns
        merged = obs_df.merge(
            self.split_df[self.covariate_columns + ["split"]], on=self.covariate_columns, how="left"
        )

        # Check if merge was successful
        if "split" not in merged.columns:
            raise ValueError(
                f"Merge failed - 'split' column not found in result. "
                f"Available columns: {list(merged.columns)}"
            )

        # Sort back to original order and extract split column
        merged = merged.sort_values("_temp_index")
        adata.obs["split"] = merged["split"].values

    def get_condition(self, split="test"):
        """Get conditions for a given split.

        Parameters
        ----------
        split : str
            Split name (e.g., 'train', 'val', 'test')

        Returns
        -------
        list
            For single covariate: list of condition values
            For multiple covariates: list of tuples of covariate values
        """
        if self.split_df is None:
            raise ValueError("Split data not found.")

        split_subset = self.split_df[self.split_df["split"] == split]

        if len(self.covariate_columns) == 1:
            # Return list of conditions for single covariate
            return split_subset[self.covariate_columns[0]].tolist()
        else:
            # Return list of tuples for multiple covariates
            return [tuple(row) for row in split_subset[self.covariate_columns].values]


def sample_x0_from_ctrl(
    adata, num_cells, ctrl_name="ctrl", additional_filt=None, selected_genes=None
):
    """
    Sample cells from control condition.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (can be backed)
    num_cells : int
        Number of cells to sample
    ctrl_name : str
        Control condition name (default: 'ctrl')
    additional_filt : np.ndarray, optional
        Additional boolean filter to apply (e.g., for batch or context filtering)
    selected_genes : list, optional
        List of gene names to select

    Returns
    -------
    x0 : np.ndarray
        Sampled expression matrix
    """

    if additional_filt is not None:
        filt = (adata.obs["condition"] == ctrl_name) & additional_filt
    else:
        filt = adata.obs["condition"] == ctrl_name

    # Get indices that match the filter
    ctrl_indices = np.where(filt)[0]

    if len(ctrl_indices) == 0:
        raise ValueError(f"No cells found matching ctrl_name={ctrl_name}")

    # Sample from these indices
    selected_indices = np.random.choice(ctrl_indices, num_cells, replace=False)

    # Sort indices for backed h5ad compatibility (HDF5 requires sorted indices)
    selected_indices = np.sort(selected_indices)

    # Extract data - handle both backed and in-memory AnnData
    # Index directly into X to avoid view-of-view issue with backed AnnData
    if selected_genes is not None:
        gene_mask = adata.var_names.isin(selected_genes)
        if hasattr(adata.X[:2, :2], "toarray"):
            x0 = adata.X[selected_indices, :][:, gene_mask].toarray()
        else:
            x0 = adata.X[selected_indices, :][:, gene_mask]
    else:
        if hasattr(adata.X[:2, :2], "toarray"):
            x0 = adata.X[selected_indices, :].toarray()
        else:
            x0 = adata.X[selected_indices, :]

    return x0


def sample_x0_from_ctrl_tensor(
    adata, num_cells, ctrl_name="ctrl", additional_filt=None, device="cpu"
):
    x0 = sample_x0_from_ctrl(adata, num_cells, ctrl_name=ctrl_name, additional_filt=additional_filt)
    x0 = torch.tensor(x0, dtype=torch.float32).to(device)
    return x0


def sample_x0_from_train_non_ctrl(
    adata, num_cells, ctrl_name="ctrl", additional_filt=None, selected_genes=None
):
    """
    Sample cells from training set excluding control condition.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (can be backed)
    num_cells : int
        Number of cells to sample
    ctrl_name : str
        Control condition name to exclude (default: 'ctrl')
    additional_filt : np.ndarray, optional
        Additional boolean filter to apply (e.g., for batch or context filtering)
    selected_genes : list, optional
        List of gene names to select

    Returns
    -------
    x0 : np.ndarray
        Sampled expression matrix
    """

    # Filter for training split and exclude control
    train_filt = adata.obs["split"] == "train"
    non_ctrl_filt = adata.obs["condition"] != ctrl_name

    if additional_filt is not None:
        filt = train_filt & non_ctrl_filt & additional_filt
    else:
        filt = train_filt & non_ctrl_filt

    # Get indices that match the filter
    train_indices = np.where(filt)[0]

    if len(train_indices) == 0:
        raise ValueError(f"No training cells found after excluding ctrl_name={ctrl_name}")

    if num_cells > len(train_indices):
        num_cells = len(train_indices)

    # Sample from these indices
    selected_indices = np.random.choice(train_indices, num_cells, replace=False)

    # Sort indices for backed h5ad compatibility (HDF5 requires sorted indices)
    selected_indices = np.sort(selected_indices)

    # Extract data - handle both backed and in-memory AnnData
    # Index directly into X to avoid view-of-view issue with backed AnnData
    if selected_genes is not None:
        gene_mask = adata.var_names.isin(selected_genes)
        if hasattr(adata.X[:2, :2], "toarray"):
            x0 = adata.X[selected_indices, :][:, gene_mask].toarray()
        else:
            x0 = adata.X[selected_indices, :][:, gene_mask]
    else:
        if hasattr(adata.X[:2, :2], "toarray"):
            x0 = adata.X[selected_indices, :].toarray()
        else:
            x0 = adata.X[selected_indices, :]

    return x0


def sample_z0_from_ctrl(
    adata, num_cells, ctrl_name="ctrl", additional_filt=None, selected_genes=None
):
    if additional_filt is not None:
        filt = (adata.obs["condition"] == ctrl_name) & additional_filt
    else:
        filt = adata.obs["condition"] == ctrl_name

    # Get indices that match the filter
    ctrl_indices = np.where(filt)[0]

    if len(ctrl_indices) == 0:
        raise ValueError(f"No cells found matching ctrl_name={ctrl_name}")

    # Sample from these indices
    selected_indices = np.random.choice(ctrl_indices, num_cells, replace=False)

    # Sort indices for backed h5ad compatibility (HDF5 requires sorted indices)
    selected_indices = np.sort(selected_indices)

    # Extract latent representations directly
    z0 = adata.obsm["X_vae"][selected_indices]

    # Filter by genes if needed
    if selected_genes is not None:
        z0 = z0[:, selected_genes]

    return z0


def sample_z0_from_ctrl_tensor(
    adata, num_cells, ctrl_name="ctrl", additional_filt=None, device="cpu"
):
    z0 = sample_z0_from_ctrl(adata, num_cells, ctrl_name=ctrl_name, additional_filt=additional_filt)
    z0 = torch.tensor(z0, dtype=torch.float32).to(device)
    return z0


def make_data_loader(
    adata,
    ot_sampler,
    condition_labels,
    context_to_idx=None,
    split="train",
    covariate_columns=["condition"],
    batch_size=32,
    shuffle=True,
    device="cpu",
    train_with_ctrl=False,
    flow_from_ctrl=True,
    ot_replace=False,
):
    ot_chunk_size = 10000

    with torch.no_grad():
        print(
            f"Preparing data loader for split='{split}' with flow_from_ctrl={flow_from_ctrl} and train_with_ctrl={train_with_ctrl}..."
        )
        split_filt = adata.obs["split"] == split
        z0_coupled_list = []
        z1_coupled_list = []
        y_list = []

        # Find condition column and context column
        condition_col = None
        context_col = None
        for col in covariate_columns:
            if col == "condition":
                condition_col = col
                condition_col_pos = covariate_columns.index(col)
            else:
                context_col = col
                context_col_pos = covariate_columns.index(col)

        # Check if we need context indices (has a second covariate that's not 'condition')
        has_context = len(covariate_columns) == 2 and context_col is not None
        c_list = [] if has_context else None

        unique_groups = adata.obs.loc[split_filt, covariate_columns].drop_duplicates().values

        for group in unique_groups:
            if has_context:
                condition = group[condition_col_pos]
                context = group[context_col_pos]
                cond_filt = adata.obs[condition_col] == condition
                context_filt = adata.obs[context_col] == context
                filt = cond_filt & context_filt & split_filt

            else:
                # Extract scalar value from group array (group is np array with 1 element)
                condition = group[0] if isinstance(group, np.ndarray) else group
                filt = (adata.obs[condition_col] == condition) & split_filt

            if sum(filt) == 0:
                print(f"No cells found for group {group}, skipping.")
                continue

            z1 = torch.tensor(adata.obsm["X_vae"][filt], dtype=torch.float32)
            if flow_from_ctrl:
                if not train_with_ctrl and condition == "ctrl":
                    print("Skipping ctrl condition in training data loader.")
                    continue
                else:
                    additional_filt = context_filt if has_context else None
                    # Sample on CPU
                    z0_np = sample_z0_from_ctrl(
                        adata, z1.shape[0], ctrl_name="ctrl", additional_filt=additional_filt
                    )
                    z0 = torch.tensor(z0_np, dtype=torch.float32)
            else:
                if not train_with_ctrl and condition == "ctrl":
                    print("Skipping ctrl condition in training data loader.")
                    continue
                else:
                    z0 = torch.randn_like(z1)

            if z1.shape[0] > ot_chunk_size:
                print(f"Chunking OT sampling for group {group} with {z1.shape[0]} cells.")
                z0_chunks = []
                z1_chunks = []
                for start_idx in range(0, z1.shape[0], ot_chunk_size):
                    end_idx = min(start_idx + ot_chunk_size, z1.shape[0])
                    z0_chunk, z1_chunk = ot_sampler.sample_plan(
                        z0[start_idx:end_idx],
                        z1[start_idx:end_idx],
                        replace=ot_replace,
                    )
                    z0_chunks.append(z0_chunk)
                    z1_chunks.append(z1_chunk)
                z0_coupled = torch.cat(z0_chunks, dim=0)
                z1_coupled = torch.cat(z1_chunks, dim=0)
            else:
                z0_coupled, z1_coupled = ot_sampler.sample_plan(z0, z1, replace=ot_replace)

            label = condition_labels[condition]

            y = torch.tensor(np.tile(label, (z0_coupled.shape[0], 1)), dtype=torch.float32)

            z0_coupled_list.append(z0_coupled)
            z1_coupled_list.append(z1_coupled)
            y_list.append(y)

            # Add context indices if needed
            if has_context and context_to_idx is not None:
                c_idx = context_to_idx[context]
                c = torch.full((z0_coupled.shape[0],), c_idx, dtype=torch.long)
                c_list.append(c)

    z0_coupled = torch.cat(z0_coupled_list, dim=0)
    z1_coupled = torch.cat(z1_coupled_list, dim=0)
    y = torch.cat(y_list, dim=0)

    if has_context and c_list:
        c = torch.cat(c_list, dim=0)
        dataset = torch.utils.data.TensorDataset(z0_coupled, z1_coupled, y, c)
    else:
        dataset = torch.utils.data.TensorDataset(z0_coupled, z1_coupled, y)

    # Use pin_memory for faster CPU-to-GPU transfer during training
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=(device != "cpu"), num_workers=0
    )

    return data_loader


def make_condot_data_loader(
    adata,
    condition_labels,
    context_to_idx=None,
    covariate_columns=["condition"],
    split="train",
    batch_size=32,
    shuffle=True,
    device="cpu",
):
    with torch.no_grad():
        split_filt = adata.obs["split"] == split
        z1_list = []
        y_list = []

        # Find condition column and context column
        condition_col = "condition"
        context_col = None
        context_col_pos = None
        condition_col_pos = None

        for col in covariate_columns:
            if col == "condition":
                condition_col_pos = covariate_columns.index(col)
            else:
                context_col = col
                context_col_pos = covariate_columns.index(col)

        # Check if we need context indices
        has_context = len(covariate_columns) == 2 and context_col is not None
        c_list = [] if has_context else None

        # Get unique groups from adata
        unique_groups = adata.obs.loc[split_filt, covariate_columns].drop_duplicates().values
        for group in unique_groups:
            if has_context:
                condition = group[condition_col_pos]
                context = group[context_col_pos]
                cond_filt = adata.obs[condition_col] == condition
                context_filt = adata.obs[context_col] == context
                filt = cond_filt & context_filt & split_filt
            else:
                condition = group
                filt = (adata.obs[condition_col] == condition) & split_filt

            if sum(filt) == 0:
                print(f"No cells found for group {group}, skipping.")
                continue

            # Keep data on CPU for memory efficiency
            z1 = torch.tensor(adata.obsm["X_vae"][filt], dtype=torch.float32)

            # Get condition label
            label = condition_labels[condition]
            y = torch.tensor(np.tile(label, (z1.shape[0], 1)), dtype=torch.float32)

            z1_list.append(z1)
            y_list.append(y)

            # Add context indices if needed
            if has_context and context_to_idx is not None:
                c_idx = context_to_idx[context]
                c = torch.full((z1.shape[0],), c_idx, dtype=torch.long)
                c_list.append(c)

    z1 = torch.cat(z1_list, dim=0)
    y = torch.cat(y_list, dim=0)

    if has_context and c_list:
        c = torch.cat(c_list, dim=0)
        dataset = torch.utils.data.TensorDataset(z1, y, c)
    else:
        dataset = torch.utils.data.TensorDataset(z1, y)

    # Use pin_memory for faster CPU-to-GPU transfer during training
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=(device != "cpu"), num_workers=0
    )

    return data_loader


def make_prediction_data_loader(
    pred_df,
    pert_names_graph,
    adata=None,
    context_to_idx=None,
    covariate_columns=["condition"],
    latent_dim=50,
    n_cells=100,
    ctrl_name="ctrl",
    batch_size=32,
    shuffle=True,
    device="cpu",
    flow_from_ctrl=True,
    use_scvi_vae=False,
    use_my_scvi_vae=False,
):
    # Both scvi backends need a sampled library size in the batch.
    include_library = bool(use_scvi_vae or use_my_scvi_vae)

    with torch.no_grad():
        z0_list = []
        y_list = []

        # Find condition column and context column
        context_col = None
        context_col_pos = None
        condition_col_pos = None

        for col in covariate_columns:
            if col == "condition":
                condition_col_pos = covariate_columns.index(col)
            else:
                context_col = col
                context_col_pos = covariate_columns.index(col)

        # Check if we need context indices
        has_context = len(covariate_columns) == 2 and context_col is not None
        c_list = [] if has_context else None

        unique_groups = pred_df[covariate_columns].drop_duplicates().values

        libraries = [] if include_library else None

        for group in unique_groups:
            if has_context:
                condition = group[condition_col_pos]
                context = group[context_col_pos]
            else:
                condition = group[0]

            if flow_from_ctrl:
                # Build additional filter for sampling ctrl cells from same context
                if has_context:
                    additional_filt = adata.obs[context_col] == context
                else:
                    additional_filt = None

                # Sample on CPU
                z0_np = sample_z0_from_ctrl(
                    adata, n_cells, ctrl_name=ctrl_name, additional_filt=additional_filt
                )
                z0 = torch.tensor(z0_np, dtype=torch.float32)
            else:
                z0 = torch.randn(n_cells, latent_dim)

            # Get condition label
            label = pert_names_graph[condition]
            y = torch.tensor(np.tile(label, (z0.shape[0], 1)), dtype=torch.float32)

            if include_library:
                # Also need to get library size for generative model input
                if has_context:
                    additional_filt = adata.obs[context_col] == context
                else:
                    additional_filt = None

                library_np = sample_x0_from_ctrl(
                    adata, n_cells, ctrl_name=ctrl_name, additional_filt=additional_filt
                )
                library = (
                    torch.tensor(library_np, dtype=torch.float32).sum(dim=1).log().unsqueeze(1)
                )
                # randomly subsample library to fit the shape
                library = library[torch.randperm(z0.shape[0])]
                libraries.append(library)

            z0_list.append(z0)
            y_list.append(y)

            # Add context indices if needed
            if has_context and context is not None and context_to_idx is not None:
                c_idx = context_to_idx[context]
                c = torch.full((z0.shape[0],), c_idx, dtype=torch.long)
                c_list.append(c)

    z0 = torch.cat(z0_list, dim=0)
    y = torch.cat(y_list, dim=0)
    if include_library:
        library = torch.cat(libraries, dim=0)

    if has_context and c_list:
        c = torch.cat(c_list, dim=0)
        if include_library:
            dataset = torch.utils.data.TensorDataset(z0, y, c, library)
        else:
            dataset = torch.utils.data.TensorDataset(z0, y, c)
    else:
        if include_library:
            dataset = torch.utils.data.TensorDataset(z0, y, library)
        else:
            dataset = torch.utils.data.TensorDataset(z0, y)

    # Use pin_memory for faster CPU-to-GPU transfer
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=(device != "cpu"), num_workers=0
    )

    return data_loader


def get_go_graph(device, k=None, graph_dir="./data", randomize=False):
    k = 20 if k is None else k
    _ensure_go_graph_artifacts(graph_dir)

    gene2go = pickle.load(open(f"{graph_dir}/gene2go_all.pkl", "rb"))
    essential_genes = pickle.load(open(f"{graph_dir}/essential_all_data_pert_genes.pkl", "rb"))

    gene2go = {i: gene2go[i] for i in essential_genes if i in gene2go}
    pert_names_go = np.unique(list(gene2go.keys()))
    node_map_pert = {x: it for it, x in enumerate(pert_names_go)}

    df_jaccard = pd.read_csv(f"{graph_dir}/go_essential_all/go_essential_all.csv")

    edge_list = pd.concat(
        [group.nlargest(k + 1, "importance") for _, group in df_jaccard.groupby("target")]
    ).reset_index(drop=True)

    if randomize:
        edge_list["source"] = np.random.permutation(edge_list["source"].to_numpy())

    import networkx as nx

    G = nx.DiGraph(
        nx.from_pandas_edgelist(
            edge_list,
            source="source",
            target="target",
            edge_attr="importance",
        )
    )
    for n in pert_names_go:
        if n not in G.nodes():
            G.add_node(n)

    edge_index_ = [(node_map_pert[e[0]], node_map_pert[e[1]]) for e in G.edges]
    edge_index = torch.tensor(edge_index_, dtype=torch.long).T.to(device)

    edge_attr = nx.get_edge_attributes(G, "importance")
    importance = np.array([edge_attr[e] for e in G.edges])
    edge_weight = torch.tensor(importance, dtype=torch.float32).to(device)

    return edge_index, edge_weight, edge_list, pert_names_go


def get_ppi_graph(k=None, device="cpu", graph_dir="./data"):
    k = 20 if k is None else k

    # Check if processed CSV exists; if not, download and process from STRING
    csv_path = f"{graph_dir}/9606.protein.links.v12.0_with_gene_names.csv"
    if not os.path.exists(csv_path):
        os.makedirs(graph_dir, exist_ok=True)
        STRING_URL = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
        gz_path = f"{graph_dir}/9606.protein.links.v12.0.txt.gz"
        txt_path = f"{graph_dir}/9606.protein.links.v12.0.txt"

        # Download STRING data
        print("Downloading STRING protein interaction network...")
        response = requests.get(STRING_URL, stream=True)
        response.raise_for_status()
        with open(gz_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Extract gzip file
        print("Extracting gzipped file...")
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())

        # Read and process
        print("Reading STRING data...")
        ppi_net = pd.read_csv(txt_path, sep=" ")

        # Extract ENSP IDs (remove "9606." prefix)
        print("Mapping protein IDs to gene symbols...")
        ppi_net["ensp1"] = ppi_net["protein1"].str.replace("9606.", "")
        ppi_net["ensp2"] = ppi_net["protein2"].str.replace("9606.", "")

        # Query mygene to get gene symbols
        mg = mygene.MyGeneInfo()
        unique_proteins = pd.concat([ppi_net["ensp1"], ppi_net["ensp2"]]).unique()

        print(f"Querying {len(unique_proteins)} unique protein IDs against MyGene...")
        results = mg.querymany(
            unique_proteins,
            scopes="ensembl.protein",
            fields="symbol",
            species="human",
            verbose=False,
        )
        protein_to_gene_dict = {r["query"]: r.get("symbol", None) for r in results}

        # Map to gene names
        ppi_net["gene1"] = ppi_net["ensp1"].map(protein_to_gene_dict)
        ppi_net["gene2"] = ppi_net["ensp2"].map(protein_to_gene_dict)

        # Remove rows where mapping failed
        ppi_net = ppi_net.dropna(subset=["gene1", "gene2"])

        # Rename columns to match expected format
        ppi_net = ppi_net.rename(
            columns={"gene1": "source", "gene2": "target", "combined_score": "importance"}
        )

        # Save processed data
        print(f"Saving processed PPI network to {csv_path}...")
        ppi_net.to_csv(csv_path)

        # Clean up temporary files
        if os.path.exists(gz_path):
            os.remove(gz_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)
    else:
        ppi_net = pd.read_csv(csv_path, index_col=0)

    edge_list = pd.concat(
        [group.nlargest(k + 1, "importance") for _, group in ppi_net.groupby("target")]
    ).reset_index(drop=True)
    edge_list = edge_list[["source", "target", "importance"]]
    pert_names_graph = np.array(list(set(edge_list["source"]).union(set(edge_list["target"]))))
    gene_to_idx = {gene: idx for idx, gene in enumerate(pert_names_graph)}

    valid_edges = edge_list[
        edge_list["source"].isin(gene_to_idx) & edge_list["target"].isin(gene_to_idx)
    ]

    source_indices = valid_edges["source"].map(gene_to_idx).values
    target_indices = valid_edges["target"].map(gene_to_idx).values
    edge_weight = valid_edges["importance"].values

    edge_index = torch.tensor(np.array([source_indices, target_indices]), dtype=torch.long).to(
        device
    )

    return edge_index, edge_weight, edge_list, pert_names_graph


def get_pert_similarity_graph(
    pert_adata_path=None, graph_dir="./data", pert_column=None, ctrl_name=None, k=None, device=None
):
    k = 20 if k is None else k
    pert_column = pert_column if pert_column is not None else "condition"
    ctrl_name = ctrl_name if ctrl_name is not None else "ctrl"
    device = device if device is not None else "cpu"
    if (
        os.path.exists(os.path.join(graph_dir, f"pert_graph_edge_list_{k}.csv"))
        and os.path.exists(os.path.join(graph_dir, f"pert_graph_edge_index_{k}.pt"))
        and os.path.exists(os.path.join(graph_dir, f"pert_graph_edge_weight_{k}.pt"))
        and os.path.exists(os.path.join(graph_dir, f"pert_graph_pert_names_graph_{k}.npy"))
    ):
        print("Found precomputed graph files, loading...")
        edge_list = pd.read_csv(os.path.join(graph_dir, f"pert_graph_edge_list_{k}.csv"))
        edge_index = torch.load(os.path.join(graph_dir, f"pert_graph_edge_index_{k}.pt")).to(device)
        edge_weight = torch.load(os.path.join(graph_dir, f"pert_graph_edge_weight_{k}.pt")).to(
            device
        )
        pert_names_graph = np.load(os.path.join(graph_dir, f"pert_graph_pert_names_graph_{k}.npy"))
        return edge_list, edge_index, edge_weight, pert_names_graph

    else:

        def _mean_expression_for_indices(matrix, indices):
            indices = np.sort(np.asarray(indices))
            x = matrix[indices, :]
            if hasattr(x, "toarray"):
                # Sparse matrix mean keeps this memory-light compared to full densification.
                return np.asarray(x.mean(axis=0)).ravel().astype(np.float32, copy=False)
            return np.asarray(x).mean(axis=0).astype(np.float32, copy=False)

        if pert_adata_path is None:
            raise ValueError(
                "pert_adata_path must be provided when building perturbation similarity graph."
            )
        adata = sc.read_h5ad(pert_adata_path, backed="r")
        cond_series = adata.obs[pert_column]
        condition_to_indices = cond_series.groupby(cond_series).indices
        print("Building mean expression profiles for each condition...")
        mean_expr_per_condition = {
            cond: _mean_expression_for_indices(adata.X, idx)
            for cond, idx in tqdm(condition_to_indices.items())
        }

        if ctrl_name not in mean_expr_per_condition:
            raise ValueError(
                f"Control condition '{ctrl_name}' not found in adata.obs['{pert_column}']"
            )

        # Build unique perturbation list while preserving insertion order.
        pert_names_graph = np.array(
            list(
                dict.fromkeys(
                    pert
                    for cond in cond_series.unique()
                    if cond != ctrl_name
                    for pert in trim_cond(cond).split("+")
                    if pert != ctrl_name
                )
            )
        )

        pert_to_matching_conditions = {}
        for cond in mean_expr_per_condition:
            for pert in trim_cond(cond).split("+"):
                if pert == ctrl_name:
                    continue
                pert_to_matching_conditions.setdefault(pert, []).append(cond)

        ctrl = mean_expr_per_condition[ctrl_name]
        signatures = []
        valid_pert_names = []
        pert_to_idx = {pert: i for i, pert in enumerate(pert_names_graph)}

        for pert in pert_names_graph:
            matching_conditions = pert_to_matching_conditions.get(pert, [])
            if not matching_conditions:
                continue

            mean_pert = np.mean(
                [mean_expr_per_condition[cond] for cond in matching_conditions], axis=0
            )
            delta = (mean_pert - ctrl).astype(np.float32, copy=False)
            if np.linalg.norm(delta) == 0:
                continue

            signatures.append(delta)
            valid_pert_names.append(pert)

        if k <= 0 or len(signatures) <= 1:
            edge_list = pd.DataFrame(columns=["source", "target", "importance"])
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            edge_weight = torch.empty((0,), dtype=torch.float32).to(device)
            return edge_list, edge_index, edge_weight, pert_names_graph

        signature_matrix = np.vstack(signatures).astype(np.float32, copy=False)
        norms = np.linalg.norm(signature_matrix, axis=1, keepdims=True)
        signature_matrix = signature_matrix / np.clip(norms, 1e-12, None)

        print("Find top-k similar perturbations based on cosine similarity...")
        # Avoid allocating a full NxN similarity matrix by querying top-k neighbors directly.
        n_neighbors = min(k + 1, len(valid_pert_names))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        knn.fit(signature_matrix)
        distances, neighbor_indices = knn.kneighbors(signature_matrix, return_distance=True)

        edge_rows = []
        source_indices = []
        target_indices = []
        edge_weights = []

        for target_pos, target in enumerate(valid_pert_names):
            for rank, source_pos in enumerate(neighbor_indices[target_pos]):
                if source_pos == target_pos:
                    continue

                source = valid_pert_names[source_pos]
                importance = 1.0 - float(distances[target_pos, rank])
                edge_rows.append({"source": source, "target": target, "importance": importance})
                source_indices.append(pert_to_idx[source])
                target_indices.append(pert_to_idx[target])
                edge_weights.append(importance)

        edge_list = pd.DataFrame(edge_rows, columns=["source", "target", "importance"])
        if len(edge_weights) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
            edge_weight = torch.empty((0,), dtype=torch.float32).to(device)
        else:
            edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long).to(device)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32).to(device)

        # save computed graph for future use
        edge_list.to_csv(os.path.join(graph_dir, f"pert_graph_edge_list_{k}.csv"), index=False)
        torch.save(edge_index.cpu(), os.path.join(graph_dir, f"pert_graph_edge_index_{k}.pt"))
        torch.save(edge_weight.cpu(), os.path.join(graph_dir, f"pert_graph_edge_weight_{k}.pt"))
        np.save(os.path.join(graph_dir, f"pert_graph_pert_names_graph_{k}.npy"), pert_names_graph)

        return edge_list, edge_index, edge_weight, pert_names_graph


def get_edge_index_and_weight(edge_list, pert_names_graph, relation_type=None, device=None):
    edge_list = edge_list.copy()
    if relation_type is not None:
        edge_list = edge_list[edge_list["relation_type"] == relation_type]

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        if device is not None:
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
        return edge_index, edge_weight

    # Map gene names → integer indices
    name_to_idx = {name: idx for idx, name in enumerate(pert_names_graph)}

    src_indices = edge_list["source"].map(name_to_idx).values
    tgt_indices = edge_list["target"].map(name_to_idx).values

    # Drop edges where source or target is not in pert_names_graph
    valid_mask = (~pd.isnull(src_indices)) & (~pd.isnull(tgt_indices))
    src_indices = src_indices[valid_mask].astype(int)
    tgt_indices = tgt_indices[valid_mask].astype(int)
    weights = edge_list["importance"].values[valid_mask].astype(float)

    edge_index = torch.tensor(np.array([src_indices, tgt_indices]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    if device is not None:
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

    return edge_index, edge_weight


def make_pert_labels(cond, pert_names_graph):
    gene_list = cond.split("+")  # accept ctrl
    idx_list = np.full(2, -1)  # max 2 perturbations at once
    for i, gene in enumerate(gene_list):
        if gene in pert_names_graph:
            idx = int(np.where(pert_names_graph == gene)[0][0])
            idx_list[i] = idx
        elif gene == "ctrl":
            idx_list[i] = -1
        else:
            print(f"{gene} not in the graph")
    return idx_list


def make_pert_labels_drug(cond, pert_names_graph):
    idx_list = np.full(1, -1)  # max 1 perturbation for now
    if cond in pert_names_graph:
        idx = int(np.where(pert_names_graph == cond)[0][0])
        idx_list[0] = idx
    elif cond == "ctrl":
        idx_list[0] = -1
    else:
        print(f"{cond} not in the graph")
    return idx_list


def make_condition_labels_graph(pert_name, pert_names_graph, drug_graph=False):
    condition_labels_graph = {}
    for cond in pert_name:
        if drug_graph:
            idx_list = make_pert_labels_drug(cond, pert_names_graph)
        else:
            idx_list = make_pert_labels(cond, pert_names_graph)
        condition_labels_graph[cond] = idx_list

    return condition_labels_graph


def build_graph(
    graph_type="go",
    pert_encoding="gat",
    pert_names=None,
    k=20,
    randomize_graph=False,
    adata=None,
    split_dict=None,
    graph_dir=None,
    pert_adata_path=None,
    device="cpu",
):
    if pert_names is None:
        raise ValueError("pert_names must be provided to build the graph.")
    if graph_dir is None:
        graph_dir = "./data"
    if graph_type == "go":
        edge_index, edge_weight, edge_list, pert_names_graph = get_go_graph(
            device, k=k, graph_dir=graph_dir, randomize=randomize_graph
        )
        condition_labels_graph = make_condition_labels_graph(pert_names, pert_names_graph)
    elif graph_type == "ppi":
        edge_index, edge_weight, edge_list, pert_names_graph = get_ppi_graph(
            k=k, device=device, graph_dir=graph_dir
        )
        condition_labels_graph = make_condition_labels_graph(pert_names, pert_names_graph)
    elif graph_type == "go+ppi":
        _, _, edge_list_go, _ = get_go_graph(device, graph_dir=graph_dir)
        _, _, edge_list_ppi, _ = get_ppi_graph(k=10, device=device, graph_dir=graph_dir)

        if pert_encoding == "hgnn":
            edge_list_go["relation"] = "gene_gene"
            edge_list_go["relation_type"] = "go_sim"
            edge_list_ppi["relation"] = "gene_gene"
            edge_list_ppi["relation_type"] = "ppi"
            edge_list = pd.concat([edge_list_go, edge_list_ppi], ignore_index=True)
            if randomize_graph:
                edge_list["source"] = np.random.permutation(edge_list["source"].to_numpy())
            pert_names_graph = np.array(
                list(set(edge_list["source"]).union(set(edge_list["target"])))
            )

            edge_index_go, edge_weight_go = get_edge_index_and_weight(
                edge_list, pert_names_graph, relation_type="go_sim", device=device
            )
            edge_index_ppi, edge_weight_ppi = get_edge_index_and_weight(
                edge_list, pert_names_graph, relation_type="ppi", device=device
            )
            edge_index = {
                ("gene", "go_sim", "gene"): edge_index_go,
                ("gene", "ppi", "gene"): edge_index_ppi,
            }
            edge_weight = {
                ("gene", "go_sim", "gene"): edge_weight_go,
                ("gene", "ppi", "gene"): edge_weight_ppi,
            }

        else:
            edge_list = pd.concat([edge_list_go, edge_list_ppi], ignore_index=True)
            if randomize_graph:
                edge_list["source"] = np.random.permutation(edge_list["source"].to_numpy())
            pert_names_graph = np.array(
                list(set(edge_list["source"]).union(set(edge_list["target"])))
            )

            edge_index, edge_weight = get_edge_index_and_weight(
                edge_list, pert_names_graph, device=device
            )

        condition_labels_graph = make_condition_labels_graph(pert_names, pert_names_graph)
    elif graph_type == "pert":
        edge_list, edge_index, edge_weight, pert_names_graph = get_pert_similarity_graph(
            pert_adata_path=pert_adata_path,
            graph_dir=graph_dir,
            pert_column="perturbation",
            ctrl_name="control",
            device=device,
            k=20,
        )
        condition_labels_graph = make_condition_labels_graph(pert_names, pert_names_graph)
    elif graph_type == "go+pert":
        _, _, edge_list_go, _ = get_go_graph(device, graph_dir=graph_dir)
        edge_list_pert, _, _, _ = get_pert_similarity_graph(
            pert_adata_path=pert_adata_path,
            graph_dir=graph_dir,
            pert_column="perturbation",
            ctrl_name="control",
            device=device,
            k=20,
        )

        if pert_encoding == "hgnn":
            # Combine and deduplicate edge lists
            edge_list_go["relation"] = "gene_gene"
            edge_list_go["relation_type"] = "go_sim"
            edge_list_pert["relation"] = "gene_gene"
            edge_list_pert["relation_type"] = "pert_sim"
            edge_list = pd.concat([edge_list_go, edge_list_pert], ignore_index=True)
            if randomize_graph:
                edge_list["source"] = np.random.permutation(edge_list["source"].to_numpy())

            # Create unified pert_names_graph and reindex edges
            pert_names_graph = np.array(
                list(set(edge_list["source"]).union(set(edge_list["target"])))
            )

            edge_index_go, edge_weight_go = get_edge_index_and_weight(
                edge_list, pert_names_graph, relation_type="go_sim", device=device
            )
            edge_index_pert, edge_weight_pert = get_edge_index_and_weight(
                edge_list, pert_names_graph, relation_type="pert_sim", device=device
            )
            edge_index = {
                ("gene", "go_sim", "gene"): edge_index_go,
                ("gene", "pert_sim", "gene"): edge_index_pert,
            }
            edge_weight = {
                ("gene", "go_sim", "gene"): edge_weight_go,
                ("gene", "pert_sim", "gene"): edge_weight_pert,
            }
        else:
            edge_list = pd.concat([edge_list_go, edge_list_pert], ignore_index=True)
            pert_names_graph = np.array(
                list(set(edge_list["source"]).union(set(edge_list["target"])))
            )

            edge_index, edge_weight = get_edge_index_and_weight(
                edge_list, pert_names_graph, device=device
            )

        condition_labels_graph = make_condition_labels_graph(pert_names, pert_names_graph)

    elif graph_type == "go+pert+ppi":
        _, _, edge_list_go, _ = get_go_graph(device, graph_dir=graph_dir)
        _, _, edge_list_ppi, _ = get_ppi_graph(k=10, device=device, graph_dir=graph_dir)
        edge_list_pert, _, _, _ = get_pert_similarity_graph(
            pert_adata_path=pert_adata_path,
            graph_dir=graph_dir,
            pert_column="perturbation",
            ctrl_name="control",
            device=device,
            k=20,
        )

        if pert_encoding == "hgnn":
            # Combine and deduplicate edge lists
            edge_list_go["relation"] = "gene_gene"
            edge_list_go["relation_type"] = "go_sim"
            edge_list_ppi["relation"] = "gene_gene"
            edge_list_ppi["relation_type"] = "ppi"
            edge_list_pert["relation"] = "gene_gene"
            edge_list_pert["relation_type"] = "pert_sim"
            edge_list = pd.concat([edge_list_go, edge_list_ppi, edge_list_pert], ignore_index=True)
            if randomize_graph:
                edge_list["source"] = np.random.permutation(edge_list["source"].to_numpy())

            # Create unified pert_names_graph and reindex edges
            pert_names_graph = np.array(
                list(set(edge_list["source"]).union(set(edge_list["target"])))
            )

            edge_index_go, edge_weight_go = get_edge_index_and_weight(
                edge_list, pert_names_graph, relation_type="go_sim", device=device
            )
            edge_index_ppi, edge_weight_ppi = get_edge_index_and_weight(
                edge_list, pert_names_graph, relation_type="ppi", device=device
            )
            edge_index_pert, edge_weight_pert = get_edge_index_and_weight(
                edge_list, pert_names_graph, relation_type="pert_sim", device=device
            )
            edge_index = {
                ("gene", "go_sim", "gene"): edge_index_go,
                ("gene", "ppi", "gene"): edge_index_ppi,
                ("gene", "pert_sim", "gene"): edge_index_pert,
            }
            edge_weight = {
                ("gene", "go_sim", "gene"): edge_weight_go,
                ("gene", "ppi", "gene"): edge_weight_ppi,
                ("gene", "pert_sim", "gene"): edge_weight_pert,
            }
        else:
            edge_list = pd.concat([edge_list_go, edge_list_pert, edge_list_ppi], ignore_index=True)
            if randomize_graph:
                edge_list["source"] = np.random.permutation(edge_list["source"].to_numpy())
            pert_names_graph = np.array(
                list(set(edge_list["source"]).union(set(edge_list["target"])))
            )

            edge_index, edge_weight = get_edge_index_and_weight(
                edge_list, pert_names_graph, device=device
            )
        condition_labels_graph = make_condition_labels_graph(pert_names, pert_names_graph)

    return edge_index, edge_weight, edge_list, pert_names_graph, condition_labels_graph


def trim_cond(cond):
    filtered = "+".join(p for p in cond.split("+") if p != "ctrl")
    return filtered if filtered else "ctrl"


def find_source_nodes(edge_list, single_pert, n_gnn_layers=3):
    """
    Find all nodes with directed paths into a perturbation within n message-passing hops.
    """

    if n_gnn_layers <= 0:
        return set()

    # Build a dictionary for O(1) lookups: target -> set of incoming source nodes.
    cache_key = "_target_to_sources_cache"
    if cache_key not in edge_list.attrs:
        edge_list.attrs[cache_key] = edge_list.groupby("target")["source"].apply(set).to_dict()

    target_to_sources = edge_list.attrs[cache_key]
    frontier = set(trim_cond(single_pert).split("+"))
    frontier.discard("ctrl")
    source_nodes_set = set()

    for _ in range(n_gnn_layers):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(target_to_sources.get(node, set()))

        next_frontier.difference_update(source_nodes_set)
        source_nodes_set.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    return source_nodes_set


def get_pert_graph_info(gfm, test_conds):
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    pert_graph_info = dict()

    perts_in_train = set(
        [item for cond in gfm.split.split_dict["train"] for item in trim_cond(cond).split("+")]
    )

    expr_array = gfm.adata.X[:, :].toarray()
    expr_df = pd.DataFrame(expr_array, columns=gfm.adata.var_names, index=gfm.adata.obs.index)

    expr_df["condition"] = gfm.adata.obs["condition"].values
    mean_expr_per_condition = expr_df.groupby("condition").mean()

    # Pre-compute control expression
    ctrl = np.array(mean_expr_per_condition.loc["ctrl", :])

    for cond in test_conds:
        cond_dict = dict()

        pert = trim_cond(cond)
        if pert == "":
            continue
        source_nodes = find_source_nodes(gfm.edge_list, pert, n_gnn_layers=2)
        if len(source_nodes) == 0:
            cond_dict["source_nodes"] = source_nodes
            continue
        source_nodes_trained = source_nodes.intersection(perts_in_train)
        train_node_ratio = len(source_nodes_trained) / len(source_nodes)

        # get cosine similarity, considering delta expression from control
        a_delta = np.array(mean_expr_per_condition.loc[cond, :]) - ctrl

        ## only consider single perturbations
        pert_list = []
        sim_list = []
        for perts in source_nodes_trained:
            # Find all conditions containing this perturbation
            matching_conditions = [c for c in mean_expr_per_condition.index if perts in c]
            if matching_conditions:
                b = mean_expr_per_condition.loc[matching_conditions, :].mean(axis=0).values
                b_delta = b - ctrl
                sim = cosine_similarity(a_delta, b_delta)
                pert_list.append(perts)
                sim_list.append(sim)

        cond_dict["source_nodes"] = source_nodes
        cond_dict["source_nodes_trained"] = source_nodes_trained
        cond_dict["train_node_ratio"] = train_node_ratio
        cond_dict["pert_list"] = pert_list
        cond_dict["sim_list"] = sim_list

        pert_graph_info[cond] = cond_dict

    return pert_graph_info


def add_pert_graph_metrics(result_df, pert_graph_info):
    n_nodes_trained_list = []
    ratio_list = []
    max_sim_list = []
    median_sim_list = []
    for cond in result_df["condition"]:
        if cond not in pert_graph_info.keys():
            n_nodes_trained_list.append(0)
            ratio_list.append(0)
            max_sim_list.append(0)
            median_sim_list.append(0)
            continue
        if len(pert_graph_info[cond]["source_nodes"]) == 0:
            ratio_list.append(0)
            n_nodes_trained_list.append(0)
            max_sim_list.append(0)
            median_sim_list.append(0)
            continue

        n_nodes_trained_list.append(len(pert_graph_info[cond]["source_nodes_trained"]))
        ratio_list.append(pert_graph_info[cond]["train_node_ratio"])
        if len(pert_graph_info[cond]["source_nodes_trained"]) == 0:
            max_sim_list.append(0)
            median_sim_list.append(0)
            continue

        max_sim_list.append(np.max(pert_graph_info[cond]["sim_list"]))
        median_sim_list.append(np.median(pert_graph_info[cond]["sim_list"]))

    result_df["n_source_nodes_trained"] = n_nodes_trained_list
    result_df["train_node_ratio"] = ratio_list
    result_df["max_sim"] = max_sim_list
    result_df["median_sim"] = median_sim_list


def calculate_delta_result(
    result_df, result_no_pred, result_pos_ctrl=None, normalize=False, return_detail=False
):
    """
    Calculate delta between test results and no-prediction baseline.

    The 'condition' column always contains perturbations only.
    Additional covariates (e.g., 'cell_line', 'top_n') are in separate columns.
    All covariate columns are used as a multi-index for alignment.

    Parameters
    ----------
    result_df : pd.DataFrame
        Test results with 'condition' column and metrics
    result_no_pred : pd.DataFrame
        No-prediction baseline with 'condition' column and metrics
    normalize : bool
        Whether to normalize by dynamic range
    return_detail : bool
        Whether to return improvement and dynamic_range details

    Returns
    -------
    result : pd.DataFrame
        Delta results with all index columns preserved
    """
    result_no_pred = result_no_pred.copy()
    result_no_pred["pearson_delta"] = 0.0

    # Identify all covariate columns (condition + any additional covariates)
    # Look for common non-metric columns between the two dataframes
    metric_cols = ["mse", "pearson", "pearson_delta", "spearman", "mmd", "w2d", "e_distance"]

    # Get non-metric columns (these are the covariate columns)
    result_df_cols = set(result_df.columns) - set(metric_cols)
    result_no_pred_cols = set(result_no_pred.columns) - set(metric_cols)
    covariate_cols = sorted(result_df_cols & result_no_pred_cols)  # Common covariate columns

    if not covariate_cols:
        raise ValueError("No common covariate columns found between result_df and result_no_pred")

    # 'condition' should always be present
    if "condition" not in covariate_cols:
        raise ValueError("'condition' column must be present in both dataframes")

    # Use all covariate columns as multi-index
    index_cols = covariate_cols
    num_index_cols = len(index_cols)

    # Use reindex to maintain order and align the two dataframes
    test_result = (
        result_df.set_index(index_cols)
        .reindex(result_no_pred.set_index(index_cols).index)
        .reset_index()
    )
    result_no_pred = (
        result_no_pred.set_index(index_cols)
        .reindex(result_no_pred.set_index(index_cols).index)
        .reset_index()
    )

    if normalize:
        if result_pos_ctrl is None:
            raise ValueError("result_pos_ctrl must be provided when normalize=True")
        improvement = test_result.iloc[:, num_index_cols:] - result_no_pred.iloc[:, num_index_cols:]
        dynamic_range = (
            result_pos_ctrl.iloc[:, num_index_cols:] - result_no_pred.iloc[:, num_index_cols:]
        )
        # Avoid division by zero
        dynamic_range = dynamic_range.replace(0, np.nan)
        result = improvement / dynamic_range
    else:
        result = test_result.iloc[:, num_index_cols:] - result_no_pred.iloc[:, num_index_cols:]

    # Add back all index columns
    for col in index_cols:
        result[col] = test_result[col]

    if return_detail:
        # Add index columns to detail dataframes too
        for col in index_cols:
            improvement[col] = test_result[col]
            dynamic_range[col] = test_result[col]
        return result, improvement, dynamic_range
    else:
        return result


#################################################################################################################################
# VAE functions
#################################################################################################################################


def get_cell_embedding(vae, adata, batch_size=256, device="cpu"):
    vae.eval()
    with torch.no_grad():
        loader = make_vae_dataloader(adata, batch_size=batch_size, shuffle=False, device=device)
        mus = []
        for batch in loader:
            data = batch[0].to(device, non_blocking=True)
            h = vae.encoder(data)
            mu = vae.fc_mu(h)
            # logvar = self.vae.fc_logvar(h)
            # z = self.vae.reparameterize(mu, logvar)
            # z.cpu().numpy()
            mu = mu.cpu().numpy()
            mus.append(mu)
        mu = np.concatenate(mus, axis=0)
    return mu


def reconstruct_cell_embedding(vae, cell_embed, device="cpu"):
    vae.eval()
    with torch.no_grad():
        z = torch.tensor(cell_embed, dtype=torch.float32).to(device)
        recon_x = vae.decoder(z)
        recon_x = recon_x.cpu().numpy()
    return recon_x


def reconstruct_expression(vae, adata, batch_size=256, device="cpu"):
    vae.eval()
    with torch.no_grad():
        loader = make_vae_dataloader(adata, batch_size=batch_size, shuffle=False, device=device)
        recon_xs = []
        for batch in loader:
            data = batch[0].to(device, non_blocking=True)
            h = vae.encoder(data)
            mu = vae.fc_mu(h)
            recon_x = vae.decoder(mu)
            recon_x = recon_x.cpu().numpy()
            recon_xs.append(recon_x)
        recon_x = np.concatenate(recon_xs, axis=0)
    adata_pred = sc.AnnData(recon_x, obs=adata.obs.copy(), var=adata.var.copy())
    return adata_pred


def get_cell_umap_vae(adata, embed_name="X_vae"):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    adata.obsm["X_umap_vae"] = umap_model.fit_transform(adata.obsm[embed_name])


##################################################################################################################################
# evaluation functions
##################################################################################################################################


def mmd_distance(x, y, gamma=1.0):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(x, y, gammas=[2, 1, 0.5, 0.1, 0.01, 0.005]):
    mmds = [mmd_distance(x, y, gamma) for gamma in gammas]
    return np.mean(mmds)


def wasserstein_2(x0, x1):
    n = x0.shape[0]
    m = x1.shape[0]
    cost_matrix = ot.dist(x0, x1, metric="sqeuclidean")
    a = np.ones(n) / n  # Uniform distribution for x0
    b = np.ones(m) / m  # Uniform distribution for x1
    w2_squared = ot.emd2(a, b, cost_matrix)

    return np.sqrt(w2_squared)


def e_distance(x, y):
    xx = pairwise_distances(x, x)
    xy = pairwise_distances(x, y)
    yy = pairwise_distances(y, y)

    return 2 * xy.mean() - xx.mean() - yy.mean()


def _compute_metrics_single_group(
    group,
    adata,
    adata_pred,
    control_type,
    covariate_columns,
    condition_col,
    context_col,
    condition_col_pos,
    context_col_pos,
    has_context,
    adata_pred_genes,
    layer,
    top_n,
    is_drug,
    n_pred,
):
    """
    Helper function to compute metrics for a single group.
    This is extracted to enable parallelization.
    """
    # Extract values from group
    if has_context:
        condition = group[condition_col_pos]
        context = group[context_col_pos]
        cond_values = {condition_col: condition, context_col: context}
    else:
        condition = group if isinstance(group, str) else group[0]
        cond_values = {condition_col: condition}

    # Extract condition value for perturbation name
    cond_value = cond_values.get("condition")

    # Generate perturbation name (strip 'ctrl' from condition)
    if isinstance(cond_value, str) and not is_drug:
        pert = "+".join([pert_gene for pert_gene in cond_value.split("+") if pert_gene != "ctrl"])
    else:
        pert = str(cond_value)

    # Skip if ctrl or not in predictions
    if cond_value == "ctrl" or pert not in adata_pred.obs["condition"].unique():
        return None

    # Get top genes - construct key from group
    if has_context:
        # For multi-covariate, join with '_'
        cond_key = "_".join(str(v) for v in group)
    else:
        cond_key = condition

    top_genes = adata.uns["rank_genes_groups_list"][cond_key][:top_n]
    top_genes = [gene for gene in top_genes if gene in adata_pred_genes]

    # Build filter for adata based on all covariate values
    filt = np.ones(adata.shape[0], dtype=bool)
    for col, val in cond_values.items():
        filt = filt & (adata.obs[col] == val)

    # n_pred = sum(filt)

    # Build additional filter for sampling control cells from same context
    if has_context and context_col is not None:
        # Filter for same context (non-condition covariates)
        additional_filt = adata.obs[context_col] == cond_values[context_col]
    else:
        additional_filt = None

    # Extract truth data using filter - handle backed files properly
    # Convert boolean filter to sorted indices (required for HDF5)
    truth_indices = np.where(filt)[0]
    truth_indices = np.sort(truth_indices)

    # Get gene indices for top_genes
    gene_mask = adata.var_names.isin(top_genes)

    # Extract data with single-dimension indexing to avoid HDF5 fancy indexing issues
    if layer is None:
        if hasattr(adata.X[:2, :2], "toarray"):
            truth = adata.X[truth_indices, :][:, gene_mask].toarray()
        else:
            truth = adata.X[truth_indices, :][:, gene_mask]
    else:
        if hasattr(adata.layers[layer][:2, :2], "toarray"):
            truth = adata.layers[layer][truth_indices, :][:, gene_mask].toarray()
        else:
            truth = adata.layers[layer][truth_indices, :][:, gene_mask]

    gene_indices = np.where(gene_mask)[0]
    filt_pred = (
        (adata_pred.obs[condition_col] == pert)
        & (adata_pred.obs[context_col] == cond_values[context_col])
        if has_context
        else (adata_pred.obs[condition_col] == pert)
    )
    if layer is None:
        if hasattr(adata_pred.X[:2, :2], "toarray"):
            pred = adata_pred.X[filt_pred, :][:, gene_indices].toarray()
        else:
            pred = adata_pred.X[filt_pred, :][:, gene_indices]
    else:
        if hasattr(adata_pred.layers[layer][:2, :2], "toarray"):
            pred = adata_pred.layers[layer][filt_pred, :][:, gene_indices].toarray()
        else:
            pred = adata_pred.layers[layer][filt_pred, :][:, gene_indices]

    if control_type == "ctrl":
        ctrl = sample_x0_from_ctrl(
            adata,
            n_pred,
            ctrl_name="ctrl",
            additional_filt=additional_filt,
            selected_genes=top_genes,
        )
    elif control_type == "pert_train":
        ctrl = sample_x0_from_train_non_ctrl(
            adata,
            n_pred,
            ctrl_name="ctrl",
            additional_filt=additional_filt,
            selected_genes=top_genes,
        )

    # Compute metrics
    result_row = {}

    # Add covariate values to result
    for col in covariate_columns:
        if col in cond_values:
            result_row[col] = cond_values[col]
        else:
            result_row[col] = None

    result_row["mse"] = mse(truth.mean(0), pred.mean(0))
    result_row["pearson"] = pearsonr(truth.mean(0), pred.mean(0))[0]
    result_row["spearman"] = spearmanr(truth.mean(0), pred.mean(0))[0]
    result_row["mmd"] = compute_scalar_mmd(truth, pred)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        result_row["pearson_delta"] = pearsonr(
            truth.mean(0) - ctrl.mean(0), pred.mean(0) - ctrl.mean(0)
        )[0]

    result_row["w2d"] = wasserstein_2(truth, pred)
    result_row["e_distance"] = e_distance(truth, pred)

    return result_row


def compute_metrics(
    adata,
    adata_pred,
    pred_df,
    control_type="pert_train",
    covariate_columns=["condition"],
    top_n=20,
    is_drug=False,
    save_path=None,
    layer=None,
    n_jobs=1,
    n_pred=100,
):
    """
    Compute metrics for predictions.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing ground truth
    adata_pred : AnnData
        Annotated data object containing predictions
    pred_df : pd.DataFrame
        DataFrame containing conditions to evaluate
    control_type : str
        Type of control to use for pearson_delta calculation. Options: 'ctrl' or 'pert_train'
    covariate_columns : list or str
        Column name(s) for covariates
    top_n : int
        Number of top differentially expressed genes to use
    n_cells : int
        Not used (kept for backwards compatibility)
    save_path : str, optional
        Path to save results CSV
    n_jobs : int
        Number of parallel jobs. 1 for sequential, -1 for all cores.
        Note: If AnnData objects are backed (backed="r"), threading backend
        will be used automatically to avoid pickling issues.
    n_pred : int
        Number of cells to sample for control (used in pearson_delta calculation)

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with computed metrics for each condition
    """

    # Normalize covariate_columns to list
    if isinstance(covariate_columns, str):
        covariate_columns = [covariate_columns]

    adata_pred_genes = adata_pred.var_names.tolist()

    # Find condition column and context column
    condition_col = None
    context_col = None
    context_col_pos = None
    condition_col_pos = None

    for col in covariate_columns:
        if col == "condition":
            condition_col = col
            condition_col_pos = covariate_columns.index(col)
        else:
            context_col = col
            context_col_pos = covariate_columns.index(col)

    # Check if we need context handling
    has_context = len(covariate_columns) == 2 and context_col is not None

    # Get unique groups from pred_df
    unique_groups = pred_df[covariate_columns].drop_duplicates().values

    if n_jobs == 1:
        # Sequential execution with progress bar
        results_list = []
        for group in tqdm(unique_groups, desc="Computing metrics"):
            result_row = _compute_metrics_single_group(
                group,
                adata,
                adata_pred,
                control_type,
                covariate_columns,
                condition_col,
                context_col,
                condition_col_pos,
                context_col_pos,
                has_context,
                adata_pred_genes,
                top_n,
                layer,
                is_drug,
                n_pred,
            )
            if result_row is not None:
                results_list.append(result_row)
    else:
        # Parallel execution
        from joblib import Parallel, delayed

        # Detect if AnnData is backed (has HDF5 file handles that can't be pickled)
        is_backed = (hasattr(adata, "isbacked") and adata.isbacked) or (
            hasattr(adata_pred, "isbacked") and adata_pred.isbacked
        )

        if is_backed:
            print(
                f"Detected backed AnnData - using threading backend for parallelization (n_jobs={n_jobs})..."
            )
            backend = "threading"
        else:
            print(f"Computing metrics in parallel with multiprocessing (n_jobs={n_jobs})...")
            backend = "loky"

        results_list = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_compute_metrics_single_group)(
                group,
                adata,
                adata_pred,
                control_type,
                covariate_columns,
                condition_col,
                context_col,
                condition_col_pos,
                context_col_pos,
                has_context,
                adata_pred_genes,
                layer,
                top_n,
                is_drug,
                n_pred,
            )
            for group in unique_groups
        )
        # Filter out None results
        results_list = [r for r in results_list if r is not None]

    # Convert list of dicts to DataFrame
    result_df = pd.DataFrame(results_list)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        result_df.to_csv(save_path, index=False)
    return result_df


def _compute_no_change_single_group(
    group,
    adata,
    control_type,
    covariate_columns,
    condition_col,
    context_col,
    condition_col_pos,
    context_col_pos,
    has_context,
    top_n,
    n_pred,
):
    """
    Helper function to compute no-change metrics for a single group.
    This is extracted to enable parallelization.
    """
    # Extract values from group
    if has_context:
        condition = group[condition_col_pos]
        context = group[context_col_pos]
        cond_values = {condition_col: condition, context_col: context}
    else:
        condition = group if isinstance(group, str) else group[0]
        cond_values = {condition_col: condition}

    # Extract condition value for perturbation name
    cond_value = cond_values.get("condition")

    # Skip if ctrl
    if cond_value == "ctrl":
        return None

    # Get top genes - construct key from group
    if has_context:
        # For multi-covariate, join with '_'
        cond_key = "_".join(str(v) for v in group)
    else:
        cond_key = condition

    top_genes = adata.uns["rank_genes_groups_list"][cond_key][:top_n]
    top_genes = [gene for gene in top_genes if gene in adata.var_names]

    # Build filter for adata based on all covariate values
    filt = np.ones(adata.shape[0], dtype=bool)
    for col, val in cond_values.items():
        filt = filt & (adata.obs[col] == val)

    # n_pred = sum(filt)

    # Build additional filter for sampling control cells from same context
    if has_context and context_col is not None:
        # Filter for same context (non-condition covariates)
        additional_filt = adata.obs[context_col] == cond_values[context_col]
    else:
        additional_filt = None

    # Extract truth data using filter - handle backed files properly
    # Convert boolean filter to sorted indices (required for HDF5)
    truth_indices = np.where(filt)[0]
    truth_indices = np.sort(truth_indices)

    # Get gene indices for top_genes
    gene_mask = adata.var_names.isin(top_genes)

    # Extract data with single-dimension indexing to avoid HDF5 fancy indexing issues
    if hasattr(adata.X[:2, :2], "toarray"):
        truth = adata.X[truth_indices, :][:, gene_mask].toarray()
    else:
        truth = adata.X[truth_indices, :][:, gene_mask]

    if control_type == "ctrl":
        pred = sample_x0_from_ctrl(
            adata,
            n_pred,
            ctrl_name="ctrl",
            additional_filt=additional_filt,
            selected_genes=top_genes,
        )
        ctrl = pred
    elif control_type == "pert_train":
        pred = sample_x0_from_train_non_ctrl(
            adata,
            n_pred,
            ctrl_name="ctrl",
            additional_filt=additional_filt,
            selected_genes=top_genes,
        )
        ctrl = pred
    else:
        raise ValueError(f"Invalid control_type: {control_type}. Must be 'ctrl' or 'pert_train'.")

    # Compute metrics
    result_row = {}

    # Add covariate values to result
    for col in covariate_columns:
        if col in cond_values:
            result_row[col] = cond_values[col]
        else:
            result_row[col] = None

    result_row["mse"] = mse(truth.mean(0), pred.mean(0))
    result_row["pearson"] = pearsonr(truth.mean(0), pred.mean(0))[0]
    result_row["spearman"] = spearmanr(truth.mean(0), pred.mean(0))[0]
    result_row["mmd"] = compute_scalar_mmd(truth, pred)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        result_row["pearson_delta"] = pearsonr(
            truth.mean(0) - ctrl.mean(0), pred.mean(0) - ctrl.mean(0)
        )[0]

    result_row["w2d"] = wasserstein_2(truth, pred)
    result_row["e_distance"] = e_distance(truth, pred)

    return result_row


def compute_no_change_metrics(
    adata,
    pred_df,
    control_type="ctrl",
    covariate_columns=["condition"],
    top_n=20,
    save_path=None,
    n_jobs=1,
    n_pred=100,
):
    """
    Compute metrics using control cells as predictions (no-change baseline).
    This creates a baseline where we use control cells from the same context
    as the "prediction" and compare them to the true perturbed cells.

    Parameters
    ----------
    adata : AnnData
        The annotated data object containing ground truth
    pred_df : pd.DataFrame
        DataFrame containing the conditions to evaluate
    covariate_columns : list or str
        Column name(s) for covariates (e.g., ['condition'] or ['condition', 'cell_line'])
    top_n : int
        Number of top differentially expressed genes to use
    save_path : str, optional
        Path to save the results CSV
    n_jobs : int
        Number of parallel jobs. 1 for sequential, -1 for all cores.
        Note: If AnnData is backed, threading backend will be used automatically.
    n_pred : int
        Number of control cells to sample for each condition

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with computed metrics for each condition
    """

    # Normalize covariate_columns to list
    if isinstance(covariate_columns, str):
        covariate_columns = [covariate_columns]

    # Find condition column and context column
    condition_col = None
    context_col = None
    context_col_pos = None
    condition_col_pos = None

    for col in covariate_columns:
        if col == "condition":
            condition_col = col
            condition_col_pos = covariate_columns.index(col)
        else:
            context_col = col
            context_col_pos = covariate_columns.index(col)

    # Check if we need context handling
    has_context = len(covariate_columns) == 2 and context_col is not None

    # Get unique groups from pred_df
    unique_groups = pred_df[covariate_columns].drop_duplicates().values

    if n_jobs == 1:
        # Sequential execution with progress bar
        results_list = []
        for group in tqdm(unique_groups, desc="Computing no-change metrics"):
            result_row = _compute_no_change_single_group(
                group,
                adata,
                control_type,
                covariate_columns,
                condition_col,
                context_col,
                condition_col_pos,
                context_col_pos,
                has_context,
                top_n,
                n_pred,
            )
            if result_row is not None:
                results_list.append(result_row)
    else:
        # Parallel execution
        from joblib import Parallel, delayed

        # Detect if AnnData is backed
        is_backed = hasattr(adata, "isbacked") and adata.isbacked

        if is_backed:
            print(f"Detected backed AnnData - using threading backend (n_jobs={n_jobs})...")
            backend = "threading"
        else:
            print(f"Computing no-change metrics in parallel (n_jobs={n_jobs})...")
            backend = "loky"

        results_list = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_compute_no_change_single_group)(
                group,
                adata,
                control_type,
                covariate_columns,
                condition_col,
                context_col,
                condition_col_pos,
                context_col_pos,
                has_context,
                top_n,
                n_pred,
            )
            for group in unique_groups
        )
        # Filter out None results
        results_list = [r for r in results_list if r is not None]

    # Convert list of dicts to DataFrame
    result_df = pd.DataFrame(results_list)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        result_df.to_csv(save_path, index=False)
    return result_df


def _compute_pos_ctrl_single_group(
    group,
    adata,
    control_type,
    covariate_columns,
    condition_col,
    context_col,
    condition_col_pos,
    context_col_pos,
    has_context,
    top_n,
):
    """
    Helper function to compute no-change metrics for a single group.
    This is extracted to enable parallelization.
    """
    # Extract values from group
    if has_context:
        condition = group[condition_col_pos]
        context = group[context_col_pos]
        cond_values = {condition_col: condition, context_col: context}
    else:
        condition = group if isinstance(group, str) else group[0]
        cond_values = {condition_col: condition}

    # Extract condition value for perturbation name
    cond_value = cond_values.get("condition")

    # Skip if ctrl
    if cond_value == "ctrl":
        return None

    # Get top genes - construct key from group
    if has_context:
        # For multi-covariate, join with '_'
        cond_key = "_".join(str(v) for v in group)
    else:
        cond_key = condition

    top_genes = adata.uns["rank_genes_groups_list"][cond_key][:top_n]
    top_genes = [gene for gene in top_genes if gene in adata.var_names]

    # Build filter for adata based on all covariate values
    filt = np.ones(adata.shape[0], dtype=bool)
    for col, val in cond_values.items():
        filt = filt & (adata.obs[col] == val)

    n_pred = sum(filt)

    # Build additional filter for sampling control cells from same context
    if has_context and context_col is not None:
        # Filter for same context (non-condition covariates)
        additional_filt = adata.obs[context_col] == cond_values[context_col]
    else:
        additional_filt = None

    # Extract truth data using filter - handle backed files properly
    # Convert boolean filter to sorted indices (required for HDF5)
    truth_indices = np.where(filt)[0]
    truth_indices = np.sort(truth_indices)

    # Get gene indices for top_genes
    gene_mask = adata.var_names.isin(top_genes)

    # Extract data with single-dimension indexing to avoid HDF5 fancy indexing issues
    if hasattr(adata.X[:2, :2], "toarray"):
        truth = adata.X[truth_indices, :][:, gene_mask].toarray()
    else:
        truth = adata.X[truth_indices, :][:, gene_mask]

    # randomly split truth into two halves
    pred_indices = np.random.choice(truth.shape[0], size=n_pred // 2, replace=False)
    pred = truth[pred_indices, :]
    truth = truth[~np.isin(np.arange(truth.shape[0]), pred_indices), :]

    if control_type == "ctrl":
        ctrl = sample_x0_from_ctrl(
            adata,
            n_pred,
            ctrl_name="ctrl",
            additional_filt=additional_filt,
            selected_genes=top_genes,
        )
    elif control_type == "pert_train":
        ctrl = sample_x0_from_train_non_ctrl(
            adata,
            n_pred,
            ctrl_name="ctrl",
            additional_filt=additional_filt,
            selected_genes=top_genes,
        )
    else:
        raise ValueError(f"Invalid control_type: {control_type}. Must be 'ctrl' or 'pert_train'.")

    # Compute metrics
    result_row = {}

    # Add covariate values to result
    for col in covariate_columns:
        if col in cond_values:
            result_row[col] = cond_values[col]
        else:
            result_row[col] = None

    result_row["mse"] = mse(truth.mean(0), pred.mean(0))
    result_row["pearson"] = pearsonr(truth.mean(0), pred.mean(0))[0]
    result_row["spearman"] = spearmanr(truth.mean(0), pred.mean(0))[0]
    result_row["mmd"] = compute_scalar_mmd(truth, pred)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        result_row["pearson_delta"] = pearsonr(
            truth.mean(0) - ctrl.mean(0), pred.mean(0) - ctrl.mean(0)
        )[0]

    result_row["w2d"] = wasserstein_2(truth, pred)
    result_row["e_distance"] = e_distance(truth, pred)

    return result_row


def compute_pos_ctrl_metrics(
    adata,
    pred_df,
    control_type="pert_train",
    covariate_columns=["condition"],
    top_n=20,
    save_path=None,
    n_jobs=1,
):
    """
    Compute metrics using control cells as predictions (no-change baseline).
    This creates a baseline where we use control cells from the same context
    as the "prediction" and compare them to the true perturbed cells.

    Parameters
    ----------
    adata : AnnData
        The annotated data object containing ground truth
    pred_df : pd.DataFrame
        DataFrame containing the conditions to evaluate
    covariate_columns : list or str
        Column name(s) for covariates (e.g., ['condition'] or ['condition', 'cell_line'])
    top_n : int
        Number of top differentially expressed genes to use
    save_path : str, optional
        Path to save the results CSV
    n_jobs : int
        Number of parallel jobs. 1 for sequential, -1 for all cores.
        Note: If AnnData is backed, threading backend will be used automatically.

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with computed metrics for each condition
    """

    # Normalize covariate_columns to list
    if isinstance(covariate_columns, str):
        covariate_columns = [covariate_columns]

    # Find condition column and context column
    condition_col = None
    context_col = None
    context_col_pos = None
    condition_col_pos = None

    for col in covariate_columns:
        if col == "condition":
            condition_col = col
            condition_col_pos = covariate_columns.index(col)
        else:
            context_col = col
            context_col_pos = covariate_columns.index(col)

    # Check if we need context handling
    has_context = len(covariate_columns) == 2 and context_col is not None

    # Get unique groups from pred_df
    unique_groups = pred_df[covariate_columns].drop_duplicates().values

    if n_jobs == 1:
        # Sequential execution with progress bar
        results_list = []
        for group in tqdm(unique_groups, desc="Computing positive control metrics"):
            result_row = _compute_pos_ctrl_single_group(
                group,
                adata,
                control_type,
                covariate_columns,
                condition_col,
                context_col,
                condition_col_pos,
                context_col_pos,
                has_context,
                top_n,
            )
            if result_row is not None:
                results_list.append(result_row)
    else:
        # Parallel execution
        from joblib import Parallel, delayed

        # Detect if AnnData is backed
        is_backed = hasattr(adata, "isbacked") and adata.isbacked

        if is_backed:
            print(f"Detected backed AnnData - using threading backend (n_jobs={n_jobs})...")
            backend = "threading"
        else:
            print(f"Computing positive control metrics in parallel (n_jobs={n_jobs})...")
            backend = "loky"

        results_list = Parallel(n_jobs=n_jobs, backend=backend, verbose=10)(
            delayed(_compute_pos_ctrl_single_group)(
                group,
                adata,
                control_type,
                covariate_columns,
                condition_col,
                context_col,
                condition_col_pos,
                context_col_pos,
                has_context,
                top_n,
            )
            for group in unique_groups
        )
        # Filter out None results
        results_list = [r for r in results_list if r is not None]

    # Convert list of dicts to DataFrame
    result_df = pd.DataFrame(results_list)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        result_df.to_csv(save_path, index=False)
    return result_df


def compute_metrics_over_de(
    adata,
    adata_pred,
    pred_df,
    control_type="ctrl",
    covariate_columns=["condition"],
    compute_baseline=True,
    save_dir=None,
    suffix="",
    n_jobs=1,
):
    """
    Compute metrics over different numbers of top DE genes.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing ground truth
    adata_pred : AnnData
        Annotated data object containing predictions
    pred_df : pd.DataFrame
        DataFrame containing conditions to evaluate
    covariate_columns : list or str
        Column name(s) for covariates
    compute_baseline : bool
        Whether to compute no-change baseline metrics
    save_dir : str, optional
        Directory to save results
    suffix : str
        Suffix for saved file names
    n_jobs : int
        Number of parallel jobs for parallelizing over groups within each top_n.
        1 for sequential, -1 for all cores. Parallelization happens at the
        group level (not over different top_n values).

    Returns
    -------
    de_results : pd.DataFrame
        Results for predicted data
    baseline_results : pd.DataFrame (optional)
        Results for no-change baseline if compute_baseline=True
    """
    if save_dir is not None and os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # Sequential loop over n_values, but parallelize within each compute_metrics call
    n_values = [10, 20, 40, 60, 80, 100]

    de_results = []
    for n in n_values:
        print(f"Computing metrics for top {n} DE genes...")
        result_df = compute_metrics(
            adata,
            adata_pred,
            pred_df,
            covariate_columns=covariate_columns,
            top_n=n,
            n_jobs=n_jobs,  # Pass n_jobs to parallelize over groups
        )
        result_df["top_n"] = n
        de_results.append(result_df)

    de_results = pd.concat(de_results, axis=0)

    if compute_baseline:
        baseline_results = []
        for n in n_values:
            print(f"Computing baseline metrics for top {n} DE genes...")
            result_baseline = compute_no_change_metrics(
                adata,
                pred_df,
                control_type=control_type,
                covariate_columns=covariate_columns,
                top_n=n,
                n_jobs=n_jobs,  # Pass n_jobs to parallelize over groups
            )
            result_baseline["top_n"] = n
            baseline_results.append(result_baseline)

        baseline_results = pd.concat(baseline_results, axis=0)

    if save_dir is not None:
        de_results.to_csv(os.path.join(save_dir, f"results_de_over_n{suffix}.csv"), index=False)

    if compute_baseline and save_dir is not None:
        baseline_results.to_csv(
            os.path.join(save_dir, f"results_no_change_over_n{suffix}.csv"), index=False
        )
        return de_results, baseline_results
    else:
        return de_results


def read_metric_csv(path):
    if os.path.exists(path):
        result_df = pd.read_csv(path)
        return result_df
    else:
        print(f"File {path} does not exist.")
        return None


def _get_available_metric_columns(df, metrics=None):
    default_metrics = ["mse", "pearson", "pearson_delta", "spearman", "mmd", "w2d", "e_distance"]
    metric_list = default_metrics if metrics is None else metrics
    return [metric for metric in metric_list if metric in df.columns]


def _perform_paired_sample_test(
    result_df,
    metrics,
    reference_model="GFM",
    exclude_models=None,
    method="wilcoxon",
    alternative="two-sided",
    p_correction="fdr_bh",
):
    exclude_models = [] if exclude_models is None else exclude_models

    if "model" not in result_df.columns:
        raise ValueError("result_df must contain a 'model' column to perform paired sample tests.")

    if "split" not in result_df.columns:
        return None

    metrics = _get_available_metric_columns(result_df, metrics)
    if not metrics:
        return None

    all_models = result_df["model"].unique()
    comparison_models = [m for m in all_models if m != reference_model and m not in exclude_models]

    non_metric_cols = [col for col in result_df.columns if col not in set(metrics) | {"model"}]
    pairing_cols = [col for col in non_metric_cols if col != "split"]
    if "split" in non_metric_cols:
        pairing_cols.append("split")

    if not pairing_cols:
        raise ValueError("Unable to determine pairing columns for paired sample tests.")

    results_list = []
    better_alternative = {
        "mse": "less",
        "pearson": "greater",
        "spearman": "greater",
        "pearson_delta": "greater",
        "w2d": "less",
        "e_distance": "less",
        "mmd": "less",
    }

    for model in comparison_models:
        for metric in metrics:
            ref_data = result_df[result_df["model"] == reference_model][
                pairing_cols + [metric]
            ].copy()
            ref_data = ref_data.rename(columns={metric: "ref_value"})

            comp_data = result_df[result_df["model"] == model][pairing_cols + [metric]].copy()
            comp_data = comp_data.rename(columns={metric: "comp_value"})

            paired_data = ref_data.merge(comp_data, on=pairing_cols, how="inner")
            paired_data = paired_data.dropna(subset=["ref_value", "comp_value"])

            if len(paired_data) < 2:
                print(
                    f"Warning: Not enough paired samples for {model} vs {reference_model} on {metric}"
                )
                continue

            alt = (
                "two-sided"
                if alternative == "two-sided"
                else better_alternative.get(metric, "two-sided")
            )
            if method == "wilcoxon":
                stat, p_value = stats.wilcoxon(
                    paired_data["ref_value"],
                    paired_data["comp_value"],
                    alternative=alt,
                )
                m_diff = (paired_data["ref_value"] - paired_data["comp_value"]).median()
                ref_m = paired_data["ref_value"].median()
                comp_m = paired_data["comp_value"].median()
            elif method == "t-test":
                stat, p_value = stats.ttest_rel(
                    paired_data["ref_value"],
                    paired_data["comp_value"],
                    alternative=alt,
                )
                m_diff = (paired_data["ref_value"] - paired_data["comp_value"]).mean()
                ref_m = paired_data["ref_value"].mean()
                comp_m = paired_data["comp_value"].mean()
            else:
                raise ValueError(f"Unsupported paired test method: {method}")

            diff = paired_data["ref_value"] - paired_data["comp_value"]
            diff_std = diff.std(ddof=1)
            cohens_d = np.nan if diff_std == 0 or np.isnan(diff_std) else diff.mean() / diff_std

            results_list.append(
                {
                    "comparison_model": model,
                    "metric": metric,
                    "n_pairs": len(paired_data),
                    "statistic": stat,
                    "p_value": p_value,
                    "m_difference": m_diff,
                    "cohens_d": cohens_d,
                    "ref_m": ref_m,
                    "comp_m": comp_m,
                }
            )

    results_df = pd.DataFrame(results_list)
    if results_df.empty:
        return results_df

    if p_correction is not None and len(results_df) > 1:
        corrected = multipletests(results_df["p_value"], method=p_correction)
        results_df["adj_p_value"] = corrected[1]

    def get_sig_stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    results_df["significance"] = results_df["adj_p_value"].apply(get_sig_stars)
    return results_df


def _aggregate_results_by_split(result_df, metrics=None, groupby_cols=None):
    if "split" not in result_df.columns:
        return None

    metrics = _get_available_metric_columns(result_df, metrics)
    if not metrics:
        return None

    if groupby_cols is None:
        groupby_cols = ["model", "split"]

    missing_cols = [col for col in groupby_cols if col not in result_df.columns]
    if missing_cols:
        raise ValueError(f"Aggregation columns not found in result_df: {missing_cols}")

    aggregated = (
        result_df.groupby(groupby_cols, observed=True)[metrics].agg(["mean", "sem"]).reset_index()
    )
    aggregated.columns = [
        col if isinstance(col, str) else "_".join(str(part) for part in col if part)
        for col in aggregated.columns.to_flat_index()
    ]
    return aggregated


def _prepare_aggregated_mean_results_for_testing(aggregated_results, metrics):
    if aggregated_results is None:
        return None

    mean_metric_map = {
        f"{metric}_mean": metric
        for metric in metrics
        if f"{metric}_mean" in aggregated_results.columns
    }
    if not mean_metric_map:
        return None

    base_cols = [
        col
        for col in aggregated_results.columns
        if col not in mean_metric_map and not col.endswith("_sem")
    ]
    test_df = aggregated_results[base_cols + list(mean_metric_map.keys())].copy()
    return test_df.rename(columns=mean_metric_map)


def combine_results(
    result_dict,
    calculate_delta=True,
    normalize=False,
    baseline_model=None,
    pos_ctrl_model=None,
    return_test_results=False,
    reference_model="GFM",
    exclude_models=None,
    method="wilcoxon",
    alternative="two-sided",
    aggregate_by_split=False,
    aggregate_groupby_cols=None,
):
    names = list(result_dict.keys())
    categories = names.copy()
    df_list = []
    result_pos_ctrl = None
    if calculate_delta:
        if baseline_model not in categories:
            raise ValueError(
                f"Invalid baseline_model: {baseline_model}. Expected one of: {sorted(categories)}"
            )
        if normalize:
            categories.remove(baseline_model)
            categories.remove(pos_ctrl_model)
            for name in names:
                if name == baseline_model or name == pos_ctrl_model:
                    continue
                if isinstance(result_dict[name], list):
                    result_delta_list = []
                    for i, res_path in enumerate(result_dict[name]):
                        result = read_metric_csv(res_path)
                        result_no_pred = read_metric_csv(result_dict[baseline_model][i])
                        result_pos_ctrl = read_metric_csv(result_dict[pos_ctrl_model][i])

                        delta_out = calculate_delta_result(
                            result,
                            result_no_pred,
                            result_pos_ctrl=result_pos_ctrl,
                            normalize=normalize,
                        )
                        result_delta = delta_out[0] if isinstance(delta_out, tuple) else delta_out
                        result_delta["split"] = i + 1
                        result_delta_list.append(result_delta)
                    if result_delta_list:  # Only concatenate if list is not empty
                        result_delta = pd.concat(result_delta_list, axis=0, ignore_index=True)
                        result_delta["model"] = name
                        df_list.append(result_delta)
                else:
                    raise ValueError("result_dict values must be either str or list of str")
        else:  # not normalize
            categories.remove(baseline_model)
            for name in names:
                if name == baseline_model:
                    continue
                if isinstance(result_dict[name], list):
                    result_delta_list = []
                    for i, res_path in enumerate(result_dict[name]):
                        result = read_metric_csv(res_path)
                        result_no_pred = read_metric_csv(result_dict[baseline_model][i])
                        delta_out = calculate_delta_result(
                            result, result_no_pred, normalize=normalize
                        )
                        result_delta = delta_out[0] if isinstance(delta_out, tuple) else delta_out
                        result_delta["split"] = i + 1
                        result_delta_list.append(result_delta)
                    if result_delta_list:  # Only concatenate if list is not empty
                        result_delta = pd.concat(result_delta_list, axis=0, ignore_index=True)
                        result_delta["model"] = name
                        df_list.append(result_delta)
                else:
                    raise ValueError("result_dict values must be either str or list of str")
    else:
        categories = names
        for name in names:
            if isinstance(result_dict[name], list):
                for i, res_path in enumerate(result_dict[name]):
                    result = read_metric_csv(res_path)
                    if result is None:
                        continue
                    result["split"] = i + 1
                    result["model"] = name
                    df_list.append(result)
            else:
                raise ValueError("result_dict values must be either str or list of str")
    combined_df = pd.concat(df_list, axis=0, ignore_index=True)
    combined_df["model"] = combined_df["model"].astype("category")
    combined_df["model"] = combined_df["model"].cat.reorder_categories(categories)

    available_metrics = _get_available_metric_columns(combined_df)

    aggregated_results = None
    aggregated_test_df = None
    if aggregate_by_split:
        aggregated_results = _aggregate_results_by_split(
            combined_df,
            metrics=available_metrics,
            groupby_cols=aggregate_groupby_cols,
        )
        aggregated_test_df = _prepare_aggregated_mean_results_for_testing(
            aggregated_results,
            available_metrics,
        )

    test_results = None
    if return_test_results:
        test_result_df = aggregated_test_df if aggregate_by_split else combined_df
        test_results = _perform_paired_sample_test(
            test_result_df,
            metrics=available_metrics,
            reference_model=reference_model,
            exclude_models=exclude_models,
            method=method,
            alternative=alternative,
        )

    if return_test_results or aggregate_by_split:
        return {
            "combined_results": combined_df,
            "test_results": test_results,
            "aggregated_results": aggregated_results,
        }

    return combined_df


def multi_hot_label(
    label_sets, num_classes, normalization=False, device: str | torch.device = "cpu"
):
    """
    Vectorized multi-label one-hot encoding.
    label_sets: Tensor of shape (batch_size, num_labels_per_sample)
    """
    label_sets = torch.as_tensor(label_sets, device=device)
    if label_sets.ndim == 1:
        label_sets = label_sets.unsqueeze(0)
    batch_size = label_sets.shape[0]
    one_hot = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=device)

    mask = (label_sets >= 0) & (label_sets < num_classes)
    valid_labels = label_sets.clone()
    valid_labels[~mask] = 0  # Set invalids to 0 to avoid indexing errors

    # Scatter to one-hot
    one_hot.scatter_(1, valid_labels, mask.float())
    if normalization:
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        one_hot = one_hot / counts
    return one_hot


def make_LM_data_loader(
    pdata_train, condition_labels, pert_emb_tensor, batch_size=100, shuffle=True, device="cpu"
):
    metadata_subset = pdata_train.obs
    with torch.no_grad():
        pert_emb_list = []
        truth_delta_x_list = []
        unique_pairs = metadata_subset[["condition", "cell_type"]].drop_duplicates().values
        for cond, ct in unique_pairs:
            if cond not in condition_labels.keys():
                print(f"{cond} is not in the PCA embedding, cannot be predicted")
                continue
            else:
                mohe = multi_hot_label(
                    condition_labels[cond], len(pdata_train.var_names), device=device
                )
                pert_emb_cond = torch.matmul(mohe, pert_emb_tensor)

                x_truth = pdata_train[
                    (pdata_train.obs["condition"] == cond) & (pdata_train.obs["cell_type"] == ct)
                ].X
                x_ctrl = pdata_train[
                    (pdata_train.obs["condition"] == "ctrl") & (pdata_train.obs["cell_type"] == ct)
                ].X
                delta_x = x_truth - x_ctrl
                n_cells = delta_x.shape[0]

                # Repeat the perturbation embedding for each cell
                pert_emb_cond_repeated = pert_emb_cond.repeat(n_cells, 1)
                pert_emb_list.append(pert_emb_cond_repeated)

                truth_delta_x_list.append(torch.tensor(delta_x, dtype=torch.float32).to(device))

        pert_emb_all = torch.cat(pert_emb_list, dim=0)
        truth_delta_x_all = torch.cat(truth_delta_x_list, dim=0)
        dataset = torch.utils.data.TensorDataset(pert_emb_all, truth_delta_x_all)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader


def train_and_predict_linear_model(adata, drug_graph=False, device="cpu"):
    print("Creating perturbation embeddings using PCA...")
    # make perturbation embeddings
    import decoupler as dc

    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "all_cells"

    pdata = dc.pp.pseudobulk(adata, sample_col="condition", groups_col="cell_type", mode="mean")
    pdata_train = pdata[pdata.obs["split"] == "train"].copy()
    pdata_val = pdata[pdata.obs["split"] == "val"].copy()

    pdata_ctrl = pdata[pdata.obs["condition"] == "ctrl"].copy()
    pdata_val = sc.concat([pdata_val, pdata_ctrl], axis=0)

    # PCA for perturbation embeddings
    from sklearn.decomposition import PCA

    X_train_T = pdata_train.X.T

    # if the number of training perturbations is less than 50, set n_components to number of perturbations - 1 to avoid errors
    if X_train_T.shape[1] >= 50:
        n_components = 50
    else:
        n_components = X_train_T.shape[1] - 1

    pca = PCA(n_components=n_components)
    pert_emb = pca.fit_transform(X_train_T)
    pert_emb_tensor = torch.tensor(pert_emb, dtype=torch.float32).to(device)

    pert_names = np.array(adata.obs["condition"].unique().tolist())
    condition_labels = make_condition_labels_graph(
        pert_names, pdata_train.var_names, drug_graph=drug_graph
    )

    train_loader = make_LM_data_loader(
        pdata_train, condition_labels, pert_emb_tensor, device=device
    )

    val_loader = make_LM_data_loader(pdata_val, condition_labels, pert_emb_tensor, device=device)

    print("Training linear perturbation model...")
    import torch.optim as optim

    from gfm.models import LinearPerturbationModel

    # Initialize model
    model = LinearPerturbationModel(
        n_pc=pert_emb_tensor.shape[1], gene_dim=pdata_train.shape[1]
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    early_stopping_patience = 100
    best_val_loss = float("inf")
    num_epochs = 1000
    patience_counter = 0
    for epoch in range(num_epochs):
        train_loss = 0
        for pert_embs_batch, true_data_batch in train_loader:
            model.train()

            pred = model(pert_embs_batch)
            loss = criterion(pred, true_data_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = 0
        for pert_embs_batch, true_data_batch in val_loader:
            model.eval()

            pred = model(pert_embs_batch)
            val_loss += criterion(pred, true_data_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

        if (epoch + 1) % 100 == 1:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    print(f"\nFinal Loss: {loss.item():.4f}")

    print("Making predictions with linear model...")
    # make predictions
    model.eval()
    pdata_test = pdata[pdata.obs["split"] == "test"].copy()
    metadata_subset = pdata_test.obs
    unique_pairs = metadata_subset[["condition", "cell_type"]].drop_duplicates().values
    with torch.no_grad():
        pred_list = []
        cond_list = []
        ct_list = []
        for cond, ct in unique_pairs:
            if cond not in condition_labels.keys():
                print(f"{cond} is not in the PCA embedding, cannot be predicted")
                continue
            else:
                mohe = multi_hot_label(
                    condition_labels[cond], len(pdata_test.var_names), device=device
                )
                pert_emb_cond = torch.matmul(mohe, pert_emb_tensor)

                pred_delta = model(pert_emb_cond).cpu().numpy()
                ctrl = pdata[(pdata.obs["condition"] == "ctrl") & (pdata.obs["cell_type"] == ct)].X
                pred = pred_delta + ctrl
                pred_list.append(pred)

                n_cells = sum(
                    (pdata_test.obs["condition"] == cond) & (pdata_test.obs["cell_type"] == ct)
                )
                cond_name = "+".join([p for p in cond.split("+") if p != "ctrl"])
                cond_list.extend([cond_name] * n_cells)
                ct_list.extend([ct] * n_cells)

        adata_pred_lm = sc.AnnData(
            np.vstack(pred_list), obs={"condition": cond_list, "cell_type": ct_list}
        )
        adata_pred_lm.var_names = pdata_test.var_names.tolist()

    return adata_pred_lm


def run_additive_model(adata, split_dict_subgroup_path, result_path):
    split_dict_subgroup = pickle.load(open(split_dict_subgroup_path, "rb"))
    single_pert_set = set(
        [
            item
            for cond in split_dict_subgroup["test_subgroup"]["combo_seen2"]
            for item in cond.split("+")
        ]
    )

    # Derive the regular split_dict_path from the subgroup path
    split_dict_path = split_dict_subgroup_path.replace("_subgroup.pkl", ".pkl")

    # Add split information to adata if it doesn't already exist
    if "split" not in adata.obs.columns:
        split_handler = SplitHandler(split_dict_path=split_dict_path)
        split_handler.add_split_to_adata(adata)

    pert_to_single_cond = {
        pert: next(
            (cond for cond in adata.obs["condition"].unique() if pert in cond and "ctrl" in cond),
            None,
        )
        for pert in single_pert_set
    }
    preds = []
    perts = []

    ctrl = adata[adata.obs["condition"] == "ctrl", :].X.toarray().mean(axis=0)
    for pert in split_dict_subgroup["test_subgroup"]["combo_seen2"]:
        p1 = pert.split("+")[0]
        p2 = pert.split("+")[1]

        single_pert_1 = pert_to_single_cond[p1]
        single_pert_2 = pert_to_single_cond[p2]

        if single_pert_1 is None and single_pert_2 is not None:
            dX_p2 = (
                adata[adata.obs["condition"] == single_pert_2, :].X.toarray().mean(axis=0) - ctrl
            )
            X_pred = ctrl + dX_p2
        elif single_pert_2 is None and single_pert_1 is not None:
            dX_p1 = (
                adata[adata.obs["condition"] == single_pert_1, :].X.toarray().mean(axis=0) - ctrl
            )
            X_pred = ctrl + dX_p1
        elif single_pert_1 is None or single_pert_2 is None:
            print(
                f"Could not find single perturbation condition for {pert}, use control as prediction"
            )
            X_pred = ctrl
        else:
            dX_p1 = (
                adata[adata.obs["condition"] == single_pert_1, :].X.toarray().mean(axis=0) - ctrl
            )
            dX_p2 = (
                adata[adata.obs["condition"] == single_pert_2, :].X.toarray().mean(axis=0) - ctrl
            )

            X_pred = ctrl + dX_p1 + dX_p2

        perts.append(pert)
        preds.append(X_pred)

    adata_pred_add = sc.AnnData(np.vstack(preds), obs={"condition": perts})
    adata_pred_add.var_names = adata.var_names.tolist()

    pred_df = pd.DataFrame({"condition": split_dict_subgroup["test_subgroup"]["combo_seen2"]})
    df = compute_metrics(
        adata,
        adata_pred_add,
        pred_df,
        control_type="pert_train",
        covariate_columns=["condition"],
        n_jobs=-1,
        top_n=20,
    )
    df["w2d"] = None
    df["mmd"] = None

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    df.to_csv(result_path, index=False)
