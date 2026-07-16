import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import umap
import seaborn as sns
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import spearmanr

from gfm.helpers import sample_x0_from_ctrl

def plot_correlation_dotplot(
        result_delta,
        x_cols = ['n_source_nodes_trained', 'train_node_ratio', 'max_sim', 'median_sim'],
        y_cols = ['mse', 'pearson', 'pearson_delta', 'mmd', 'w2d'],
        x_cols_labels = ['Number of training-set neighbors', 'Fraction of training-set neighbors', 'Maximum Similarity', 'Median Similarity'],
        save_path=None
        ):
    """
    Create a dot plot showing Spearman correlations between variables.
    Dot size represents |rho| and color represents -log10(p-value).
    """

    y_label_dict = {
        'mse': 'Δ MSE',
        'pearson': 'Δ Pearson',
        'pearson_delta': 'Pearson Delta',
        'spearman': 'Δ Spearman',
        'mmd': 'Δ MMD',
        'w2d': 'Δ 2-WD',
        'e_distance': 'Δ E-Distance'
    }
    
    # Calculate correlations for all x/y pairs
    correlations = []
    for i, y_col in enumerate(y_cols):
        for j, x_col in enumerate(x_cols):
            # Get data and remove NaN values
            x_data = result_delta[x_col].values
            y_data = result_delta[y_col].values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) > 1:
                rho, p_value = spearmanr(x_clean, y_clean)
                # Avoid log10(0) by setting a floor for p-values
                p_value = max(p_value, 1e-300)
                neg_log_p = -np.log10(p_value)
            else:
                rho = np.nan
                neg_log_p = 0
            
            correlations.append({
                'x_idx': j,
                'y_idx': i,
                'x_col': x_col,
                'y_col': y_col,
                'x_label': x_cols_labels[j],
                'y_label': y_label_dict.get(y_col, y_col),
                'rho': rho,
                'neg_log_p': neg_log_p
            })
    
    corr_df = pd.DataFrame(correlations)
    
    # Create figure with more width for legends
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create scatter plot where position is based on indices
    # Size based on |rho|, color based on -log10(p)
    # Reduce size scaling to prevent dot overlap
    sizes = np.abs(corr_df['rho']) * 300  # Reduced from 500
    colors = corr_df['neg_log_p']
    
    scatter = ax.scatter(
        corr_df['x_idx'], 
        corr_df['y_idx'], 
        s=sizes, 
        c=colors,
        cmap='viridis',
        alpha=0.7,
    )
    
    # Add smaller colorbar for p-values at the top right
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="3%", height="40%", loc='upper left', 
                     bbox_to_anchor=(1.02, 0.0, 1, 1), bbox_transform=ax.transAxes)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label('-log10(p-value)', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # Add size legend for |rho| (positioned below the colorbar)
    rho_values = [0.2, 0.6, 1.0]
    legend_elements = []
    for rho_val in rho_values:
        legend_elements.append(
            plt.scatter([], [], s=rho_val * 300, c='gray', alpha=0.7, 
                        label=f'{rho_val:.1f}')
        )
    
    # Position size legend below colorbar, not overlapping
    size_legend = ax.legend(
        handles=legend_elements,
        title='|ρ| (Spearman)',
        loc='lower left',
        bbox_to_anchor=(1.0, 0.0, 1, 1),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        labelspacing=1.5,  # Increase vertical spacing between legend items
        scatterpoints=1,
        borderpad=1.2
    )
    
    # Set ticks and labels with smaller fonts
    ax.set_xticks(range(len(x_cols)))
    ax.set_yticks(range(len(y_cols)))
    ax.set_xticklabels(x_cols_labels, rotation=30, ha='right', fontsize=9)
    y_tick_labels: list[str] = []
    for y_col in y_cols:
        if y_col is None:
            y_tick_labels.append('')
        elif y_col in y_label_dict:
            y_tick_labels.append(y_label_dict[y_col])
        else:
            y_tick_labels.append(y_col)
    ax.set_yticklabels(y_tick_labels, fontsize=9)
    
    # Invert y-axis so first metric is at top
    ax.invert_yaxis()
    
    # Set axis limits to reduce empty space
    ax.set_xlim(-0.5, len(x_cols) - 0.5)
    ax.set_ylim(len(y_cols) - 0.5, -0.5)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Spearman Correlation Analysis', fontsize=11, pad=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()
    
    return corr_df



def get_adata_condition_subset(adata_path_dict, condition):
    adata_list = []
    missing_models = []
    for model, path in adata_path_dict.items():
        adata = sc.read_h5ad(path, backed='r')
        
        if model == "Truth":
            filt = adata.obs['condition'] == condition
            # split_handler = SplitHandler(split_dict_path = split_dict_path)
            # split_handler.add_split_to_adata(adata)
            # get negative control
            # x0 = sample_x0_from_train_non_ctrl(adata, sum(filt))
            # adata_ctrl = sc.AnnData(X=x0, obs=adata.obs[filt], var=adata.var)
            # adata_ctrl.obs['model'] = 'Pert train'
            x0 = sample_x0_from_ctrl(adata, sum(filt))
            adata_ctrl = sc.AnnData(X=x0, var=adata.var)
            adata_ctrl.obs['condition'] = 'ctrl'
            # adata_ctrl = adata[adata.obs['condition'] == 'ctrl'].to_memory()
            adata_ctrl.obs['model'] = 'Control'
            adata_list.append(adata_ctrl)
            # get rank_genes_groups_list
            uns_rank_genes_groups_list = adata.uns['rank_genes_groups_list']

            adata_subset = adata[filt].to_memory()
            if adata_subset.n_obs == 0:
                missing_models.append(model)
                continue
            adata_subset.obs['model'] = model
            adata_list.append(adata_subset)
        else:
            pert = '+'.join([pert_gene for pert_gene in condition.split('+') if pert_gene != 'ctrl'])
            filt = adata.obs['condition'] == pert
            adata_subset = adata[filt].to_memory()
            if adata_subset.n_obs == 0:
                print(f"Warning: No cells found for model '{model}' under condition '{condition}' (looking for perturbation '{pert}'). Skipping this model.")
                missing_models.append(model)
                continue
            adata_subset.obs['model'] = model
            adata_list.append(adata_subset)

    if not adata_list:
        raise ValueError(
            f"No cells found for condition '{condition}'. Checked models: {', '.join(adata_path_dict.keys())}."
        )

    adata = sc.concat(adata_list, axis=0)
    adata.obs_names_make_unique()
    adata.uns['rank_genes_groups_list'] = uns_rank_genes_groups_list

    n_pca_components = min(50, adata.n_obs - 1, adata.n_vars - 1)
    if n_pca_components < 1:
        missing_models_msg = ""
        if missing_models:
            missing_models_msg = f" Missing model subsets: {', '.join(missing_models)}."
        raise ValueError(
            f"Need at least 2 cells and 2 genes to compute PCA/UMAP for condition '{condition}', "
            f"but got {adata.n_obs} cells and {adata.n_vars} genes after concatenation.{missing_models_msg}"
        )

    sc.pp.pca(adata, n_comps=n_pca_components)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    return adata

def plot_delta_expression_boxplot_model(
        adata, 
        cond, 
        model_name_list,
        top_n=10,
        palette=None,
        save_path=None,
        return_plot_df=False,
        figsize=(7, 2)
):
    colormap_name = 'tab10'
    
    if palette is None:
        palette = sns.color_palette(colormap_name, n_colors=len(model_name_list))
    
    pert = '+'.join([pert_gene for pert_gene in cond.split('+') if pert_gene != 'ctrl'])

    top_genes = adata.uns['rank_genes_groups_list'][cond][:top_n]

    plot_df_ctrl_mean = adata[adata.obs['model'] == 'Control', top_genes].to_df().mean()

    plot_df_list = []
    for model_name in model_name_list:
        if model_name == 'Control':
            continue
        if model_name == 'Truth':
            df = adata[(adata.obs['condition'] == cond) & (adata.obs['model'] == model_name), top_genes].to_df()
            plot_df_norm = df - plot_df_ctrl_mean
        else:
            df = adata[(adata.obs['condition'] == pert) & (adata.obs['model'] == model_name), top_genes].to_df()
            plot_df_norm = df - plot_df_ctrl_mean

        plot_df_melted = plot_df_norm.melt(var_name='Gene', value_name='Expression')
        plot_df_melted['Model'] = model_name
        plot_df_list.append(plot_df_melted)

    combined_plot_df = pd.concat(plot_df_list, ignore_index=True)
    combined_plot_df['Model'] = pd.Categorical(combined_plot_df['Model'], categories=model_name_list)

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(data=combined_plot_df, x='Gene', y='Expression', hue='Model', 
                palette=palette, fill=True, dodge=True, gap=0.1, 
                linewidth=0.25, fliersize=0.25, ax=ax, legend=False)
    if isinstance(palette, list):
        legend_elements = [Patch(facecolor=palette[i], label=model_name) for i, model_name in enumerate(model_name_list)]
    else:
        legend_elements = [Patch(facecolor=palette[model_name], label=model_name) for model_name in model_name_list]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(model_name_list),
        frameon=False,
    )
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.set_title(f"Perturbed gene: {pert}")
    ax.set_xlabel('')
    current_labels = ax.get_xticklabels()
    ax.set_xticklabels(current_labels, rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylabel('LogFC (normalized to control)')

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    if return_plot_df:
        return combined_plot_df
    
def plot_umap_by_model(adata, adata_paths, model_color_dict, condition, 
                       models = ['GFM', 'CellFlow', 'GEARS', 'scGPT'],
                       save_path=None, umap_kwargs=None, figsize=(7.08, 2), size=10):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    categories = list(adata_paths.keys())
    categories.append('Control')

    if model_color_dict is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        if not default_colors:
            default_colors = list(mcolors.TABLEAU_COLORS.values())
        model_color_dict = {
            model_name: default_colors[i % len(default_colors)]
            for i, model_name in enumerate(categories)
        }

    adata.obs['model'] = pd.Categorical(adata.obs['model'], categories=categories)
    adata.uns['model_colors'] = [model_color_dict[m] for m in adata.obs['model'].cat.categories]
    pert = '+'.join([pert_gene for pert_gene in condition.split('+') if pert_gene != 'ctrl'])
    plot_kwargs = {
        'color': ['model'],
        'alpha': 0.6,
        'size': size,
        'frameon': False,
        'title': '',
        'legend_loc': None,
    }
    if umap_kwargs is not None:
        plot_kwargs.update(umap_kwargs)

    fig, axes = plt.subplots(1, len(models), figsize=figsize)
    if len(models) == 1:
        axes = [axes]
    for i, model in enumerate(models):
        adata_plot = adata[adata.obs['model'].isin([model, 'Truth', 'Control'])].copy()
        sc.pl.umap(adata_plot, ax=axes[i], show=False, **plot_kwargs)

    # Create a single legend on the right side with all models
    all_models = adata.obs['model'].cat.categories
    legend_elements = [Patch(facecolor=model_color_dict[m], label=m) for m in all_models]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(all_models),
        frameon=False,
    )
    fig.suptitle(f'Perturbed Gene: {pert}', fontsize=14, y=1.02)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()