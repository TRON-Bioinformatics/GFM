import torch
import numpy as np
import pickle
import scanpy as sc
import os
from tqdm.auto import tqdm
import pandas as pd

from torchcfm import OTPlanSampler
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import ODESolver

from gfm.models import VAE, ConditionAwareVAE, SCVIVAE, ConditionalODE, ODEWrapper, ModelGuidanceWrapper
from gfm.helpers import (build_graph, make_condition_labels_graph, make_condot_data_loader, make_data_loader, make_prediction_data_loader,
                         get_cell_embedding, SplitHandler)
from gfm.vae_training_utils import (make_vae_dataloader, make_condition_aware_vae_dataloader, train_vae, train_condition_aware_vae,
                                    make_scvi_vae_dataloader, train_scvi_vae)
from gfm.train import evaluate_metrics_condot, evaluate_metrics_no_fm, evaluate_one_epoch_condot, evaluate_one_epoch_no_fm, train_one_epoch, evaluate_one_epoch, evaluate_metrics, train_one_epoch_condot, train_one_epoch_no_fm


class GFM:
    def __init__(
            self,
            adata: sc.AnnData | None = None,
            vae_hidden_dim: int = 500,
            latent_dim: int = 50,
            data_hidden_dim: int = 200,
            pert_hidden_dim: int = 50,
            pert_latent_dim: int = 50,
            gnn_num_layers: int = 2,
            time_embed_dim: int = 50,
            condition_method: str = 'concat',
            aggregation_method: str = 'sum',
            split_dict_path: str | None = None, 
            split_dict: dict | None = None, 
            split_df_path: str | None = None, 
            split_df: pd.DataFrame | None = None,
            output_dir: str | None = None,
            vae_save_path: str | None = None,
            vae_name: str | None = None,
            model_name: str | None = None,
            embed_name: str = 'X_vae',
            device: str = 'cpu',
            train_with_ctrl: bool = True,
            flow_from_ctrl: bool = False,
            no_fm: bool = False,
            use_contrastive: bool = False,
            use_condition_classifier: bool = False,
            use_null_embedding: bool = False,
            use_scvi_vae: bool = False,
            use_my_scvi_vae: bool = False,
            scvi_likelihood: str = 'nb'
    ):

        self.adata = adata
        self.latent_dim = latent_dim
        self.data_hidden_dim = data_hidden_dim
        self.pert_hidden_dim = pert_hidden_dim
        self.pert_latent_dim = pert_latent_dim
        self.gnn_num_layers = gnn_num_layers
        self.time_embed_dim = time_embed_dim
        self.condition_method = condition_method
        self.aggregation_method = aggregation_method
        self.output_dir = output_dir
        self.embed_name = embed_name
        self.vae_hidden_dim = vae_hidden_dim
        self.device = device
        self.train_with_ctrl = train_with_ctrl
        self.model_name = 'model.pt' if model_name is None else model_name
        self.vae_name = 'vae.pt' if vae_name is None else vae_name
        self.flow_from_ctrl = flow_from_ctrl
        self.no_fm = no_fm
        self.use_contrastive = use_contrastive
        self.use_condition_classifier = use_condition_classifier
        self.use_null_embedding = use_null_embedding
        self.use_scvi_vae = use_scvi_vae
        self.use_my_scvi_vae = use_my_scvi_vae
        self.scvi_likelihood = scvi_likelihood

        if self.adata is None:
            raise ValueError("Anndata must be provided to initialize GFM.")

        if self.use_scvi_vae and self.use_my_scvi_vae:
            raise ValueError("use_scvi_vae and use_my_scvi_vae are mutually exclusive")

        if self.output_dir is None:
                self.output_dir = './'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if vae_save_path is not None:
            self.vae_save_path = vae_save_path
        else:
            self.vae_save_path = os.path.join(self.output_dir, self.vae_name)
        
        self.split = SplitHandler(
            split_dict_path=split_dict_path, 
            split_dict=split_dict, 
            split_df_path=split_df_path, 
            split_df=split_df
        )
        self.split.add_split_to_adata(self.adata)
        
        self.model_save_path = os.path.join(self.output_dir, self.model_name)

        if self.use_my_scvi_vae:
            print(f"Initializing custom SCVIVAE (likelihood={self.scvi_likelihood})")
            self.my_scvi_vae = SCVIVAE(
                input_dim=len(self.adata.var_names),
                hidden_dim=self.vae_hidden_dim,
                latent_dim=self.latent_dim,
                likelihood=self.scvi_likelihood,
            ).to(self.device)
            # Initialize library priors from training counts if available
            counts = self.adata.layers['counts'] if 'counts' in self.adata.layers else self.adata.X
            try:
                if hasattr(counts, 'toarray'):
                    counts_arr = counts.toarray()  # type: ignore[reportAttributeAccessIssue]
                else:
                    counts_arr = np.asarray(counts)
                self.my_scvi_vae.set_library_priors_from_counts(counts_arr)
            except Exception as e:
                print(f"Warning: could not set library priors from counts ({e}). Using defaults.")
        elif not self.use_scvi_vae:
            if not self.use_contrastive and not self.use_condition_classifier:
                print("Initializing VAE without contrastive loss or condition classifier")
                self.vae = VAE(
                            input_dim=len(self.adata.var_names), 
                            hidden_dim=self.vae_hidden_dim, 
                            latent_dim=self.latent_dim
                            ).to(self.device)
                
            elif self.use_contrastive and not self.use_condition_classifier:
                print("Initializing VAE with contrastive loss but without condition classifier")
                self.condition_aware_vae = ConditionAwareVAE(
                            input_dim=len(self.adata.var_names),
                            hidden_dim=512,
                            latent_dim=50,
                            num_conditions=self.adata.obs['condition'].nunique(),
                            use_contrastive=True,
                            use_condition_classifier=False,  # Try False first, then True
                            temperature=0.1  # Lower = tighter clusters (0.07-0.2)
                        ).to(self.device)
            elif self.use_contrastive and self.use_condition_classifier:
                print("Initializing VAE with contrastive loss and condition classifier")
                self.condition_aware_vae = ConditionAwareVAE(
                            input_dim=len(self.adata.var_names),
                            hidden_dim=512,
                            latent_dim=50,
                            num_conditions=self.adata.obs['condition'].nunique(),
                            use_contrastive=True,
                            use_condition_classifier=True,
                            temperature=0.1
                        ).to(self.device)
        else:
            print(f"Initializing VAE from scVI package (likelihood={self.scvi_likelihood})")
            from scvi.model import SCVI
            SCVI.setup_anndata(self.adata, layer="counts")
            self.scvi_vae = SCVI(self.adata, n_latent=self.latent_dim, n_hidden=self.vae_hidden_dim)
            

        self.pert_names = np.array(self.adata.obs['condition'].unique().tolist())
        if not self.train_with_ctrl:
            self.pert_names = [cond for cond in self.pert_names if cond != 'ctrl']
        
        if len(self.split.covariate_columns) > 1:
            context_column = [x for x in self.split.covariate_columns if x != 'condition'][0]
            self.context_names = np.array(self.adata.obs[context_column].unique().tolist())
            self.context_to_idx = {ctx: idx for idx, ctx in enumerate(self.context_names)}
        else:
            self.context_names = None
            self.context_to_idx = None


    def initialize_fm(
            self, 
            pert_adata_path = None,
            graph_type = 'go', 
            k = 20, 
            pert_encoding = 'gat',
            graph_dir = None,
            randomize_graph = False):
        
        # Common arguments for all ConditionalODE instantiations
        common_args = {
            'data_latent_dim': self.latent_dim,
            'data_hidden_dim': self.data_hidden_dim,
            'pert_latent_dim': self.pert_latent_dim,
            'pert_hidden_dim': self.pert_hidden_dim,
            'time_embed_dim': self.time_embed_dim,
            'condition_method': self.condition_method,
            'aggregation_method': self.aggregation_method,
            'use_null_embedding': self.use_null_embedding,
            'context_size': len(self.context_names) if self.context_names is not None else 0
        }
    
        if graph_type == 'one_hot':
            self.pert_names_graph = np.array([cond.replace('+', '').replace('ctrl', '') for cond in self.pert_names if cond != 'ctrl'])
            self.condition_labels_graph = make_condition_labels_graph(self.pert_names, self.pert_names_graph)

            self.model = ConditionalODE(
                pert_input_size=len(self.pert_names_graph),
                pert_encoding=graph_type,
                **common_args
            ).to(self.device)

        else:
            self.edge_index, self.edge_weight, self.edge_list, self.pert_names_graph, self.condition_labels_graph = build_graph(
                graph_type=graph_type,
                pert_encoding=pert_encoding,
                pert_names=self.pert_names,
                k=k,
                randomize_graph=randomize_graph,
                adata=self.adata,
                split_dict=self.split.split_dict,
                graph_dir=graph_dir,
                pert_adata_path=pert_adata_path,
                device=self.device,
            )

            self.model = ConditionalODE(
                pert_input_size=len(self.pert_names_graph),
                pert_graph=self.edge_index,
                pert_graph_weight=self.edge_weight,
                pert_encoding=pert_encoding,
                gnn_num_layers=self.gnn_num_layers,
                **common_args
            ).to(self.device)
        

    def pretrain_vae(
            self,
            batch_size=512,
            max_epochs=500,
            early_stopping_patience=100,
            min_beta=1.0,
            max_beta=1.0,
            warmup_epochs=1,
            contrastive_weight=0.3,
            classifier_weight=0.05,
            return_training_history=False,
            lr=1e-3
            ):
        """
        Pretrain VAE with optional validation.
        
        Args:
            batch_size: Batch size for training
            max_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping (only used with validation)
            min_beta: Starting beta value for KL warmup
            max_beta: Final beta value for KL
            warmup_epochs: Number of epochs to warmup beta
            contrastive_weight: Weight for contrastive loss (if using contrastive VAE)
            classifier_weight: Weight for classifier loss (if using classifier)
            return_training_history: Whether to return training history
            lr: Learning rate
        """
        if self.adata is None:
            raise ValueError("Anndata must be provided to pretrain VAE.")

        # Prepare data splits using boolean masks (memory-efficient, no copying)
        train_mask = self.adata.obs['split'] == 'train'
        val_mask = self.adata.obs['split'] == 'val'
        
        # Check if we have validation data
        has_validation = val_mask.sum() > 0
        
        if has_validation:
            print(f"Training VAE with validation ({val_mask.sum()} validation samples)")
        else:
            print("Training VAE without validation")

        if self.use_my_scvi_vae:
            print("Training custom SCVIVAE model")
            train_loader = make_scvi_vae_dataloader(
                self.adata, batch_size=batch_size, shuffle=True,
                device=self.device, indices=train_mask,
            )
            val_loader = make_scvi_vae_dataloader(
                self.adata, batch_size=batch_size, shuffle=False,
                device=self.device, indices=val_mask,
            ) if has_validation else None
            optimizer = torch.optim.AdamW(self.my_scvi_vae.parameters(), lr=lr)
            try:
                training_history = train_scvi_vae(
                    self.my_scvi_vae, train_loader, optimizer,
                    val_loader=val_loader,
                    device=self.device,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    beta=max_beta,
                    return_training_history=return_training_history,
                )
            finally:
                torch.save(self.my_scvi_vae.state_dict(), self.vae_save_path)
                print(f"SCVIVAE model saved to {self.vae_save_path}")
                self.adata.obsm['X_vae'] = self.my_scvi_vae.get_latent_representation(
                    self.adata, batch_size=batch_size, device=self.device,
                )
                print(f"SCVIVAE embeddings stored in adata.obsm['X_vae']")
            if return_training_history:
                return training_history
            return

        if not self.use_scvi_vae:
            try:
                # Standard VAE (no contrastive, no classifier)
                if not self.use_contrastive and not self.use_condition_classifier:
                    print("Training standard VAE")
                    train_loader = make_vae_dataloader(self.adata, batch_size=batch_size, shuffle=True, device=self.device, indices=train_mask)
                    val_loader = make_vae_dataloader(self.adata, batch_size=batch_size, shuffle=False, device=self.device, indices=val_mask) if has_validation else None
                    optimizer = torch.optim.AdamW(self.vae.parameters(), lr=lr)

                    training_history = train_vae(
                        self.vae, train_loader, optimizer, 
                        val_loader=val_loader,
                        device=self.device,
                        batch_size=batch_size, 
                        max_epochs=max_epochs, 
                        early_stopping_patience=early_stopping_patience,
                        min_beta=min_beta, 
                        max_beta=max_beta, 
                        warmup_epochs=warmup_epochs,
                        return_training_history=return_training_history
                    )
                
                # Condition-aware VAE (with contrastive and/or classifier)
                else:
                    print(f"Training condition-aware VAE (contrastive={self.use_contrastive}, classifier={self.use_condition_classifier})")
                    train_loader, group_to_idx = make_condition_aware_vae_dataloader(
                        self.adata, batch_size=batch_size, shuffle=True, device=self.device, indices=train_mask
                    )
                    if has_validation:
                        val_loader, _ = make_condition_aware_vae_dataloader(
                            self.adata, batch_size=batch_size, shuffle=False, device=self.device, indices=val_mask
                        )
                    else:
                        val_loader = None
                    
                    optimizer = torch.optim.AdamW(self.condition_aware_vae.parameters(), lr=lr)

                    training_history = train_condition_aware_vae(
                        self.condition_aware_vae, 
                        train_loader, 
                        optimizer,
                        val_loader=val_loader,
                        device=self.device,
                        batch_size=batch_size,
                        max_epochs=max_epochs,
                        early_stopping_patience=early_stopping_patience,
                        beta_start=min_beta,
                        beta_final=max_beta,
                        beta_warmup_epochs=warmup_epochs,
                        contrastive_weight=contrastive_weight,
                        classifier_weight=classifier_weight,
                        return_training_history=return_training_history
                    )

            finally:
                if self.use_contrastive or self.use_condition_classifier:
                    torch.save(self.condition_aware_vae.state_dict(), self.vae_save_path)
                    print(f"VAE model saved to {self.vae_save_path}")
                    self.adata.obsm['X_vae'] = get_cell_embedding(self.condition_aware_vae, self.adata, device=self.device)
                    print(f"VAE embeddings stored in adata.obsm['X_vae']")
                    
                else:
                    torch.save(self.vae.state_dict(), self.vae_save_path)
                    print(f"VAE model saved to {self.vae_save_path}")
                    self.adata.obsm['X_vae'] = get_cell_embedding(self.vae, self.adata, device=self.device)
                    print(f"VAE embeddings stored in adata.obsm['X_vae']")


            if return_training_history:
                return training_history
        
        else: # use scvi
            print("Training scVI VAE model")
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0] if has_validation else np.array([])
            rest_mask = ~(train_mask | val_mask)
            rest_idx = np.where(rest_mask)[0]
            
            datasplitter_kwargs = {
                'external_indexing': [train_idx, val_idx, rest_idx]
            }
            
            try:
                self.scvi_vae.train(
                    max_epochs=max_epochs,
                    early_stopping=True,
                    early_stopping_patience=early_stopping_patience,
                    datasplitter_kwargs=datasplitter_kwargs
                )

            finally:
                # Save model and embeddings after successful training
                self.scvi_vae.save(self.output_dir, prefix=self.vae_name, overwrite=True)
                print(f"SCVI VAE model saved to {self.output_dir} with prefix {self.vae_name}")

                self.adata.obsm['X_vae'] = self.scvi_vae.get_latent_representation()
                print(f"SCVI VAE embeddings stored in adata.obsm['X_vae']")

    def prepare_training(self, batch_size=1000, lr=1e-3, ot_replace=True, use_condot=False):
        """
        Prepare dataloaders for training.
        
        Args:
            batch_size: Batch size for training
            lr: Learning rate for optimizer
            ot_replace: Whether to use replacement in OT sampling
            use_condot: If True, use conditional OT sampling
        """


        if self.use_my_scvi_vae:
            if self.vae_save_path is not None and os.path.exists(self.vae_save_path):
                print(f"Loading pre-trained SCVIVAE model from {self.vae_save_path}")
                self.my_scvi_vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))
                if 'X_vae' not in self.adata.obsm:
                    self.adata.obsm['X_vae'] = self.my_scvi_vae.get_latent_representation(
                        self.adata, device=self.device,
                    )
                    print("SCVIVAE embeddings stored in adata.obsm['X_vae']")
        elif not self.use_scvi_vae:
            if self.vae_save_path is not None:
                if os.path.exists(self.vae_save_path):
                    print(f"Loading pre-trained VAE model from {self.vae_save_path}")
                    if self.use_contrastive or self.use_condition_classifier:
                        self.condition_aware_vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))
                    else:
                        self.vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))
                    if 'X_vae' not in self.adata.obsm:
                        self.adata.obsm['X_vae'] = get_cell_embedding(self.vae, self.adata, device=self.device)
                        print(f"VAE model loaded and embeddings stored in adata.obsm['X_vae']")
        else:
            self.scvi_vae = self.scvi_vae.load(self.output_dir, prefix=self.vae_name, adata=self.adata)
            print(f"SCVI VAE model loaded from {self.output_dir} with prefix {self.vae_name}")
            if 'X_vae' not in self.adata.obsm:
                self.adata.obsm['X_vae'] = self.scvi_vae.get_latent_representation()
                print(f"SCVI VAE embeddings stored in adata.obsm['X_vae']")

        self.use_condot = use_condot

        self.train_with_val = 'val' in self.adata.obs['split'].unique()

        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.ot_sampler = OTPlanSampler(method='exact', normalize_cost=False)
        
        # Common arguments for data loaders
        common_loader_args = {
            'batch_size': batch_size,
            'device': self.device
        }
        
        if use_condot:
            print("Using conditional OT path")
            self.train_loader = make_condot_data_loader(
                self.adata,
                self.condition_labels_graph,
                context_to_idx=self.context_to_idx,
                split='train',
                covariate_columns=self.split.covariate_columns,
                shuffle=True,
                **common_loader_args)
            if self.train_with_val:
                self.val_loader = make_condot_data_loader(
                    self.adata,
                    self.condition_labels_graph,
                    context_to_idx=self.context_to_idx,
                    split='val',
                    covariate_columns=self.split.covariate_columns,
                    shuffle=False,
                    **common_loader_args)
        else:
            # Use original approach (stores all coupled pairs in memory)
            print("Using standard OT sampling")
            # Additional common arguments for standard OT
            standard_ot_args = {
                **common_loader_args,
                'train_with_ctrl': self.train_with_ctrl,
                'flow_from_ctrl': self.flow_from_ctrl,
                'ot_replace': ot_replace
            }
            
            self.train_loader = make_data_loader(
                self.adata,
                self.ot_sampler, 
                self.condition_labels_graph,
                context_to_idx=self.context_to_idx,
                split='train',
                covariate_columns=self.split.covariate_columns,
                shuffle=True,
                **standard_ot_args)
            if self.train_with_val:
                self.val_loader = make_data_loader(
                    self.adata,
                    self.ot_sampler, 
                    self.condition_labels_graph,
                    context_to_idx=self.context_to_idx,
                    split='val',
                    covariate_columns=self.split.covariate_columns,
                    shuffle=False,
                    **standard_ot_args)
 
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
    
    
    def train_model(
            self,
            max_epochs=500,
            early_stopping_patience=5,
            eval_freq=10,
            save_path=None
            ):
        
        if save_path is None:
            save_path = self.model_save_path

        best_val_w2d = float('inf')
        warmup_epochs = 100
        patience_counter = 0
        model_save_epoch = None
        

        try:
            if self.train_with_val:
                print("Starting training with validation...")
                evaluation_df = {
                        'epoch': [],
                        'train_loss': [],
                        'val_loss': [],
                        'val_mse': [],
                        'val_w2d': [],
                        'val_mmd': []
                    }
                for epoch in range(max_epochs):
                    if self.no_fm:
                        train_loss = train_one_epoch_no_fm(self.model, self.train_loader, self.optimizer)
                        val_loss = evaluate_one_epoch_no_fm(self.model, self.val_loader)
                    else:
                        if self.use_condot:
                            train_loss = train_one_epoch_condot(self.model, self.train_loader, self.optimizer, self.path)
                            val_loss = evaluate_one_epoch_condot(self.model, self.val_loader, self.path)
                        else:
                            train_loss = train_one_epoch(self.model, self.train_loader, self.optimizer, self.path)
                            val_loss = evaluate_one_epoch(self.model, self.val_loader, self.path)
                    evaluation_df['epoch'].append(epoch + 1)
                    evaluation_df['train_loss'].append(train_loss)
                    evaluation_df['val_loss'].append(val_loss)

                    if epoch % eval_freq == 0 or epoch == max_epochs - 1:
                        if self.no_fm:
                            val_mse, val_w2d, val_mmd = evaluate_metrics_no_fm(self.model, self.val_loader)
                        else:
                            if self.use_condot:
                                val_mse, val_w2d, val_mmd = evaluate_metrics_condot(self.model, self.val_loader)
                            else:
                                val_mse, val_w2d, val_mmd = evaluate_metrics(self.model, self.val_loader)
                        evaluation_df['val_mse'].append(val_mse)
                        evaluation_df['val_w2d'].append(val_w2d)
                        evaluation_df['val_mmd'].append(val_mmd)

                        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Error: {val_mse:.4f}, Val W2D: {val_w2d:.4f}, Val MMD: {val_mmd:.4f}")
                        if (val_w2d < best_val_w2d) and (warmup_epochs <= 0):
                            best_val_w2d = val_w2d
                            patience_counter = 0
                            if save_path:
                                torch.save(self.model.state_dict(), save_path)
                                model_save_epoch = epoch + 1
                        else:
                            patience_counter += 1

                        if patience_counter >= early_stopping_patience:
                            print("Early stopping triggered")
                            break
                        
                    else:
                        evaluation_df['val_mse'].append(None)
                        evaluation_df['val_w2d'].append(None)
                        evaluation_df['val_mmd'].append(None)
                    warmup_epochs -= 1
                if model_save_epoch is not None:
                    print(f"Best model saved at epoch {model_save_epoch}")
            else:
                print("Starting training without validation...")
                evaluation_df = {
                        'epoch': [],
                        'train_loss': []
                    }
                for epoch in range(max_epochs):
                    if self.no_fm:
                        train_loss = train_one_epoch_no_fm(self.model, self.train_loader, self.optimizer)
                    else:
                        if self.use_condot:
                            train_loss = train_one_epoch_condot(self.model, self.train_loader, self.optimizer, self.path)
                        else:
                            train_loss = train_one_epoch(self.model, self.train_loader, self.optimizer, self.path)
                    evaluation_df['epoch'].append(epoch + 1)
                    evaluation_df['train_loss'].append(train_loss)
                    print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}")

        finally:
            evaluation_df = pd.DataFrame(evaluation_df)
            if not os.path.exists(save_path):
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
            return evaluation_df

    def load_model(self, save_path=None):
        if self.use_my_scvi_vae:
            if self.vae_save_path is not None and os.path.exists(self.vae_save_path):
                print(f"Loading SCVIVAE model from {self.vae_save_path}")
                self.my_scvi_vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))
        elif not self.use_scvi_vae:
            if self.vae_save_path is not None:
                if os.path.exists(self.vae_save_path):
                    if self.use_contrastive or self.use_condition_classifier:
                        print(f"Loading condition-aware VAE model from {self.vae_save_path}")
                        self.condition_aware_vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))
                    else:
                        print(f"Loading VAE model from {self.vae_save_path}")
                        self.vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))
        else:
            self.scvi_vae = self.scvi_vae.load(self.output_dir, prefix=self.vae_name, adata=self.adata)
            print(f"SCVI VAE model loaded from {self.output_dir} with prefix {self.vae_name}")

        if save_path is None:
            save_path = self.model_save_path
        print(f"Loading FM model from {save_path}")
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def predict(self, pred_df, n_cells=100, ctrl_name='ctrl',
                guidance=1.5):
        """
        Predict the outcomes for all conditions in the adata.
        """


        conditions = pred_df['condition'].unique().tolist()
        # perturbations to predict might not be in the training data
        condition_labels_graph = make_condition_labels_graph(conditions, self.pert_names_graph)
        dataloader = make_prediction_data_loader(
            pred_df,
            condition_labels_graph,
            adata=self.adata,
            context_to_idx=self.context_to_idx,
            covariate_columns=self.split.covariate_columns,
            n_cells=n_cells,
            ctrl_name=ctrl_name,
            batch_size=8000, shuffle=False, device=self.device,
            flow_from_ctrl=self.flow_from_ctrl,
            use_scvi_vae=self.use_scvi_vae,
            use_my_scvi_vae=self.use_my_scvi_vae)
        
        preds = []
        perts = []
        contexts = []

        device = next(self.model.parameters()).device
        if self.no_fm:
            print("Predicting without flow matching...")

        # Either backend needs a per-cell library size to be passed in the batch.
        needs_library = self.use_scvi_vae or self.use_my_scvi_vae

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Unpack batch - may have 2 or 3 elements depending on context
                if len(self.split.covariate_columns) == 2:
                    if needs_library:
                        z0, y, c, l = batch
                        c = c.to(device=device, non_blocking=True)
                        l = l.to(device=device, non_blocking=True)
                    else:
                        z0, y, c = batch
                        c = c.to(device=device, non_blocking=True)
                else:
                    if needs_library:
                        z0, y, l = batch
                        l = l.to(device=device, non_blocking=True)
                        c = None
                    else:
                        z0, y = batch
                        c = None
                
                # Move batch tensors to the same device as the model to avoid device mismatch
                z0 = z0.to(device=device, non_blocking=True)
                y = y.to(device=device, non_blocking=True)
                
                if self.no_fm:
                    z1 = self.model(torch.zeros(z0.shape[0]).to(device), z0, y, c)
                else:
                    if self.flow_from_ctrl:
                        wrapped_ode = ODEWrapper(self.model, y, c)
                    else:
                        wrapped_ode = ModelGuidanceWrapper(self.model, y, guidance=guidance, c=c)

                    T = torch.linspace(0, 1, 2).to(device)
                    solver = ODESolver(velocity_model = wrapped_ode)
                    
                    z1 = solver.sample(time_grid=T,
                                        x_init = z0,
                                        step_size=None,
                                        method='dopri5')
                
                if self.use_scvi_vae:
                    generative_inputs = self._prepare_scvi_generative_inputs(z1, l)
                    generative_outputs = self.scvi_vae.module.generative(**generative_inputs)
                    pred = generative_outputs['px'].sample().cpu().numpy()
                elif self.use_my_scvi_vae:
                    # `l` is log-library-size sampled from ctrl cells (workaround:
                    # flow matching does not predict library latent).
                    x_counts = self.my_scvi_vae.sample_expression(z1, l)
                    pred = x_counts.cpu().numpy()
                else:
                    if self.use_contrastive or self.use_condition_classifier:
                        x1 = self.condition_aware_vae.decoder(z1)
                    else:
                        x1 = self.vae.decoder(z1)
                    pred = x1.cpu().numpy()
                preds.append(pred)

                y_np = y.cpu().numpy()
                genes = []
                for row in y_np:
                    non_neg_indices = np.where(row >= 0)[0]
                    if len(non_neg_indices) == 0:
                        # All values are -1, indicating ctrl condition
                        name = ctrl_name
                    else:
                        idxs = row[non_neg_indices].astype(int)
                        selected_labels = self.pert_names_graph[idxs]
                        name = '+'.join(selected_labels)
                    genes.append(name)
                perts.extend(genes)

                if c is not None:
                    contexts.extend([self.context_names[idx] for idx in c.cpu().numpy()])
        
        md = {}
        for col in self.split.covariate_columns:
            if col == 'condition':
                md[col] = perts
            else:
                md[col] = contexts
        adata_pred = sc.AnnData(np.vstack(preds), obs=md)
        adata_pred.var_names = self.adata.var_names.tolist()
        if self.use_scvi_vae or self.use_my_scvi_vae:
            adata_pred.layers['counts'] = adata_pred.X.copy()
            sc.pp.normalize_total(adata_pred)
            sc.pp.log1p(adata_pred)
        
        return adata_pred
    
    def _prepare_scvi_generative_inputs(self, z1, library):
        # For SCVI, we need to prepare the generative inputs in a specific way
        generative_inputs = {
            'z': z1,
            'library': library,
            'y': torch.zeros_like(library, dtype=torch.long).to(z1.device),  # Dummy condition labels (not used in generative mode)
            'batch_index': torch.zeros_like(library, dtype=torch.long).to(z1.device),
            'cont_covs': None, 'cat_covs': None, 'size_factor': None
        }
        return generative_inputs

