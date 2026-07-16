"""
Training utilities for condition-aware VAE
"""
import torch
import numpy as np

def make_vae_dataloader(adata, batch_size=128, shuffle=True, device='cpu', indices=None):
    """
    Create VAE dataloader with optional index-based subsetting for memory efficiency.
    
    Args:
        adata: AnnData object
        batch_size: batch size
        shuffle: whether to shuffle
        device: device for pin_memory optimization
        indices: optional boolean mask or integer indices to subset adata (avoids copying)
    """
    # Subset data if indices provided (much faster than adata[indices])
    if indices is not None:
        if hasattr(adata[:].X, 'toarray'):
            X = torch.tensor(adata[indices].X.toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata[indices].X, dtype=torch.float32)
    else:
        if hasattr(adata.X[:], 'toarray'):
            X = torch.tensor(adata.X[:].toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X[:], dtype=torch.float32)
    
    # Keep data on CPU, will be moved to device during training
    dataset = torch.utils.data.TensorDataset(X)
    # Use pin_memory for faster CPU-to-GPU transfer during training
    # drop_last=True when shuffling (training) to avoid batch size 1 with BatchNorm
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       drop_last=shuffle,
                                       pin_memory=(device != 'cpu'), num_workers=0)

def train_one_epoch_vae(vae, train_loader, optimizer, device='cpu', batch_size=128, beta=1.0):
    vae.train()
    train_loss = 0
    recon_loss = 0
    kld = 0
    for batch in train_loader:
        optimizer.zero_grad()
        data = batch[0].to(device, non_blocking=True)
        recon_batch, mu, logvar = vae(data)
        loss, recon_loss_batch, kld_batch = vae.loss(recon_batch, data, mu, logvar, beta=beta)
        recon_loss += recon_loss_batch.item()
        kld += kld_batch.item()

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)*batch_size
    recon_loss /= len(train_loader)*batch_size
    kld /= len(train_loader)*batch_size
    return train_loss, recon_loss, kld

def evaluate_vae(vae, data_loader, device='cpu', batch_size=128, beta=1.0):
    vae.eval()
    val_loss = 0
    recon_loss = 0
    kld = 0
    with torch.no_grad():
        for batch in data_loader:
            data = batch[0].to(device, non_blocking=True)
            recon_batch, mu, logvar = vae(data)
            loss, recon_loss_batch, kld_batch = vae.loss(recon_batch, data, mu, logvar, beta=beta)
            recon_loss += recon_loss_batch.item()
            kld += kld_batch.item()
            val_loss += loss.item()
    val_loss /= len(data_loader)*batch_size
    recon_loss /= len(data_loader)*batch_size
    kld /= len(data_loader)*batch_size
    return val_loss, recon_loss, kld

def train_vae(
        vae, 
        train_loader, 
        optimizer,
        val_loader=None,
        device='cpu',
        batch_size=128, 
        max_epochs=500, 
        early_stopping_patience=25,
        min_beta=1.0,
        max_beta=1.0,
        warmup_epochs=1,
        return_training_history=False
    ):
    """
    Train VAE with optional validation.
    
    Args:
        vae: VAE model
        train_loader: training data loader
        val_loader: validation data loader (optional, set to None to skip validation)
        optimizer: optimizer
        device: device
        batch_size: batch size
        max_epochs: maximum epochs
        early_stopping_patience: patience for early stopping (only used if val_loader is provided)
        min_beta: starting beta value
        max_beta: final beta value
        warmup_epochs: epochs to warmup beta
        return_training_history: whether to return training history
    """
    patience_counter = 0
    best_val_loss = float('inf')
    use_validation = val_loader is not None

    if return_training_history:
        training_history = {
            'train_loss': [],
            'recon_loss': [],
            'kld': [],
            'beta': []
        }
        if use_validation:
            training_history['val_loss'] = []
            training_history['val_recon_loss'] = []
            training_history['val_kld'] = []

    for epoch in range(max_epochs):
        # Linear warmup: interpolate from min_beta to max_beta over warmup_epochs
        if warmup_epochs > 0:
            warmup_progress = min(1.0, epoch / warmup_epochs)
            beta = min_beta + (max_beta - min_beta) * warmup_progress
        else:
            beta = max_beta
        
        train_loss, recon_loss, kld = train_one_epoch_vae(vae, train_loader, optimizer, device=device, batch_size=batch_size, beta=beta)

        # Optional validation
        if use_validation:
            val_loss, val_recon_loss, val_kld = evaluate_vae(vae, val_loader, device=device, batch_size=batch_size, beta=beta)

        if return_training_history:
            training_history['train_loss'].append(train_loss)
            training_history['recon_loss'].append(recon_loss)
            training_history['kld'].append(kld)
            training_history['beta'].append(beta)
            if use_validation:
                training_history['val_loss'].append(val_loss)
                training_history['val_recon_loss'].append(val_recon_loss)
                training_history['val_kld'].append(val_kld)

        if epoch % 10 == 0:
            if use_validation:
                print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}")

        # Early stopping (only if validation is enabled)
        if use_validation:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    print(f"Best Val Loss: {best_val_loss:.4f}")
                    if return_training_history:
                        return training_history
                    break
        
    if return_training_history:
        return training_history

def train_vae_no_val(
        vae, 
        train_loader, 
        optimizer, 
        device='cpu', 
        batch_size=128, 
        max_epochs=500, 
        min_beta=1.0, 
        max_beta=1.0, 
        warmup_epochs=1,
        return_training_history=False
    ):
    if return_training_history:
        training_history = {
            'train_loss': [],
            'recon_loss': [],
            'kld': [],
            'beta': []
        }
    
    for epoch in range(max_epochs):
        # Linear warmup: interpolate from min_beta to max_beta over warmup_epochs
        if warmup_epochs > 0:
            warmup_progress = min(1.0, epoch / warmup_epochs)
            beta = min_beta + (max_beta - min_beta) * warmup_progress
        else:
            beta = max_beta
        
        train_loss, recon_loss, kld = train_one_epoch_vae(vae, train_loader, optimizer, device=device, batch_size=batch_size, beta=beta)

        if return_training_history:
            training_history['train_loss'].append(train_loss)
            training_history['recon_loss'].append(recon_loss)
            training_history['kld'].append(kld)
            training_history['beta'].append(beta)

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss}")
    
    if return_training_history:
        return training_history


def make_condition_aware_vae_dataloader(adata, batch_size=128, shuffle=True, device='cpu', indices=None,
                                        covariate_columns=['condition']):
    """
    Create dataloader with condition labels for supervised training.
    
    Args:
        adata: AnnData object with 'condition' in obs
        batch_size: batch size
        shuffle: whether to shuffle
        device: device for pin_memory optimization
        indices: optional boolean mask or integer indices to subset adata (avoids copying)
    
    Returns:
        DataLoader with (expression, condition_idx) pairs, condition_to_idx mapping
    """
    
    if indices is not None:
        if hasattr(adata.X, 'toarray'):
            X = torch.tensor(adata.X[indices].toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X[indices], dtype=torch.float32)
    else:
        if hasattr(adata.X, 'toarray'):
            X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X, dtype=torch.float32)
    
    # Encode conditions to indices
    adata.obs['group'] = adata.obs[covariate_columns].astype(str).agg('_'.join, axis=1)
    unique_groups = adata.obs['group'].drop_duplicates().tolist()
    group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
    condition_indices = torch.tensor(
        [group_to_idx[c] for c in adata.obs.loc[indices, 'group'].values], 
        dtype=torch.long
    )
    
    dataset = torch.utils.data.TensorDataset(X, condition_indices)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=(device != 'cpu'), 
        num_workers=0
    ), group_to_idx


def train_one_epoch_condition_aware_vae(
    vae, 
    train_loader, 
    optimizer, 
    device='cpu', 
    batch_size=128, 
    beta=1.0,
    contrastive_weight=0.1,
    classifier_weight=0.1
):
    """Train one epoch with condition-aware VAE."""
    vae.train()
    total_loss = 0
    loss_components = {'recon': 0, 'kld': 0, 'contrastive': 0, 'classifier': 0}
    
    for batch_idx, (data, condition_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device, non_blocking=True)
        condition_labels = condition_labels.to(device, non_blocking=True)
        
        # Forward pass
        recon_batch, mu, logvar, z = vae(data)
        
        # Compute loss
        loss, loss_dict = vae.loss(
            recon_batch, data, mu, logvar, z=z,
            condition_labels=condition_labels,
            beta=beta,
            contrastive_weight=contrastive_weight,
            classifier_weight=classifier_weight
        )
        
        # Backward pass
        loss.backward()
        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for key, val in loss_dict.items():
            if key in loss_components:
                loss_components[key] += val
    
    # Average losses
    n_samples = len(train_loader) * batch_size
    total_loss /= n_samples
    for key in loss_components:
        loss_components[key] /= n_samples
    
    return total_loss, loss_components


def evaluate_condition_aware_vae(
    vae, 
    val_loader, 
    device='cpu', 
    batch_size=128, 
    beta=1.0,
    contrastive_weight=0.1,
    classifier_weight=0.1
):
    """Evaluate condition-aware VAE."""
    vae.eval()
    total_loss = 0
    loss_components = {'recon': 0, 'kld': 0, 'contrastive': 0, 'classifier': 0}
    
    with torch.no_grad():
        for data, condition_labels in val_loader:
            data = data.to(device, non_blocking=True)
            condition_labels = condition_labels.to(device, non_blocking=True)
            
            recon_batch, mu, logvar, z = vae(data)
            loss, loss_dict = vae.loss(
                recon_batch, data, mu, logvar, z=z,
                condition_labels=condition_labels,
                beta=beta,
                contrastive_weight=contrastive_weight,
                classifier_weight=classifier_weight
            )
            
            total_loss += loss.item()
            for key, val in loss_dict.items():
                if key in loss_components:
                    loss_components[key] += val
    
    n_samples = len(val_loader) * batch_size
    total_loss /= n_samples
    for key in loss_components:
        loss_components[key] /= n_samples
    
    return total_loss, loss_components


def train_condition_aware_vae(
    vae,
    train_loader,
    optimizer,
    val_loader=None,
    device='cpu',
    batch_size=128,
    max_epochs=500,
    early_stopping_patience=25,
    beta_start=1.0,
    beta_final=1.0,
    beta_warmup_epochs=1,
    contrastive_weight=0.1,
    classifier_weight=0.1,
    return_training_history=False
):
    """
    Train condition-aware VAE with optional validation.
    
    Args:
        vae: ConditionAwareVAE model
        train_loader: training data loader (with condition labels)
        optimizer: optimizer
        val_loader: validation data loader (optional, set to None to skip validation)
        device: device
        batch_size: batch size
        max_epochs: maximum epochs
        early_stopping_patience: patience for early stopping (only used if val_loader is provided)
        beta_start: starting beta value
        beta_final: final beta value
        beta_warmup_epochs: epochs to warmup beta
        contrastive_weight: weight for contrastive loss (0.05-0.2 recommended)
        classifier_weight: weight for classifier loss (0.05-0.1 recommended)
        return_training_history: whether to return training history
    """
    patience_counter = 0
    best_val_loss = float('inf')
    use_validation = val_loader is not None
    
    if return_training_history:
        training_history = {
            'train_loss': [],
            'train_recon': [],
            'train_kld': [],
            'train_contrastive': [],
            'train_classifier': [],
            'beta': []
        }
        if use_validation:
            training_history['val_loss'] = []
            training_history['val_recon'] = []
            training_history['val_kld'] = []
            training_history['val_contrastive'] = []
            training_history['val_classifier'] = []
    
    for epoch in range(max_epochs):
        # Linear warmup schedule
        if beta_warmup_epochs > 0:
            warmup_progress = min(1.0, epoch / beta_warmup_epochs)
            beta = beta_start + (beta_final - beta_start) * warmup_progress
        else:
            beta = beta_final
        
        # Train
        train_loss, train_components = train_one_epoch_condition_aware_vae(
            vae, train_loader, optimizer, 
            device=device, batch_size=batch_size, beta=beta,
            contrastive_weight=contrastive_weight,
            classifier_weight=classifier_weight
        )
        
        # Optional validation
        if use_validation:
            val_loss, val_components = evaluate_condition_aware_vae(
                vae, val_loader, 
                device=device, batch_size=batch_size, beta=beta,
                contrastive_weight=contrastive_weight,
                classifier_weight=classifier_weight
            )
        
        # Record history
        if return_training_history:
            training_history['train_loss'].append(train_loss)
            training_history['train_recon'].append(train_components['recon'])
            training_history['train_kld'].append(train_components['kld'])
            training_history['train_contrastive'].append(train_components.get('contrastive', 0))
            training_history['train_classifier'].append(train_components.get('classifier', 0))
            training_history['beta'].append(beta)
            if use_validation:
                training_history['val_loss'].append(val_loss)
                training_history['val_recon'].append(val_components['recon'])
                training_history['val_kld'].append(val_components['kld'])
                training_history['val_contrastive'].append(val_components.get('contrastive', 0))
                training_history['val_classifier'].append(val_components.get('classifier', 0))
        
        # Print progress
        if epoch % 10 == 0:
            if use_validation:
                print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}")
        
        # Early stopping (only if validation is enabled)
        if use_validation:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    print(f"Best Val Loss: {best_val_loss:.4f}")
                    if return_training_history:
                        return training_history
                    break
    
    if return_training_history:
        return training_history


# =============================================================================
# SCVI-like VAE training utilities (NB / ZINB on raw counts)
# =============================================================================
def make_scvi_vae_dataloader(
    adata,
    batch_size=128,
    shuffle=True,
    device='cpu',
    indices=None,
    counts_layer='counts',
):
    """Create dataloader of raw counts for SCVIVAE training.

    Pulls counts from ``adata.layers[counts_layer]`` if present, otherwise
    falls back to ``adata.X``.
    """
    if counts_layer is not None and counts_layer in adata.layers:
        X_src = adata.layers[counts_layer]
    else:
        X_src = adata.X

    if indices is not None:
        # `indices` may be a pandas boolean Series; convert to numpy for
        # consistent fancy/boolean indexing across sparse and dense backends.
        if hasattr(indices, 'values'):
            indices = indices.values
        indices = np.asarray(indices)
        X_src = X_src[indices]

    if hasattr(X_src, 'toarray'):
        X = torch.tensor(X_src.toarray(), dtype=torch.float32)
    else:
        X = torch.tensor(np.asarray(X_src), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=shuffle,
        pin_memory=(device != 'cpu'), num_workers=0,
    )


def train_one_epoch_scvi_vae(vae, train_loader, optimizer, device='cpu',
                              batch_size=128, beta=1.0):
    vae.train()
    total = 0.0
    components = {'recon': 0.0, 'kld_z': 0.0, 'kld_l': 0.0}
    for batch in train_loader:
        optimizer.zero_grad()
        x = batch[0].to(device, non_blocking=True)
        fwd = vae(x)
        loss, loss_dict = vae.loss(x, fwd, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        for k in components:
            components[k] += loss_dict[k]
    n = len(train_loader) * batch_size
    total /= n
    for k in components:
        components[k] /= n
    return total, components


def evaluate_scvi_vae(vae, data_loader, device='cpu', batch_size=128, beta=1.0):
    vae.eval()
    total = 0.0
    components = {'recon': 0.0, 'kld_z': 0.0, 'kld_l': 0.0}
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device, non_blocking=True)
            fwd = vae(x)
            loss, loss_dict = vae.loss(x, fwd, beta=beta)
            total += loss.item()
            for k in components:
                components[k] += loss_dict[k]
    n = len(data_loader) * batch_size
    total /= n
    for k in components:
        components[k] /= n
    return total, components


def train_scvi_vae(
    vae,
    train_loader,
    optimizer,
    val_loader=None,
    device='cpu',
    batch_size=128,
    max_epochs=500,
    early_stopping_patience=25,
    beta=1.0,
    return_training_history=False,
):
    """Train SCVIVAE.  Mirrors ``train_vae`` but with a fixed beta (no
    annealing) and NB/ZINB-aware loss components.
    """
    patience_counter = 0
    best_val_loss = float('inf')
    use_validation = val_loader is not None

    if return_training_history:
        training_history = {
            'train_loss': [], 'recon': [], 'kld_z': [], 'kld_l': [],
        }
        if use_validation:
            training_history.update({
                'val_loss': [], 'val_recon': [], 'val_kld_z': [], 'val_kld_l': [],
            })

    for epoch in range(max_epochs):
        train_loss, train_comp = train_one_epoch_scvi_vae(
            vae, train_loader, optimizer,
            device=device, batch_size=batch_size, beta=beta,
        )

        if use_validation:
            val_loss, val_comp = evaluate_scvi_vae(
                vae, val_loader, device=device, batch_size=batch_size, beta=beta,
            )

        if return_training_history:
            training_history['train_loss'].append(train_loss)
            training_history['recon'].append(train_comp['recon'])
            training_history['kld_z'].append(train_comp['kld_z'])
            training_history['kld_l'].append(train_comp['kld_l'])
            if use_validation:
                training_history['val_loss'].append(val_loss)
                training_history['val_recon'].append(val_comp['recon'])
                training_history['val_kld_z'].append(val_comp['kld_z'])
                training_history['val_kld_l'].append(val_comp['kld_l'])

        if epoch % 10 == 0:
            if use_validation:
                print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}")

        if use_validation:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    print(f"Best Val Loss: {best_val_loss:.4f}")
                    if return_training_history:
                        return training_history
                    break

    if return_training_history:
        return training_history


def train_scvi_vae_no_val(
    vae,
    train_loader,
    optimizer,
    device='cpu',
    batch_size=128,
    max_epochs=500,
    beta=1.0,
    return_training_history=False,
):
    """Train SCVIVAE without validation."""
    if return_training_history:
        training_history = {
            'train_loss': [], 'recon': [], 'kld_z': [], 'kld_l': [],
        }

    for epoch in range(max_epochs):
        train_loss, train_comp = train_one_epoch_scvi_vae(
            vae, train_loader, optimizer,
            device=device, batch_size=batch_size, beta=beta,
        )
        if return_training_history:
            training_history['train_loss'].append(train_loss)
            training_history['recon'].append(train_comp['recon'])
            training_history['kld_z'].append(train_comp['kld_z'])
            training_history['kld_l'].append(train_comp['kld_l'])
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {train_loss:.4f}")

    if return_training_history:
        return training_history
