import numpy as np
import torch
from flow_matching.solver import ODESolver

from gfm.helpers import mmd_distance, wasserstein_2
from gfm.models import ODEWrapper


def train_one_epoch(model, train_loader, optimizer, path):
    device = next(model.parameters()).device
    train_loss = 0
    for batch in train_loader:
        model.train()
        optimizer.zero_grad()

        # Unpack batch - may have 3 or 4 elements depending on context
        if len(batch) == 4:
            z0_coupled, z1_coupled, y, c = batch
            c = c.to(device, non_blocking=True)
        else:
            z0_coupled, z1_coupled, y = batch
            c = None

        # Move data to device
        z0_coupled = z0_coupled.to(device, non_blocking=True)
        z1_coupled = z1_coupled.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t = torch.rand(z0_coupled.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=z0_coupled, x_1=z1_coupled)
        vt = model(path_sample.t, path_sample.x_t, y, c)
        loss = torch.pow(vt - path_sample.dx_t, 2).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def evaluate_one_epoch(model, val_loader, path):
    device = next(model.parameters()).device
    val_loss = 0
    for batch in val_loader:
        model.eval()

        # Unpack batch - may have 3 or 4 elements depending on context
        if len(batch) == 4:
            z0_coupled, z1_coupled, y, c = batch
            c = c.to(device, non_blocking=True)
        else:
            z0_coupled, z1_coupled, y = batch
            c = None

        # Move data to device
        z0_coupled = z0_coupled.to(device, non_blocking=True)
        z1_coupled = z1_coupled.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t = torch.rand(z0_coupled.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=z0_coupled, x_1=z1_coupled)
        vt = model(path_sample.t, path_sample.x_t, y, c)
        loss = torch.pow(vt - path_sample.dx_t, 2).mean()

        val_loss += loss.item()

    return val_loss / len(val_loader)


def evaluate_metrics(model, val_loader):
    device = next(model.parameters()).device
    model.eval()
    mse_list = []
    w2d_list = []
    mmd_list = []
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch - may have 3 or 4 elements depending on context
            if len(batch) == 4:
                z0_coupled, z1_coupled, y, c = batch
                c = c.to(device, non_blocking=True)
            else:
                z0_coupled, z1_coupled, y = batch
                c = None

            # Move data to device
            z0_coupled = z0_coupled.to(device, non_blocking=True)
            z1_coupled = z1_coupled.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            wrapped_ode = ODEWrapper(model, y, c)

            T = torch.linspace(0, 1, 2).to(device)
            solver = ODESolver(velocity_model=wrapped_ode)
            z1 = solver.sample(time_grid=T, x_init=z0_coupled, step_size=None, method="dopri5")
            z1 = z1.cpu().numpy()
            z1_coupled = z1_coupled.cpu().numpy()
            mse = ((z1 - z1_coupled) ** 2).mean()
            w2d = wasserstein_2(z1, z1_coupled)
            mmd = mmd_distance(z1, z1_coupled)
            w2d_list.append(w2d)
            mse_list.append(mse)
            mmd_list.append(mmd)

    return np.mean(mse_list), np.mean(w2d_list), np.mean(mmd_list)


def train_one_epoch_condot(model, train_loader, optimizer, path):
    device = next(model.parameters()).device
    train_loss = 0
    for batch in train_loader:
        model.train()
        optimizer.zero_grad()

        # Unpack batch - may have 2 or 3 elements depending on context
        if len(batch) == 3:
            z1, y, c = batch
            c = c.to(device, non_blocking=True)
        else:
            z1, y = batch
            c = None

        # Move data to device
        z0 = torch.randn_like(z1).to(device, non_blocking=True)  # Sample z0 as random noise
        z1 = z1.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t = torch.rand(z0.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=z0, x_1=z1)
        vt = model(path_sample.t, path_sample.x_t, y, c)
        loss = torch.pow(vt - path_sample.dx_t, 2).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def evaluate_one_epoch_condot(model, val_loader, path):
    device = next(model.parameters()).device
    val_loss = 0
    for batch in val_loader:
        model.eval()

        # Unpack batch - may have 2 or 3 elements depending on context
        if len(batch) == 3:
            z1, y, c = batch
            c = c.to(device, non_blocking=True)
        else:
            z1, y = batch
            c = None

        # Move data to device
        z0 = torch.randn_like(z1).to(device, non_blocking=True)  # Sample z0 as random noise
        z1 = z1.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t = torch.rand(z0.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=z0, x_1=z1)
        vt = model(path_sample.t, path_sample.x_t, y, c)
        loss = torch.pow(vt - path_sample.dx_t, 2).mean()

        val_loss += loss.item()

    return val_loss / len(val_loader)


def evaluate_metrics_condot(model, val_loader):
    device = next(model.parameters()).device
    model.eval()
    mse_list = []
    w2d_list = []
    mmd_list = []
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch - may have 2 or 3 elements depending on context
            if len(batch) == 3:
                z1, y, c = batch
                c = c.to(device, non_blocking=True)
            else:
                z1, y = batch
                c = None

            # Move data to device
            z0 = torch.randn_like(z1).to(device, non_blocking=True)  # Sample z0 as random noise
            z1 = z1.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            wrapped_ode = ODEWrapper(model, y, c)

            T = torch.linspace(0, 1, 2).to(device)
            solver = ODESolver(velocity_model=wrapped_ode)
            z1_pred = solver.sample(time_grid=T, x_init=z0, step_size=None, method="dopri5")
            z1_pred = z1_pred.cpu().numpy()
            z1 = z1.to("cpu").numpy()
            mse = ((z1_pred - z1) ** 2).mean()
            w2d = wasserstein_2(z1_pred, z1)
            mmd = mmd_distance(z1_pred, z1)
            w2d_list.append(w2d)
            mse_list.append(mse)
            mmd_list.append(mmd)

    return np.mean(mse_list), np.mean(w2d_list), np.mean(mmd_list)


def train_one_epoch_no_fm(model, train_loader, optimizer):
    device = next(model.parameters()).device
    train_loss = 0
    for batch in train_loader:
        model.train()
        optimizer.zero_grad()

        # Unpack batch - may have 3 or 4 elements depending on context
        if len(batch) == 4:
            z0_coupled, z1_coupled, y, c = batch
            c = c.to(device, non_blocking=True)
        else:
            z0_coupled, z1_coupled, y = batch
            c = None

        # Move data to device
        z0_coupled = z0_coupled.to(device, non_blocking=True)
        z1_coupled = z1_coupled.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t = torch.zeros(z0_coupled.shape[0]).to(device)
        z1 = model(t, z0_coupled, y, c)
        loss = torch.pow(z1 - z1_coupled, 2).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(train_loader)


def evaluate_one_epoch_no_fm(model, val_loader):
    device = next(model.parameters()).device
    val_loss = 0
    for batch in val_loader:
        model.eval()

        # Unpack batch - may have 3 or 4 elements depending on context
        if len(batch) == 4:
            z0_coupled, z1_coupled, y, c = batch
            c = c.to(device, non_blocking=True)
        else:
            z0_coupled, z1_coupled, y = batch
            c = None

        # Move data to device
        z0_coupled = z0_coupled.to(device, non_blocking=True)
        z1_coupled = z1_coupled.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        t = torch.zeros(z0_coupled.shape[0]).to(device)
        z1 = model(t, z0_coupled, y, c)
        loss = torch.pow(z1 - z1_coupled, 2).mean()

        val_loss += loss.item()

    return val_loss / len(val_loader)


def evaluate_metrics_no_fm(model, val_loader):
    device = next(model.parameters()).device
    model.eval()
    mse_list = []
    w2d_list = []
    mmd_list = []
    with torch.no_grad():
        for batch in val_loader:
            # Unpack batch - may have 3 or 4 elements depending on context
            if len(batch) == 4:
                z0_coupled, z1_coupled, y, c = batch
                c = c.to(device, non_blocking=True)
            else:
                z0_coupled, z1_coupled, y = batch
                c = None

            # Move data to device
            z0_coupled = z0_coupled.to(device, non_blocking=True)
            z1_coupled = z1_coupled.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            t = torch.zeros(z0_coupled.shape[0]).to(device)
            z1 = model(t, z0_coupled, y, c)
            z1 = z1.cpu().numpy()
            z1_coupled = z1_coupled.cpu().numpy()
            mse = ((z1 - z1_coupled) ** 2).mean()
            w2d = wasserstein_2(z1, z1_coupled)
            mmd = mmd_distance(z1, z1_coupled)
            w2d_list.append(w2d)
            mse_list.append(mse)
            mmd_list.append(mmd)

    return np.mean(mse_list), np.mean(w2d_list), np.mean(mmd_list)
