#!/usr/bin/env python3

import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import torch
from gears import PertData
from gears.utils import create_cell_graph_dataset_for_prediction
from torch import nn
from torch_geometric.loader import DataLoader
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torchtext.vocab import Vocab

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
)
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id, set_seed

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

set_seed(42)

pert_data_dir = sys.argv[1]
split_path = sys.argv[2]
seed = sys.argv[3]
save_dir = sys.argv[4]
load_model = sys.argv[5]  # if None, train from scratch

if load_model == "":
    load_model = None

# settings for data processing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2

n_hvg = 0  # number of highly variable genes
include_zero_gene = (
    "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
)
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True

load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 10
eval_batch_size = 10
epochs = 15
schedule_interval = 1
early_stop = 5

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100

# check device
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
    print(f"  Cached:    {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
gpu = torch.cuda.current_device()
# gpu = 1
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
print(device)

# load pert_data
pert_data = PertData(data_path=os.path.dirname(pert_data_dir))

train_start = time.time()

pert_data.load(data_path=pert_data_dir)
# only single perturbation
pert_data.prepare_split(split_dict_path=split_path, split="custom", seed=seed)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

# prepare save_dir
model_save_dir = Path(f"{save_dir}/model_split_{seed}/")
model_save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {model_save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, model_save_dir / "run.log")
# log running date and current git commit
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")

if load_model is not None:
    # with pretrained
    model_dir = Path(load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    genes = pert_data.adata.var["gene_name"].tolist()

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)
else:
    # if no pretrained, run this
    genes = pert_data.adata.var["gene_name"].tolist()
    vocab = Vocab(VocabPybind(genes + special_tokens, None))  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

# initialize model
ntokens = len(vocab)  # size of vocabulary
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    do_mvc=MVC,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    use_fast_transformer=use_fast_transformer,
)
model.to(device)

# load pretrained model
if load_model is not None:
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    for k, v in pretrained_dict.items():
        logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)


criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=amp)


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = torch.zeros(batch_size, n_genes).long().to(device)
        for b, pert_idx in enumerate(batch_data.pert_idx):
            if pert_idx[0] == -1:
                continue
            if pert_idx[0] > n_genes:
                continue
            pert_flags[b, int(pert_idx[0])] = 1
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[:max_seq_len]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(input_values, dtype=torch.bool)  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = torch.zeros(batch_size, n_genes).long().to(device)
            for b, pert_idx in enumerate(batch_data.pert_idx):
                if pert_idx[0] == -1:
                    continue
                if pert_idx[0] > n_genes:
                    continue
                pert_flags[b, int(pert_idx[0])] = 1
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
                    input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]

                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)


best_val_loss = float("inf")
best_model = None
patience = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    train(
        model,
        train_loader,
    )
    val_loss, val_mre = evaluate(
        model,
        valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | " f"valid loss/mse {val_loss:5.4f} |"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {best_val_loss:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    torch.save(
        model.state_dict(),
        model_save_dir / f"model_{epoch}.pt",
    )

    scheduler.step()
train_time = time.time() - train_start
print(f"Training time: {train_time:.2f}s")

torch.save(best_model.state_dict(), model_save_dir / "best_model.pt")


# evaluation
model_file = model_save_dir / "best_model.pt"
model.load_state_dict(torch.load(model_file))
logger.info(f"Loading all model params from {model_file}")

model.to(device)


def predict(
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    import scanpy as sc

    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    pert_list_filtered = []
    for perturbation_genes in pert_list:
        pert = []
        for gene in perturbation_genes:
            if gene in gene_list:
                pert.append(gene)
        if pert:
            pert_list_filtered.append(pert)
    for pert in pert_list_filtered:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    preds_list = []
    conds_list = []
    with torch.no_grad():
        for pert in pert_list_filtered:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0).cpu().numpy()
            preds_list.append(preds)
            pert_name = "+".join(pert)
            conds_list.extend([pert_name] * preds.shape[0])

    adata_pred = sc.AnnData(np.vstack(preds_list), obs={"condition": conds_list})
    adata_pred.var_names = adata.var["gene_name"].tolist()

    return adata_pred


# conds = pert_data.adata.obs["condition"].unique().tolist()
conds = pert_data.set2conditions["test"]
split_conds = [x.split("+") for x in conds]
split_conds = [list(filter(lambda y: y != "ctrl", x)) for x in split_conds]

infer_start = time.time()
adata_pred = predict(model, split_conds, pool_size=100)
infer_time = time.time() - infer_start
print(f"Inference time: {infer_time:.2f}s")

timing = {"train_time_s": train_time, "inference_time_s": infer_time}
with open(os.path.join(save_dir, f"timing_{seed}.json"), "w") as f:
    json.dump(timing, f, indent=2)
print(f"Timing saved to {os.path.join(save_dir, f'timing_{seed}.json')}")
adata_pred.write(f"{save_dir}/adata_pred_{seed}.h5ad")
