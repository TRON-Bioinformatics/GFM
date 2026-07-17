import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, GINEConv, GPSConv, HeteroConv
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from gfm.helpers import multi_hot_label


class ODEWrapper(nn.Module):
    def __init__(self, ode_func, y, c=None):
        super().__init__()
        self.ode_func = ode_func
        self.y = y
        self.c = c

    def forward(self, t, x):
        # Ensure t is a batch of the same size as x
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])

        return self.ode_func(t, x, self.y, self.c)


class ModelGuidanceWrapper(nn.Module):
    def __init__(self, ode_func, y, guidance, c=None):
        super().__init__()
        self.ode_func = ode_func
        self.y = y
        self.guidance = guidance
        self.c = c

    def forward(self, t, x):
        # Ensure t is a batch of the same size as x
        if t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1):
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])

        out_cond = self.ode_func(t, x, self.y, self.c)
        # Use -1 to trigger the null embedding (ctrl) instead of zeros
        # This ensures the unconditional forward uses the learned null embedding
        null_condition = torch.full_like(self.y, -1)
        out_uncond = self.ode_func(t, x, null_condition, self.c)
        return (1 - self.guidance) * out_uncond + self.guidance * out_cond


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=2):
        super().__init__()
        act_fc = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.num_layer = num_layer

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(num_layer):
            if i == 0:
                self.convs.append(GCNConv(input_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            elif i == num_layer - 1:
                self.convs.append(GCNConv(hidden_dim, output_dim))
                self.bns.append(nn.BatchNorm1d(output_dim))
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.acts.append(act_fc)

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layer):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = self.acts[i](x)
            if i < self.num_layer - 1:
                x = self.dropout(x)
        return x


class HGNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        edge_types,
        num_layer=2,
        heads=4,
        dropout=0.1,
        node_type="gene",
    ):
        super().__init__()
        if not edge_types:
            raise ValueError("HGNNEncoder requires at least one edge type.")

        self.node_type = node_type
        self.dropout = nn.Dropout(dropout)
        self.num_layer = num_layer
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        for i in range(num_layer):
            in_channels = input_dim if i == 0 else hidden_dim * heads
            is_last = i == num_layer - 1
            out_channels = output_dim if is_last else hidden_dim
            layer_heads = 1 if is_last else heads
            concat = not is_last
            layer_output_dim = out_channels if is_last else out_channels * layer_heads

            self.convs.append(
                HeteroConv(
                    {
                        edge_type: GATConv(
                            (in_channels, in_channels),
                            out_channels,
                            heads=layer_heads,
                            concat=concat,
                            dropout=dropout,
                            edge_dim=1,
                            add_self_loops=False,
                        )
                        for edge_type in edge_types
                    },
                    aggr="sum",
                )
            )
            self.bns.append(nn.BatchNorm1d(layer_output_dim))
            self.acts.append(nn.SiLU())
            if in_channels == layer_output_dim:
                self.skip_layers.append(nn.Identity())
            else:
                self.skip_layers.append(nn.Linear(in_channels, layer_output_dim))

    @staticmethod
    def _format_edge_attr_dict(edge_weight_dict):
        if edge_weight_dict is None:
            return None
        if not isinstance(edge_weight_dict, dict):
            raise TypeError("HGNNEncoder edge_weight must be a dictionary keyed by edge type.")

        edge_attr_dict = {}
        for edge_type, edge_weight in edge_weight_dict.items():
            if edge_weight is None:
                continue
            edge_attr_dict[edge_type] = (
                edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
            )
        return edge_attr_dict or None

    def forward(self, x, edge_index_dict, edge_weight_dict=None):
        if not isinstance(edge_index_dict, dict):
            raise TypeError(
                "HGNNEncoder expects edge_index_dict to be a dictionary keyed by edge type."
            )

        x_dict = {self.node_type: x}
        edge_attr_dict = self._format_edge_attr_dict(edge_weight_dict)

        for i in range(self.num_layer):
            if edge_attr_dict is None:
                out_dict = self.convs[i](x_dict, edge_index_dict)
            else:
                out_dict = self.convs[i](x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)

            out = out_dict.get(self.node_type)
            if out is None:
                out = torch.zeros_like(self.skip_layers[i](x_dict[self.node_type]))

            out = out + self.skip_layers[i](x_dict[self.node_type])
            out = self.bns[i](out)
            out = self.acts[i](out)
            if i < self.num_layer - 1:
                out = self.dropout(out)
            x_dict = {self.node_type: out}

        return x_dict[self.node_type]


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=2):
        super().__init__()
        act_fc = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.num_layer = num_layer

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()

        # GIN uses an MLP inside each layer
        for i in range(num_layer):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    act_fc,
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(mlp, edge_dim=1))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            elif i == num_layer - 1:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    act_fc,
                    nn.Linear(output_dim, output_dim),
                )
                self.convs.append(GINEConv(mlp, edge_dim=1))
                self.bns.append(nn.BatchNorm1d(output_dim))
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    act_fc,
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(mlp, edge_dim=1))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.acts.append(act_fc)

    def forward(self, x, edge_index, edge_weight=None):
        # Reshape edge_weight to [num_edges, 1] if it's 1D
        if edge_weight is not None and edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)

        for i in range(self.num_layer):
            x = self.convs[i](x, edge_index, edge_attr=edge_weight)
            x = self.bns[i](x)
            x = self.acts[i](x)
            if i < self.num_layer - 1:
                x = self.dropout(x)
        return x


class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4, num_layer=2):
        super().__init__()
        act_fc = nn.SiLU()
        self.dropout = nn.Dropout(0.1)
        self.num_layer = num_layer

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()

        # GAT with multi-head attention
        for i in range(num_layer):
            if i == 0:
                self.convs.append(
                    GATConv(input_dim, hidden_dim, heads=heads, dropout=0.1, edge_dim=1)
                )
                self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
            elif i == num_layer - 1:
                self.convs.append(
                    GATConv(
                        hidden_dim * heads,
                        output_dim,
                        heads=1,
                        concat=False,
                        dropout=0.1,
                        edge_dim=1,
                    )
                )
                self.bns.append(nn.BatchNorm1d(output_dim))
            else:
                self.convs.append(
                    GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.1, edge_dim=1)
                )
                self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
            self.acts.append(act_fc)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is not None and edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)

        for i in range(self.num_layer):
            x = self.convs[i](x, edge_index, edge_attr=edge_weight)
            x = self.bns[i](x)
            x = self.acts[i](x)
            if i < self.num_layer - 1:
                x = self.dropout(x)
        return x


class MHAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer=2, num_heads=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads or self._choose_num_heads(output_dim)
        self.input_proj = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_layer):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=output_dim,
                    num_heads=self.num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.attn_norms.append(nn.LayerNorm(output_dim))
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(output_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(output_dim))

    @staticmethod
    def _choose_num_heads(embed_dim):
        for num_heads in (8, 5, 4, 2):
            if embed_dim % num_heads == 0:
                return num_heads
        return 1

    def forward(self, x, edge_index=None, edge_weight=None):
        x = self.input_proj(x).unsqueeze(0)
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.attn_layers,
            self.attn_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            attn_out, _ = attn(x, x, x, need_weights=False)
            x = attn_norm(x + self.dropout(attn_out))
            x = ffn_norm(x + self.dropout(ffn(x)))
        return x.squeeze(0)


class GPSEncoder(nn.Module):
    def __init__(
        self,
        pert_input_size,
        hidden_dim,
        output_dim,
        num_layers=4,
        num_heads=4,
        pe_dim=16,
        dropout=0.0,
    ):
        super().__init__()

        self.pe_dim = pe_dim
        self.num_layers = num_layers
        self.pert_input_size = pert_input_size

        self.pert_emb = nn.Embedding(pert_input_size, hidden_dim - pe_dim)

        # linear projection
        self.node_in = nn.Linear(hidden_dim, hidden_dim)
        self.edge_in = nn.Linear(1, hidden_dim)

        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            # local message passing network
            mpnn = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=hidden_dim,
            )

            # GPS layer
            self.gps_layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=mpnn,
                    heads=num_heads,
                    attn_type="multihead",
                    attn_kwargs={"dropout": dropout},
                )
            )

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, pert_ids, edge_index, pe, edge_weight=None, batch=None):
        x = self.pert_emb(pert_ids)
        if self.pe_dim > 0:
            x = torch.cat([x, pe], dim=-1)

        x = self.node_in(x)

        if edge_weight is not None:
            edge_attr = edge_weight.unsqueeze(-1)
            edge_attr = self.edge_in(edge_attr)
        else:
            edge_attr = None

        for layer in self.gps_layers:
            x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)

        x = self.out_proj(x)
        return x


class DeepFiLMBlock(nn.Module):
    def __init__(self, data_dim, cond_dim):
        super().__init__()

        self.cond_net = nn.Linear(cond_dim, data_dim * 2)  # Outputs gamma and beta

        self.ln = nn.LayerNorm(data_dim)
        self.layer = nn.Linear(data_dim, data_dim)
        self.act_fc = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, cond):
        mod = self.cond_net(cond)
        gamma, beta = mod.chunk(2, dim=-1)

        h = self.layer(x)
        h = self.ln(h)
        h = h * (1 + gamma) + beta  # Using (1 + gamma) for identity preservation
        h = self.act_fc(h)
        h = self.dropout(h)

        return x + h


class DeepSetAggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

        self.rho = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x is [batch_size, num_elements, input_dim]
        phi_x = self.phi(x)  # [batch_size, num_elements, output_dim]
        sum_phi = phi_x.sum(dim=1)  # [batch_size, output_dim]
        return self.rho(sum_phi)  # [batch_size, output_dim]


class ConditionalODE(nn.Module):
    def __init__(
        self,
        pert_input_size,
        data_latent_dim=128,
        data_hidden_dim=256,
        pert_graph=None,
        pert_graph_weight=None,
        pert_latent_dim=50,
        pert_hidden_dim=25,
        pert_encoding="gat",
        gnn_num_layers=2,
        pe_dim=16,
        aggregation_method="sum",
        condition_method="concat",
        time_embed_dim=50,
        use_null_embedding=True,
        context_size=1,
    ):
        super().__init__()

        self.pert_input_size = pert_input_size
        self.pert_latent_dim = pert_latent_dim
        self.data_latent_dim = data_latent_dim
        self.pert_graph = pert_graph
        self.pert_graph_weight = pert_graph_weight
        self.pert_encoding = pert_encoding
        self.aggregation_method = aggregation_method
        self.condition_method = condition_method
        self.time_embed_dim = time_embed_dim
        self.time_dim = 512
        self.use_null_embedding = use_null_embedding
        self.context_size = context_size

        act_fc = nn.SiLU()

        # perturbation embedding and encoder
        if pert_encoding == "one_hot":
            self.pert_encoder = nn.Sequential(
                nn.Linear(pert_input_size, pert_hidden_dim),
                act_fc,
                nn.Linear(pert_hidden_dim, pert_hidden_dim),
                act_fc,
                nn.Linear(pert_hidden_dim, pert_latent_dim),
            )

        elif pert_encoding == "gcn":
            self.pert_emb = nn.Embedding(pert_input_size, pert_latent_dim)
            self.pert_encoder = GCNEncoder(
                pert_latent_dim, pert_hidden_dim, pert_latent_dim, num_layer=gnn_num_layers
            )

        elif pert_encoding == "gin":
            self.pert_emb = nn.Embedding(pert_input_size, pert_latent_dim)
            self.pert_encoder = GINEncoder(
                pert_latent_dim, pert_hidden_dim, pert_latent_dim, num_layer=gnn_num_layers
            )
        elif pert_encoding == "gat":
            self.pert_emb = nn.Embedding(pert_input_size, pert_latent_dim)
            self.pert_encoder = GATEncoder(
                pert_latent_dim, pert_hidden_dim, pert_latent_dim, num_layer=gnn_num_layers
            )

        elif pert_encoding in ["hgnn", "hetero_gat"]:
            if not isinstance(pert_graph, dict):
                raise TypeError(
                    "HGNN perturbation encoding requires pert_graph to be an edge_index dictionary."
                )
            self.pert_emb = nn.Embedding(pert_input_size, pert_latent_dim)
            # Infer node type from edge type tuples (e.g. ('drug', 'rel', 'drug') -> 'drug')
            node_type = list(pert_graph.keys())[0][0]
            self.pert_encoder = HGNNEncoder(
                pert_latent_dim,
                pert_hidden_dim,
                pert_latent_dim,
                edge_types=list(pert_graph.keys()),
                num_layer=gnn_num_layers,
                node_type=node_type,
            )

        elif pert_encoding == "mha":
            self.pert_emb = nn.Embedding(pert_input_size, pert_latent_dim)
            self.pert_encoder = MHAEncoder(
                pert_latent_dim, pert_hidden_dim, pert_latent_dim, num_layer=gnn_num_layers
            )

        elif pert_encoding == "gps":
            if not isinstance(pert_graph, torch.Tensor):
                raise TypeError(
                    "GPS perturbation encoding requires pert_graph to be a tensor edge_index."
                )
            self.pert_encoder = GPSEncoder(
                pert_input_size=pert_input_size,
                hidden_dim=pert_hidden_dim,
                output_dim=pert_latent_dim,
                num_layers=4,
                num_heads=1,
                pe_dim=pe_dim,
                dropout=0.1,
            )

            pert_ids = torch.arange(pert_input_size, device=pert_graph.device)
            self.graph_data = Data(
                pert_ids=pert_ids, edge_index=pert_graph, edge_attr=pert_graph_weight
            )
            transform = AddLaplacianEigenvectorPE(k=pe_dim, attr_name="pe")
            self.graph_data = transform(self.graph_data)
            self.graph_data.pe = self.graph_data.pe.to(pert_graph.device)

        # self.pert_weight = nn.Parameter(torch.randn(self.pert_input_size) * 0.02)

        if self.use_null_embedding:
            # Learnable null embedding for control/unconditional samples
            # Use MLP to match capacity of perturbation encoder
            self.null_pert_embedding = nn.Parameter(
                torch.randn(context_size, pert_latent_dim) * 0.02
            )
            self.null_proj = nn.Sequential(
                nn.Linear(pert_latent_dim, pert_latent_dim),
                nn.SiLU(),
                nn.Linear(pert_latent_dim, pert_latent_dim),
            )

        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim // 2),
            act_fc,
            nn.Linear(self.time_dim // 2, self.time_embed_dim),
        )

        if aggregation_method == "deepset":
            self.aggregator = DeepSetAggregator(pert_latent_dim, pert_hidden_dim, pert_latent_dim)

        if condition_method == "concat":
            self.ode = nn.Sequential(
                nn.Linear(pert_latent_dim + data_latent_dim + time_embed_dim, data_hidden_dim),
                act_fc,
                nn.Linear(data_hidden_dim, data_hidden_dim),
                act_fc,
                nn.Linear(data_hidden_dim, data_latent_dim),
            )
        elif condition_method == "film":
            self.cond_in = nn.Sequential(
                nn.Linear(pert_latent_dim + time_embed_dim, pert_latent_dim + time_embed_dim),
                act_fc,
                nn.Linear(pert_latent_dim + time_embed_dim, pert_latent_dim + time_embed_dim),
            )
            self.ode_blocks = nn.ModuleList(
                [
                    DeepFiLMBlock(
                        data_dim=data_latent_dim, cond_dim=pert_latent_dim + time_embed_dim
                    )
                    for _ in range(4)
                ]
            )
            self.final_proj = nn.Linear(data_latent_dim, data_latent_dim)
            nn.init.zeros_(self.final_proj.weight)
            nn.init.zeros_(self.final_proj.bias)

    def forward(self, t, x, y, c=None):
        time_emb = self.time_encoder(self.timestep_embedding(t, self.time_dim))

        if self.pert_encoding in ["gcn", "gin", "gat", "mha", "hgnn", "hetero_gat"]:
            device = self.pert_emb.weight.device
            y_emb_total = self.pert_emb(torch.arange(self.pert_input_size, device=device))
            y_emb_total = self.pert_encoder(y_emb_total, self.pert_graph)
            # y_emb_total = self.pert_encoder(y_emb_total, self.pert_graph, self.pert_graph_weight)
            # w = mhe * nn.functional.softplus(self.pert_weight) # ensure positivity of weights
            # w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # y_emb = torch.matmul(w, y_emb_total)
            if self.aggregation_method == "sum":
                mhe = multi_hot_label(y.long(), self.pert_input_size, device=device)
                y_emb = torch.matmul(mhe, y_emb_total)
            if self.aggregation_method == "deepset":
                ctrl_mask = y.long() == -1
                y[ctrl_mask] = 0
                y_emb_set = y_emb_total[
                    y.long()
                ]  # [batch_size, num_perturbations, pert_latent_dim]
                y_emb_set[ctrl_mask] = 0
                y_emb = self.aggregator(y_emb_set)

            if self.use_null_embedding:
                # Project null embedding through MLP and add as baseline
                # Ctrl: y_emb = 0 + null_emb (baseline)
                # Pert: y_emb = pert_emb + null_emb (deviation from baseline)
                null_emb = self.null_pert_embedding[0] if c is None else self.null_pert_embedding[c]
                null_emb_projected = self.null_proj(null_emb)
                y_emb = y_emb + null_emb_projected

        elif self.pert_encoding == "gps":
            device = self.graph_data.pert_ids.device
            y_emb_total = self.pert_encoder(
                pert_ids=self.graph_data.pert_ids,
                edge_index=self.graph_data.edge_index,
                pe=self.graph_data.pe,
                edge_weight=self.graph_data.edge_attr,
            )
            # w = mhe * nn.functional.softplus(self.pert_weight)
            # w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # y_emb = torch.matmul(w, y_emb_total)
            if self.aggregation_method == "sum":
                mhe = multi_hot_label(y.long(), self.pert_input_size, device=device)
                y_emb = torch.matmul(mhe, y_emb_total)
            if self.aggregation_method == "deepset":
                ctrl_mask = y.long() == -1
                y[ctrl_mask] = 0
                y_emb_set = y_emb_total[
                    y.long()
                ]  # [batch_size, num_perturbations, pert_latent_dim]
                y_emb_set[ctrl_mask] = 0
                y_emb = self.aggregator(y_emb_set)

            if self.use_null_embedding:
                # Project null embedding through MLP and add as baseline
                null_emb = self.null_pert_embedding[0] if c is None else self.null_pert_embedding[c]
                null_emb_projected = self.null_proj(null_emb)
                y_emb = y_emb + null_emb_projected

        elif self.pert_encoding == "one_hot":
            device = next(self.pert_encoder.parameters()).device
            mhe = multi_hot_label(y.long(), self.pert_input_size, device=device)
            y_emb = self.pert_encoder(mhe)

            if self.use_null_embedding:
                # Project null embedding and add as baseline
                null_emb = self.null_pert_embedding[0] if c is None else self.null_pert_embedding[c]
                null_emb_projected = self.null_proj(null_emb)
                y_emb = y_emb + null_emb_projected

        if self.condition_method == "concat":
            joint = torch.cat([x, y_emb, time_emb], dim=1)
            out = self.ode(joint)
        elif self.condition_method == "film":
            cond_emb = torch.cat([y_emb, time_emb], dim=1)
            cond_emb = self.cond_in(cond_emb)
            h = x
            for block in self.ode_blocks:
                h = block(h, cond_emb)
            out = self.final_proj(h)

        return out

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Copied from: https://github.com/facebookresearch/flow_matching/blob/main/examples/image/models/nn.py

        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        import math

        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        norm_fc = nn.BatchNorm1d(out_features)
        act_fc = nn.ReLU()
        dropout_rate = 0.1

        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features), norm_fc, act_fc, nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.layer(x)


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            LinearLayer(input_dim, hidden_dim),
            LinearLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            LinearLayer(latent_dim, hidden_dim),
            LinearLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            LinearLayer(input_dim, hidden_dim), LinearLayer(hidden_dim, hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            LinearLayer(latent_dim, hidden_dim),
            LinearLayer(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        # KL divergence
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld, recon_loss, kld

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        return recon_x, mu, logvar


class ConditionAwareVAE(VAE):
    """
    VAE with multiple auxiliary objectives for condition-aware latent space:
    1. Supervised contrastive loss (InfoNCE) - pulls same-condition cells together
    2. Optional condition classification head
    3. Optional graph-aware perturbation encoding

    This creates a latent space that:
    - Maintains good reconstruction (standard VAE objective)
    - Groups cells by perturbation condition (contrastive loss)
    - Preserves biological structure (graph regularization)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        num_conditions=None,
        use_contrastive=True,
        use_condition_classifier=False,
        temperature=0.1,
    ):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.use_contrastive = use_contrastive
        self.use_condition_classifier = use_condition_classifier
        self.temperature = temperature

        # Contrastive learning: project latent to normalized embedding space
        if use_contrastive:
            self.contrastive_proj = nn.Sequential(
                nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
            )

        # Optional: condition classifier for multi-task learning
        if use_condition_classifier and num_conditions is not None:
            self.condition_classifier = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_conditions),
            )

    def supervised_contrastive_loss(self, z, condition_labels):
        """
        Supervised contrastive loss (SupCon) adapted for VAE latents.
        Pulls together cells with the same perturbation, pushes apart different ones.

        Args:
            z: latent embeddings [batch_size, latent_dim]
            condition_labels: condition indices [batch_size]

        Returns:
            contrastive_loss: scalar
        """
        # Project to contrastive space and normalize
        z_proj = self.contrastive_proj(z)
        z_proj = nn.functional.normalize(z_proj, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z_proj, z_proj.T) / self.temperature

        # Create mask for positive pairs (same condition)
        labels = condition_labels.unsqueeze(0)
        mask = (labels == labels.T).float()
        mask.fill_diagonal_(0)  # Exclude self-similarity

        # Compute loss
        # For each anchor, contrast with all positives vs all negatives
        exp_sim = torch.exp(sim_matrix)

        # Sum over all negatives (including other positives as negatives is fine for InfoNCE)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Average over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        loss = -mean_log_prob_pos.mean()
        return loss

    def loss(
        self,
        recon_x,
        x,
        mu,
        logvar,
        z=None,
        condition_labels=None,
        beta=1.0,
        contrastive_weight=0.1,
        classifier_weight=0.1,
    ):
        """
        Combined loss with multiple objectives.

        Args:
            recon_x: reconstructed data
            x: original data
            mu: latent mean
            logvar: latent log variance
            z: sampled latent (needed for contrastive loss)
            condition_labels: condition indices [batch_size]
            beta: KLD weight
            contrastive_weight: weight for contrastive loss
            classifier_weight: weight for classification loss
        """
        # Standard VAE losses
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kld
        loss_dict = {"recon": recon_loss.item(), "kld": kld.item()}

        # Contrastive loss
        if self.use_contrastive and z is not None and condition_labels is not None:
            contrastive_loss = self.supervised_contrastive_loss(z, condition_labels)
            total_loss = total_loss + contrastive_weight * contrastive_loss
            loss_dict["contrastive"] = contrastive_loss.item()

        # Classification loss
        if self.use_condition_classifier and z is not None and condition_labels is not None:
            logits = self.condition_classifier(z.detach())  # Detach to prevent gradient flow
            classifier_loss = nn.functional.cross_entropy(logits, condition_labels)
            total_loss = total_loss + classifier_weight * classifier_loss
            loss_dict["classifier"] = classifier_loss.item()

        return total_loss, loss_dict

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z


class LinearPerturbationModel(nn.Module):
    def __init__(self, n_pc, gene_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_pc, gene_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(1, gene_dim))

    def forward(self, pert_embs):
        # pert_embs: (batch_size, 50)
        # W: (n_pc, gene_dim)
        # B: (1, gene_dim)
        pred = pert_embs @ self.W + self.B  # (batch_size, gene_dim)
        return pred


class SCVIVAE(nn.Module):
    """
    Lightweight scVI-like VAE that models raw single-cell counts with a
    Negative Binomial (NB) or Zero-Inflated Negative Binomial (ZINB)
    likelihood. No batch / covariate conditioning.

    Latent variables:
        z       : biological state, ~ N(mu_z, sigma_z^2)
        ell     : log-library size, ~ N(mu_ell, sigma_ell^2)
                  prior is N(library_log_means, library_log_vars), typically
                  set from empirical mean/var of log total counts in training
                  data via ``set_library_priors_from_counts``.

    Decoder produces gene-wise scale rho (softmax over genes) so that the NB
    mean is mu = exp(ell) * rho. Dispersion (theta) is a per-gene learnable
    parameter (gene-wise dispersion). For ZINB, an additional per-cell-per-gene
    dropout logit is produced.

    Note: ``forward`` takes raw (integer) counts. The encoders apply log1p
    internally.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        likelihood="nb",
        dropout_rate=0.1,
        library_log_means=7.0,
        library_log_vars=1.0,
    ):
        super().__init__()
        assert likelihood in ("nb", "zinb"), "likelihood must be 'nb' or 'zinb'"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.likelihood = likelihood

        # Shared trunk could be used; keep separate trunks for clarity.
        # z encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # library encoder
        self.l_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.fc_l_mu = nn.Linear(hidden_dim, 1)
        self.fc_l_logvar = nn.Linear(hidden_dim, 1)

        # decoder trunk -> px_scale (softmax) and optional dropout logits
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.px_scale_decoder = nn.Linear(hidden_dim, input_dim)
        if likelihood == "zinb":
            self.px_dropout_decoder = nn.Linear(hidden_dim, input_dim)

        # gene-wise log-dispersion (theta = exp(px_r))
        self.px_r = nn.Parameter(torch.zeros(input_dim))

        # library priors (registered as buffers so they move with .to(device))
        self.register_buffer("library_log_means", torch.tensor([float(library_log_means)]))
        self.register_buffer("library_log_vars", torch.tensor([float(library_log_vars)]))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def set_library_priors_from_counts(self, counts):
        """Set the log-library prior from observed raw counts (numpy or tensor).

        counts : (n_cells, n_genes) raw count matrix.
        """
        if not torch.is_tensor(counts):
            counts = torch.as_tensor(counts, dtype=torch.float32)
        log_lib = torch.log(counts.sum(dim=1).clamp(min=1.0))
        m = log_lib.mean().item()
        v = log_lib.var().clamp(min=1e-6).item()
        self.library_log_means.fill_(m)
        self.library_log_vars.fill_(v)

    # ------------------------------------------------------------------
    # Encoders / decoder
    # ------------------------------------------------------------------
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """x: raw counts (B, G). Returns dict with z, library samples + params."""
        x_log = torch.log1p(x)
        h = self.encoder(x_log)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)

        h_l = self.l_encoder(x_log)
        l_mu = self.fc_l_mu(h_l)
        l_logvar = self.fc_l_logvar(h_l).clamp(min=-10.0, max=10.0)
        log_library = self.reparameterize(l_mu, l_logvar)
        return {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "log_library": log_library,
            "l_mu": l_mu,
            "l_logvar": l_logvar,
        }

    def decode(self, z, log_library):
        """Returns dict with px_rate (NB mean), px_r (theta), and px_dropout."""
        h = self.decoder(z)
        px_scale = torch.softmax(self.px_scale_decoder(h), dim=-1)
        px_rate = torch.exp(log_library) * px_scale
        # clamp rate for numerical stability
        px_rate = px_rate.clamp(min=1e-8, max=1e8)
        theta = torch.exp(self.px_r).clamp(min=1e-4, max=1e4)
        out = {"px_scale": px_scale, "px_rate": px_rate, "px_r": theta}
        if self.likelihood == "zinb":
            out["px_dropout"] = self.px_dropout_decoder(h)
        return out

    def forward(self, x):
        enc = self.encode(x)
        dec = self.decode(enc["z"], enc["log_library"])
        out = {**enc, **dec}
        return out

    # ------------------------------------------------------------------
    # Likelihood / loss
    # ------------------------------------------------------------------
    @staticmethod
    def _nb_log_prob(x, mu, theta, eps=1e-8):
        """Log-prob of NB parameterized by mean (mu) and dispersion (theta).

        Var(x) = mu + mu^2 / theta.
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
        )

    @classmethod
    def _zinb_log_prob(cls, x, mu, theta, pi_logits, eps=1e-8):
        """Log-prob of ZINB. pi_logits are dropout logits (log pi/(1-pi))."""
        log_theta_mu_eps = torch.log(theta + mu + eps)
        # log(1 - pi) and log(pi) from logits in a stable way
        log_one_minus_pi = -nn.functional.softplus(pi_logits)  # log(sigmoid(-pi_logits))
        log_pi = -nn.functional.softplus(-pi_logits)  # log(sigmoid(pi_logits))

        # Standard NB log-prob
        nb_lp = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
        )

        # NB(x=0) log-prob (= theta * log(theta/(theta+mu)))
        nb_zero_lp = theta * (torch.log(theta + eps) - log_theta_mu_eps)
        # log( pi + (1-pi) * NB(0) ) = logsumexp(log_pi, log_one_minus_pi + nb_zero_lp)
        case_zero = torch.logsumexp(
            torch.stack([log_pi, log_one_minus_pi + nb_zero_lp], dim=0), dim=0
        )
        case_non_zero = log_one_minus_pi + nb_lp

        is_zero = (x < 1e-8).float()
        return is_zero * case_zero + (1.0 - is_zero) * case_non_zero

    def reconstruction_loss(self, x, fwd):
        if self.likelihood == "nb":
            lp = self._nb_log_prob(x, fwd["px_rate"], fwd["px_r"])
        else:
            lp = self._zinb_log_prob(x, fwd["px_rate"], fwd["px_r"], fwd["px_dropout"])
        # sum over genes, sum over batch (matches existing VAE 'sum' reduction)
        return -lp.sum()

    def kl_z(self, fwd):
        return -0.5 * torch.sum(1 + fwd["logvar"] - fwd["mu"].pow(2) - fwd["logvar"].exp())

    def kl_library(self, fwd):
        prior_mean = self.library_log_means.to(fwd["l_mu"].device)
        prior_var = self.library_log_vars.to(fwd["l_mu"].device)
        var = fwd["l_logvar"].exp()
        kl = 0.5 * (
            torch.log(prior_var)
            - fwd["l_logvar"]
            + (var + (fwd["l_mu"] - prior_mean).pow(2)) / prior_var
            - 1.0
        )
        return kl.sum()

    def loss(self, x, fwd, beta=1.0):
        recon = self.reconstruction_loss(x, fwd)
        kld_z = self.kl_z(fwd)
        kld_l = self.kl_library(fwd)
        total = recon + beta * (kld_z + kld_l)
        loss_dict = {
            "recon": recon.item(),
            "kld_z": kld_z.item(),
            "kld_l": kld_l.item(),
        }
        return total, loss_dict

    # ------------------------------------------------------------------
    # Inference helpers (for downstream code)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_latent_representation(
        self,
        adata,
        batch_size: int = 256,
        device: str = "cpu",
        layer: str = "counts",
        give_mean: bool = True,
    ) -> np.ndarray:
        """Return latent z (mean by default) as numpy array for an AnnData."""
        self.eval()
        if layer is not None and layer in adata.layers:
            X = adata.layers[layer]
        else:
            X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = torch.as_tensor(X, dtype=torch.float32)
        zs = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device, non_blocking=True)
            x_log = torch.log1p(xb)
            h = self.encoder(x_log)
            mu = self.fc_mu(h)
            if give_mean:
                zs.append(mu.cpu().numpy())
            else:
                logvar = self.fc_logvar(h).clamp(-10, 10)
                zs.append(self.reparameterize(mu, logvar).cpu().numpy())
        return np.concatenate(zs, axis=0)

    @torch.no_grad()
    def sample_expression(self, z, log_library):
        """Decode latent z + log library size into NB/ZINB sampled counts."""
        dec = self.decode(z, log_library)
        mu = dec["px_rate"]
        theta = dec["px_r"]
        # NegativeBinomial parameterized by total_count=theta and logits=log(mu/theta)
        logits = torch.log(mu + 1e-8) - torch.log(theta + 1e-8)
        nb = torch.distributions.NegativeBinomial(total_count=theta, logits=logits)
        x = nb.sample()
        if self.likelihood == "zinb":
            pi = torch.sigmoid(dec["px_dropout"])
            mask = (torch.rand_like(pi) > pi).float()
            x = x * mask
        return x
