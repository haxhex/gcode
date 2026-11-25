import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class FlareInputEncoding(nn.Module):

    def __init__(self, d_sem: int=384, d_stat: int=14, d_proj: int=128, d_flare: int=128):
        super().__init__()
        self.proj_sem = nn.Linear(d_sem, d_flare)
        self.proj_stat = nn.Linear(d_stat, d_proj)
        self.init_proj = nn.Linear(d_flare + d_proj, d_flare)
        self.layer_norm = nn.LayerNorm(d_flare)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_sem: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        if x_stat.size(-1) < 14:
            pad = torch.zeros(x_stat.size(0), 14 - x_stat.size(-1), device=x_stat.device)
            x_stat = torch.cat([x_stat, pad], dim=-1)
        h_sem = F.gelu(self.proj_sem(x_sem))
        h_stat = F.relu(self.proj_stat(x_stat))
        h = self.init_proj(torch.cat([h_sem, h_stat], dim=-1))
        return self.dropout(self.layer_norm(h))

class DualBranchGatedLayer(nn.Module):

    def __init__(self, d_flare: int=128, num_heads: int=4, num_relations: int=6):
        super().__init__()
        self.d_flare = d_flare
        self.num_heads = num_heads
        self.d_k = d_flare // num_heads
        self.num_relations = num_relations
        self.w_src_sem = nn.Parameter(torch.Tensor(num_relations, d_flare, d_flare))
        self.w_dst_sem = nn.Parameter(torch.Tensor(num_relations, d_flare, d_flare))
        self.attn_vec_sem = nn.Parameter(torch.Tensor(num_relations, num_heads, self.d_k))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.gin_mlp = nn.Sequential(nn.Linear(d_flare, d_flare), nn.ReLU(), nn.Linear(d_flare, d_flare))
        self.eps = nn.Parameter(torch.zeros(num_relations))
        self.gate_fc = nn.Linear(d_flare * 2, d_flare)
        self.layer_norm = nn.LayerNorm(d_flare)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_src_sem)
        nn.init.xavier_uniform_(self.w_dst_sem)
        nn.init.xavier_uniform_(self.attn_vec_sem)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        E = edge_index.size(1)
        if E == 0:
            return h
        h_sem = torch.zeros_like(h)
        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue
            r_edges = edge_index[:, mask]
            src, dst = (r_edges[0], r_edges[1])
            s_proj = torch.matmul(h[src], self.w_src_sem[r]).view(-1, self.num_heads, self.d_k)
            d_proj = torch.matmul(h[dst], self.w_dst_sem[r]).view(-1, self.num_heads, self.d_k)
            scores = (self.attn_vec_sem[r].unsqueeze(0) * self.leaky_relu(s_proj + d_proj)).sum(dim=-1) / math.sqrt(self.d_k)
            exp_scores = torch.exp(scores - scores.max(dim=0, keepdim=True)[0])
            sum_exp = torch.zeros(N, self.num_heads, device=h.device)
            sum_exp.index_add_(0, dst, exp_scores)
            alpha = exp_scores / (sum_exp[dst] + 1e-09)
            messages = (s_proj * alpha.unsqueeze(-1)).view(-1, self.d_flare)
            h_sem.index_add_(0, dst, messages)
        h_struct = torch.zeros_like(h)
        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue
            r_edges = edge_index[:, mask]
            src, dst = (r_edges[0], r_edges[1])
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src])
            gin_out = self.gin_mlp((1.0 + self.eps[r]) * h + agg)
            h_struct += gin_out
        concat_feat = torch.cat([h_sem, h_struct], dim=-1)
        gate = torch.sigmoid(self.gate_fc(concat_feat))
        fused = gate * h_sem + (1.0 - gate) * h_struct
        return self.layer_norm(h + fused)

class MethodHierarchicalPooling(nn.Module):

    def __init__(self, d_flare: int=128):
        super().__init__()
        self.attn = nn.Linear(d_flare, 1)

    def forward(self, h_statements: torch.Tensor, method_map: Dict[int, List[int]]) -> Tuple[torch.Tensor, List[int]]:
        method_vectors = []
        method_ids = sorted(list(method_map.keys()))
        for m_id in method_ids:
            stmt_indices = method_map[m_id]
            if not stmt_indices:
                method_vectors.append(torch.zeros(1, h_statements.size(-1), device=h_statements.device))
                continue
            stmt_idx_tensor = torch.tensor(stmt_indices, dtype=torch.long, device=h_statements.device)
            h_m = h_statements[stmt_idx_tensor]
            weights = F.softmax(self.attn(h_m), dim=0)
            vec = (h_m * weights).sum(dim=0, keepdim=True)
            method_vectors.append(vec)
        if method_vectors:
            h_methods = torch.cat(method_vectors, dim=0)
        else:
            h_methods = torch.zeros(1, h_statements.size(-1), device=h_statements.device)
        return (h_methods, method_ids)

class RGCNLayer(nn.Module):

    def __init__(self, d_model: int=128, num_relations: int=6):
        super().__init__()
        self.d_model = d_model
        self.num_relations = num_relations
        self.weights = nn.Parameter(torch.Tensor(num_relations, d_model, d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        E = edge_index.size(1)
        if E == 0:
            return h
        out = torch.zeros_like(h)
        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue
            r_edges = edge_index[:, mask]
            src, dst = (r_edges[0], r_edges[1])
            h_trans = torch.matmul(h[src], self.weights[r])
            out.index_add_(0, dst, h_trans)
        return self.layer_norm(F.relu(h + out + self.bias))