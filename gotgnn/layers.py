import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GoTInputEncoding(nn.Module):

    def __init__(self, d_sem: int=384, d_abs: int=5, d_qual: int=5, num_types: int=4, d_type: int=32, d_model: int=256):
        super().__init__()
        self.proj_sem = nn.Linear(d_sem, d_model)
        self.proj_abs = nn.Linear(d_abs, d_model)
        self.proj_qual = nn.Linear(d_qual, d_model)
        self.type_embedding = nn.Embedding(num_types, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_sem: torch.Tensor, x_abs: torch.Tensor, x_qual: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        h = self.proj_sem(x_sem) + self.proj_abs(x_abs) + self.proj_qual(x_qual) + self.type_embedding(node_types)
        return self.dropout(self.layer_norm(h))

class RGATv2Layer(nn.Module):

    def __init__(self, d_model: int=256, num_heads: int=4, num_relations: int=6):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_relations = num_relations
        self.w_src = nn.Parameter(torch.Tensor(num_relations, d_model, d_model))
        self.w_dst = nn.Parameter(torch.Tensor(num_relations, d_model, d_model))
        self.attn_vec = nn.Parameter(torch.Tensor(num_relations, num_heads, self.d_k))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_src)
        nn.init.xavier_uniform_(self.w_dst)
        nn.init.xavier_uniform_(self.attn_vec)

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
            h_src = torch.matmul(h[src], self.w_src[r]).view(-1, self.num_heads, self.d_k)
            h_dst = torch.matmul(h[dst], self.w_dst[r]).view(-1, self.num_heads, self.d_k)
            scores = (self.attn_vec[r].unsqueeze(0) * self.leaky_relu(h_src + h_dst)).sum(dim=-1)
            scores = scores / math.sqrt(self.d_k)
            exp_scores = torch.exp(scores - scores.max(dim=0, keepdim=True)[0])
            sum_exp = torch.zeros(N, self.num_heads, device=h.device)
            sum_exp.index_add_(0, dst, exp_scores)
            alpha = exp_scores / (sum_exp[dst] + 1e-09)
            messages = h_src * alpha.unsqueeze(-1)
            messages = messages.view(-1, self.d_model)
            out.index_add_(0, dst, messages)
        return self.layer_norm(h + out + self.bias)

class CausalDAGGlobalAttention(nn.Module):

    def __init__(self, d_model: int=256, num_heads: int=4, alpha: float=1.0, d_cutoff: int=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.alpha = alpha
        self.d_cutoff = d_cutoff
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        q = self.w_q(h).view(N, self.num_heads, self.d_k).transpose(0, 1)
        k = self.w_k(h).view(N, self.num_heads, self.d_k).transpose(0, 1)
        v = self.w_v(h).view(N, self.num_heads, self.d_k).transpose(0, 1)
        raw_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        bias = -self.alpha * dist_matrix.transpose(0, 1)
        causal_mask = torch.isneginf(bias) | torch.isnan(bias) | (dist_matrix.transpose(0, 1) > self.d_cutoff)
        eye = torch.eye(N, dtype=torch.bool, device=h.device)
        causal_mask = causal_mask & ~eye
        scores = raw_scores + bias.unsqueeze(0)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -1000000000.0)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(0, 1).contiguous().view(N, self.d_model)
        out = self.w_out(context)
        return self.layer_norm(h + out)

class AttentionPooling(nn.Module):

    def __init__(self, d_model: int=256):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(h), dim=0)
        h_graph = (h * weights).sum(dim=0, keepdim=True)
        return h_graph