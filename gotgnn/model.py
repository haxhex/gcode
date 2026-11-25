import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
try:
    from config import GoTGNNConfig
    from gotgnn.layers import GoTInputEncoding, RGATv2Layer, CausalDAGGlobalAttention, AttentionPooling
except ImportError:
    from ..config import GoTGNNConfig
    from .layers import GoTInputEncoding, RGATv2Layer, CausalDAGGlobalAttention, AttentionPooling

class GoTGNN(nn.Module):

    def __init__(self, config: Optional[GoTGNNConfig]=None):
        super().__init__()
        self.cfg = config or GoTGNNConfig()
        self.input_encoder = GoTInputEncoding(d_sem=self.cfg.d_sem, d_abs=self.cfg.d_abs, d_qual=self.cfg.d_qual, num_types=4, d_type=self.cfg.d_type, d_model=self.cfg.d_model)
        self.local_layers = nn.ModuleList([RGATv2Layer(d_model=self.cfg.d_model, num_heads=self.cfg.num_heads, num_relations=6) for _ in range(self.cfg.num_layers)])
        self.causal_global_attn = CausalDAGGlobalAttention(d_model=self.cfg.d_model, num_heads=self.cfg.num_global_heads, alpha=self.cfg.alpha, d_cutoff=self.cfg.d_cutoff)
        self.pool = AttentionPooling(d_model=self.cfg.d_model)
        self.head_node_quality = nn.Sequential(nn.Linear(self.cfg.d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_path_policy = nn.Sequential(nn.Linear(self.cfg.d_model * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_action_type = nn.Sequential(nn.Linear(self.cfg.d_model, 64), nn.ReLU(), nn.Linear(64, 5))
        self.head_action_count = nn.Sequential(nn.Linear(self.cfg.d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, x_sem: torch.Tensor, x_abs: torch.Tensor, x_qual: torch.Tensor, node_types: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, dist_matrix: torch.Tensor, candidate_paths: Optional[List[List[int]]]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_encoder(x_sem, x_abs, x_qual, node_types)
        for layer in self.local_layers:
            h = layer(h, edge_index, edge_type)
        h_global = self.causal_global_attn(h, dist_matrix)
        h_graph = self.pool(h_global)
        y_hat_node = self.head_node_quality(h_global).squeeze(-1)
        p_hat_pi = None
        if candidate_paths is not None and len(candidate_paths) > 0:
            path_scores = []
            for path in candidate_paths:
                path_tensor = torch.tensor(path, dtype=torch.long, device=h.device)
                h_path = h_global[path_tensor].mean(dim=0, keepdim=True)
                pair_feat = torch.cat([h_path, h_graph], dim=-1)
                score = self.head_path_policy(pair_feat)
                path_scores.append(score)
            path_logits = torch.cat(path_scores, dim=-1).squeeze(0)
            p_hat_pi = torch.sigmoid(path_logits)
        a_hat_type = self.head_action_type(h_graph)
        a_hat_count = self.head_action_count(h_graph)
        return (y_hat_node, p_hat_pi, a_hat_type, a_hat_count, h_global)