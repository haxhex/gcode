import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Optional
try:
    from config import FlareGNNConfig
    from scpg.builder import SCPGGraph, SCPGEdgeType
    from flare_gnn.layers import FlareInputEncoding, DualBranchGatedLayer, MethodHierarchicalPooling
except ImportError:
    from ..config import FlareGNNConfig
    from ..scpg.builder import SCPGGraph, SCPGEdgeType
    from .layers import FlareInputEncoding, DualBranchGatedLayer, MethodHierarchicalPooling

class FlareGNN(nn.Module):

    def __init__(self, config: Optional[FlareGNNConfig]=None):
        super().__init__()
        self.cfg = config or FlareGNNConfig()
        self.input_encoder = FlareInputEncoding(d_sem=self.cfg.d_sem, d_stat=self.cfg.d_stat_post, d_proj=self.cfg.d_proj, d_flare=self.cfg.d_flare)
        self.layers = nn.ModuleList([DualBranchGatedLayer(d_flare=self.cfg.d_flare, num_heads=self.cfg.num_heads, num_relations=6) for _ in range(self.cfg.num_layers)])
        self.method_pooler = MethodHierarchicalPooling(d_flare=self.cfg.d_flare)
        self.stmt_classifier = nn.Sequential(nn.Linear(self.cfg.d_flare, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.method_classifier = nn.Sequential(nn.Linear(self.cfg.d_flare, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x_sem: torch.Tensor, x_stat: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, method_map: Dict[int, List[int]]) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
        h = self.input_encoder(x_sem, x_stat)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        y_hat_node = self.stmt_classifier(h).squeeze(-1)
        h_method, method_ids = self.method_pooler(h, method_map)
        y_hat_method = self.method_classifier(h_method).squeeze(-1)
        return (y_hat_node, y_hat_method, method_ids, h)

class FlareGNNLoss(nn.Module):

    def __init__(self, config: FlareGNNConfig):
        super().__init__()
        self.cfg = config

    def forward(self, y_hat_node: torch.Tensor, y_true_node: torch.Tensor, y_hat_method: torch.Tensor, y_true_method: torch.Tensor, method_map: Dict[int, List[int]]) -> Tuple[torch.Tensor, dict]:
        p_t = torch.where(y_true_node == 1.0, y_hat_node, 1.0 - y_hat_node)
        alpha_t = torch.where(y_true_node == 1.0, self.cfg.alpha_focal, 1.0 - self.cfg.alpha_focal)
        focal_weight = alpha_t * torch.pow(1.0 - p_t + 1e-09, self.cfg.gamma_focal)
        bce = F.binary_cross_entropy(y_hat_node, y_true_node, reduction='none')
        l_focal = (focal_weight * bce).mean()
        statement_logits = torch.logit(y_hat_node.clamp(1e-06, 1.0 - 1e-06))
        listwise_terms = []
        for method_id in sorted(method_map):
            statement_indices = method_map[method_id]
            if not statement_indices:
                continue
            idx = torch.as_tensor(statement_indices, dtype=torch.long, device=y_hat_node.device)
            targets = y_true_node[idx]
            if not torch.any(targets > 0):
                continue
            target_dist = F.softmax(self.cfg.beta_target * targets, dim=0)
            predicted_log_dist = F.log_softmax(statement_logits[idx], dim=0)
            listwise_terms.append(-(target_dist * predicted_log_dist).sum())
        l_rank = torch.stack(listwise_terms).mean() if listwise_terms else torch.tensor(0.0, device=y_hat_node.device)
        l_stmt = l_focal + self.cfg.lambda_rank * l_rank
        if len(y_hat_method) > 0 and len(y_true_method) > 0:
            l_method = F.binary_cross_entropy(y_hat_method, y_true_method)
        else:
            l_method = torch.tensor(0.0, device=y_hat_node.device)
        total_loss = self.cfg.lambda_stmt * l_stmt + self.cfg.lambda_method * l_method
        breakdown = {'total_loss': total_loss.item(), 'l_focal': l_focal.item(), 'l_rank': l_rank.item() if isinstance(l_rank, torch.Tensor) else 0.0, 'l_stmt': l_stmt.item(), 'l_method': l_method.item() if isinstance(l_method, torch.Tensor) else 0.0}
        return (total_loss, breakdown)

class FlareGNNInference:

    def __init__(self, model: FlareGNN, config: Optional[FlareGNNConfig]=None):
        self.model = model
        self.cfg = config or model.cfg
        self.model.eval()

    def infer_and_extract_repair_subgraph(self, scpg: SCPGGraph) -> Dict[str, any]:
        stmt_ids = sorted(list(scpg.nodes.keys()))
        if not stmt_ids:
            return {'stmt_scores': {}, 'method_scores': {}, 'seed_stmts': [], 'repair_subgraph': scpg}
        id_to_idx = {sid: i for i, sid in enumerate(stmt_ids)}
        x_sem = torch.tensor(np.stack([scpg.nodes[sid].semantic_feat for sid in stmt_ids]), dtype=torch.float32)
        if scpg.phase_mask == 0:
            x_stat = torch.tensor(np.stack([scpg.nodes[sid].static_feat for sid in stmt_ids]), dtype=torch.float32)
        else:
            combined_stat = []
            for sid in stmt_ids:
                s_feat = np.concatenate([scpg.nodes[sid].static_feat, scpg.nodes[sid].dynamic_feat], axis=-1)
                combined_stat.append(s_feat)
            x_stat = torch.tensor(np.stack(combined_stat), dtype=torch.float32)
        edge_list = []
        edge_types = []
        for e in scpg.edges:
            if e.src in id_to_idx and e.dst in id_to_idx:
                edge_list.append([id_to_idx[e.src], id_to_idx[e.dst]])
                edge_types.append(e.edge_type.value)
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_type_tensor = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type_tensor = torch.zeros(0, dtype=torch.long)
        method_map_idx = {}
        for m_id, s_ids in scpg.methods.items():
            method_map_idx[m_id] = [id_to_idx[s] for s in s_ids if s in id_to_idx]
        with torch.no_grad():
            y_hat_node, y_hat_method, method_ids, _ = self.model(x_sem=x_sem, x_stat=x_stat, edge_index=edge_index, edge_type=edge_type_tensor, method_map=method_map_idx)
        node_scores_np = y_hat_node.cpu().numpy()
        method_scores_np = y_hat_method.cpu().numpy() if y_hat_method.dim() > 0 else np.array([y_hat_method.item()])
        median_score = float(np.median(node_scores_np))
        mad = float(np.median(np.abs(node_scores_np - median_score)))
        tau = median_score + self.cfg.lambda_mad * max(0.0001, mad)
        seed_stmts: Set[int] = set()
        stmt_scores: Dict[int, float] = {}
        for i, sid in enumerate(stmt_ids):
            score = float(node_scores_np[i])
            stmt_scores[sid] = score
            scpg.nodes[sid].suspiciousness_score = score
            if score >= tau:
                seed_stmts.add(sid)
        if not seed_stmts and stmt_ids:
            top_stmt = stmt_ids[int(np.argmax(node_scores_np))]
            seed_stmts.add(top_stmt)
        repair_stmts: Set[int] = set(seed_stmts)
        visited = set(seed_stmts)
        queue = [(sid, 0) for sid in seed_stmts]
        adj: Dict[int, Set[int]] = {sid: set() for sid in stmt_ids}
        for e in scpg.edges:
            if e.edge_type in [SCPGEdgeType.CFG_FLOW, SCPGEdgeType.DFG_DATA, SCPGEdgeType.PDG_DEP]:
                adj[e.src].add(e.dst)
                adj[e.dst].add(e.src)
        while queue:
            curr_sid, hop = queue.pop(0)
            if hop < self.cfg.k_hop:
                for neighbor in adj.get(curr_sid, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        repair_stmts.add(neighbor)
                        queue.append((neighbor, hop + 1))
        if len(repair_stmts) > self.cfg.b_max:
            sorted_repair = sorted(list(repair_stmts), key=lambda sid: stmt_scores.get(sid, 0.0), reverse=True)
            repair_stmts = set(sorted_repair[:self.cfg.b_max])
        repair_subgraph = scpg.get_induced_subgraph(repair_stmts)
        method_scores: Dict[int, float] = {}
        for i, m_id in enumerate(method_ids):
            method_scores[m_id] = float(method_scores_np[i]) if i < len(method_scores_np) else 0.0
        return {'stmt_scores': stmt_scores, 'method_scores': method_scores, 'mad_threshold': tau, 'seed_stmts': sorted(list(seed_stmts)), 'repair_stmt_ids': sorted(list(repair_stmts)), 'repair_subgraph': repair_subgraph, 'num_repair_nodes': len(repair_stmts), 'num_repair_edges': len(repair_subgraph.edges)}