import torch
import numpy as np
from typing import List, Dict, Optional
try:
    from config import GoTGNNConfig
    from thought_graph.graph_types import ThoughtGraph, ThoughtEdgeType
    from gotgnn.model import GoTGNN
except ImportError:
    from ..config import GoTGNNConfig
    from ..thought_graph.graph_types import ThoughtGraph, ThoughtEdgeType
    from .model import GoTGNN

class GoTGNNInference:

    def __init__(self, model: GoTGNN, config: Optional[GoTGNNConfig]=None):
        self.model = model
        self.cfg = config or model.cfg
        self.model.eval()

    def extract_candidate_paths(self, graph: ThoughtGraph, max_paths: int=8, max_len: int=6) -> List[List[int]]:
        paths: List[List[int]] = []
        roots = graph.root_nodes if graph.root_nodes else [min(graph.nodes.keys())]

        def dfs(curr: int, current_path: List[int]):
            if len(paths) >= max_paths or len(current_path) >= max_len:
                paths.append(current_path.copy())
                return
            neighbors = [dst for dst, _ in graph.adj_out.get(curr, [])]
            if not neighbors:
                paths.append(current_path.copy())
                return
            for nxt in neighbors:
                if nxt not in current_path:
                    current_path.append(nxt)
                    dfs(nxt, current_path)
                    current_path.pop()
                    if len(paths) >= max_paths:
                        break
        for r in roots:
            dfs(r, [r])
            if len(paths) >= max_paths:
                break
        if not paths:
            paths = [[nid] for nid in sorted(graph.nodes.keys())[:max_paths]]
        return paths

    def _prepare_tensors(self, graph: ThoughtGraph):
        node_ids = sorted(list(graph.nodes.keys()))
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        x_sem = torch.tensor(np.stack([graph.nodes[nid].semantic_feat for nid in node_ids]), dtype=torch.float32)
        x_abs = torch.tensor(np.stack([graph.nodes[nid].abs_feat for nid in node_ids]), dtype=torch.float32)
        x_qual = torch.tensor(np.stack([graph.nodes[nid].qual_feat for nid in node_ids]), dtype=torch.float32)
        node_types = torch.tensor([graph.nodes[nid].node_type.value for nid in node_ids], dtype=torch.long)
        edge_list = []
        type_list = []
        for e in graph.edges:
            if e.src in id_to_idx and e.dst in id_to_idx:
                edge_list.append([id_to_idx[e.src], id_to_idx[e.dst]])
                type_list.append(e.edge_type.value)
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(type_list, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        _, dist_mat = graph.compute_causal_ancestors_and_distances(d_cutoff=self.cfg.d_cutoff)
        dist_tensor = torch.tensor(dist_mat, dtype=torch.float32)
        return (node_ids, id_to_idx, x_sem, x_abs, x_qual, node_types, edge_index, edge_type, dist_tensor)

    def infer(self, graph: ThoughtGraph) -> Dict[str, any]:
        node_ids, id_to_idx, x_sem, x_abs, x_qual, node_types, edge_index, edge_type, dist_tensor = self._prepare_tensors(graph)
        candidate_paths_raw = self.extract_candidate_paths(graph, max_paths=self.cfg.max_search_width, max_len=self.cfg.max_path_length)
        candidate_paths_idx = [[id_to_idx[nid] for nid in p if nid in id_to_idx] for p in candidate_paths_raw]
        with torch.no_grad():
            y_hat_node, p_hat_pi, a_hat_type, a_hat_count, _ = self.model(x_sem=x_sem, x_abs=x_abs, x_qual=x_qual, node_types=node_types, edge_index=edge_index, edge_type=edge_type, dist_matrix=dist_tensor, candidate_paths=candidate_paths_idx)
        leaf_ids = set(graph.leaf_nodes or [nid for nid in node_ids if not graph.adj_out.get(nid)])
        pruned_nodes = []
        for i, nid in enumerate(node_ids):
            score = float(y_hat_node[i].item())
            graph.nodes[nid].predicted_score = score
            is_pruned = nid in leaf_ids and score < self.cfg.prune_threshold
            graph.nodes[nid].pruned = is_pruned
            if is_pruned:
                pruned_nodes.append(nid)
        if p_hat_pi is not None and len(p_hat_pi) > 0:
            eligible = [i for i, path in enumerate(candidate_paths_raw) if path and path[-1] not in pruned_nodes]
            pool = eligible or list(range(len(candidate_paths_raw)))
            best_idx = max(pool, key=lambda i: float(p_hat_pi[i].item()))
            best_path_nodes = candidate_paths_raw[best_idx]
            best_path_prob = float(p_hat_pi[best_idx].item())
            all_frontier_low_utility = not bool(eligible)
        else:
            best_idx = 0
            best_path_nodes = candidate_paths_raw[0] if candidate_paths_raw else []
            best_path_prob = 0.0
            all_frontier_low_utility = True
        best_subgraph_nodes = set(best_path_nodes)
        queue = list(best_path_nodes)
        while queue:
            current = queue.pop()
            for parent, _ in graph.adj_in.get(current, []):
                if parent not in best_subgraph_nodes:
                    best_subgraph_nodes.add(parent)
                    queue.append(parent)
        best_subgraph_nodes = sorted(best_subgraph_nodes)
        best_subgraph_edges = [{'src': edge.src, 'dst': edge.dst, 'type': edge.edge_type.name} for edge in graph.edges if edge.src in best_subgraph_nodes and edge.dst in best_subgraph_nodes]
        action_names = ['IDEA', 'CRITIQUE', 'REFINE', 'AGGREGATE_SIMILAR', 'AGGREGATE_TASK']
        pred_action_id = int(torch.argmax(a_hat_type, dim=-1).item())
        pred_action_name = action_names[pred_action_id] if pred_action_id < len(action_names) else 'Refine'
        pred_count = min(self.cfg.max_action_children, max(1, int(round(float(a_hat_count.item())))))
        action_probs = torch.softmax(a_hat_type, dim=-1).squeeze(0).tolist()
        return {'node_predictions': {nid: graph.nodes[nid].predicted_score for nid in node_ids}, 'pruned_nodes': pruned_nodes, 'prune_ratio': len(pruned_nodes) / max(1, len(node_ids)), 'best_path': best_path_nodes, 'best_subgraph_nodes': best_subgraph_nodes, 'best_subgraph_edges': best_subgraph_edges, 'all_frontier_low_utility': all_frontier_low_utility, 'best_path_probability': best_path_prob, 'best_path_meets_threshold': best_path_prob >= self.cfg.path_success_threshold, 'path_success_threshold': self.cfg.path_success_threshold, 'path_success_probabilities': [float(v) for v in p_hat_pi.tolist()] if p_hat_pi is not None else [], 'action_type': pred_action_name, 'action_type_id': pred_action_id, 'action_probabilities': {name: float(prob) for name, prob in zip(action_names, action_probs)}, 'action_count': pred_count}