from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import numpy as np

class ThoughtNodeType(Enum):
    IDEA = 0
    CRITIQUE = 1
    REFINEMENT = 2
    AGGREGATION = 3

class ThoughtEdgeType(Enum):
    CRITIQUE = 0
    INFORMS = 1
    AGGREGATE = 2
    DEPENDS = 3
    CROSS_TASK = 4
    FEEDBACK = 5

class AbstractionLevel(Enum):
    ARCHITECTURE = 0
    MODULE = 1
    CLASS = 2
    FUNCTION = 3
    IMPLEMENTATION = 4

class QualityDimension(Enum):
    SECURITY = 0
    PERFORMANCE = 1
    ARCHITECTURE = 2
    TESTABILITY = 3
    MAINTAINABILITY = 4

@dataclass
class ThoughtNode:
    node_id: int
    task_id: str
    node_type: ThoughtNodeType
    content: str
    abstraction: AbstractionLevel = AbstractionLevel.FUNCTION
    quality_dims: List[QualityDimension] = field(default_factory=list)
    semantic_feat: np.ndarray = field(default_factory=lambda: np.zeros(384, dtype=np.float32))
    abs_feat: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    qual_feat: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    ground_truth_score: float = 0.0
    predicted_score: float = 0.0
    pruned: bool = False

    def build_feature_vectors(self):
        self.abs_feat = np.zeros(5, dtype=np.float32)
        self.abs_feat[self.abstraction.value] = 1.0
        self.qual_feat = np.zeros(5, dtype=np.float32)
        for q in self.quality_dims:
            self.qual_feat[q.value] = 1.0

@dataclass
class ThoughtEdge:
    src: int
    dst: int
    edge_type: ThoughtEdgeType
    weight: float = 1.0

class ThoughtGraph:

    def __init__(self, graph_id: str, task_name: str, config_id: str='C1', task_id: Optional[str]=None):
        self.graph_id = graph_id
        self.task_name = task_name
        self.task_id = task_id or task_name
        self.config_id = config_id
        self.nodes: Dict[int, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.adj_out: Dict[int, List[Tuple[int, ThoughtEdgeType]]] = {}
        self.adj_in: Dict[int, List[Tuple[int, ThoughtEdgeType]]] = {}
        self.leaf_nodes: List[int] = []
        self.root_nodes: List[int] = []

    def add_node(self, node: ThoughtNode):
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj_out:
            self.adj_out[node.node_id] = []
        if node.node_id not in self.adj_in:
            self.adj_in[node.node_id] = []

    def add_edge(self, src: int, dst: int, edge_type: ThoughtEdgeType, weight: float=1.0):
        if src not in self.nodes or dst not in self.nodes:
            raise ValueError(f'Edge endpoints must exist before insertion: {src} -> {dst}')
        if src == dst:
            raise ValueError('Thought graphs cannot contain self-edges')
        if any((e.src == src and e.dst == dst and (e.edge_type == edge_type) for e in self.edges)):
            return
        queue, seen = ([dst], set())
        while queue:
            current = queue.pop()
            if current == src:
                raise ValueError(f'Edge {src} -> {dst} would introduce a cycle')
            if current in seen:
                continue
            seen.add(current)
            queue.extend((n for n, _ in self.adj_out.get(current, [])))
        edge = ThoughtEdge(src, dst, edge_type, weight)
        self.edges.append(edge)
        self.adj_out[src].append((dst, edge_type))
        self.adj_in[dst].append((src, edge_type))

    def topological_sort(self) -> List[int]:
        in_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        topo_order = []
        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for neighbor, _ in self.adj_out.get(curr, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(topo_order) != len(self.nodes):
            raise ValueError('Thought graph is cyclic; GoTGNN requires a DAG')
        return topo_order

    def compute_causal_ancestors_and_distances(self, d_cutoff: int=4) -> Tuple[Dict[int, Set[int]], np.ndarray]:
        n = len(self.nodes)
        node_ids = sorted(list(self.nodes.keys()))
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        dist_matrix = np.full((n, n), np.inf, dtype=np.float32)
        np.fill_diagonal(dist_matrix, 0.0)
        for nid in node_ids:
            u_idx = id_to_idx[nid]
            queue = [(nid, 0)]
            visited = {nid}
            while queue:
                curr, d = queue.pop(0)
                c_idx = id_to_idx[curr]
                dist_matrix[u_idx, c_idx] = d
                if d < d_cutoff:
                    for neighbor, _ in self.adj_out.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))
        ancestors_map = {nid: set() for nid in node_ids}
        for u_id in node_ids:
            u_idx = id_to_idx[u_id]
            for v_id in node_ids:
                v_idx = id_to_idx[v_id]
                if u_id != v_id and dist_matrix[u_idx, v_idx] < np.inf:
                    ancestors_map[v_id].add(u_id)
        return (ancestors_map, dist_matrix)

    def to_dict(self) -> Dict[str, Any]:
        return {'graph_id': self.graph_id, 'task_name': self.task_name, 'task_id': self.task_id, 'config_id': self.config_id, 'root_nodes': list(self.root_nodes), 'leaf_nodes': list(self.leaf_nodes), 'num_nodes': len(self.nodes), 'num_edges': len(self.edges), 'nodes': [{'node_id': n.node_id, 'task_id': n.task_id, 'type': n.node_type.name, 'content': n.content, 'abstraction': n.abstraction.name, 'quality_dims': [q.name for q in n.quality_dims], 'ground_truth_score': float(n.ground_truth_score), 'predicted_score': float(n.predicted_score), 'pruned': bool(n.pruned)} for n in self.nodes.values()], 'edges': [{'src': e.src, 'dst': e.dst, 'type': e.edge_type.name, 'weight': float(e.weight)} for e in self.edges]}