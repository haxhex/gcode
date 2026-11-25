from typing import Dict, List, Optional
import numpy as np
try:
    from thought_graph.graph_types import ThoughtGraph
except ImportError:
    from .graph_types import ThoughtGraph


class CausalValuePropagator:

    @staticmethod
    def _causal_children(graph: ThoughtGraph) -> Dict[int, List[int]]:
        children = {node_id: [] for node_id in graph.nodes}
        for edge in graph.edges:
            if edge.src in children and edge.dst in graph.nodes:
                children[edge.src].append(edge.dst)
        return children

    def propagate_observed_outcomes(self, graph: ThoughtGraph, terminal_outcomes: Dict[int, float]) -> Dict[int, float]:
        children = self._causal_children(graph)
        terminal = {}
        for node_id, rate in terminal_outcomes.items():
            if node_id not in graph.nodes or not 0.0 <= float(rate) <= 1.0:
                raise ValueError(f'Invalid observed outcome for terminal node {node_id}')
            terminal[int(node_id)] = float(rate)
        memo: Dict[int, Optional[float]] = {}
        visiting = set()

        def value(node_id: int) -> Optional[float]:
            if node_id in memo:
                return memo[node_id]
            if node_id in visiting:
                raise ValueError('Thought graph must be acyclic for causal value propagation')
            if node_id in terminal:
                memo[node_id] = terminal[node_id]
                return memo[node_id]
            visiting.add(node_id)
            values = [v for child in children[node_id] if (v := value(child)) is not None]
            visiting.remove(node_id)
            memo[node_id] = float(np.mean(values)) if values else None
            return memo[node_id]

        for node_id in graph.nodes:
            value(node_id)
        observed = {nid: score for nid, score in memo.items() if score is not None}
        for node_id, score in observed.items():
            graph.nodes[node_id].ground_truth_score = score
        return observed
