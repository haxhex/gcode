from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import json
try:
    from thought_graph.graph_types import ThoughtGraph, ThoughtNode, ThoughtEdgeType, ThoughtNodeType, AbstractionLevel, QualityDimension
    from thought_graph.generator import TextEmbeddingEngine
except ImportError:
    from ..thought_graph.graph_types import ThoughtGraph, ThoughtNode, ThoughtEdgeType, ThoughtNodeType, AbstractionLevel, QualityDimension
    from ..thought_graph.generator import TextEmbeddingEngine
ACTION_TYPES = ('IDEA', 'CRITIQUE', 'REFINE', 'AGGREGATE_SIMILAR', 'AGGREGATE_TASK')
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_TYPES)}
VALID_SPLITS = {'train', 'validation', 'test'}

@dataclass
class PreparedGraphRecord:
    task_name: str
    split: str
    graph: ThoughtGraph
    candidate_paths: List[List[int]]
    path_targets: torch.Tensor
    node_targets: torch.Tensor
    node_mask: torch.Tensor
    action_type_target: torch.Tensor
    child_count_target: torch.Tensor

def causal_children(graph: ThoughtGraph) -> Dict[int, List[int]]:
    children = {node_id: [] for node_id in graph.nodes}
    for edge in graph.edges:
        parent, child = (edge.src, edge.dst)
        if parent in children and child in graph.nodes and (child not in children[parent]):
            children[parent].append(child)
    return children

def observed_path_rate(item: Dict[str, Any]) -> float:
    if 'pass_rate' in item:
        rate = float(item['pass_rate'])
    else:
        total = int(item.get('total_tests', 0))
        passed = int(item.get('passed_tests', -1))
        if total <= 0 or passed < 0 or passed > total:
            raise ValueError('Each candidate path needs valid passed_tests/total_tests or pass_rate')
        rate = passed / total
    if not 0.0 <= rate <= 1.0:
        raise ValueError('Candidate pass rates must be in [0, 1]')
    return rate

def propagate_observed_node_utility(graph: ThoughtGraph, candidate_paths: Sequence[Dict[str, Any]]) -> Dict[int, float]:
    children = causal_children(graph)
    terminal_values: Dict[int, List[float]] = {}
    for candidate in candidate_paths:
        ids = [int(x) for x in candidate.get('terminal_node_ids', [])]
        if not ids:
            ids = [int(x) for x in candidate.get('node_ids', [])[-1:]]
        if not ids:
            raise ValueError('Candidate path records must include node_ids')
        rate = observed_path_rate(candidate)
        for node_id in ids:
            if node_id not in graph.nodes:
                raise ValueError(f'Candidate refers to unknown terminal node {node_id}')
            terminal_values.setdefault(node_id, []).append(rate)
    direct = {node_id: float(np.mean(values)) for node_id, values in terminal_values.items()}
    memo: Dict[int, Optional[float]] = {}
    visiting = set()

    def value(node_id: int) -> Optional[float]:
        if node_id in memo:
            return memo[node_id]
        if node_id in visiting:
            raise ValueError('Causal thought relations contain a cycle')
        if node_id in direct:
            memo[node_id] = direct[node_id]
            return memo[node_id]
        visiting.add(node_id)
        vals = [v for child in children.get(node_id, []) if (v := value(child)) is not None]
        visiting.remove(node_id)
        memo[node_id] = float(np.mean(vals)) if vals else None
        return memo[node_id]
    for node_id in graph.nodes:
        value(node_id)
    return {node_id: score for node_id, score in memo.items() if score is not None}

def graph_from_dict(data: Dict[str, Any], embedder: Optional[TextEmbeddingEngine]=None) -> ThoughtGraph:
    embedder = embedder or TextEmbeddingEngine(dim=384)
    graph = ThoughtGraph(graph_id=str(data['graph_id']), task_name=str(data['task_name']), config_id=str(data.get('config_id', 'C1')), task_id=str(data.get('task_id', data['task_name'])))
    for item in data.get('nodes', []):
        content = str(item['content'])
        node = ThoughtNode(node_id=int(item['node_id']), task_id=str(item.get('task_id', data.get('task_id', data['task_name']))), node_type=ThoughtNodeType[str(item['type']).upper()], content=content, abstraction=AbstractionLevel[str(item.get('abstraction', 'FUNCTION')).upper()], quality_dims=[QualityDimension[str(q).upper()] for q in item.get('quality_dims', [])], semantic_feat=embedder.embed(content), ground_truth_score=float(item.get('ground_truth_score', 0.0)), predicted_score=float(item.get('predicted_score', 0.0)), pruned=bool(item.get('pruned', False)))
        node.build_feature_vectors()
        graph.add_node(node)
    for item in data.get('edges', []):
        edge_type = ThoughtEdgeType[str(item['type']).upper()]
        graph.add_edge(int(item['src']), int(item['dst']), edge_type, float(item.get('weight', 1.0)))
    graph.root_nodes = [int(x) for x in data.get('root_nodes', [])]
    graph.leaf_nodes = [int(x) for x in data.get('leaf_nodes', [])]
    if not graph.root_nodes:
        graph.root_nodes = [n for n in graph.nodes if not graph.adj_in[n]]
    if not graph.leaf_nodes:
        graph.leaf_nodes = [n for n in graph.nodes if not graph.adj_out[n]]
    return graph

def prepare_record(data: Dict[str, Any], embedder: Optional[TextEmbeddingEngine]=None) -> PreparedGraphRecord:
    required = {'task_name', 'split', 'graph', 'candidate_paths', 'next_action'}
    missing = required - data.keys()
    if missing:
        raise ValueError(f'Training record is missing required keys: {sorted(missing)}')
    split = str(data['split'])
    if split not in VALID_SPLITS:
        raise ValueError(f'Invalid split {split!r}; expected train, validation, or test')
    graph = graph_from_dict(data['graph'], embedder)
    path_items = list(data['candidate_paths'])
    if not path_items:
        raise ValueError('At least one evaluated candidate path is required')
    node_ids = sorted(graph.nodes)
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    paths, path_rates = ([], [])
    for item in path_items:
        path = [int(x) for x in item['node_ids']]
        if not path or any((node_id not in index for node_id in path)):
            raise ValueError('Candidate path contains no nodes or an unknown node ID')
        paths.append([index[node_id] for node_id in path])
        path_rates.append(observed_path_rate(item))
    label_graph = graph_from_dict(data.get('label_graph', data['graph']), embedder)
    utility = propagate_observed_node_utility(label_graph, path_items)
    missing_nodes = [node_id for node_id in node_ids if node_id not in utility]
    if missing_nodes:
        raise ValueError(f'Thought nodes {missing_nodes} lack descendant execution outcomes; evaluate all terminal thoughts so every node in V_T receives a descendant-propagated target')
    node_targets = torch.tensor([utility[node_id] for node_id in node_ids], dtype=torch.float32)
    node_mask = torch.ones(len(node_ids), dtype=torch.bool)
    action = data['next_action']
    action_name = str(action.get('type', '')).upper()
    if action_name not in ACTION_TO_ID:
        raise ValueError(f'Unknown next action {action_name!r}; expected one of {ACTION_TYPES}')
    child_ids = [int(x) for x in action.get('child_node_ids', [])]
    child_count = int(action.get('child_count', len(child_ids)))
    if child_ids and child_count != len(child_ids):
        raise ValueError('child_count must equal the number of actually recorded child_node_ids')
    if child_count < 1:
        raise ValueError('The next action must record at least one generated thought node')
    return PreparedGraphRecord(task_name=str(data['task_name']), split=split, graph=graph, candidate_paths=paths, path_targets=torch.tensor(path_rates, dtype=torch.float32), node_targets=node_targets, node_mask=node_mask, action_type_target=torch.tensor([ACTION_TO_ID[action_name]], dtype=torch.long), child_count_target=torch.tensor([float(child_count)], dtype=torch.float32))

def write_history_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    import os
    task_splits: Dict[str, str] = {}
    ready = []
    for record in records:
        if not record.get('split') or not record.get('label_graph') or (not record.get('candidate_paths')):
            raise ValueError('History records need a task split, post-action label_graph, and measured candidate outcomes')
        task = str(record['task_name'])
        split = str(record['split'])
        if split not in VALID_SPLITS:
            raise ValueError(f'Invalid split {split!r}')
        if task in task_splits and task_splits[task] != split:
            raise ValueError(f'Task {task!r} appears in multiple splits')
        task_splits[task] = split
        prepare_record(record)
        ready.append(record)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        for record in ready:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
