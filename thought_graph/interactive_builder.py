from __future__ import annotations
import json
import numpy as np
from typing import Any, Dict, List, Tuple
try:
    from thought_graph.graph_types import ThoughtGraph, ThoughtNode, ThoughtNodeType, ThoughtEdgeType, AbstractionLevel, QualityDimension
    from thought_graph.generator import TextEmbeddingEngine
    from gotgnn.inference import GoTGNNInference
    from prompts.master_prompts import PROMPT_ROOT_DECOMPOSITION, PROMPT_GNN_STEERING_INJECTION
except ImportError:
    from .graph_types import ThoughtGraph, ThoughtNode, ThoughtNodeType, ThoughtEdgeType, AbstractionLevel, QualityDimension
    from .generator import TextEmbeddingEngine
    from ..gotgnn.inference import GoTGNNInference
    from ..prompts.master_prompts import PROMPT_ROOT_DECOMPOSITION, PROMPT_GNN_STEERING_INJECTION
ACTION_TO_NODE_TYPE = {'IDEA': ThoughtNodeType.IDEA, 'CRITIQUE': ThoughtNodeType.CRITIQUE, 'REFINE': ThoughtNodeType.REFINEMENT, 'AGGREGATE_SIMILAR': ThoughtNodeType.AGGREGATION, 'AGGREGATE_TASK': ThoughtNodeType.AGGREGATION}

class InteractiveThoughtGraphExpander:

    def __init__(self, got_inference: GoTGNNInference, tau_prune: float=0.4):
        self.got_infer = got_inference
        self.tau_prune = tau_prune
        self.node_counter = 0
        self.embedder = TextEmbeddingEngine(dim=384)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denom) if denom > 0 else 0.0

    def _similar_thought_groups(self, graph: ThoughtGraph) -> List[Dict[str, Any]]:
        threshold = self.got_infer.cfg.thought_similarity_threshold
        frontier = [nid for nid in graph.nodes if not graph.adj_out.get(nid)]
        neighbors: Dict[int, List[Tuple[int, float]]] = {nid: [] for nid in frontier}
        for i, left_id in enumerate(frontier):
            for right_id in frontier[i + 1:]:
                left, right = (graph.nodes[left_id], graph.nodes[right_id])
                if left.task_id != right.task_id or left.abstraction != right.abstraction:
                    continue
                similarity = self._cosine_similarity(left.semantic_feat, right.semantic_feat)
                if similarity >= threshold:
                    neighbors[left_id].append((right_id, similarity))
                    neighbors[right_id].append((left_id, similarity))
        groups, seen = ([], set())
        for start in frontier:
            if start in seen or not neighbors[start]:
                continue
            stack, component = ([start], [])
            seen.add(start)
            while stack:
                current = stack.pop()
                component.append(current)
                for nxt, _ in neighbors[current]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            scores = [score for node_id in component for _, score in neighbors[node_id] if _ in component]
            groups.append({'node_ids': sorted(component), 'mean_cosine': float(np.mean(scores)) if scores else 0.0})
        return sorted(groups, key=lambda item: item['mean_cosine'], reverse=True)

    def initialize_graph(self, task_name: str, task_id: str='T01', n_init: int=3) -> Tuple[ThoughtGraph, str]:
        if n_init < 1:
            raise ValueError('Each subtask must start with at least one idea')
        graph = ThoughtGraph(graph_id=f'interactive_{task_id}', task_name=task_id, task_id=task_id)
        self.node_counter = 0
        prompt = PROMPT_ROOT_DECOMPOSITION.format(requirement_text=task_name, benchmark_category='rSDE-Bench Multi-File Project', complexity_level='Advanced', n_init=n_init)
        return (graph, prompt)

    @staticmethod
    def _parse_output(raw_output: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_output, str):
            try:
                raw_output = json.loads(raw_output)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Thought output must be valid JSON: {exc}') from exc
        if isinstance(raw_output, dict):
            raw_output = raw_output.get('thoughts', raw_output.get('ideas'))
        if not isinstance(raw_output, list) or not all((isinstance(x, dict) for x in raw_output)):
            raise ValueError('Thought output must be a JSON array of thought objects')
        return raw_output

    def apply_generated_thoughts(self, graph: ThoughtGraph, raw_output: Any, action_type: str='IDEA', target_node_id: int | None=None, selected_path: List[int] | None=None, expected_count: int | None=None, allowed_source_ids: List[int] | None=None) -> List[int]:
        action = str(action_type).upper()
        if action not in ACTION_TO_NODE_TYPE:
            raise ValueError(f'Unsupported thought action: {action_type}')
        thoughts = self._parse_output(raw_output)
        if expected_count is not None and len(thoughts) != int(expected_count):
            raise ValueError(f'Expected {expected_count} generated thoughts, received {len(thoughts)}')
        if action in {'AGGREGATE_SIMILAR', 'AGGREGATE_TASK'} and len(thoughts) != 1:
            raise ValueError('An aggregation action creates exactly one aggregation node')
        if not graph.nodes and action != 'IDEA':
            raise ValueError('Initial thought generation must create IDEA nodes first')
        selected_path = selected_path or []
        added: List[int] = []
        for item in thoughts:
            content = item.get('content') or item.get('summary')
            if not content and any((item.get(key) for key in ('algorithmic_technique', 'state_strategy', 'technical_risk'))):
                content = ' '.join((str(item[key]) for key in ('summary', 'algorithmic_technique', 'state_strategy', 'technical_risk') if item.get(key)))
            if not isinstance(content, str) or not content.strip():
                raise ValueError('Every generated thought needs non-empty plain-language content')
            if any((token in content for token in ('```', 'def ', 'class ', 'import '))):
                raise ValueError('Thought nodes must not contain executable code')
            try:
                abstraction = AbstractionLevel[str(item.get('abstraction', 'FUNCTION')).upper()]
                quality_dims = [QualityDimension[str(q).upper()] for q in item.get('quality_dims', [])]
            except KeyError as exc:
                raise ValueError(f'Unknown abstraction level or quality dimension: {exc}') from exc
            aggregation_sources = None
            if action in {'AGGREGATE_SIMILAR', 'AGGREGATE_TASK'}:
                aggregation_sources = [int(x) for x in item.get('source_node_ids', selected_path)]
                if len(aggregation_sources) < 2 or any((source not in graph.nodes for source in aggregation_sources)):
                    raise ValueError('Aggregation requires at least two existing source nodes')
                if len(set(aggregation_sources)) != len(aggregation_sources):
                    raise ValueError('Aggregation source IDs must be unique')
                if allowed_source_ids is not None and (not set(aggregation_sources).issubset(set(allowed_source_ids))):
                    raise ValueError('Aggregation may only use the cosine-screened source nodes')
                if action == 'AGGREGATE_TASK' and any((graph.nodes[source].node_type != ThoughtNodeType.AGGREGATION for source in aggregation_sources)):
                    raise ValueError('Task aggregation can only connect subtask aggregation nodes')
            self.node_counter = max(self.node_counter, max(graph.nodes, default=0)) + 1
            node_id = self.node_counter
            task_id = str(getattr(graph, 'task_id', graph.task_name))
            requested_type = item.get('node_type')
            expected_type = ACTION_TO_NODE_TYPE[action].name
            if requested_type and str(requested_type).upper() != expected_type:
                raise ValueError(f'Action {action} requires node_type {expected_type}, received {requested_type}')
            node = ThoughtNode(node_id=node_id, task_id=task_id, node_type=ACTION_TO_NODE_TYPE[action], content=content.strip(), abstraction=abstraction, semantic_feat=self.embedder.embed(content.strip()), quality_dims=quality_dims)
            node.build_feature_vectors()
            graph.add_node(node)
            if action == 'IDEA':
                graph.root_nodes.append(node_id)
            elif action == 'CRITIQUE':
                parent = target_node_id or (selected_path[-1] if selected_path else None)
                if parent not in graph.nodes:
                    raise ValueError('Critique action requires a valid target thought')
                graph.add_edge(parent, node_id, ThoughtEdgeType.CRITIQUE)
            elif action == 'REFINE':
                parent = target_node_id or (selected_path[-1] if selected_path else None)
                if parent not in graph.nodes:
                    raise ValueError('Refinement action requires a valid source thought')
                graph.add_edge(parent, node_id, ThoughtEdgeType.FEEDBACK)
                critique_ids = [n for n in selected_path if graph.nodes[n].node_type == ThoughtNodeType.CRITIQUE]
                for critique_id in critique_ids:
                    graph.add_edge(critique_id, node_id, ThoughtEdgeType.INFORMS)
            else:
                for source in aggregation_sources:
                    source_node = graph.nodes[source]
                    if action == 'AGGREGATE_TASK':
                        graph.add_edge(source, node_id, ThoughtEdgeType.DEPENDS)
                    else:
                        graph.add_edge(source, node_id, ThoughtEdgeType.AGGREGATE)
            added.append(node_id)
        graph.leaf_nodes = [node_id for node_id in graph.nodes if not graph.adj_out[node_id]]
        return added

    def apply_plan_output(self, graph: ThoughtGraph, plan: Dict[str, Any], raw_output: Any) -> Dict[str, Any]:
        added = self.apply_generated_thoughts(graph=graph, raw_output=raw_output, action_type=plan['step_action'], target_node_id=plan.get('target_node_id'), selected_path=plan.get('best_path', []), expected_count=plan.get('action_count'), allowed_source_ids=plan.get('aggregation_source_ids') if plan.get('step_action') == 'AGGREGATE_SIMILAR' else None)
        record = plan['history_record']
        record['task_name'] = graph.task_name
        record['next_action']['child_node_ids'] = added
        record['next_action']['child_count'] = len(added)
        record['status'] = 'awaiting_candidate_test_outcomes'
        return record

    @staticmethod
    def attach_test_outcomes(history_record: Dict[str, Any], outcomes: List[Dict[str, Any]], split: str | None=None, label_graph: Dict[str, Any] | None=None) -> Dict[str, Any]:
        if split is not None and split not in {'train', 'validation', 'test'}:
            raise ValueError('split must be train, validation, or test')
        by_path = {tuple((int(x) for x in item['node_ids'])): item for item in history_record['candidate_paths']}
        extra_entries: List[Dict[str, Any]] = []
        for outcome in outcomes:
            ids = tuple((int(x) for x in outcome['node_ids']))
            if ids not in by_path:
                raise ValueError(f'Test outcome refers to an unrecorded candidate path: {ids}')
            total, passed = (int(outcome['total_tests']), int(outcome['passed_tests']))
            if total <= 0 or not 0 <= passed <= total:
                raise ValueError('Test outcomes need valid passed_tests and total_tests')
            entry_payload = {'passed_tests': passed, 'total_tests': total, 'terminal_node_ids': [int(x) for x in outcome.get('terminal_node_ids', ids[-1:])], 'test_run_id': outcome.get('test_run_id'), 'candidate_id': outcome.get('candidate_id')}
            if 'passed_tests' not in by_path[ids]:
                by_path[ids].update(entry_payload)
            else:
                new_entry = {'node_ids': list(ids)}
                new_entry.update(entry_payload)
                extra_entries.append(new_entry)
        history_record['candidate_paths'] = [item for item in history_record['candidate_paths'] if 'passed_tests' in item] + extra_entries
        if split is not None:
            history_record['split'] = split
        if label_graph is not None:
            history_record['label_graph'] = label_graph
        history_record['status'] = 'labeled_from_execution_history' if split else 'outcomes_attached_split_pending'
        return history_record

    def step(self, graph: ThoughtGraph, environment_feedback: str='') -> Dict[str, Any]:
        if not graph.nodes:
            raise ValueError('Generate the initial ideas before asking GoTGNN to score a graph')
        plan = self.got_infer.infer(graph)
        selected = list(plan.get('best_path', []))
        action = str(plan['action_type']).upper()
        selected = list(plan.get('best_path', []))
        similar_groups = self._similar_thought_groups(graph)
        aggregation_sources = similar_groups[0]['node_ids'] if similar_groups else []
        valid_actions = ['IDEA', 'CRITIQUE', 'REFINE']
        if len(aggregation_sources) >= 2:
            valid_actions.append('AGGREGATE_SIMILAR')
        task_aggregations = [n for n in graph.nodes.values() if n.node_type == ThoughtNodeType.AGGREGATION]
        if len({n.task_id for n in task_aggregations}) >= 2:
            valid_actions.append('AGGREGATE_TASK')
        if action not in valid_actions:
            probs = plan.get('action_probabilities', {})
            action = max(valid_actions, key=lambda name: probs.get(name, 0.0))
        count = max(1, min(self.got_infer.cfg.max_action_children, int(plan['action_count'])))
        if action in {'AGGREGATE_SIMILAR', 'AGGREGATE_TASK'}:
            count = 1
        target = selected[-1] if selected else max(graph.nodes)
        state_before = graph.to_dict()
        candidates = self.got_infer.extract_candidate_paths(graph, max_paths=max(self.got_infer.cfg.max_search_width, len(graph.leaf_nodes) * 2, len(graph.nodes)), max_len=max(self.got_infer.cfg.max_path_length, len(graph.nodes)))
        selected_subgraph = list(plan.get('best_subgraph_nodes', selected))
        selected_thoughts = [{'node_id': nid, 'type': graph.nodes[nid].node_type.name, 'abstraction': graph.nodes[nid].abstraction.name, 'content': graph.nodes[nid].content, 'utility': graph.nodes[nid].predicted_score} for nid in selected_subgraph if nid in graph.nodes]
        steering_prompt = PROMPT_GNN_STEERING_INJECTION.format(num_nodes=len(graph.nodes), num_edges=len(graph.edges), path_node_ids=selected, path_prob=float(plan.get('best_path_probability', 0.0)), predicted_action_type=action, next_action_count=count, tau_prune=self.tau_prune, tau_merge=self.got_infer.cfg.thought_similarity_threshold, pruned_node_ids=plan.get('pruned_nodes', []), selected_path=selected, selected_thoughts=selected_thoughts, aggregation_candidates=[{'node_id': nid, 'content': graph.nodes[nid].content, 'group_mean_cosine': similar_groups[0]['mean_cosine']} for nid in aggregation_sources], aggregation_source_ids=aggregation_sources, environment_feedback=environment_feedback or 'No new test feedback.')
        history_record = {'graph': state_before, 'candidate_paths': [{'node_ids': path} for path in candidates], 'next_action': {'type': action, 'child_node_ids': []}, 'state_id': f"{graph.graph_id}:decision:{len(getattr(graph, 'action_history', []))}", 'selected_path': selected, 'target_node_id': target, 'aggregation_source_ids': aggregation_sources, 'requested_child_count': count, 'status': 'awaiting_generated_thoughts_and_test_outcomes'}
        return {'step_action': action, 'action_count': count, 'best_path': selected, 'best_subgraph_nodes': selected_subgraph, 'target_node_id': target, 'pruned_nodes': plan.get('pruned_nodes', []), 'path_success_probability': float(plan.get('best_path_probability', 0.0)), 'steering_prompt': steering_prompt, 'history_record': history_record, 'current_total_nodes': len(graph.nodes), 'current_total_edges': len(graph.edges)}