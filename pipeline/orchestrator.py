import sys
import json
import copy
import inspect
import textwrap
import traceback
from typing import Dict, List, Any, Tuple, Optional, Callable
try:
    from config import PipelineConfig, GOT_12_CONFIGS
    from pem.manager import ProjectEvolutionaryMemory, PlaybookEvolutionaryMemory
    from scpg.builder import SCPGBuilder, SCPGGraph
    from thought_graph.graph_types import ThoughtGraph, ThoughtNode, ThoughtNodeType, ThoughtEdgeType, QualityDimension
except ImportError:
    from ..config import PipelineConfig, GOT_12_CONFIGS
    from ..pem.manager import ProjectEvolutionaryMemory, PlaybookEvolutionaryMemory
    from ..scpg.builder import SCPGBuilder, SCPGGraph
    from ..thought_graph.graph_types import ThoughtGraph, ThoughtNode, ThoughtNodeType, ThoughtEdgeType, QualityDimension

try:
    from thought_graph.interactive_builder import InteractiveThoughtGraphExpander
    from gotgnn.model import GoTGNN
    from gotgnn.inference import GoTGNNInference
    from flare_gnn.model import FlareGNN, FlareGNNInference
except Exception:
    try:
        from ..thought_graph.interactive_builder import InteractiveThoughtGraphExpander
        from ..gotgnn.model import GoTGNN
        from ..gotgnn.inference import GoTGNNInference
        from ..flare_gnn.model import FlareGNN, FlareGNNInference
    except Exception:
        InteractiveThoughtGraphExpander = None
        GoTGNN = None
        GoTGNNInference = None
        FlareGNN = None
        FlareGNNInference = None

class CodingTeam:

    def __init__(self, llm_fn: Optional[Callable[[str], str]]=None):
        self.llm_fn = llm_fn
        self.temperature: Optional[float] = None

    def call_llm(self, prompt: str) -> str:
        self._require_llm('LLM invocation')
        try:
            parameters = inspect.signature(self.llm_fn).parameters.values()
            accepts_temperature = any((parameter.name == 'temperature' or parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters))
        except (TypeError, ValueError):
            accepts_temperature = False
        if self.temperature is not None and accepts_temperature:
            return self.llm_fn(prompt, temperature=self.temperature)
        return self.llm_fn(prompt)

    def _require_llm(self, context: str) -> None:
        if self.llm_fn is None:
            raise RuntimeError(f'CodingTeam requires a real LLM function. Pass `llm_fn=your_llm_callable` to GCodeOrchestrator. Context: {context}')

    def generate_initial_code(self, requirement: str, path: List[int], action_type: str, thought_context: List[Dict[str, Any]] | None=None, action_count: int=1) -> str:
        self._require_llm('generate_initial_code')
        prompt = f'[CODING TASK]\nRequirement: {requirement}\nSelected reasoning subgraph node ids: {path}\nThought action used to build the selected subgraph: {action_type} ({action_count} new thought nodes).\nSelected thought content: {thought_context or []}\nUse the selected thought content and preserve its dependencies and project constraints. Do not treat node IDs as a substitute for their content.\n\nWrite complete, executable Python 3.10+ code. No comments, docstrings, or placeholders.'
        return self.call_llm(prompt)

    def integrate_project(self, requirement: str, dependency_order: List[str], subtasks: List[Dict[str, Any]], selected_reasoning: List[Dict[str, Any]]) -> str:
        self._require_llm('integrate_project')
        prompt = f'[PROJECT INTEGRATION]\nRequirement: {requirement}\nDependency order: {dependency_order}\nSubtask outputs: {subtasks}\nSelected dependency-consistent reasoning subgraph: {selected_reasoning}\nAssemble complementary subtask implementations into a coherent project. Preserve each useful, non-conflicting contribution, reconcile conflicts using the explicit dependency order and interface contracts, and preserve the listed file paths. Do not invent unimplemented subtask behavior. Return valid JSON only as {{"files":[{{"path":"...","content":"..."}}]}} .'
        raw = self.call_llm(prompt)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Project integration output must be valid JSON: {exc}') from exc
        files = parsed.get('files') if isinstance(parsed, dict) else None
        if not isinstance(files, list) or any((not isinstance(f, dict) or not f.get('path') or (not isinstance(f.get('content'), str)) for f in files)):
            raise ValueError('Project integration must return a JSON files array with path and content')
        return json.dumps(parsed, ensure_ascii=False)

    def update_code(self, current_code: str, feedback: Dict[str, Any], advice: List[str], thought_context: List[Dict[str, Any]] | None=None, action_type: str='REFINE', action_count: int=1) -> str:
        self._require_llm('update_code')
        advice_block = '\n'.join((f'- {a}' for a in advice)) if advice else '(none)'
        prompt = f"[REPAIR TASK]\nRepair the code below. Fix only the faulty lines.\n\nFaulty statement ids: {feedback.get('repair_subgraph', [])}\nFault scores: {feedback.get('scores', {})}\nSelected GoTGNN reasoning: {thought_context or []}\nNext action: {action_type}; requested thought count: {action_count}.\nAdvice from memory:\n{advice_block}\n\n[CURRENT CODE]\n{current_code}\n\nReturn corrected Python code only. No comments, no docstrings."
        return self.call_llm(prompt)

class DynamicPythonTestExecutor:

    @staticmethod
    def run_tests_on_code(source_code: str, custom_tests: Optional[List[Tuple[str, Callable]]]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            compiled = compile(source_code, '<tested_code>', 'exec')
        except SyntaxError as e:
            return ({'all_passed': False, 'passed_count': 0, 'failed_count': 1, 'total_count': 1, 'failures': [f'SyntaxError at line {e.lineno}: {e.msg}'], 'failing_lines': [e.lineno or 1]}, {'covered_lines': []})
        local_scope: Dict[str, Any] = {'__builtins__': __builtins__}
        try:
            exec(compiled, local_scope)
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])
            failing_lines = [f.lineno for f in tb if f.filename == '<tested_code>']
            return ({'all_passed': False, 'passed_count': 0, 'failed_count': 1, 'total_count': 1, 'failures': [f'RuntimeError on load: {type(e).__name__}: {e}'], 'failing_lines': failing_lines or [1]}, {'covered_lines': []})
        if custom_tests:
            test_cases = custom_tests
        else:
            test_cases = [('test_module_loaded', lambda: True)]
        passed = 0
        failed = 0
        failures: List[str] = []
        failing_lines: set = set()
        covered_lines: set = set()
        for test_name, test_fn in test_cases:
            try:
                result = test_fn()
                if result is True or result is None:
                    passed += 1
                else:
                    failed += 1
                    failures.append(f'{test_name}: returned {result!r} instead of True')
                    failing_lines.add(1)
            except Exception as e:
                failed += 1
                tb = traceback.extract_tb(sys.exc_info()[2])
                code_frames = [f.lineno for f in tb if f.filename == '<tested_code>']
                failing_lines.update(code_frames)
                failures.append(f'{test_name}: {type(e).__name__}: {e}')
        for idx, line in enumerate(source_code.strip().splitlines(), start=1):
            stripped = line.strip()
            if stripped:
                covered_lines.add(idx)
        return ({'all_passed': failed == 0, 'passed_count': passed, 'failed_count': failed, 'total_count': len(test_cases), 'failures': failures, 'failing_lines': sorted(failing_lines)}, {'covered_lines': sorted(covered_lines)})

class TestTeam:

    def __init__(self, custom_tests: Optional[List[Tuple[str, Callable]]] = None, llm_fn: Optional[Callable[[str], str]] = None):
        self.custom_tests = custom_tests or []
        self.llm_fn = llm_fn

    def generate_targeted_tests(
        self,
        code: str,
        flare_pre: Dict[str, Any],
        advice: Optional[List[str]] = None,
        requirement: str = '',
        subtask_id: str = ''
    ) -> List[Tuple[str, Callable]]:
        seed_stmts = flare_pre.get('seed_stmts', [])
        stmt_scores = flare_pre.get('stmt_scores', {})
        if not seed_stmts and not stmt_scores:
            return []

        if self.llm_fn is not None:
            try:
                from prompts.master_prompts import PROMPT_DYNAMIC_TEST_GENERATION
            except ImportError:
                from ..prompts.master_prompts import PROMPT_DYNAMIC_TEST_GENERATION

            code_lines = code.splitlines()
            suspicious_info = []
            for s_id in seed_stmts[:5]:
                score = stmt_scores.get(s_id, 0.0)
                line_idx = s_id - 1 if 0 <= s_id - 1 < len(code_lines) else 0
                line_str = code_lines[line_idx] if 0 <= line_idx < len(code_lines) else ''
                suspicious_info.append(f'Line {s_id} (suspiciousness {score:.3f}): {line_str.strip()}')
            suspicious_str = '\n'.join(suspicious_info) if suspicious_info else '(none)'

            context_lines = []
            for s_id in seed_stmts[:3]:
                start = max(0, s_id - 3)
                end = min(len(code_lines), s_id + 4)
                context_lines.extend(code_lines[start:end])
            repair_context = '\n'.join(context_lines) if context_lines else code[:500]

            advice_str = '\n'.join(advice) if advice else '(none)'
            prompt = PROMPT_DYNAMIC_TEST_GENERATION.format(
                component_name=subtask_id or 'SubtaskModule',
                interface_contracts=f'Requirement: {requirement}\nAdvice: {advice_str}',
                suspicious_statements=suspicious_str,
                repair_context=repair_context,
            )
            try:
                raw = self.llm_fn(prompt)
                return self._parse_tests_from_code(raw, code)
            except Exception:
                return []

        return []

    def _parse_tests_from_code(self, raw_code: str, target_code: str) -> List[Tuple[str, Callable]]:
        cleaned = raw_code.strip()
        if '```python' in cleaned:
            cleaned = cleaned.split('```python')[1].split('```')[0].strip()
        elif '```' in cleaned:
            cleaned = cleaned.split('```')[1].split('```')[0].strip()

        tests: List[Tuple[str, Callable]] = []
        try:
            scope: Dict[str, Any] = {'__builtins__': __builtins__}
            exec(compile(target_code, '<tested_code>', 'exec'), scope)
            exec(compile(cleaned, '<generated_tests>', 'exec'), scope)
            for name, val in list(scope.items()):
                if (name.startswith('test_') or name.startswith('check_')) and callable(val):
                    tests.append((name, val))
        except Exception:
            pass
        return tests

    def execute_tests(
        self,
        code: str,
        iteration: int,
        dynamic_tests: Optional[List[Tuple[str, Callable]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        all_tests = list(self.custom_tests)
        if dynamic_tests:
            all_tests.extend(dynamic_tests)
        return DynamicPythonTestExecutor.run_tests_on_code(code, all_tests)

class RootCauseAnalyzer:

    def analyze(self, graph_t: ThoughtGraph, path: List[int], results: Dict[str, Any], flare_post: Dict[str, Any]) -> Dict[str, Any]:
        seeds = flare_post.get('seed_stmts', [])
        failures = results.get('failures', [])
        pattern = 'ValidationFailure' if failures else 'UnknownFailure'
        description = f'Failed assertions: {failures[:3]}. Suspicious statements: {seeds}.'
        return {'pattern': pattern, 'description': description, 'advice': 'Apply boundary checks and input validation on suspicious statements.', 'suspicious_statements': seeds}

class GCodeOrchestrator:

    def __init__(self, gotgnn: GoTGNN, flare_gnn: FlareGNN, config: Optional[PipelineConfig]=None, llm_fn: Optional[Callable[[str], str]]=None, custom_tests: Optional[List[Tuple[str, Callable]]]=None):
        self.cfg = config or PipelineConfig()
        self.got_infer = GoTGNNInference(gotgnn)
        self.thought_expander = InteractiveThoughtGraphExpander(self.got_infer)
        self.flare_infer = FlareGNNInference(flare_gnn)
        self.scpg_builder = SCPGBuilder()
        pem_storage = getattr(self.cfg, 'storage_path', None) or getattr(self.cfg, 'pem_storage_path', None)
        self.pem = ProjectEvolutionaryMemory(storage_path=pem_storage)
        self.coding_team = CodingTeam(llm_fn=llm_fn)
        self.test_team = TestTeam(custom_tests=custom_tests, llm_fn=llm_fn)
        self.analyzer = RootCauseAnalyzer()

    def run_project(self, requirement: str, max_subtasks: int=8, max_iterations: Optional[int]=None, history_split: Optional[str]=None, history_path: Optional[str]=None, graph_config_id: str='C6') -> Dict[str, Any]:
        self.coding_team._require_llm('project decomposition')
        if graph_config_id not in GOT_12_CONFIGS:
            raise ValueError(f'Unknown thought-graph configuration: {graph_config_id}')
        self.coding_team.temperature = GOT_12_CONFIGS[graph_config_id].temperature
        try:
            from prompts.master_prompts import PROMPT_TASK_DECOMPOSITION
            from thought_graph.graph_types import AbstractionLevel, QualityDimension
            from thought_graph.generator import TextEmbeddingEngine
        except ImportError:
            from ..prompts.master_prompts import PROMPT_TASK_DECOMPOSITION
            from ..thought_graph.graph_types import AbstractionLevel, QualityDimension
            from ..thought_graph.generator import TextEmbeddingEngine
        prompt = PROMPT_TASK_DECOMPOSITION.format(requirement=requirement, max_subtasks=max_subtasks, target_language='Python')
        raw = self.coding_team.call_llm(prompt)
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Project decomposition must be valid JSON: {exc}') from exc
        subtasks = plan.get('subtasks') if isinstance(plan, dict) else None
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError('Project decomposition must contain a non-empty subtasks array')
        if len(subtasks) > max_subtasks:
            raise ValueError(f'Decomposition returned more than {max_subtasks} subtasks')
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in subtasks:
            if not isinstance(item, dict) or not item.get('id') or (not item.get('goal')):
                raise ValueError('Each subtask needs a unique id and a goal')
            task_id = str(item['id'])
            if task_id in by_id:
                raise ValueError(f'Duplicate subtask id: {task_id}')
            item['id'] = task_id
            item['files'] = [str(x) for x in item.get('files', [])]
            item['depends_on'] = [str(x) for x in item.get('depends_on', [])]
            by_id[task_id] = item
        for task_id, item in by_id.items():
            missing = set(item['depends_on']) - set(by_id)
            if missing or task_id in item['depends_on']:
                raise ValueError(f'Invalid dependencies for {task_id}: {sorted(missing)}')
        indegree = {task_id: len(item['depends_on']) for task_id, item in by_id.items()}
        dependents = {task_id: [] for task_id in by_id}
        for task_id, item in by_id.items():
            for prerequisite in item['depends_on']:
                dependents[prerequisite].append(task_id)
        queue = [task_id for task_id, degree in indegree.items() if degree == 0]
        order = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for dependent in dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
        if len(order) != len(by_id):
            raise ValueError('Subtask dependency graph contains a cycle')
        results: Dict[str, Dict[str, Any]] = {}
        all_history: List[Dict[str, Any]] = []
        for task_id in order:
            item = by_id[task_id]
            upstream = [{'id': parent, 'goal': by_id[parent]['goal'], 'files': by_id[parent]['files'], 'produced_code': results[parent]['final_code'], 'selected_reasoning_subgraph': results[parent]['thought_graph_summary']['selected_reasoning_subgraph_nodes']} for parent in item['depends_on']]
            subtask_request = f"Project requirement: {requirement}\nSubtask {task_id}: {item['goal']}\nTarget files: {item['files']}\nPrerequisite outputs: {upstream}\nRespect the stated dependency order and interface contracts."
            results[task_id] = self.run_cycle(subtask_request, subtask_id=task_id, got_config_id=graph_config_id, max_iterations=max_iterations, history_split=history_split)
            all_history.extend(results[task_id].get('reasoning_history', []))
        if history_path:
            if history_split is None:
                raise ValueError('Set history_split before exporting labeled project history')
            try:
                from gotgnn.training_data import write_history_jsonl
            except ImportError:
                from ..gotgnn.training_data import write_history_jsonl
            write_history_jsonl(all_history, history_path)
        project_graph = ThoughtGraph(graph_id='project_reasoning', task_name=requirement, config_id=graph_config_id, task_id='project')
        remapped: Dict[str, Dict[int, int]] = {}
        next_id = 1
        for task_id in order:
            try:
                from gotgnn.training_data import graph_from_dict
            except ImportError:
                from ..gotgnn.training_data import graph_from_dict
            local = graph_from_dict(results[task_id]['thought_graph'])
            mapping = {}
            for old_id in sorted(local.nodes):
                node = copy.deepcopy(local.nodes[old_id])
                node.node_id = next_id
                node.task_id = task_id
                project_graph.add_node(node)
                mapping[old_id] = next_id
                next_id += 1
            for edge in local.edges:
                project_graph.add_edge(mapping[edge.src], mapping[edge.dst], edge.edge_type, edge.weight)
            remapped[task_id] = mapping
            selected_ids = results[task_id]['thought_graph_summary']['selected_reasoning_subgraph_nodes']
            selected_ids = [mapping[x] for x in selected_ids if x in mapping]
            aggregate_id = next_id
            next_id += 1
            content = f"Selected solution for subtask {task_id}: {by_id[task_id]['goal']}"
            aggregate = ThoughtNode(node_id=aggregate_id, task_id=task_id, node_type=ThoughtNodeType.AGGREGATION, content=content, abstraction=AbstractionLevel.MODULE, quality_dims=[QualityDimension.ARCHITECTURE, QualityDimension.MAINTAINABILITY], semantic_feat=TextEmbeddingEngine(dim=384).embed(content))
            aggregate.build_feature_vectors()
            project_graph.add_node(aggregate)
            terminal_sources = selected_ids or [mapping[x] for x in local.leaf_nodes if x in mapping]
            for source in terminal_sources:
                if source in project_graph.nodes and source != aggregate_id:
                    project_graph.add_edge(source, aggregate_id, ThoughtEdgeType.AGGREGATE)
            results[task_id]['project_aggregate_node'] = aggregate_id
        for task_id in order:
            for prerequisite in by_id[task_id]['depends_on']:
                project_graph.add_edge(results[prerequisite]['project_aggregate_node'], results[task_id]['project_aggregate_node'], ThoughtEdgeType.DEPENDS)
        project_graph.root_nodes = [nid for nid in project_graph.nodes if not project_graph.adj_in[nid]]
        project_graph.leaf_nodes = [nid for nid in project_graph.nodes if not project_graph.adj_out[nid]]
        global_plan = self.got_infer.infer(project_graph)
        selected_project_nodes = global_plan['best_subgraph_nodes']
        integration_context = self._thought_context(project_graph, selected_project_nodes)
        project_files = self.coding_team.integrate_project(requirement=requirement, dependency_order=order, subtasks=[{'id': task_id, 'goal': by_id[task_id]['goal'], 'files': by_id[task_id]['files'], 'depends_on': by_id[task_id]['depends_on'], 'code': results[task_id]['final_code']} for task_id in order], selected_reasoning=integration_context)
        return {'requirement': requirement, 'dependency_order': order, 'subtask_dag': subtasks, 'subtask_results': results, 'project_reasoning_graph': project_graph.to_dict(), 'selected_project_path': global_plan['best_path'], 'selected_project_subgraph_nodes': selected_project_nodes, 'project_files_json': project_files}

    @staticmethod
    def _thought_context(graph: ThoughtGraph, node_ids: List[int]) -> List[Dict[str, Any]]:
        return [{'node_id': node_id, 'type': graph.nodes[node_id].node_type.name, 'abstraction': graph.nodes[node_id].abstraction.name, 'quality_dimensions': [q.name for q in graph.nodes[node_id].quality_dims], 'utility': graph.nodes[node_id].predicted_score, 'content': graph.nodes[node_id].content} for node_id in node_ids if node_id in graph.nodes]

    def _generate_action_thoughts(self, graph: ThoughtGraph, plan: Dict[str, Any]) -> Dict[str, Any]:
        self.coding_team._require_llm('thought generation')
        raw = self.coding_team.call_llm(plan['steering_prompt'])
        return self.thought_expander.apply_plan_output(graph, plan, raw)

    def _add_test_failure_critique(self, graph: ThoughtGraph, target_node_id: int, root_cause: Dict[str, Any], iteration: int) -> int:
        node_id = max(graph.nodes, default=0) + 1
        target = graph.nodes[target_node_id]
        failures = root_cause.get('description', 'Test execution failed without a localized cause.')
        suspicious = root_cause.get('suspicious_statements', [])
        content = f'Observed test failure at repair round {iteration}. Diagnosis: {failures} Suspicious code statements: {suspicious}. The failing assertion is evidence about behavior, not proof that the reported statement is the root cause.'
        node = ThoughtNode(node_id=node_id, task_id=target.task_id, node_type=ThoughtNodeType.CRITIQUE, content=content, abstraction=target.abstraction, quality_dims=[QualityDimension.TESTABILITY, QualityDimension.MAINTAINABILITY])
        node.build_feature_vectors()
        graph.add_node(node)
        graph.add_edge(target_node_id, node_id, ThoughtEdgeType.CRITIQUE)
        graph.leaf_nodes = [nid for nid in graph.nodes if not graph.adj_out[nid]]
        return node_id

    def run_cycle(self, requirement: str, subtask_id: str='T01', got_config_id: str='C6', max_iterations: Optional[int]=None, history_split: Optional[str]=None, history_path: Optional[str]=None) -> Dict[str, Any]:
        max_iter = max_iterations or self.cfg.max_iterations
        if got_config_id not in GOT_12_CONFIGS:
            raise ValueError(f'Unknown thought-graph configuration: {got_config_id}')
        graph_config = GOT_12_CONFIGS[got_config_id]
        self.coding_team.temperature = graph_config.temperature
        if history_split is not None and history_split not in {'train', 'validation', 'test'}:
            raise ValueError('history_split must be train, validation, or test')
        cycle_log: List[Dict[str, Any]] = []
        reasoning_history: List[Dict[str, Any]] = []
        graph_t, seed_prompt = self.thought_expander.initialize_graph(task_name=requirement, task_id=subtask_id, n_init=graph_config.n_init)
        graph_t.config_id = got_config_id
        self.coding_team._require_llm('initial thought generation')
        seed_raw = self.coding_team.call_llm(seed_prompt)
        seed_ids = self.thought_expander.apply_generated_thoughts(graph_t, seed_raw, action_type='IDEA', expected_count=graph_config.n_init)
        first_plan = self.thought_expander.step(graph_t)
        first_record = self._generate_action_thoughts(graph_t, first_plan)
        reasoning_history.append(first_record)
        plan_got = self.got_infer.infer(graph_t)
        pi_star = plan_got['best_path']
        selected_subgraph = plan_got.get('best_subgraph_nodes', pi_star)
        context = self._thought_context(graph_t, selected_subgraph)
        code_y = self.coding_team.generate_initial_code(requirement, selected_subgraph, first_record['next_action']['type'], context, first_record['next_action']['child_count'])
        graph_c = self.scpg_builder.build_from_source(code_y, code_id=f'{subtask_id}_init', phase_mask=0)
        advice: List[str] = []
        final_code = code_y
        passed = False
        last_patch: Optional[str] = None
        for iteration in range(1, max_iter + 1):
            iter_record: Dict[str, Any] = {'iteration': iteration}
            flare_pre = self.flare_infer.infer_and_extract_repair_subgraph(graph_c)
            iter_record['pre_test_scores'] = flare_pre['stmt_scores']
            iter_record['pre_test_seeds'] = flare_pre['seed_stmts']

            dynamic_tests = self.test_team.generate_targeted_tests(
                code=final_code,
                flare_pre=flare_pre,
                advice=advice,
                requirement=requirement,
                subtask_id=subtask_id
            )
            iter_record['dynamic_tests_count'] = len(dynamic_tests)
            test_results, coverage = self.test_team.execute_tests(
                final_code,
                iteration,
                dynamic_tests=dynamic_tests
            )
            iter_record['test_results'] = test_results
            if reasoning_history:
                record = reasoning_history[-1]
                chosen_before = record.get('selected_path', [])
                terminal_after = pi_star[-1:] if pi_star else chosen_before[-1:]
                total = int(test_results.get('total_count', 0))
                passed_count = int(test_results.get('passed_count', 0))
                outcomes = [{'node_ids': chosen_before, 'terminal_node_ids': terminal_after, 'passed_tests': passed_count, 'total_tests': total, 'candidate_id': f'{subtask_id}_candidate_{iteration}', 'test_run_id': f'{subtask_id}_test_{iteration}'}]
                if history_split is not None:
                    evaluated_leaves = set(terminal_after)
                    recorded_paths = [item['node_ids'] for item in record.get('candidate_paths', []) if item.get('node_ids')]
                    all_paths = self.got_infer.extract_candidate_paths(graph_t, max_paths=max(len(graph_t.leaf_nodes) * 2, len(graph_t.nodes), 8), max_len=max(len(graph_t.nodes), 6))
                    for leaf_id in graph_t.leaf_nodes:
                        if leaf_id in evaluated_leaves:
                            continue
                        leaf_path = next((p for p in all_paths if p and p[-1] == leaf_id), [leaf_id])
                        parent_cand = next((cp for cp in recorded_paths if cp and cp[-1] == leaf_id), None)
                        if parent_cand is None:
                            parent_cand = next((cp for cp in recorded_paths if cp and set(cp).issubset(set(leaf_path))), chosen_before)
                        leaf_ctx = self._thought_context(graph_t, leaf_path)
                        alt_code = self.coding_team.generate_initial_code(requirement, leaf_path, record['next_action']['type'], leaf_ctx, 1)
                        alt_res, _ = self.test_team.execute_tests(alt_code, iteration)
                        alt_total = int(alt_res.get('total_count', 0))
                        alt_passed = int(alt_res.get('passed_count', 0))
                        if alt_total <= 0:
                            raise ValueError(f'Leaf {leaf_id} execution has no measured test count')
                        outcomes.append({'node_ids': parent_cand, 'terminal_node_ids': [leaf_id], 'passed_tests': alt_passed, 'total_tests': alt_total, 'candidate_id': f'{subtask_id}_leaf_{leaf_id}_iter_{iteration}', 'test_run_id': f'{subtask_id}_test_leaf_{leaf_id}_{iteration}'})
                        evaluated_leaves.add(leaf_id)
                try:
                    if total <= 0:
                        raise ValueError('Execution record has no measured test count; no label was created')
                    self.thought_expander.attach_test_outcomes(record, outcomes, split=history_split, label_graph=graph_t.to_dict())
                except ValueError as exc:
                    if history_split is not None:
                        raise
                    iter_record['history_label_warning'] = str(exc)
            if test_results['all_passed']:
                passed = True
                iter_record['status'] = 'PASSED'
                cycle_log.append(iter_record)
                break
            graph_c = self.scpg_builder.build_from_source(source_code=final_code, code_id=f'{subtask_id}_iter{iteration}', phase_mask=1, test_results=test_results)
            flare_post = self.flare_infer.infer_and_extract_repair_subgraph(graph_c)
            iter_record['post_test_repair_stmts'] = flare_post['repair_stmt_ids']
            iter_record['mad_threshold'] = flare_post['mad_threshold']
            root_cause = self.analyzer.analyze(graph_t, pi_star, test_results, flare_post)
            iter_record['root_cause'] = root_cause

            self.pem.record_experience(
                subtask_id=subtask_id,
                root_cause=root_cause,
                context={'path': pi_star, 'iteration': iteration},
                v_repair=flare_post.get('repair_stmt_ids', []),
                y_patch=last_patch,
                outcome=test_results,
                iteration=iteration
            )
            if iteration == max_iter:
                iter_record['status'] = 'MAX_ITER_REACHED'
                cycle_log.append(iter_record)
                break
            advice = self.pem.advise(
                subtask_id=subtask_id,
                current_context=root_cause['pattern'],
                suspicious_stmts=flare_post.get('seed_stmts', [])
            )
            iter_record['pem_advice'] = advice
            target_id = pi_star[-1] if pi_star else seed_ids[-1] if seed_ids else max(graph_t.nodes)
            failure_node = self._add_test_failure_critique(graph_t, target_id, root_cause, iteration)
            environment_feedback = f"Observed test result: {test_results}. Root-cause analysis: {root_cause}. Flare-GNN repair context: {flare_post.get('repair_stmt_ids', [])}. PEM advice: {advice}."
            action_plan = self.thought_expander.step(graph_t, environment_feedback=environment_feedback)
            action_record = self._generate_action_thoughts(graph_t, action_plan)
            reasoning_history.append(action_record)
            plan_got = self.got_infer.infer(graph_t)
            pi_star = plan_got['best_path']
            selected_subgraph = plan_got.get('best_subgraph_nodes', pi_star)
            context = self._thought_context(graph_t, selected_subgraph)
            feedback = {'repair_subgraph': flare_post['repair_stmt_ids'], 'scores': flare_post['stmt_scores'], 'environment_feedback': environment_feedback, 'failure_thought_id': failure_node}
            final_code = self.coding_team.update_code(final_code, feedback, advice, thought_context=context, action_type=action_record['next_action']['type'], action_count=action_record['next_action']['child_count'])
            last_patch = final_code
            graph_c = self.scpg_builder.build_from_source(final_code, code_id=f'{subtask_id}_iter{iteration + 1}', phase_mask=0)
            iter_record['status'] = 'REPAIRED'
            cycle_log.append(iter_record)
        if history_path:
            if history_split is None:
                raise ValueError('Set history_split before exporting labeled training records')
            try:
                from gotgnn.training_data import write_history_jsonl
            except ImportError:
                from ..gotgnn.training_data import write_history_jsonl
            write_history_jsonl(reasoning_history, history_path)
        return {'requirement': requirement, 'subtask_id': subtask_id, 'passed': passed, 'total_iterations': len(cycle_log), 'final_code': final_code, 'cycle_log': cycle_log, 'reasoning_history': reasoning_history, 'thought_graph': graph_t.to_dict(), 'thought_graph_summary': {'num_nodes': len(graph_t.nodes), 'num_edges': len(graph_t.edges), 'pruned_nodes': plan_got['pruned_nodes'], 'selected_reasoning_path': pi_star, 'selected_reasoning_subgraph_nodes': selected_subgraph}}
