from typing import Dict, List, Any, Optional, Tuple, Callable
try:
    from thought_graph.graph_types import ThoughtGraph, ThoughtNode, ThoughtEdge, ThoughtNodeType, ThoughtEdgeType, AbstractionLevel, QualityDimension
    from prompts.master_prompts import DynamicCodingTeamPromptBuilder, DynamicTextualBackpropBuilder
except ImportError:
    from ..thought_graph.graph_types import ThoughtGraph, ThoughtNode, ThoughtEdge, ThoughtNodeType, ThoughtEdgeType, AbstractionLevel, QualityDimension
    from ..prompts.master_prompts import DynamicCodingTeamPromptBuilder, DynamicTextualBackpropBuilder

class GoTOperationType:
    GENERATE = 'Generate'
    SCORE = 'Score'
    VALIDATE = 'Validate'
    AGGREGATE = 'Aggregate'
    KEEP_BEST_N = 'KeepBestN'
    GROUND_TRUTH = 'GroundTruth'

class GoTOperation:

    def __init__(self, op_id: str, op_type: str, predecessors: Optional[List[str]]=None, params: Optional[Dict[str, Any]]=None):
        self.op_id = op_id
        self.op_type = op_type
        self.predecessors = predecessors or []
        self.params = params or {}

class GraphOfOperations:

    def __init__(self, name: str):
        self.name = name
        self.operations: Dict[str, GoTOperation] = {}
        self.execution_order: List[str] = []

    def add_operation(self, op: GoTOperation):
        self.operations[op.op_id] = op
        self.execution_order.append(op.op_id)

    @classmethod
    def create_default_plan(cls) -> 'GraphOfOperations':
        goo = cls('Default_GoO')
        goo.add_operation(GoTOperation('op_gen_init', GoTOperationType.GENERATE, params={'count': 3}))
        goo.add_operation(GoTOperation('op_score_gotgnn', GoTOperationType.SCORE, predecessors=['op_gen_init']))
        goo.add_operation(GoTOperation('op_prune_best', GoTOperationType.KEEP_BEST_N, predecessors=['op_score_gotgnn'], params={'tau_prune': 0.4}))
        goo.add_operation(GoTOperation('op_gen_critique', GoTOperationType.GENERATE, predecessors=['op_prune_best']))
        goo.add_operation(GoTOperation('op_aggregate', GoTOperationType.AGGREGATE, predecessors=['op_gen_critique']))
        goo.add_operation(GoTOperation('op_validate_cpg', GoTOperationType.VALIDATE, predecessors=['op_aggregate']))
        return goo

class GraphReasoningState:

    def __init__(self, task_description: str):
        self.task_description = task_description
        self.thoughts: Dict[int, Dict[str, Any]] = {}
        self.edges: List[Tuple[int, int, str]] = []
        self.operation_history: List[Dict[str, Any]] = []

    def add_thought(self, thought_id: int, content: str, node_type: str, score: float=0.0) -> Dict[str, Any]:
        record = {'thought_id': thought_id, 'content': content, 'node_type': node_type, 'score': score, 'pruned': False, 'validated': False}
        self.thoughts[thought_id] = record
        return record

    def add_edge(self, src: int, dst: int, relation: str):
        self.edges.append((src, dst, relation))

    def get_thought(self, thought_id: int) -> Optional[Dict[str, Any]]:
        return self.thoughts.get(thought_id)

    def to_thought_graph(self, task_id: str='T01') -> ThoughtGraph:
        graph = ThoughtGraph(graph_id=f'got_{task_id}_trace', task_name=self.task_description, config_id='C12')
        type_map = {'Idea': ThoughtNodeType.IDEA, 'Critique': ThoughtNodeType.CRITIQUE, 'Refine': ThoughtNodeType.REFINEMENT, 'Refinement': ThoughtNodeType.REFINEMENT, 'Aggregate': ThoughtNodeType.AGGREGATION, 'Aggregation': ThoughtNodeType.AGGREGATION, 'AggregateSimilar': ThoughtNodeType.AGGREGATION, 'AggregateTask': ThoughtNodeType.AGGREGATION}
        edge_map = {'informs': ThoughtEdgeType.INFORMS, 'critique': ThoughtEdgeType.CRITIQUE, 'critiques': ThoughtEdgeType.CRITIQUE, 'refine': ThoughtEdgeType.INFORMS, 'refines': ThoughtEdgeType.INFORMS, 'aggregate': ThoughtEdgeType.AGGREGATE, 'aggregates': ThoughtEdgeType.AGGREGATE, 'depends': ThoughtEdgeType.DEPENDS, 'depends_on': ThoughtEdgeType.DEPENDS, 'feedback': ThoughtEdgeType.FEEDBACK}
        for tid, tdata in self.thoughts.items():
            ntype = type_map.get(tdata.get('node_type', 'Idea'), ThoughtNodeType.IDEA)
            node = ThoughtNode(node_id=tid, task_id=task_id, node_type=ntype, content=tdata.get('content', ''), ground_truth_score=tdata.get('score', 0.0), abstraction=AbstractionLevel.MODULE, quality_dims=[QualityDimension.ARCHITECTURE])
            node.build_feature_vectors()
            graph.add_node(node)
        for src, dst, rel in self.edges:
            etype = edge_map.get(rel.lower(), ThoughtEdgeType.INFORMS)
            graph.add_edge(src, dst, etype)
        return graph

class GoTController:

    def __init__(self, goo: GraphOfOperations, grs: GraphReasoningState, lm_callable: Optional[Callable[[str], str]]=None):
        self.goo = goo
        self.grs = grs
        self.lm = lm_callable

    def step(self, operation: GoTOperation) -> Dict[str, Any]:
        op_log = {'op_id': operation.op_id, 'op_type': operation.op_type, 'status': 'executed'}
        self.grs.operation_history.append(op_log)
        return op_log

    def run_all(self) -> GraphReasoningState:
        for op_id in self.goo.execution_order:
            op = self.goo.operations[op_id]
            self.step(op)
        return self.grs

class AgentRole:
    PROJECT_MANAGER = 'ProjectManager'
    ARCHITECT = 'Architect'
    MODULE_ENGINEER = 'ModuleEngineer'
    QA_ENGINEER = 'QAEngineer'
    GRADIENT_AGENT = 'GradientAgent'
    UPDATER_AGENT = 'UpdaterAgent'

class TextualGradient:

    def __init__(self, iteration: int, loss_metric: float, failing_assertions: List[str], flare_suspicious_stmts: List[int], repair_subgraph_summary: str, root_cause_pattern: str):
        self.iteration = iteration
        self.loss_metric = loss_metric
        self.failing_assertions = failing_assertions
        self.flare_suspicious_stmts = flare_suspicious_stmts
        self.repair_subgraph_summary = repair_subgraph_summary
        self.root_cause_pattern = root_cause_pattern

    def to_critique_prompt(self) -> str:
        return f'[GRADIENT - ITERATION {self.iteration}]\nLoss (Error Rate): {self.loss_metric:.2%}\nFailure Mode: {self.root_cause_pattern}\nFailing Assertions:\n' + '\n'.join([f'- {a}' for a in self.failing_assertions]) + f'\nAttributed Statements: {self.flare_suspicious_stmts}\nRepair Context:\n{self.repair_subgraph_summary}\n'

class TextualBackpropOptimizer:

    def __init__(self):
        self.gradient_history: List[TextualGradient] = []
        self.prompt_updates_log: List[Dict[str, Any]] = []

    def backward(self, iteration: int, test_results: Dict[str, Any], flare_post: Dict[str, Any], root_cause: Dict[str, Any], localized_subgraph: str) -> TextualGradient:
        total = max(test_results.get('total_count', 1), 1)
        passed = test_results.get('passed_count', 0)
        loss = 1.0 - passed / total
        grad = TextualGradient(iteration=iteration, loss_metric=loss, failing_assertions=test_results.get('failures', []), flare_suspicious_stmts=flare_post.get('seed_stmts', []), repair_subgraph_summary=localized_subgraph, root_cause_pattern=root_cause.get('pattern', 'UnknownFailure'))
        self.gradient_history.append(grad)
        prompt_update = DynamicTextualBackpropBuilder.build_prompt_update(agent_role=AgentRole.MODULE_ENGINEER, iteration=iteration, failure_pattern=grad.root_cause_pattern, critique=grad.to_critique_prompt(), advice=root_cause.get('advice', 'Apply defensive validation.'))
        self.prompt_updates_log.append({'iteration': iteration, 'target_role': AgentRole.MODULE_ENGINEER, 'updated_prompt_directive': prompt_update})
        return grad

class EvoMACCollaborationNetwork:

    def __init__(self):
        self.roles = [AgentRole.PROJECT_MANAGER, AgentRole.ARCHITECT, AgentRole.MODULE_ENGINEER, AgentRole.QA_ENGINEER]
        self.active_edges = [(AgentRole.PROJECT_MANAGER, AgentRole.ARCHITECT), (AgentRole.ARCHITECT, AgentRole.MODULE_ENGINEER), (AgentRole.MODULE_ENGINEER, AgentRole.QA_ENGINEER), (AgentRole.QA_ENGINEER, AgentRole.GRADIENT_AGENT), (AgentRole.GRADIENT_AGENT, AgentRole.UPDATER_AGENT)]
        self.system_prompts: Dict[str, str] = {}

    def update_agent_prompt(self, role: str, prompt_directive: str):
        self.system_prompts[role] = prompt_directive

class UnifiedGraphBridge:

    def __init__(self):
        self.goo = GraphOfOperations.create_default_plan()
        self.grs: Optional[GraphReasoningState] = None
        self.network = EvoMACCollaborationNetwork()
        self.backprop_optimizer = TextualBackpropOptimizer()

    def initialize_task(self, requirement: str):
        self.grs = GraphReasoningState(task_description=requirement)

    def record_thought(self, thought_id: int, content: str, node_type: str, score: float=0.0) -> Dict[str, Any]:
        if self.grs is None:
            self.grs = GraphReasoningState(task_description='Default Task')
        return self.grs.add_thought(thought_id=thought_id, content=content, node_type=node_type, score=score)

    def record_edge(self, src: int, dst: int, relation: str):
        if self.grs is None:
            self.grs = GraphReasoningState(task_description='Default Task')
        self.grs.add_edge(src, dst, relation)

    def extract_thought_graph(self, task_id: str='T01') -> ThoughtGraph:
        if self.grs is None:
            raise RuntimeError('Cannot extract thought graph: task has not been initialized.')
        return self.grs.to_thought_graph(task_id=task_id)

    def execute_forward_step(self, subtask_id: str, requirement: str, plan_got: Dict[str, Any], scpg_contracts: str, pem_advice: List[str]) -> str:
        return DynamicCodingTeamPromptBuilder.build_initial(requirement=requirement, subtask_id=subtask_id, target_files=['module.py'], plan_got=plan_got, scpg_contracts=scpg_contracts, pem_advice=pem_advice)

    def execute_backward_step(self, iteration: int, test_results: Dict[str, Any], flare_post: Dict[str, Any], root_cause: Dict[str, Any], localized_subgraph: str) -> TextualGradient:
        grad = self.backprop_optimizer.backward(iteration=iteration, test_results=test_results, flare_post=flare_post, root_cause=root_cause, localized_subgraph=localized_subgraph)
        if self.backprop_optimizer.prompt_updates_log:
            last_update = self.backprop_optimizer.prompt_updates_log[-1]
            self.network.update_agent_prompt(role=last_update['target_role'], prompt_directive=last_update['updated_prompt_directive'])
        return grad