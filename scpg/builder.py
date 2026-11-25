import ast
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional

class SCPGEdgeType(Enum):
    CFG_FLOW = 0
    DFG_DATA = 1
    CALL_INVOKE = 2
    PDG_DEP = 3
    AST_PARENT = 4
    EXEC_TRACE = 5

@dataclass
class CodeStatementNode:
    stmt_id: int
    line_no: int
    stmt_type: str
    code_text: str
    method_id: int
    file_id: int
    is_defective: bool = False
    static_feat: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    dynamic_feat: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    semantic_feat: np.ndarray = field(default_factory=lambda: np.zeros(384, dtype=np.float32))
    suspiciousness_score: float = 0.0

@dataclass
class SCPGEdge:
    src: int
    dst: int
    edge_type: SCPGEdgeType
    weight: float = 1.0

class SCPGGraph:

    def __init__(self, code_id: str, phase_mask: int=0):
        self.code_id = code_id
        self.phase_mask = phase_mask
        self.nodes: Dict[int, CodeStatementNode] = {}
        self.edges: List[SCPGEdge] = []
        self.methods: Dict[int, List[int]] = {}
        self.method_names: Dict[int, str] = {}
        self.total_passed_tests: int = 0
        self.total_failed_tests: int = 0

    def add_node(self, node: CodeStatementNode):
        self.nodes[node.stmt_id] = node
        if node.method_id not in self.methods:
            self.methods[node.method_id] = []
        self.methods[node.method_id].append(node.stmt_id)

    def add_edge(self, src: int, dst: int, edge_type: SCPGEdgeType, weight: float=1.0):
        self.edges.append(SCPGEdge(src, dst, edge_type, weight))

    def get_induced_subgraph(self, subset_nodes: Set[int]) -> 'SCPGGraph':
        sub = SCPGGraph(f'{self.code_id}_sub', self.phase_mask)
        for nid in subset_nodes:
            if nid in self.nodes:
                sub.add_node(self.nodes[nid])
        for e in self.edges:
            if e.src in subset_nodes and e.dst in subset_nodes:
                sub.add_edge(e.src, e.dst, e.edge_type, e.weight)
        return sub

class StatementEmbeddingEngine:

    def __init__(self, dim: int=384):
        self.dim = dim
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Install sentence-transformers to build real SCPG semantic features') from exc
        self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if int(self._model.get_sentence_embedding_dimension()) != dim:
            raise ValueError(f'Embedding model dimension must be {dim}')

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)

class SCPGBuilder:

    def __init__(self, seed: int=42):
        self.rng = np.random.default_rng(seed)
        self.cfg_edges = []
        self.embed_engine = StatementEmbeddingEngine(dim=384)

    def _extract_cfg_edges(self, node: ast.AST, stmt_id_map: Dict[ast.AST, int]) -> List[Tuple[int, int]]:
        edges = []

        def process_node(n: ast.AST, exit_stmts: List[int]):
            if isinstance(n, ast.If):
                test_stmts = []
                for child in ast.iter_child_nodes(n):
                    if isinstance(child, ast.stmt):
                        test_stmts.append(stmt_id_map.get(child, -1))
                body_exits = []
                for stmt in n.body:
                    if isinstance(stmt, ast.stmt):
                        body_exits.append(stmt_id_map.get(stmt, -1))
                        process_node(stmt, body_exits)
                orelse_exits = []
                for stmt in n.orelse:
                    if isinstance(stmt, ast.stmt):
                        orelse_exits.append(stmt_id_map.get(stmt, -1))
                        process_node(stmt, orelse_exits)
                if test_stmts and body_exits:
                    edges.append((test_stmts[-1], body_exits[0]))
                if test_stmts and orelse_exits:
                    edges.append((test_stmts[-1], orelse_exits[0]))
                return body_exits + orelse_exits
            elif isinstance(n, (ast.For, ast.While)):
                body_exits = []
                for stmt in n.body:
                    if isinstance(stmt, ast.stmt):
                        body_exits.append(stmt_id_map.get(stmt, -1))
                        process_node(stmt, body_exits)
                orelse_exits = []
                for stmt in n.orelse:
                    if isinstance(stmt, ast.stmt):
                        orelse_exits.append(stmt_id_map.get(stmt, -1))
                        process_node(stmt, orelse_exits)
                if body_exits:
                    edges.append((body_exits[-1], body_exits[0]))
                return body_exits + orelse_exits
            elif isinstance(n, ast.Try):
                body_exits = []
                for stmt in n.body:
                    if isinstance(stmt, ast.stmt):
                        body_exits.append(stmt_id_map.get(stmt, -1))
                        process_node(stmt, body_exits)
                handler_exits = []
                for handler in n.handlers:
                    for stmt in handler.body:
                        if isinstance(stmt, ast.stmt):
                            handler_exits.append(stmt_id_map.get(stmt, -1))
                            process_node(stmt, handler_exits)
                orelse_exits = []
                for stmt in n.orelse:
                    if isinstance(stmt, ast.stmt):
                        orelse_exits.append(stmt_id_map.get(stmt, -1))
                        process_node(stmt, orelse_exits)
                finally_exits = []
                for stmt in n.finalbody:
                    if isinstance(stmt, ast.stmt):
                        finally_exits.append(stmt_id_map.get(stmt, -1))
                        process_node(stmt, finally_exits)
                if body_exits and handler_exits:
                    edges.append((body_exits[-1], handler_exits[0]))
                if body_exits and orelse_exits:
                    edges.append((body_exits[-1], orelse_exits[0]))
                return body_exits + handler_exits + orelse_exits + finally_exits
            return exit_stmts
        process_node(node, [])
        return edges

    def build_from_source(self, source_code: str, code_id: str='snippet_01', phase_mask: int=0, test_results: Optional[Dict[str, any]]=None) -> SCPGGraph:
        graph = SCPGGraph(code_id=code_id, phase_mask=phase_mask)
        try:
            tree = ast.parse(source_code)
        except Exception:
            lines = [l for l in source_code.splitlines() if l.strip()]
            for idx, line in enumerate(lines):
                node = CodeStatementNode(stmt_id=idx + 1, line_no=idx + 1, stmt_type='RawStatement', code_text=line.strip(), method_id=0, file_id=0, semantic_feat=self.embed_engine.embed(line))
                graph.add_node(node)
                if idx > 0:
                    graph.add_edge(idx, idx + 1, SCPGEdgeType.CFG_FLOW)
            return graph
        stmt_counter = 0
        current_method = 0
        current_method_name = 'global'
        graph.method_names[0] = current_method_name
        var_defs: Dict[str, List[int]] = {}
        stmt_id_map: Dict[ast.AST, int] = {}
        function_def_ids: Dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current_method += 1
                current_method_name = node.name
                graph.method_names[current_method] = current_method_name
                stmt_counter += 1
                stmt_id_map[node] = stmt_counter
                function_def_ids[node.name] = stmt_counter
            if isinstance(node, ast.stmt) and (not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                stmt_counter += 1
                stmt_id_map[node] = stmt_counter
        stmt_counter = 0
        current_method = 0
        prev_stmt_id = None
        call_edges = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current_method += 1
                stmt_counter += 1
                line_no = getattr(node, 'lineno', stmt_counter)
                stmt_type = type(node).__name__
                code_text = ast.unparse(node).splitlines()[0] if hasattr(ast, 'unparse') else stmt_type
                static_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                stmt_node = CodeStatementNode(stmt_id=stmt_counter, line_no=line_no, stmt_type=stmt_type, code_text=code_text, method_id=current_method, file_id=0, static_feat=static_vec, semantic_feat=self.embed_engine.embed(code_text))
                graph.add_node(stmt_node)
                prev_stmt_id = stmt_counter
            if isinstance(node, ast.stmt) and (not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                stmt_counter += 1
                line_no = getattr(node, 'lineno', stmt_counter)
                stmt_type = type(node).__name__
                code_text = ast.unparse(node).splitlines()[0] if hasattr(ast, 'unparse') else stmt_type
                is_ctrl = 1.0 if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)) else 0.0
                is_assign = 1.0 if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) else 0.0
                is_call = 1.0 if any((isinstance(c, ast.Call) for c in ast.walk(node))) else 0.0
                is_ret = 1.0 if isinstance(node, ast.Return) else 0.0
                depth = float(getattr(node, 'col_offset', 0) // 4)
                tokens = float(len(code_text.split()))
                vars_in_stmt = [c.id for c in ast.walk(node) if isinstance(c, ast.Name)]
                num_vars = float(len(set(vars_in_stmt)))
                complexity = 1.0 + (1.0 if is_ctrl else 0.0) + (1.0 if is_call else 0.0)
                static_vec = np.array([is_ctrl, is_assign, is_call, is_ret, depth, tokens, num_vars, complexity], dtype=np.float32)
                stmt_node = CodeStatementNode(stmt_id=stmt_counter, line_no=line_no, stmt_type=stmt_type, code_text=code_text, method_id=current_method, file_id=0, static_feat=static_vec, semantic_feat=self.embed_engine.embed(code_text))
                graph.add_node(stmt_node)
                if prev_stmt_id is not None:
                    graph.add_edge(prev_stmt_id, stmt_counter, SCPGEdgeType.CFG_FLOW)
                prev_stmt_id = stmt_counter
                for v in vars_in_stmt:
                    if is_assign and v not in var_defs:
                        var_defs[v] = [stmt_counter]
                    elif v in var_defs:
                        for def_id in var_defs[v]:
                            graph.add_edge(def_id, stmt_counter, SCPGEdgeType.DFG_DATA)
                        if is_assign:
                            var_defs[v].append(stmt_counter)
                if is_call:
                    for call_node in ast.walk(node):
                        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
                            func_name = call_node.func.id
                            if func_name in function_def_ids:
                                call_edges.append((stmt_counter, function_def_ids[func_name]))
        cfg_branch_edges = self._extract_cfg_edges(tree, stmt_id_map)
        for src, dst in cfg_branch_edges:
            if src > 0 and dst > 0:
                graph.add_edge(src, dst, SCPGEdgeType.CFG_FLOW)
        for src, dst in call_edges:
            if src > 0 and dst > 0:
                graph.add_edge(src, dst, SCPGEdgeType.CALL_INVOKE)
        if phase_mask == 1 and test_results:
            n_pass_total = test_results.get('total_passed', 0)
            n_fail_total = test_results.get('total_failed', 0)
            failing_lines = set(test_results.get('failing_lines', []))
            graph.total_passed_tests = n_pass_total
            graph.total_failed_tests = n_fail_total
            for s_id, s_node in graph.nodes.items():
                is_faulty = s_node.line_no in failing_lines
                s_node.is_defective = is_faulty
                if is_faulty:
                    n_fail_cov = float(n_fail_total)
                    n_pass_cov = float(max(0, n_pass_total - 1))
                else:
                    n_fail_cov = 0.0
                    n_pass_cov = float(n_pass_total)
                denom = math.sqrt(max(1e-09, n_fail_total * (n_fail_cov + n_pass_cov)))
                ochiai = n_fail_cov / denom if denom > 0 else 0.0
                tar_denom = n_fail_cov / max(1, n_fail_total) + n_pass_cov / max(1, n_pass_total)
                tarantula = n_fail_cov / max(1, n_fail_total) / tar_denom if tar_denom > 0 else 0.0
                s_node.dynamic_feat = np.array([n_pass_cov, n_fail_cov, ochiai, tarantula, float(n_fail_cov + n_pass_cov), 0.0], dtype=np.float32)
                s_node.suspiciousness_score = float(ochiai)
        return graph