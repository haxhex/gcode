import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

try:
    from config import PEMConfig
except ImportError:
    from ..config import PEMConfig


@dataclass
class PlaybookRule:
    rule_id: str
    subtask_type: str
    error_pattern: str
    root_cause_summary: str
    prescribed_advice: str
    confidence: float = 0.8
    success_count: int = 1
    failure_count: int = 0
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))


@dataclass
class ProjectExperienceRecord:
    record_id: str
    subtask_id: str
    path: List[int]
    root_cause: Dict[str, Any]
    v_repair: List[int]
    y_patch: Optional[str]
    outcome: Dict[str, Any]
    iteration: int = 1
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.8
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'record_id': self.record_id,
            'subtask_id': self.subtask_id,
            'path': self.path,
            'root_cause': self.root_cause,
            'v_repair': self.v_repair,
            'y_patch': self.y_patch,
            'outcome': self.outcome,
            'iteration': self.iteration,
            'timestamp': self.timestamp,
            'confidence': float(self.confidence),
            'embedding': self.embedding.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectExperienceRecord':
        emb = np.array(data.get('embedding', [0.0] * 64), dtype=np.float32)
        return cls(
            record_id=data.get('record_id', ''),
            subtask_id=data.get('subtask_id', ''),
            path=data.get('path', []),
            root_cause=data.get('root_cause', {}),
            v_repair=data.get('v_repair', []),
            y_patch=data.get('y_patch'),
            outcome=data.get('outcome', {}),
            iteration=data.get('iteration', 1),
            timestamp=data.get('timestamp', 0.0),
            confidence=float(data.get('confidence', 0.8)),
            embedding=emb,
        )


PEMRecord = ProjectExperienceRecord


class ProjectEvolutionaryMemory:
    def __init__(
        self,
        config: Optional[PEMConfig] = None,
        seed: int = 42,
        storage_path: Optional[Union[str, Path]] = None,
    ):
        self.cfg = config or PEMConfig()
        self.rules: Dict[str, PlaybookRule] = {}
        self.records: List[ProjectExperienceRecord] = []
        self.history: List[Dict[str, Any]] = []
        self.rng = np.random.default_rng(seed)
        self.storage_path = Path(storage_path) if storage_path else getattr(self.cfg, 'storage_path', None)
        if self.storage_path:
            self.storage_path = Path(self.storage_path)
            self.load_from_disk()

    def _embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        vec = np.zeros(64, dtype=np.float32)
        for w in words:
            idx = abs(hash(w)) % 64
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec

    def record_experience(
        self,
        subtask_id: str,
        root_cause: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        v_repair: Optional[List[int]] = None,
        y_patch: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
        iteration: int = 1,
    ):
        ctx = context or {}
        path = ctx.get('path', [])
        test_outcome = outcome if outcome is not None else ctx.get('results', {})
        v_rep = v_repair if v_repair is not None else ctx.get('v_repair', [])
        patch = y_patch if y_patch is not None else ctx.get('y_patch', None)
        iter_num = iteration or ctx.get('iteration', 1)

        pattern = root_cause.get('pattern', 'LogicError')
        cause = root_cause.get('description', 'Unchecked boundary condition')
        advice = root_cause.get('advice', 'Add strict boundary checks and state guards')

        rule_key = f'{subtask_id}_{pattern}'
        emb = self._embed(f'{subtask_id} {pattern} {cause} stmts:{v_rep}')

        if rule_key in self.rules:
            rule = self.rules[rule_key]
            rule.confidence = min(0.99, rule.confidence + 0.05)
            rule.success_count += 1
        else:
            rule = PlaybookRule(
                rule_id=rule_key,
                subtask_type=subtask_id,
                error_pattern=pattern,
                root_cause_summary=cause,
                prescribed_advice=advice,
                confidence=0.8,
                embedding=emb,
            )
            self.rules[rule_key] = rule

        rec_id = f'rec_{subtask_id}_{len(self.records) + 1}_{int(time.time() * 1000) % 100000}'
        new_record = ProjectExperienceRecord(
            record_id=rec_id,
            subtask_id=subtask_id,
            path=list(path),
            root_cause=copy.deepcopy(root_cause),
            v_repair=list(v_rep),
            y_patch=patch,
            outcome=copy.deepcopy(test_outcome),
            iteration=iter_num,
            timestamp=time.time(),
            confidence=rule.confidence,
            embedding=emb,
        )

        is_duplicate = False
        for existing in self.records:
            if existing.subtask_id == subtask_id:
                sim = float(np.dot(emb, existing.embedding))
                if sim >= self.cfg.similarity_threshold:
                    existing.confidence = min(0.99, existing.confidence + 0.05)
                    existing.outcome = copy.deepcopy(test_outcome)
                    if patch is not None:
                        existing.y_patch = patch
                    is_duplicate = True
                    break

        if not is_duplicate:
            self.records.append(new_record)

        self.history.append({
            'subtask_id': subtask_id,
            'root_cause': root_cause,
            'context': ctx,
            'v_repair': v_rep,
            'y_patch': patch,
            'outcome': test_outcome,
        })

        self.decay_and_prune()

        if self.storage_path:
            self.save_to_disk()

    def advise(
        self,
        subtask_id: str,
        current_context: Optional[str] = None,
        suspicious_stmts: Optional[List[int]] = None,
    ) -> List[str]:
        if not self.rules and not self.records:
            return ['Enforce modular boundaries and explicit error handling.']

        query_text = f"{subtask_id} {current_context or ''}"
        if suspicious_stmts:
            query_text += f" stmts:{suspicious_stmts}"
        query_emb = self._embed(query_text)

        scored_rules = []
        for r in self.rules.values():
            cos_sim = float(np.dot(query_emb, r.embedding))
            final_score = cos_sim * r.confidence
            scored_rules.append((final_score, r))
        scored_rules.sort(key=lambda x: x[0], reverse=True)
        top_rules = scored_rules[:self.cfg.top_k_retrieval]
        advice_list = [
            f'[{r.error_pattern} Warning (conf={r.confidence:.2f})]: {r.prescribed_advice}'
            for _, r in top_rules
        ]

        failed_patch_warnings = []
        for rec in reversed(self.records):
            if rec.subtask_id == subtask_id and rec.y_patch:
                pattern = rec.root_cause.get('pattern', 'Failure')
                stmts = rec.v_repair
                if stmts:
                    failed_patch_warnings.append(
                        f"[Previous Patch Failure Warning]: Earlier patch at statements {stmts} failed under {pattern}. Avoid repeating identical edits."
                    )
                if len(failed_patch_warnings) >= 2:
                    break

        return advice_list + failed_patch_warnings

    def decay_and_prune(self):
        to_prune = []
        for k, r in self.rules.items():
            r.confidence -= self.cfg.decay_rate
            if r.confidence < self.cfg.min_confidence:
                to_prune.append(k)
        for k in to_prune:
            del self.rules[k]

        max_recs = getattr(self.cfg, 'max_rules_per_cluster', 10) * 10
        if len(self.records) > max_recs:
            self.records = self.records[-max_recs:]

    def save_to_disk(self, path: Optional[Union[str, Path]] = None):
        target = Path(path) if path else self.storage_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'rules': {
                k: {
                    'rule_id': r.rule_id,
                    'subtask_type': r.subtask_type,
                    'error_pattern': r.error_pattern,
                    'root_cause_summary': r.root_cause_summary,
                    'prescribed_advice': r.prescribed_advice,
                    'confidence': float(r.confidence),
                    'success_count': r.success_count,
                    'failure_count': r.failure_count,
                    'embedding': r.embedding.tolist(),
                }
                for k, r in self.rules.items()
            },
            'records': [r.to_dict() for r in self.records],
        }
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load_from_disk(self, path: Optional[Union[str, Path]] = None):
        target = Path(path) if path else self.storage_path
        if not target or not target.is_file():
            return
        try:
            with open(target, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            self.rules = {}
            for k, r in payload.get('rules', {}).items():
                emb = np.array(r.get('embedding', [0.0] * 64), dtype=np.float32)
                self.rules[k] = PlaybookRule(
                    rule_id=r['rule_id'],
                    subtask_type=r.get('subtask_type', ''),
                    error_pattern=r.get('error_pattern', ''),
                    root_cause_summary=r.get('root_cause_summary', ''),
                    prescribed_advice=r.get('prescribed_advice', ''),
                    confidence=float(r.get('confidence', 0.8)),
                    success_count=int(r.get('success_count', 1)),
                    failure_count=int(r.get('failure_count', 0)),
                    embedding=emb,
                )
            self.records = [
                ProjectExperienceRecord.from_dict(r)
                for r in payload.get('records', [])
            ]
        except Exception as e:
            print(f'Warning: Failed to load PEM records from {target}: {e}')


PlaybookEvolutionaryMemory = ProjectEvolutionaryMemory
