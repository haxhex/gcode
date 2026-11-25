import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

@dataclass
class BenchmarkTask:
    task_id: str
    domain: str
    suite: str
    task_name: str
    num_unit_tests: int
    spec_token_length: int

def _load_tasks(data_path: Optional[Union[str, Path]], benchmark_name: str) -> List[BenchmarkTask]:
    if data_path is None:
        raise FileNotFoundError(f'Provide a JSON metadata file for {benchmark_name}; the repository does not include or generate benchmark data.')
    path = Path(data_path)
    with path.open('r', encoding='utf-8') as stream:
        payload = json.load(stream)
    rows = payload.get('tasks') if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON list or an object containing a 'tasks' list.")
    required_fields = set(BenchmarkTask.__dataclass_fields__)
    tasks = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Task at index {index} must be a JSON object.')
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(f'Task at index {index} is missing fields: {sorted(missing)}')
        tasks.append(BenchmarkTask(**{field: row[field] for field in required_fields}))
    return tasks

def load_rsde_bench_tasks(data_path: Optional[Union[str, Path]]=None) -> List[BenchmarkTask]:
    return _load_tasks(data_path, 'rSDE-Bench')

def load_humaneval_tasks(data_path: Optional[Union[str, Path]]=None) -> List[BenchmarkTask]:
    return _load_tasks(data_path, 'HumanEval')