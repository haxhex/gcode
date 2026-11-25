from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch.optim import AdamW
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import FlareGNNConfig
from scpg.builder import SCPGBuilder
from flare_gnn.model import FlareGNN, FlareGNNLoss

@dataclass
class Record:
    graph: object
    faulty_lines: set[int]
    split: str
    dataset: str
    phase: str
    project_id: str
    program_id: str

    @property
    def project_key(self) -> str:
        return f'{self.dataset}:{self.project_id}'

def read_records(path: str) -> list[Record]:
    builder = SCPGBuilder()
    project_splits: dict[str, set[str]] = defaultdict(set)
    records: list[Record] = []
    with open(path, encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                required = ('dataset', 'phase', 'project_id', 'split', 'program_id', 'source_code', 'faulty_lines')
                missing = [key for key in required if key not in row]
                if missing:
                    raise ValueError(f"missing required field(s): {', '.join(missing)}")
                dataset = str(row['dataset']).lower()
                phase = str(row['phase']).lower()
                if dataset not in {'pytracebugs', 'bugsinpy'}:
                    raise ValueError('dataset must be pytracebugs or bugsinpy')
                if phase not in {'pre', 'post'}:
                    raise ValueError('phase must be pre or post')
                if dataset == 'pytracebugs' and phase != 'pre':
                    raise ValueError('PyTraceBugs records are used in pre-test mode')
                if row['split'] not in {'train', 'validation', 'test'}:
                    raise ValueError('split must be train, validation, or test')
                faulty_lines = {int(value) for value in row['faulty_lines']}
                if not faulty_lines:
                    raise ValueError('faulty_lines must contain independently verified fault locations')
                test_results = None
                if phase == 'post':
                    test_results = row.get('test_results')
                    if not isinstance(test_results, dict) or not all((key in test_results for key in ('total_passed', 'total_failed', 'failing_lines'))):
                        raise ValueError('post-test rows require measured test_results and failing_lines')
                    if int(test_results['total_failed']) < 1:
                        raise ValueError('post-test rows require at least one measured failing test')
                    test_results = {**test_results, 'failing_lines': [int(value) for value in test_results['failing_lines']]}
                graph = builder.build_from_source(row['source_code'], code_id=str(row['program_id']), phase_mask=int(phase == 'post'), test_results=test_results)
                if not graph.nodes or any((node.stmt_type == 'RawStatement' for node in graph.nodes.values())):
                    raise ValueError('source must parse as Python; parser fallback is not accepted for training')
                if not faulty_lines & {node.line_no for node in graph.nodes.values()}:
                    raise ValueError('faulty_lines do not point to parsed statement nodes')
                project_id = str(row['project_id'])
                project_key = f'{dataset}:{project_id}'
                project_splits[project_key].add(str(row['split']))
                records.append(Record(graph=graph, faulty_lines=faulty_lines, split=str(row['split']), dataset=dataset, phase=phase, project_id=project_id, program_id=str(row['program_id'])))
            except Exception as exc:
                raise ValueError(f'Invalid Flare-GNN record at line {line_no}: {exc}') from exc
    leakage = {project: splits for project, splits in project_splits.items() if len(splits) != 1}
    if leakage:
        raise ValueError(f'Project leakage across splits: {leakage}')
    for split in ('train', 'validation', 'test'):
        if not any((record.split == split for record in records)):
            raise ValueError(f'No records supplied for the {split} split')
    train_pytrace = [r for r in records if r.split == 'train' and r.dataset == 'pytracebugs' and (r.phase == 'pre')]
    train_bugs_post = [r for r in records if r.split == 'train' and r.dataset == 'bugsinpy' and (r.phase == 'post')]
    if not train_pytrace or not train_bugs_post:
        raise ValueError('Training needs PyTraceBugs pre-test and BugsInPy post-test records for 70/30 batches')
    return records

def tensors(record: Record, device: torch.device):
    graph = record.graph
    ids = sorted(graph.nodes)
    index = {sid: i for i, sid in enumerate(ids)}
    x_sem = torch.tensor(np.stack([graph.nodes[s].semantic_feat for s in ids]), dtype=torch.float32, device=device)
    x_stat = torch.tensor(np.stack([np.concatenate([graph.nodes[s].static_feat, graph.nodes[s].dynamic_feat]) if record.phase == 'post' else graph.nodes[s].static_feat for s in ids]), dtype=torch.float32, device=device)
    edges = [(index[e.src], index[e.dst], e.edge_type.value) for e in graph.edges if e.src in index and e.dst in index]
    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long, device=device) if edges else torch.empty((2, 0), dtype=torch.long, device=device)
    edge_type = torch.tensor([e[2] for e in edges], dtype=torch.long, device=device) if edges else torch.empty((0,), dtype=torch.long, device=device)
    method_map = {m: [index[s] for s in statements if s in index] for m, statements in graph.methods.items()}
    methods = [m for m, statements in method_map.items() if statements]
    y_node = torch.tensor([float(graph.nodes[s].line_no in record.faulty_lines) for s in ids], dtype=torch.float32, device=device)
    y_method = torch.tensor([float(any((graph.nodes[s].line_no in record.faulty_lines for s in graph.methods[m]))) for m in methods], dtype=torch.float32, device=device)
    return (x_sem, x_stat, edge_index, edge_type, method_map, methods, y_node, y_method)

def evaluate(model: FlareGNN, records: Iterable[Record], criterion: FlareGNNLoss, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for record in records:
            x_sem, x_stat, edge_index, edge_type, method_map, _, y_node, y_method = tensors(record, device)
            pred_node, pred_method, _, _ = model(x_sem, x_stat, edge_index, edge_type, method_map)
            loss, _ = criterion(pred_node, y_node, pred_method, y_method, method_map)
            losses.append(float(loss.item()))
    if not losses:
        raise ValueError('Validation fold contains no records')
    return float(np.mean(losses))

def mixed_batches(records: list[Record], projects: set[str], batch_size: int, rng: random.Random):
    pytrace = [r for r in records if r.project_key in projects and r.dataset == 'pytracebugs' and (r.phase == 'pre')]
    bugs_post = [r for r in records if r.project_key in projects and r.dataset == 'bugsinpy' and (r.phase == 'post')]
    if not pytrace or not bugs_post:
        raise ValueError('Each training fold must contain both PyTraceBugs pre-test and BugsInPy post-test projects')
    if batch_size < 10 or batch_size % 10:
        raise ValueError('batch_size must be a positive multiple of 10 so each batch can preserve a 70/30 ratio')
    pytrace_count = batch_size * 7 // 10
    bugs_count = batch_size - pytrace_count
    steps = max(1, math.ceil((len(pytrace) + len(bugs_post)) / batch_size))
    for _ in range(steps):
        yield ([rng.choice(pytrace) for _ in range(pytrace_count)] + [rng.choice(bugs_post) for _ in range(bugs_count)])

def train_one(records: list[Record], train_projects: set[str], validation: list[Record], epochs: int, batch_size: int, seed: int, save_path: Path, device: torch.device, cfg: FlareGNNConfig) -> float:
    model = FlareGNN(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = FlareGNNLoss(cfg)
    rng = random.Random(seed)
    best = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in mixed_batches(records, train_projects, batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for record in batch:
                x_sem, x_stat, edge_index, edge_type, method_map, _, y_node, y_method = tensors(record, device)
                pred_node, pred_method, _, _ = model(x_sem, x_stat, edge_index, edge_type, method_map)
                loss, _ = criterion(pred_node, y_node, pred_method, y_method, method_map)
                batch_losses.append(loss)
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            optimizer.step()
            epoch_losses.append(float(batch_loss.detach().item()))
        val_loss = evaluate(model, validation, criterion, device)
        print(f'epoch {epoch}/{epochs}: train_loss={np.mean(epoch_losses):.6f} validation_loss={val_loss:.6f}')
        if val_loss < best:
            best = val_loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': model.state_dict(), 'config': cfg.__dict__}, save_path)
    return best

def stratified_project_folds(records: list[Record], folds: int, seed: int) -> list[set[str]]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.split == 'train':
            by_dataset[record.dataset].append(record.project_key)
    fold_projects = [set() for _ in range(folds)]
    for dataset, project_keys in sorted(by_dataset.items()):
        unique = sorted(set(project_keys))
        if len(unique) < folds:
            raise ValueError(f'{folds}-fold project CV needs at least {folds} training projects in {dataset}; found {len(unique)}')
        rng.shuffle(unique)
        for index, project_key in enumerate(unique):
            fold_projects[index % folds].add(project_key)
    return fold_projects

def train(data_path: str, epochs: int, lr: float, batch_size: int, folds: int, seed: int, save_path: str):
    records = read_records(data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = FlareGNNConfig(lr=lr)
    output = Path(save_path)
    heldout_folds = stratified_project_folds(records, folds, seed)
    train_records = [r for r in records if r.split == 'train']
    for fold_index, heldout_projects in enumerate(heldout_folds, 1):
        fold_train_projects = {r.project_key for r in train_records} - heldout_projects
        fold_validation = [r for r in train_records if r.project_key in heldout_projects]
        fold_path = output.with_name(f"{output.stem}_fold_{fold_index}{output.suffix or '.pt'}")
        score = train_one(records, fold_train_projects, fold_validation, epochs, batch_size, seed + fold_index, fold_path, device, cfg)
        print(f'fold {fold_index}/{folds}: project-held-out validation_loss={score:.6f}')
    final_train_projects = {r.project_key for r in records if r.split == 'train'}
    final_validation = [r for r in records if r.split == 'validation']
    score = train_one(records, final_train_projects, final_validation, epochs, batch_size, seed + folds + 1, output, device, cfg)
    print(f'Saved final checkpoint {output} (best validation_loss={score:.6f}, device={device})')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='JSONL with real labeled PyTraceBugs and BugsInPy records')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=10, help='multiple of 10 for exact 70/30 source batches')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', required=True, help='Final checkpoint path; fold checkpoints are saved beside it')
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.folds < 2:
        parser.error('epochs and lr must be positive and folds must be at least 2')
    train(args.data, args.epochs, args.lr, args.batch_size, args.folds, args.seed, args.output)
if __name__ == '__main__':
    main()