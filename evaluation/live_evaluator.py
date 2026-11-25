from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GoTGNNConfig
from gotgnn.inference import GoTGNNInference
from gotgnn.model import GoTGNN
from gotgnn.training_data import prepare_record
from train_gotgnn import forward_record

def evaluate(data_path: str, checkpoint: str, split: str='test'):
    records = []
    task_splits = {}
    with open(data_path, encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                record = prepare_record(raw)
            except Exception as exc:
                raise ValueError(f'Invalid history record at line {line_no}: {exc}') from exc
            task_splits.setdefault(record.task_name, set()).add(record.split)
            if record.split == split:
                records.append(record)
    if any((len(splits) != 1 for splits in task_splits.values())):
        raise ValueError('Task leakage across data splits')
    if not records:
        raise ValueError(f'No records for requested split {split!r}')
    cfg = GoTGNNConfig()
    model = GoTGNN(cfg)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload.get('state_dict', payload))
    model.eval()
    infer = GoTGNNInference(model, cfg)
    node_pred, node_true = ([], [])
    action_correct = count_abs_error = 0.0
    with torch.no_grad():
        for record in records:
            yn, yt, nm, _pp, _pt, at, att, ac, act = forward_record(model, infer, record)
            node_pred.extend(yn[nm].tolist())
            node_true.extend(yt[nm].tolist())
            action_correct += float(at.argmax(-1).item() == att.item())
            count_abs_error += float(torch.abs(ac.reshape(-1)[0] - act.reshape(-1)[0]).item())

    def rho(a, b):
        if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return None
        return float(spearmanr(a, b).statistic)
    return {'split': split, 'records': len(records), 'labeled_thought_nodes': len(node_true), 'thought_utility_spearman': rho(node_pred, node_true), 'next_action_accuracy': action_correct / len(records), 'child_count_mae': count_abs_error / len(records), 'note': 'Spearman is used only to measure ranking of thought-node utility, matching the thesis. Candidate-path success is not reported with Spearman.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='Labeled GoTGNN execution-history JSONL')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    parser.add_argument('--output', help='Optional JSON metrics destination')
    args = parser.parse_args()
    result = evaluate(args.data, args.checkpoint, args.split)
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(serialized + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()