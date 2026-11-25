import argparse
import json
import os
import sys
from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import GoTGNNConfig
from gotgnn.inference import GoTGNNInference
from gotgnn.loss import GoTMultiTaskLoss
from gotgnn.model import GoTGNN
from gotgnn.training_data import PreparedGraphRecord, TextEmbeddingEngine, prepare_record

def read_records(path):
    embedder = TextEmbeddingEngine(dim=384)
    records = []
    task_splits = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                item = prepare_record(raw, embedder)
            except Exception as exc:
                raise ValueError(f'Invalid GoTGNN history record at line {line_number}: {exc}') from exc
            task_splits[item.task_name].add(item.split)
            records.append(item)
    leakage = {task: splits for task, splits in task_splits.items() if len(splits) != 1}
    if leakage:
        raise ValueError(f'Task leakage across data splits: {leakage}')
    if not records:
        raise ValueError('No labeled execution-history records found')
    return records

def forward_record(model, infer, record: PreparedGraphRecord):
    graph = record.graph
    node_ids, _, x_sem, x_abs, x_qual, node_types, edge_index, edge_type, dist = infer._prepare_tensors(graph)
    y_node, p_path, a_type, a_count, _ = model(x_sem=x_sem, x_abs=x_abs, x_qual=x_qual, node_types=node_types, edge_index=edge_index, edge_type=edge_type, dist_matrix=dist, candidate_paths=record.candidate_paths)
    target_node = record.node_targets.to(y_node.device)
    node_mask = record.node_mask.to(y_node.device)
    target_path = record.path_targets.to(y_node.device)
    return (y_node, target_node, node_mask, p_path, target_path, a_type, record.action_type_target.to(y_node.device), a_count, record.child_count_target.to(y_node.device))

def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

def train(data_path, epochs=100, lr=0.001, save_path='checkpoints/gotgnn.pt', warmup_epochs=5, patience=10):
    records = read_records(data_path)
    train_records = [r for r in records if r.split == 'train']
    val_records = [r for r in records if r.split == 'validation']
    if not train_records:
        raise ValueError('The history file contains no train records')
    cfg = GoTGNNConfig(lr=lr)
    model = GoTGNN(cfg)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = GoTMultiTaskLoss(cfg)
    infer = GoTGNNInference(model, cfg)
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs, epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-06)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for record in train_records:
            optimizer.zero_grad()
            args = forward_record(model, infer, record)
            loss, _ = criterion(*args)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item())
        avg = total / len(train_records)
        print(f'Epoch {epoch}/{epochs} | train loss={avg:.4f}')
        if val_records:
            model.eval()
            with torch.no_grad():
                val_total = 0.0
                for record in val_records:
                    val_total += float(criterion(*forward_record(model, infer, record))[0].item())
            val_avg = val_total / len(val_records)
            print(f'Epoch {epoch}/{epochs} | validation loss={val_avg:.4f}')
            if val_avg < best_val_loss:
                best_val_loss = val_avg
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print('Loaded best model from early stopping')
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'action_types': ['IDEA', 'CRITIQUE', 'REFINE', 'AGGREGATE_SIMILAR', 'AGGREGATE_TASK']}, save_path)
    print(f'Saved checkpoint: {save_path}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GoTGNN from execution-history labels')
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', default='checkpoints/gotgnn.pt')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    args = parser.parse_args()
    train(args.data, epochs=args.epochs, lr=args.lr, save_path=args.save, warmup_epochs=args.warmup_epochs, patience=args.patience)