import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import importlib
from config import GOT_12_CONFIGS, PipelineConfig
from gotgnn.model import GoTGNN
from flare_gnn.model import FlareGNN
from pipeline.orchestrator import GCodeOrchestrator

def load_callable(spec: str):
    module_name, sep, name = spec.partition(':')
    if not sep:
        raise ValueError('Provider must use module:callable notation')
    provider = getattr(importlib.import_module(module_name), name)
    result = provider()
    if not isinstance(result, dict) or not callable(result.get('llm_fn')):
        raise ValueError("Provider factory must return {'llm_fn': callable, 'custom_tests': [(name, callable), ...]}")
    tests = result.get('custom_tests')
    if not isinstance(tests, list) or not tests:
        raise ValueError('A real custom test suite is required')
    return (result['llm_fn'], tests)

def discover_tasks(tasks_dir: Path) -> List[str]:
    tasks = []
    for category in ['website', 'game']:
        category_dir = tasks_dir / category / 'EvoMAC'
        if category_dir.exists():
            for task_dir in category_dir.iterdir():
                if task_dir.is_dir():
                    tasks.append(task_dir.name)
    return sorted(tasks)

def load_requirement(tasks_dir: Path, task_id: str) -> str:
    for category in ['website', 'game']:
        task_dir = tasks_dir / category / 'EvoMAC' / task_id
        if task_dir.exists():
            for prompt_file in task_dir.glob('*.prompt'):
                return prompt_file.read_text(encoding='utf-8').strip()
    requirement_file = tasks_dir / f'{task_id}.txt'
    if requirement_file.exists():
        return requirement_file.read_text(encoding='utf-8').strip()
    requirement_file = tasks_dir / task_id / 'requirement.txt'
    if requirement_file.exists():
        return requirement_file.read_text(encoding='utf-8').strip()
    raise FileNotFoundError(f'Requirement file not found for task {task_id} in {tasks_dir}')

def run_task_with_all_configs(task_id: str, requirement: str, split: str, orchestrator: GCodeOrchestrator, output_dir: Path, max_iterations: int=5) -> Dict[str, Any]:
    task_results = {'task_id': task_id, 'split': split, 'configs': {}}
    for config_id in sorted(GOT_12_CONFIGS.keys()):
        print(f"\n{'=' * 60}")
        print(f'Task: {task_id} | Config: {config_id} | Split: {split}')
        print(f"{'=' * 60}")
        try:
            history_path = output_dir / 'history' / split / f'{task_id}_{config_id}.jsonl'
            history_path.parent.mkdir(parents=True, exist_ok=True)
            result = orchestrator.run_cycle(requirement=requirement, subtask_id=task_id, got_config_id=config_id, max_iterations=max_iterations, history_split=split, history_path=str(history_path))
            task_results['configs'][config_id] = {'status': 'success', 'history_path': str(history_path), 'iterations': result.get('iterations', 0), 'final_status': result.get('final_status', 'unknown')}
            print(f"✓ Config {config_id} completed: {result.get('final_status', 'unknown')}")
        except Exception as e:
            print(f'✗ Config {config_id} failed: {e}')
            task_results['configs'][config_id] = {'status': 'error', 'error': str(e)}
    return task_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--llm-provider', required=True)
    parser.add_argument('--got-checkpoint')
    parser.add_argument('--flare-checkpoint')
    parser.add_argument('--max-iterations', type=int, default=5)
    parser.add_argument('--train-ratio', type=float, default=0.73)
    parser.add_argument('--val-ratio', type=float, default=0.11)
    args = parser.parse_args()
    tasks_dir = Path(args.tasks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = discover_tasks(tasks_dir)
    n_total = len(all_tasks)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    train_tasks = all_tasks[:n_train]
    val_tasks = all_tasks[n_train:n_train + n_val]
    test_tasks = all_tasks[n_train + n_val:]
    llm_fn, custom_tests = load_callable(args.llm_provider)
    if args.got_checkpoint and Path(args.got_checkpoint).is_file():
        import torch
        checkpoint = torch.load(args.got_checkpoint, map_location='cpu', weights_only=False)
        got = GoTGNN()
        got.load_state_dict(checkpoint.get('state_dict', checkpoint))
        got.eval()
        print(f'Loaded GoTGNN checkpoint from {args.got_checkpoint}')
    else:
        from gotgnn.model import GoTGNN
        got = GoTGNN()
        print('Using untrained GoTGNN (random initialization)')
    if args.flare_checkpoint and Path(args.flare_checkpoint).is_file():
        import torch
        checkpoint = torch.load(args.flare_checkpoint, map_location='cpu', weights_only=False)
        flare = FlareGNN()
        flare.load_state_dict(checkpoint.get('state_dict', checkpoint))
        flare.eval()
        print(f'Loaded Flare-GNN checkpoint from {args.flare_checkpoint}')
    else:
        from flare_gnn.model import FlareGNN
        flare = FlareGNN()
        print('Using untrained Flare-GNN (random initialization)')
    orchestrator = GCodeOrchestrator(gotgnn=got, flare_gnn=flare, config=PipelineConfig(max_iterations=args.max_iterations), llm_fn=llm_fn, custom_tests=custom_tests)
    all_results = {'train': {}, 'validation': {}, 'test': {}}
    print('\n' + '=' * 60)
    print('GENERATING G-CODE TRAINING CORPUS')
    print('=' * 60)
    print(f'Total tasks: {n_total}')
    print(f'Train tasks: {len(train_tasks)}')
    print(f'Validation tasks: {len(val_tasks)}')
    print(f'Test tasks: {len(test_tasks)}')
    print(f'Total configurations per task: {len(GOT_12_CONFIGS)}')
    print(f'Expected total traces: {n_total} × {len(GOT_12_CONFIGS)} = {n_total * len(GOT_12_CONFIGS)}')
    print('=' * 60)
    for task_id in train_tasks:
        try:
            requirement = load_requirement(tasks_dir, task_id)
            result = run_task_with_all_configs(task_id, requirement, 'train', orchestrator, output_dir, args.max_iterations)
            all_results['train'][task_id] = result
        except Exception as e:
            print(f'✗ Train task {task_id} failed: {e}')
            all_results['train'][task_id] = {'status': 'error', 'error': str(e)}
    for task_id in val_tasks:
        try:
            requirement = load_requirement(tasks_dir, task_id)
            result = run_task_with_all_configs(task_id, requirement, 'validation', orchestrator, output_dir, args.max_iterations)
            all_results['validation'][task_id] = result
        except Exception as e:
            print(f'✗ Validation task {task_id} failed: {e}')
            all_results['validation'][task_id] = {'status': 'error', 'error': str(e)}
    for task_id in test_tasks:
        try:
            requirement = load_requirement(tasks_dir, task_id)
            result = run_task_with_all_configs(task_id, requirement, 'test', orchestrator, output_dir, args.max_iterations)
            all_results['test'][task_id] = result
        except Exception as e:
            print(f'✗ Test task {task_id} failed: {e}')
            all_results['test'][task_id] = {'status': 'error', 'error': str(e)}
    report_path = output_dir / 'corpus_generation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print('\n' + '=' * 60)
    print('CORPUS GENERATION COMPLETE')
    print('=' * 60)
    print(f'Report saved to: {report_path}')
    print(f"History files in: {output_dir / 'history'}")
    print('\nMerging history files into training JSONL...')
    for split in ['train', 'validation', 'test']:
        split_dir = output_dir / 'history' / split
        merged_path = output_dir / f'{split}_history.jsonl'
        if not split_dir.exists():
            print(f'Warning: No history directory for split {split}')
            continue
        all_records = []
        for history_file in split_dir.glob('*.jsonl'):
            with open(history_file, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))
        with open(merged_path, 'w', encoding='utf-8') as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'✓ {split}: {len(all_records)} records -> {merged_path}')
    print('\n✓ Ready for GoTGNN training with:')
    print(f'  python train_gotgnn.py --data {output_dir}/train_history.jsonl --epochs 20')
if __name__ == '__main__':
    main()