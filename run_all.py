from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import torch
from config import GoTGNNConfig, FlareGNNConfig, PipelineConfig
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
        raise ValueError('A real custom test suite is required; the built-in module-load smoke check is not evaluation')
    return (result['llm_fn'], tests)

def load_model(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requirement', required=True, help='Path to a real project requirement text file')
    parser.add_argument('--provider', required=True, help='Python provider factory as module:callable')
    parser.add_argument('--got-checkpoint', required=True)
    parser.add_argument('--flare-checkpoint', required=True)
    parser.add_argument('--config', choices=[f'C{i}' for i in range(1, 13)], default='C6')
    parser.add_argument('--output', required=True, help='Directory for generated project files and measured run report')
    parser.add_argument('--max-iterations', type=int, default=5)
    args = parser.parse_args()
    requirement_path = Path(args.requirement)
    requirement = requirement_path.read_text(encoding='utf-8').strip()
    if not requirement:
        parser.error('requirement file is empty')
    for checkpoint in (args.got_checkpoint, args.flare_checkpoint):
        if not Path(checkpoint).is_file():
            parser.error(f'trained checkpoint does not exist: {checkpoint}')
    llm_fn, tests = load_callable(args.provider)
    got = load_model(GoTGNN(GoTGNNConfig()), args.got_checkpoint)
    flare = load_model(FlareGNN(FlareGNNConfig()), args.flare_checkpoint)
    orchestrator = GCodeOrchestrator(got, flare, PipelineConfig(max_iterations=args.max_iterations), llm_fn, tests)
    result = orchestrator.run_project(requirement, graph_config_id=args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    try:
        project = json.loads(result['project_files_json'])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError('The orchestration result did not contain valid integrated project files') from exc
    for file in project.get('files', []):
        target = (out / file['path']).resolve()
        if out.resolve() not in target.parents:
            raise ValueError(f"Generated path escapes output directory: {file['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file['content'], encoding='utf-8')
    report = out / 'run_report.json'
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(f'Run report: {report}')
if __name__ == '__main__':
    main()