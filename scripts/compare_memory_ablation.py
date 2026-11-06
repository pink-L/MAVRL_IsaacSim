"""Compare completed, equally budgeted past/current and current-only training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_counts(left, right):
    a = {(r['density'], r['scene_seed']): r for r in left}
    b = {(r['density'], r['scene_seed']): r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys():
        raise ValueError('Evaluation tasks must be unique and paired')
    output = {}
    for task, row in a.items():
        counts = output.setdefault(task[0], {'both_success': 0, 'past_current_only_success': 0, 'current_only_success': 0, 'both_fail': 0})
        x = row['reason'] == 'success'; y = b[task]['reason'] == 'success'
        key = 'both_success' if x and y else 'past_current_only_success' if x else 'current_only_success' if y else 'both_fail'
        counts[key] += 1
    return output


def compare(baseline, variant, out):
    roots = [Path(baseline), Path(variant)]; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sources = {}
    for root in roots:
        if read(root/'long_run/status.json')['state'] != 'completed':
            raise ValueError(f'Pipeline has not completed: {root}')
    for relative in ['dataset/manifest.json', 'vae/best.pt', 'initial/latest.pt']:
        hashes = [digest(root/relative) for root in roots]
        if hashes[0] != hashes[1]: raise ValueError(f'Paired representation ablation must share {relative}')
        for root, value in zip(roots, hashes): sources[str(root/relative)] = value
    for stage in ['lstm', 'final']:
        configs = [read(root/stage/'config.json') for root in roots]
        for index, config in enumerate(configs):
            expected = [1, 1, 0] if index == 0 else [0, 1, 0]
            if config['lstm']['reconstruction_members'] != expected: raise ValueError('Unexpected reconstruction objective')
        reduced = []
        for config in configs:
            config = copy.deepcopy(config); config.pop('_run', None)
            config['lstm'].pop('reconstruction_members'); reduced.append(config)
        if reduced[0] != reduced[1]: raise ValueError(f'Other {stage} training settings differ')
    reports = [read(root/'evaluation_full/summary.json') for root in roots]
    if reports[0]['training_timesteps'] != reports[1]['training_timesteps']:
        raise ValueError('Unequal PPO training budgets')
    episodes = [read(root/'evaluation_full/episodes.json') for root in roots]
    pairs = paired_counts(*episodes)
    for density in pairs:
        if reports[0]['results'][density]['seeds'] != reports[1]['results'][density]['seeds']:
            raise ValueError('Evaluation seed order differs')
    for root in roots:
        for relative in ['lstm/config.json', 'final/config.json', 'evaluation_full/summary.json', 'evaluation_full/episodes.json']:
            sources[str(root/relative)] = digest(root/relative)
    result = {'baseline': str(roots[0]), 'current_only': str(roots[1]), 'paired_counts': pairs,
              'results': {'past_current': reports[0]['results'], 'current_only': reports[1]['results']},
              'source_sha256': sources,
              'scope': 'One shared-data/shared-VAE training seed with two reconstruction objectives and equally budgeted PPO retraining. Current-only still uses an LSTM. Descriptive paired results; no multi-seed or statistical-significance claim.'}
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Paired reconstruction-objective ablation', '', result['scope'], '',
             '| Density | Past + current arrival | Current-only arrival | Past/current-only wins | Current-only wins |',
             '|---|---:|---:|---:|---:|']
    for density, counts in pairs.items():
        left, right = (r['results'][density] for r in reports)
        lines.append(f"| {density} | {left['success_rate']:.3f} | {right['success_rate']:.3f} | {counts['past_current_only_success']} | {counts['current_only_success']} |")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['paired_counts'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline'); parser.add_argument('current_only'); parser.add_argument('--out', required=True)
    args = parser.parse_args(); compare(args.baseline, args.current_only, args.out)
