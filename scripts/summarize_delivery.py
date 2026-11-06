"""Export completed three-way results and source hashes; no new GPU experiments."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    queue = Path('artifacts/capture_recovery_20261105')
    if read(queue/'status.json')['state'] != 'experiments_complete':
        raise ValueError('The experiment queue has not completed its paired checks')
    roots = {label: Path(f'artifacts/capture2_{variant}_seed0') for label, variant in
             [('baseline', 'baseline'), ('current_only', 'current'), ('fixed', 'fixed_speed')]}
    density_order = ['sparse', 'medium', 'dense']
    sources = {}
    weights = {}
    metrics = {}
    logs = {}
    for label, root in roots.items():
        if read(root/'long_run/status.json')['state'] != 'completed':
            raise ValueError(f'Incomplete pipeline: {label}')
        validation = read(root/'validation.json')
        required = ['checkpoint_load_action_identity', 'train_validation_scene_disjoint',
                    'finite_losses', 'representation_exactly_frozen', 'separate_critic_present',
                    'evaluation_disjoint_from_logged_training_tasks', 'evaluation_available']
        if not all(validation.get(key) for key in required):
            raise ValueError(f'Missing artifact validation: {label}')
        if not read(root/'memory_figures_full/provenance.json')['exact_replay']:
            raise ValueError(f'Replay failed: {label}')
        report = read(root/'evaluation_full/summary.json')
        checkpoint = root/'final/latest.pt'
        if report['training_timesteps'] != 8000000 or report['sha256'] != digest(checkpoint):
            raise ValueError(f'Evaluation budget or weight differs: {label}')
        episodes = read(root/'evaluation_full/episodes.json')
        tasks = {(r['density'], r['scene_seed']) for r in episodes}
        if len(tasks) != 300 or len(episodes) != 300:
            raise ValueError(f'Expected 300 unique tasks: {label}')
        reset = read(root/'evaluation_full_reset/summary.json')
        metrics[label] = {}
        for density in density_order:
            selected = [row for row in episodes if row['density'] == density]
            counts = Counter(row['reason'] for row in selected)
            saved = report['results'][density]
            if len(selected) != 100 or saved['episodes'] != 100:
                raise ValueError('Expected 100 tasks per density')
            if counts['success']/100 != saved['success_rate'] or counts['collision']/100 != saved['collision_rate']:
                raise ValueError('Summary outcome count differs from episodes')
            mean_speed = float(np.mean([row['mean_speed'] for row in selected]))
            if not np.isclose(mean_speed, saved['mean_speed_all_episodes'], rtol=0, atol=1e-12):
                raise ValueError('Summary speed differs from raw episode means')
            metrics[label][density] = {**{k: v for k, v in saved.items() if k != 'seeds'},
                'counts': dict(counts), 'reset_memory_success_rate': reset['results'][density]['success_rate']}
        logs[label] = [json.loads(line) for line in (root/'final/train.jsonl').read_text().splitlines()]
        for stage, filename in [('initial', 'latest.pt'), ('vae', 'best.pt'), ('lstm', 'best.pt'), ('final', 'latest.pt')]:
            path = root/stage/filename
            weights[f'{label}/{stage}'] = {'path': str(path), 'resolved_path': str(path.resolve()),
                'bytes': path.stat().st_size, 'sha256': digest(path)}
        for relative in ['final/config.json', 'final/train.jsonl', 'evaluation_full/summary.json',
                         'evaluation_full/episodes.json', 'evaluation_full/trajectories.json',
                         'evaluation_full_reset/summary.json', 'evaluation_full_replay/summary.json',
                         'memory_figures_full/provenance.json', 'validation.json', 'REUSED_INPUTS.json']:
            path = root/relative; sources[str(path)] = digest(path)
    comparisons = {key: read(queue/key/'comparison.json') for key in ['memory_comparison', 'speed_comparison']}
    for key in comparisons:
        path = queue/key/'comparison.json'; sources[str(path)] = digest(path)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    result = {'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'script_sha256': digest(__file__), 'roots': {k: str(v) for k, v in roots.items()},
              'metrics': metrics, 'ppo_last_records': {k: v[-1] for k, v in logs.items()},
              'memory_pairs': comparisons['memory_comparison']['paired_counts'],
              'speed_pairs': comparisons['speed_comparison']['paired'],
              'weights': weights, 'source_sha256': sources,
              'scope': 'One training seed, 100 paired scenes per density, three completed 8m PPO runs. '
                       'Current-only retains LSTM; reset-memory tests are inference diagnostics. '
                       'Fixed uses one project-selected 2 m/s target. No multi-seed, equal-speed Pareto, '
                       'original AvoidBench numerical reproduction, or real-flight claim.'}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'WEIGHTS.json').write_text(json.dumps(weights, indent=2)+'\n')
    lines = ['# 三组完成实验的结果汇总', '', result['scope'], '',
             '| 配置 | 密度 | 到达/100 | 碰撞/100 | 平均速度m/s | 清空记忆到达率 |',
             '|---|---|---:|---:|---:|---:|']
    for label, densities in metrics.items():
        for density, row in densities.items():
            lines.append(f"| {label} | {density} | {row['counts'].get('success',0)} | {row['counts'].get('collision',0)} | {row['mean_speed_all_episodes']:.3f} | {row['reset_memory_success_rate']:.0%} |")
    lines += ['', '权重及SHA256：`WEIGHTS.json`。原始统计、配对成败计数、源文件摘要：`summary.json`。', '',
              '每组速度先按episode求均值，再对100个任务等权平均。到达率区间为场景任务上的95% Wilson区间，不能表示跨训练种子的误差。', '',
              '结论须区分历史状态使用、过去帧重建监督和目标速度奖励三个问题；不以训练完成代替消融收益成立。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    colors = ['#0072B2', '#D55E00', '#009E73']; labels = ['Past + current / adaptive', 'Current-only / adaptive', 'Past + current / target 2 m/s']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    x = np.arange(3)
    for index, (label, color, title) in enumerate(zip(roots, colors, labels)):
        rows = [metrics[label][d] for d in density_order]
        rates = np.array([r['success_rate'] for r in rows]); ci = np.array([r['success_95ci'] for r in rows]).T
        errors = np.array([rates-ci[0], ci[1]-rates])
        if np.any(errors < -1e-12): raise ValueError('Invalid arrival interval')
        offset = (index-1)*.25
        axes[0].bar(x+offset, rates, .24, color=color, label=title)
        axes[0].errorbar(x+offset, rates, yerr=np.maximum(errors, 0), fmt='none', ecolor='black', capsize=3)
        axes[1].plot(x, [r['mean_speed_all_episodes'] for r in rows], marker=['o', 's', '^'][index], color=color, label=title)
    axes[0].set(ylabel='Arrival fraction (95% Wilson interval)', ylim=(0, 1.05))
    axes[1].set(ylabel='All-episode mean horizontal speed (m/s)', ylim=(0, 2.5))
    axes[0].legend(fontsize=7, loc='lower left')
    for ax in axes:
        ax.set_xticks(x, ['Sparse\nn=100', 'Medium\nn=100', 'Dense\nn=100'])
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('MAVRL in Isaac Sim cylinders: one training seed, paired held-out tasks')
    for extension in ['png', 'pdf', 'svg']: fig.savefig(out/f'comparison.{extension}', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    for (label, rows), color, title in zip(logs.items(), colors, labels):
        ax.plot([r['timesteps']/1e6 for r in rows], [r['success_rate'] for r in rows], color=color, label=title, linewidth=1)
    ax.axvline(1.216, color='gray', linestyle='--', linewidth=.8)
    ax.set(xlabel='Environment transitions (millions)', ylabel='Training episode arrival fraction',
           ylim=(0, 1.05), title='Raw rollout means; curriculum switches at 1.216M transitions')
    ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    for extension in ['png', 'pdf']: fig.savefig(out/f'ppo_training.{extension}', dpi=180)
    plt.close(fig)
    (out/'figure_provenance.json').write_text(json.dumps({'source_sha256': sources,
        'summary_sha256': digest(out/'summary.json'), 'script_sha256': digest(__file__),
        'transformations': 'Raw rollout means without smoothing. Navigation arrival fractions and Wilson intervals from saved independent evaluations; speed is all-episode mean. No rescaling or selecting best checkpoints.',
        'software': {'matplotlib': matplotlib.__version__, 'numpy': np.__version__}}, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'verified_completed_runs': list(roots)}, indent=2))


if __name__ == '__main__':
    main()
