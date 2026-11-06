"""Plot preselected held-out tasks from recorded positions; does not run a policy."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from mavrl_isaac.scenes import make_scene


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('root')
parser.add_argument('--normal', default='evaluation_sync')
parser.add_argument('--reset', default='evaluation_sync_reset')
parser.add_argument('--output', default='figures_sync')
args = parser.parse_args()
root = Path(args.root)
out = root / args.output
out.mkdir(parents=True, exist_ok=True)
runs = {}
inputs = {}
for name in [args.normal, args.reset]:
    runs[name] = {}
    for part in ['summary', 'episodes', 'trajectories']:
        path = root / name / f'{part}.json'
        runs[name][part] = json.loads(path.read_text())
        inputs[str(path)] = digest(path)
normal = runs[args.normal]['summary']
reset = runs[args.reset]['summary']
assert normal['sha256'] == reset['sha256']
cfg = normal['config']
assert cfg['scene'] == reset['config']['scene']
assert cfg['_run']['source_sha256']['mavrl_isaac/scenes.py'] == digest('mavrl_isaac/scenes.py'), 'Use the recorded scene generator version'

plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(14, 7))
fig.subplots_adjust(left=.055, right=.985, bottom=.18, top=.78, wspace=.24)
selected = []
colors = ['#0072B2', '#D55E00']
for ax, density in zip(axes, ['sparse', 'medium', 'dense']):
    # Choose by seed order alone, before inspecting outcome or trajectory.
    seed = min(normal['results'][density]['seeds'])
    assert seed == min(reset['results'][density]['seeds'])
    scene = make_scene(seed, cfg['scene'], cfg['simulation']['drone_radius'], density)
    for x, y, radius in scene.cylinders:
        ax.add_patch(Circle((x, y), radius, facecolor='#d9d9d9', edgecolor='#777777', linewidth=.5))
    labels = []
    selection = {'density': density, 'scene_seed': seed, 'outcomes': {}}
    for index, name in enumerate([args.normal, args.reset]):
        record = next(r for r in runs[name]['episodes'] if r['density'] == density and r['scene_seed'] == seed)
        trajectory = sorted((r for r in runs[name]['trajectories'] if r['density'] == density and r['scene_seed'] == seed), key=lambda r: r['time_s'])
        points = np.array([r['position'] for r in trajectory])
        np.testing.assert_allclose(points[0], scene.start, rtol=0, atol=1e-10)
        assert record['num_obstacles'] == len(scene.cylinders)
        np.testing.assert_allclose(record['traversability'], scene.traversability, rtol=0, atol=1e-10)
        # Points only: 1 Hz samples do not reveal the exact 10 Hz curve or terminal position.
        ax.scatter(points[:, 0], points[:, 1], s=15, color=colors[index], marker=['o', 'x'][index], zorder=3)
        ax.scatter(*points[-1, :2], s=65, facecolors='none', edgecolors=colors[index], linewidths=1.4, zorder=4)
        label = ['Memory', 'Reset'][index]
        labels.append(f"{label}: {record['reason']} ({record['duration_s']:.1f}s)")
        selection['outcomes'][name] = {'reason': record['reason'], 'last_plotted_time_s': trajectory[-1]['time_s'], 'terminal_time_s': record['duration_s']}
    ax.scatter(*scene.start[:2], marker='s', color='black', s=48, zorder=5)
    ax.scatter(*scene.goal[:2], marker='*', color='#009E73', edgecolors='black', linewidths=.5, s=150, zorder=5)
    ax.set(title=f'{density.capitalize()}, seed {seed}\n' + '\n'.join(labels), xlabel='World x (m)', ylabel='World y (m)', xlim=(-23, 23), ylim=(-3, 46), aspect='equal')
    ax.spines[['top', 'right']].set_visible(False)
    selected.append(selection)
legend = [Line2D([], [], marker='s', color='black', ls='', label='Start'),
          Line2D([], [], marker='*', color='#009E73', ls='', label='Goal'),
          Line2D([], [], marker='o', color=colors[0], ls='', label='Memory (1 Hz samples)'),
          Line2D([], [], marker='x', color=colors[1], ls='', label='Reset (1 Hz samples)')]
fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .03), ncol=4, frameon=False)
fig.suptitle('Isaac-trained short-run policy: first evaluation seed in each density', y=.98, fontsize=14)
fig.text(.5, .93, 'Hollow circles mark the last recorded sample, not the exact terminal position', ha='center', fontsize=11)
for extension in ['png', 'pdf']:
    fig.savefig(out / f'trajectories.{extension}', dpi=180)
plt.close(fig)
(out / 'trajectories_provenance.json').write_text(json.dumps({
    'selection': 'Smallest evaluation seed in each density; no outcome-based selection.',
    'scope': 'Actual logged positions sampled at 1 Hz; geometry regenerated with the same source/config/seed and checked against episode metadata. No interpolation, inferred terminal position or generated policy path.',
    'tasks': selected, 'inputs_sha256': inputs,
    'scene_generator_sha256': digest('mavrl_isaac/scenes.py'),
}, indent=2) + '\n')
print(json.dumps(selected, indent=2))
