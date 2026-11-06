"""Test render settling under the trained policy without changing production capture.

Each observation is rendered twice at exactly the same kinematic state. The policy
uses the first image. The extra render changes renderer timing, so these navigation
results are diagnostic only and cannot replace the standard evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run import record_run
from mavrl_isaac.training import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--subframes', type=int, default=1)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path('configs/isaac_mavrl.json').read_text())
    cfg.update(seed=200, num_envs=16)
    if args.subframes < 1:
        parser.error('--subframes must be positive')
    cfg['camera']['rt_subframes'] = args.subframes
    record_run(out, cfg, args)
    seed_all(cfg['seed'])
    from mavrl_isaac.renderer import launch, IsaacDepthRenderer
    app = launch()
    status = {'state': 'running', 'captures': 0, 'differing_capture_pairs': 0,
              'scope': 'Diagnostic extra render at unchanged kinematic state; not a production evaluation or a fix.'}
    try:
        from mavrl_isaac.environment import NavigationEnv
        from mavrl_isaac.evaluation import evaluate
        renderer = IsaacDepthRenderer(cfg)
        env = NavigationEnv(cfg, renderer)
        render = renderer.render
        with (out/'capture_pairs.jsonl').open('w', buffering=1) as log:
            def paired_capture(positions, rotations):
                first = render(positions, rotations).copy()
                second = render(positions, rotations).copy()
                delta = np.abs(first.astype(np.int16) - second.astype(np.int16))
                changed = np.flatnonzero((delta > 2).reshape(env.n, -1).any(axis=1))
                status['captures'] += 1
                record = {'capture': status['captures'], 'max_gray_delta': int(delta.max()),
                          'pixels_gt_2': int((delta > 2).sum())}
                if len(changed):
                    status['differing_capture_pairs'] += 1
                    record['environments'] = [
                        {'index': int(i), 'scene_seed': env.scenes[i].seed,
                         'density': env.scenes[i].density, 'step': int(env.steps[i]),
                         'position': positions[i].tolist(), 'rotation': rotations[i].tolist(),
                         'max_gray_delta': int(delta[i].max()),
                         'pixels_gt_2': int((delta[i] > 2).sum())}
                        for i in changed]
                    if status['differing_capture_pairs'] <= 10:
                        np.savez_compressed(out/f"frames_{status['captures']:06d}.npz",
                                            indices=changed, first=first[changed], second=second[changed])
                log.write(json.dumps(record)+'\n')
                if status['captures'] % 100 == 0 or len(changed):
                    (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
                return first
            renderer.render = paired_capture
            evaluate(env, cfg, args.policy, out/'diagnostic_evaluation', args.episodes,
                     ['sparse', 'medium', 'dense'])
        status['state'] = 'completed'
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
        app.close()


if __name__ == '__main__':
    main()
