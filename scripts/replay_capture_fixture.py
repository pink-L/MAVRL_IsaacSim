"""Rebuild the saved anomalous state in a fresh renderer and compare its images."""
import argparse
import json
from pathlib import Path

import numpy as np
from run import record_run

p = argparse.ArgumentParser()
p.add_argument('--probe', default='artifacts/formal_seed0/policy_capture_probe')
p.add_argument('--out', required=True)
a = p.parse_args()
source, out = Path(a.probe), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
records = [json.loads(s) for s in (source/'capture_pairs.jsonl').read_text().splitlines()]
record = next(x for x in records if x['pixels_gt_2'])
frames = np.load(source/f"frames_{record['capture']:06d}.npz")
cfg = json.loads((source/'config.json').read_text()); cfg.pop('_run', None)
record_run(out, cfg, a)
from mavrl_isaac.renderer import launch, IsaacDepthRenderer
from mavrl_isaac.scenes import make_scene
app = launch()
try:
    renderer = IsaacDepthRenderer(cfg)
    states = {x['index']: x for x in record['environments']}
    assert len(states) == cfg['num_envs'], 'Fixture requires recorded poses for every camera'
    positions = np.array([states[i]['position'] for i in range(cfg['num_envs'])])
    rotations = np.array([states[i]['rotation'] for i in range(cfg['num_envs'])])
    for i, state in states.items():
        renderer.set_scene(i, make_scene(state['scene_seed'], cfg['scene'],
                           cfg['simulation']['drone_radius'], state['density']))
    rows = []; previous = None
    for step in range(10):
        image = renderer.render(positions, rotations)
        row = {'step': step}
        for name in ['first', 'second']:
            diff = np.abs(image[frames['indices']].astype(np.int16) - frames[name].astype(np.int16))
            row[name] = {'mae_gray': float(diff.mean()), 'pixels_gt_2': int((diff > 2).sum()),
                         'max_gray_delta': int(diff.max())}
        row['equal_previous'] = previous is not None and np.array_equal(previous, image)
        rows.append(row); previous = image.copy()
    np.savez_compressed(out/'reference.npz', depth=previous)
    import carb
    settings = carb.settings.get_settings()
    result = {'source_capture': record['capture'], 'comparisons': rows,
              'settings': {k: settings.get(k) for k in [
                  '/app/asyncRendering', '/app/asyncRenderingLowLatency',
                  '/exts/isaacsim.core.throttling/enable_async',
                  '/omni/replicator/asyncRendering', '/omni/replicator/captureOnPlay']},
              'scope': 'Fresh static-state renderer reference; diagnoses anomalous depth only.'}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
finally:
    app.close()
