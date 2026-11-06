"""Bounded throughput probes; benchmark weights are discarded, not research results.

Temporarily suspends this project's supervisor and active training process, then
restores both in finally. Never changes model hyperparameters or system services.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from mavrl_isaac.models import VAE, Representation
    from mavrl_isaac.representation_training import vae_loss, reconstruction_loss
    from mavrl_isaac.training import seed_all

    seed_all(700 + args.index)
    folder = Path(args.out)
    manifest = json.loads((Path(args.dataset) / 'manifest.json').read_text())
    files = [r for r in manifest['files'] if r['path'].startswith('train/')][:16]
    arrays = [np.load(Path(args.dataset) / r['path'])['image'] for r in files]
    if args.worker == 'vae':
        model = VAE(1, 64).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        raw = torch.from_numpy(np.concatenate(arrays))
        iteration = 0
        def step():
            nonlocal iteration
            offset = (iteration * 32) % (len(raw) - 32)
            iteration += 1
            x = raw[offset:offset+32].cuda().float() / 255.
            mask = torch.rand(len(x), device='cuda') < .5
            x[mask] = x[mask].flip(-1)
            recon, mu, logsigma = model(x)
            loss = vae_loss(recon, x, mu, logsigma)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # Preserve the production loop's per-minibatch scalar synchronizations.
            value = loss.item()
            F.mse_loss(recon, x, reduction='sum').item()
            (recon-x).abs().sum().item()
            return 32, value
    else:
        model = Representation().cuda()
        model.encoder.requires_grad_(False)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0001, eps=1e-5)
        cached = []
        for raw in arrays:
            x = torch.from_numpy(raw)
            with torch.no_grad(): mu, _ = model.encoder(x.cuda().float()/255.)
            cached.append((mu.cpu(), x))
        iteration = 0
        def step():
            nonlocal iteration
            mu, raw = cached[iteration % len(cached)]
            iteration += 1
            loss, _, _ = reconstruction_loss(model, mu.cuda(), raw.cuda().float()/255., 10, [1, 1, 0])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            return len(raw), loss.item()
    for _ in range(10): step()
    torch.cuda.synchronize()
    (folder / f'ready_{args.index}').write_text(str(os.getpid()))
    while not (folder / 'go.json').exists(): time.sleep(.02)
    timing = json.loads((folder / 'go.json').read_text())
    while time.monotonic() < timing['start']: time.sleep(.005)
    start = time.monotonic(); count = updates = 0
    while time.monotonic() < timing['end']:
        n, value = step(); count += n; updates += 1
        if not np.isfinite(value): raise FloatingPointError('Nonfinite benchmark loss')
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    result = {'worker': args.worker, 'index': args.index, 'frames': count,
              'updates': updates, 'seconds': elapsed, 'frames_per_second': count/elapsed,
              'torch_peak_mb': torch.cuda.max_memory_allocated()/2**20}
    (folder / f'worker_{args.index}.json').write_text(json.dumps(result, indent=2))


def orchestrate(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    status_path = Path(args.status)
    before = json.loads(status_path.read_text())
    supervisor = before['pid']
    assert str(Path.cwd()) in Path(f'/proc/{supervisor}/cmdline').read_text()
    stopped = []; children = []; logfiles = []; records = []
    started = time.time()
    try:
        os.kill(supervisor, signal.SIGSTOP); stopped.append(supervisor)
        before = json.loads(status_path.read_text())
        child = before.get('child_pid')
        if child and Path(f'/proc/{child}').exists():
            assert str(Path.cwd()) in Path(f'/proc/{child}/cmdline').read_text()
            os.kill(child, signal.SIGSTOP); stopped.append(child)
        gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        assert {int(p.strip()) for p in gpu_pids.splitlines() if p.strip()} <= set(stopped), 'Another GPU job is present'
        (out/'suspended.json').write_text(json.dumps({'pids': stopped, 'original_status': before, 'started_at_unix': started}, indent=2))
        for workload in ([] if args.environments else ['vae', 'lstm']):
            for trial, workers in enumerate([1, 2, 1]):
                folder = out/f'{workload}_{trial}_{workers}'; folder.mkdir()
                group = []
                for index in range(workers):
                    log = (folder/f'worker_{index}.log').open('w'); logfiles.append(log)
                    command = [sys.executable, __file__, '--worker', workload, '--index', str(index), '--dataset', args.dataset, '--out', str(folder)]
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    children.append(proc); group.append(proc)
                deadline = time.monotonic()+120
                while not all((folder/f'ready_{i}').exists() for i in range(workers)):
                    if any(p.poll() is not None for p in group): raise RuntimeError(f'Benchmark worker exited: {folder}')
                    if time.monotonic()>deadline: raise TimeoutError('Benchmark warmup')
                    time.sleep(.1)
                start = time.monotonic()+.2
                (folder/'go.json').write_text(json.dumps({'start': start, 'end': start+args.seconds}))
                samples=[]
                while any(p.poll() is None for p in group):
                    samples.append(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True).strip())
                    if time.monotonic()>start+args.seconds+60: raise TimeoutError('Benchmark measurement')
                    time.sleep(1)
                assert all(p.returncode==0 for p in group), f'Worker failure: {folder}'
                rows=[json.loads((folder/f'worker_{i}.json').read_text()) for i in range(workers)]
                record={'workload':workload,'trial':trial,'workers':workers,'aggregate_frames_per_second':sum(r['frames_per_second'] for r in rows),'worker_results':rows,'gpu_samples_util_memory_power':samples}
                records.append(record); print(json.dumps(record),flush=True)
                (out/'results.json').write_text(json.dumps({'trials':records,'scope':'Representative actual-model training steps on recorded data; excludes full-dataset loading, validation and checkpoint I/O. Suspended original process retains its allocated VRAM. No MPS or hyperparameter changes.'},indent=2))
        for count in args.environments or []:
            log=(out/f'env_{count}.log').open('w'); logfiles.append(log)
            command=[sys.executable,'scripts/benchmark_env.py','--num-envs',str(count),'--steps','500','--warmup','20','--tiled','--policy','artifacts/corrected/final/latest.pt','--out',str(out)]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);children.append(process)
            process.wait(timeout=600)
            if process.returncode:raise RuntimeError(f'Environment benchmark failed: {count}')
            record=json.loads((out/f'benchmark_{count}_tiled.json').read_text())
            records.append(record)
            print('ENV',count,record['transitions_per_second'],record['gpu'],flush=True)
            (out/'results.json').write_text(json.dumps({'trials':records,'scope':'Source-model rollout with actual episode resets, excluding gradient updates. Suspended original process retains GPU memory; per-process memory is reported separately.'},indent=2))
    finally:
        for process in children:
            if process.poll() is None: process.terminate()
        for process in children:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        for log in logfiles: log.close()
        for pid in reversed(stopped):
            try: os.kill(pid, signal.SIGCONT)
            except ProcessLookupError: pass
        (out/'restored.json').write_text(json.dumps({'resumed_pids':stopped,'elapsed_seconds':time.time()-started},indent=2))
        print('RESTORED',stopped,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--dataset',default='artifacts/formal_seed0/dataset')
    parser.add_argument('--status',default='artifacts/formal_seed0/long_run/status.json')
    parser.add_argument('--seconds',type=float,default=20)
    parser.add_argument('--environments',type=int,nargs='+')
    parser.add_argument('--worker',choices=['vae','lstm'])
    parser.add_argument('--index',type=int,default=0)
    options=parser.parse_args()
    if options.worker: worker(options)
    else: orchestrate(options)
