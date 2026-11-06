"""Project-local overnight queue and a scheduled, evidence-based progress report.

GPU work remains serial. Reports are local files, not external notifications.
09:00 is a reporting time, not an instruction to interrupt training.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ZONE = ZoneInfo('Asia/Shanghai')


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def latest_record(path):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def snapshot_run(root):
    root = Path(root)
    return {'root': str(root), 'status': read_json(root/'long_run/status.json'),
            'training': {stage: latest_record(root/stage/'train.jsonl') for stage in ['vae', 'lstm', 'final']},
            'evaluation': read_json(root/'evaluation_full/summary.json')}


def write_report(plan, folder, filename):
    timestamp = datetime.now(ZONE).isoformat()
    variants = ['baseline', 'current_only'] + (['fixed_speed'] if 'fixed_speed' in plan else [])
    snapshots = {name: snapshot_run(plan[name]) for name in variants}
    titles = {'baseline': '主基线：过去＋当前帧重建', 'current_only': '重训消融：仅当前帧重建',
              'fixed_speed': '速度对照：过去＋当前帧重建，目标速度奖励'}
    data = {'generated_at': timestamp, 'report_at': plan['report_at'], 'snapshots': snapshots,
            'scope': 'Actual files only. Missing evaluations are unavailable, not zero. A current-only LSTM is retrained; it is not hidden-state reset.'}
    if plan.get('queue_status'):
        data['queue'] = read_json(plan['queue_status'])
    if plan.get('comparison_root'):
        data['comparisons'] = {name: read_json(Path(plan['comparison_root'])/name/'comparison.json')
                               for name in ['memory_comparison', 'speed_comparison']}
    if plan.get('delivery_status'):
        data['delivery'] = read_json(plan['delivery_status'])
    lines = ['# MAVRL 夜间阶段报告', '', f'生成时间：{timestamp}',
             '', '北京时间09:00为阶段报告时间；按照用户明确选择，之后继续执行训练。', '']
    if data.get('queue'):
        queue = data['queue']
        lines += [f"训练队列：`{queue.get('state')}`；阶段：`{queue.get('stage')}`。", '']
        if queue.get('error'): lines += [f"队列错误：{queue['error']}", '']
    for label, run in snapshots.items():
        state = run['status'] or {'state': 'not_started'}
        title = titles[label]
        lines += [f'## {title}', '', f"目录：`{run['root']}`", f"状态：`{state.get('state')}`；阶段：`{state.get('stage', 'not_started')}`", '']
        if label == 'current_only':
            lines += ['此对照复用主基线的基础策略、数据集和VAE；下方若显示VAE日志，代表共享训练产物，不是新训练的VAE。', '']
        elif label == 'fixed_speed':
            lines += ['此对照复用主基线的数据、VAE和LSTM，只重训PPO；下方表征训练日志属于共享产物。目标速度为本项目预先选定的2.0m/s，采用奖励约束，实际飞行速度不保证恒定。', '']
        if state.get('error'): lines += [f"错误：{state['error']}", '']
        for stage, record in run['training'].items():
            if record is None:
                if stage == 'final' and state.get('stage') == 'train' and state.get('state') == 'running':
                    lines.append('- PPO：已启动，尚未完成并记录第一轮rollout。')
                else:
                    lines.append(f'- {stage}：尚无已完成的训练日志记录。')
            elif stage == 'final':
                lines.append(f"- PPO：{record['timesteps']}步；日志累计吞吐{record['fps']:.1f}步/秒。训练内部指标不能代替独立评估成功率。")
            else:
                lines.append(f"- {stage}：最近完成第{record['epoch']}轮。")
        report = run['evaluation']
        if report:
            lines += ['', '| 密度 | 任务数 | 到达率 | 碰撞率 | 平均速度m/s |', '|---|---:|---:|---:|---:|']
            for density, result in report['results'].items():
                lines.append(f"| {density} | {result['episodes']} | {result['success_rate']:.3f} | {result['collision_rate']:.3f} | {result['mean_speed_all_episodes']:.3f} |")
        else:
            lines += ['', '尚无正式导航评估结果；不能用此前短程结果代替。']
        lines.append('')
    lines += ['## 配对结果与后续交付', '']
    if 'fixed_speed' not in plan:
        lines += ['- 固定速度奖励消融：已核对论文公式，具体系数和目标速度仍需明确实验设计。']
    for name, result in data.get('comparisons', {}).items():
        path = Path(plan['comparison_root'])/name/'comparison.json'
        lines.append(f"- {name}：" + (f'已有汇总，见`{path}`。' if result else '尚无完成的配对汇总。'))
    delivered = data.get('delivery') or {}
    delivery_line = (f"- 本轮交付文档已完成：`{delivered['report']}`；复跑指南：`{delivered['handoff']}`。科学结论及限制以报告为准。"
                     if delivered.get('state') == 'completed'
                     else '- 单训练种子的结果解释与最终复现报告仍需核对完整证据。')
    lines += [delivery_line,
              '', '用户于2026-11-05取消多训练种子验证；仍保留不同地图任务和重复评估。',
              '', '当前任务不引入C3BF或动态障碍。']
    (folder/f'{filename}.json').write_text(json.dumps(data, indent=2)+'\n')
    (folder/f'{filename}.md').write_text('\n'.join(lines)+'\n')
    return data


def watch_report(plan, folder):
    """Observe an existing queue; never launch, stop, or restart GPU jobs."""
    deadline = datetime.fromisoformat(plan['report_at'])
    state = {'pid': os.getpid(), 'state': 'watching', 'report_at': plan['report_at'],
             'mode': 'report_only_watch'}
    while True:
        write_report(plan, folder, 'progress')
        done = datetime.now(ZONE) >= deadline
        if done:
            write_report(plan, folder, 'MORNING_REPORT')
            state.update(state='completed', morning_report=str(folder/'MORNING_REPORT.md'))
        state['updated_at'] = datetime.now(ZONE).isoformat()
        (folder/'report_status.json').write_text(json.dumps(state, indent=2)+'\n')
        if done: return
        time.sleep(30)


def run_queue(plan, folder):
    state_path = folder/'status.json'
    if state_path.exists(): raise ValueError('Queue status already exists; do not start a duplicate scheduler')
    state = {'pid': os.getpid(), 'state': 'waiting_for_baseline', 'started_at': datetime.now(ZONE).isoformat(), 'child_pid': None}
    deadline = datetime.fromisoformat(plan['report_at'])
    child = None
    child_log = None
    report_written = False
    terminal = False
    try:
        while True:
            main = read_json(Path(plan['baseline'])/'long_run/status.json')
            if main and main['state'] == 'failed' and child is None:
                state.update(state='baseline_failed', error=main.get('error')); terminal = True
            if not terminal and child is None and main and main['state'] == 'completed':
                checks = read_json(Path(plan['baseline'])/'validation.json')
                if not checks or not all(checks.get(k) for k in ['finite_losses', 'representation_exactly_frozen', 'separate_critic_present', 'checkpoint_load_action_identity']):
                    raise RuntimeError('Baseline completion lacks required artifact validation')
                occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
                if occupied:
                    state.update(state='waiting_for_free_gpu', occupied_pids=occupied)
                else:
                    child_log = (folder/'current_only_supervisor.log').open('ab', buffering=0)
                    child = subprocess.Popen(plan['command'], stdin=subprocess.DEVNULL, stdout=child_log, stderr=subprocess.STDOUT)
                    state.update(state='current_only_running', child_pid=child.pid, command=plan['command'])
            if child is not None and not terminal and child.poll() is not None:
                if child.returncode != 0:
                    state.update(state='current_only_failed', exit_code=child.returncode); terminal = True
                else:
                    command = ['scripts/python.sh', 'scripts/compare_memory_ablation.py', plan['baseline'], plan['current_only'], '--out', str(folder/'memory_comparison')]
                    with (folder/'comparison.log').open('w') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    state.update(state='completed', comparison=str(folder/'memory_comparison')); terminal = True
            write_report(plan, folder, 'progress')
            if not report_written and datetime.now(ZONE) >= deadline:
                write_report(plan, folder, 'MORNING_REPORT')
                report_written = True
                state['morning_report'] = str(folder/'MORNING_REPORT.md')
            state['updated_at'] = datetime.now(ZONE).isoformat()
            temporary = folder/'status.tmp'
            temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(state_path)
            if terminal and report_written: break
            time.sleep(30)
    except BaseException as exc:
        state.update(state='scheduler_failed', error=repr(exc), updated_at=datetime.now(ZONE).isoformat())
        state_path.write_text(json.dumps(state, indent=2)+'\n')
        write_report(plan, folder, 'progress')
        raise
    finally:
        if child_log: child_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--report-only', action='store_true')
    modes.add_argument('--watch-report', action='store_true', help='Only observe an existing queue and write the scheduled report')
    args = parser.parse_args()
    plan_path = Path(args.plan); plan = json.loads(plan_path.read_text())
    if args.report_only: write_report(plan, plan_path.parent, 'progress')
    elif args.watch_report: watch_report(plan, plan_path.parent)
    else: run_queue(plan, plan_path.parent)
