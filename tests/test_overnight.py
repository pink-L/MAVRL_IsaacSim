"""Scheduler and paired-task bookkeeping tests; no training or simulated results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


queue = load('overnight_queue_test', 'scripts/overnight_queue.py')
comparison = load('memory_ablation_test', 'scripts/compare_memory_ablation.py')


class OvernightTests(unittest.TestCase):
    def test_completed_valid_baseline_starts_followup_once_and_writes_report(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root = Path(temp); baseline = root/'baseline'; (baseline/'long_run').mkdir(parents=True)
            (baseline/'long_run/status.json').write_text(json.dumps({'state': 'completed'}))
            keys = ['finite_losses', 'representation_exactly_frozen', 'separate_critic_present', 'checkpoint_load_action_identity']
            (baseline/'validation.json').write_text(json.dumps(dict.fromkeys(keys, True)))
            plan = {'baseline': str(baseline), 'current_only': str(root/'variant'), 'report_at': '2000-01-01T09:00:00+08:00', 'command': ['TEST_ONLY_DO_NOT_EXECUTE']}
            child = Mock(pid=123, returncode=0); child.poll.return_value = 0
            with patch.object(queue.subprocess, 'check_output', return_value=''), patch.object(queue.subprocess, 'Popen', return_value=child) as launch, patch.object(queue.subprocess, 'run') as compare:
                queue.run_queue(plan, root)
            launch.assert_called_once(); compare.assert_called_once()
            self.assertEqual(json.loads((root/'status.json').read_text())['state'], 'completed')
            self.assertIn('尚无正式导航评估结果', (root/'MORNING_REPORT.md').read_text())

    def test_failed_baseline_does_not_launch_gpu_followup(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root = Path(temp); baseline = root/'baseline'; (baseline/'long_run').mkdir(parents=True)
            (baseline/'long_run/status.json').write_text(json.dumps({'state': 'failed', 'error': 'test failure'}))
            plan = {'baseline': str(baseline), 'current_only': str(root/'variant'), 'report_at': '2000-01-01T09:00:00+08:00', 'command': ['TEST_ONLY_DO_NOT_EXECUTE']}
            with patch.object(queue.subprocess, 'Popen') as launch:
                queue.run_queue(plan, root)
            launch.assert_not_called()
            self.assertEqual(json.loads((root/'status.json').read_text())['state'], 'baseline_failed')
            self.assertIn('test failure', (root/'MORNING_REPORT.md').read_text())

    def test_paired_counts_and_missing_task_rejection(self):
        left = [{'density': 'sparse', 'scene_seed': i, 'reason': reason} for i, reason in enumerate(['success', 'success', 'collision', 'collision'])]
        right = [{'density': 'sparse', 'scene_seed': i, 'reason': reason} for i, reason in enumerate(['success', 'collision', 'success', 'collision'])]
        result = comparison.paired_counts(left, right)['sparse']
        self.assertEqual(list(result.values()), [1, 1, 1, 1])
        with self.assertRaises(ValueError): comparison.paired_counts(left, right[:-1])
        with self.assertRaises(ValueError): comparison.paired_counts(left, right+[right[0]])

    def test_report_watcher_includes_unstarted_fixed_without_launching_jobs(self):
        with tempfile.TemporaryDirectory(dir='.') as temp:
            root = Path(temp)
            plan = {name: str(root/name) for name in ['baseline', 'current_only', 'fixed_speed']}
            plan.update(report_at='2000-01-01T09:00:00+08:00', comparison_root=str(root/'comparisons'))
            with patch.object(queue.subprocess, 'Popen') as launch, patch.object(queue.subprocess, 'run') as run:
                queue.watch_report(plan, root)
            launch.assert_not_called(); run.assert_not_called()
            report = json.loads((root/'MORNING_REPORT.json').read_text())
            self.assertIsNone(report['snapshots']['fixed_speed']['evaluation'])
            self.assertIsNone(report['comparisons']['speed_comparison'])
            self.assertEqual(json.loads((root/'report_status.json').read_text())['state'], 'completed')
            self.assertIn('目标速度奖励', (root/'MORNING_REPORT.md').read_text())


if __name__ == '__main__': unittest.main()
