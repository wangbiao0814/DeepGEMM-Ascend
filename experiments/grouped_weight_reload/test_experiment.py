"""CPU-only checks for workload controls and faithful profiler export handling."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from cann_experiment import build_cases, collect_trace, export_metric_rows, summarize
from run import describe_case


def arguments(**updates):
    values = dict(experts=16, tokens=64, topk=8, n=2048, k=2048,
                  flush_mib=None, kernel_pattern=r'grouped.*(?:matmul|gemm)', case=[])
    values.update(updates)
    return argparse.Namespace(**values)


class WorkloadTests(unittest.TestCase):
    def test_controls_and_permutation(self):
        cases = build_cases(arguments())
        self.assertEqual(cases[0]['counts'], [32] * 16)
        for case in cases:
            self.assertEqual(case['routed_rows'], 512)
            self.assertEqual(case['active_experts'], 16)
            self.assertLessEqual(max(case['counts']), 64)
            self.assertEqual(case['group_list'][-1], 512)
            self.assertIsNone(case['actual_block_m'])
            self.assertIsNone(case['actual_output_tiles'])
            self.assertIsNone(case['model_weight_gm_to_l1_bytes'])
        self.assertEqual(len({case['useful_flops'] for case in cases}), 1)
        self.assertEqual(len({case['active_weight_bytes'] for case in cases}), 1)
        self.assertEqual(cases[1]['counts'][::-1], cases[2]['counts'])
        self.assertGreater(cases[1]['load_cv'], cases[0]['load_cv'])

    def test_boundary_suite_without_assuming_vendor_tiling(self):
        cases = build_cases(arguments(tokens=512))
        self.assertEqual(cases[0]['counts'], [256] * 16)
        self.assertGreater(max(cases[1]['counts']), 256)
        self.assertLess(min(cases[1]['counts']), 256)
        self.assertIsNone(cases[1]['actual_output_tiles'])

    def test_custom_zero_group_and_tight_prefix_sum(self):
        case = build_cases(arguments(experts=3, tokens=500, topk=2,
                                    case=['abc:100,400,500']))[0]
        self.assertEqual(case['group_list'], [100, 500, 1000])
        zero = build_cases(arguments(experts=3, tokens=4, topk=1,
                                    case=['empty:0,2,2']))[0]
        self.assertEqual(zero['group_list'], [0, 2, 4])
        self.assertEqual(zero['active_experts'], 2)

    def test_invalid_routing_and_names(self):
        for updates in [dict(topk=17), dict(tokens=0), dict(flush_mib=float('nan')),
                        dict(case=['../bad:1,2']), dict(kernel_pattern='['),
                        dict(case=['bad:' + ','.join(['33'] * 16)]),
                        dict(case=['bad:' + ','.join(['512'] + ['0'] * 15)]),
                        dict(case=['same:' + ','.join(['32'] * 16)] * 2)]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                build_cases(arguments(**updates))

    def test_original_deepgemm_padding(self):
        case = describe_case(3, 100, 1024, 512)
        self.assertEqual(case['psum'], [100, 356, 612])
        self.assertEqual(case['model_weight_gm_to_l1_bytes'], 3 * 1024 * 512 * 2)

    def test_statistics_include_dispersion(self):
        values = summarize([1, 2, 3, 4])
        self.assertEqual(values['mean_us'], 2.5)
        self.assertEqual(values['p50_us'], 2.5)
        self.assertAlmostEqual(values['p95_us'], 3.85)
        self.assertGreater(values['cv'], 0)
        self.assertEqual(summarize([2, 2, 2])['std_us'], 0)

    def test_cli_without_torch_or_deepgemm(self):
        script = Path(__file__).with_name('run.py')
        result = subprocess.run([sys.executable, str(script), '--operator-backend', 'cann',
                                 '--dry-run'], capture_output=True, text=True, check=True)
        cases = json.loads(result.stdout)
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0]['experts'], 16)
        old = subprocess.run([sys.executable, str(script), '--dry-run', '--m', '257'],
                             capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(old.stdout)[0]['m_tiles_per_expert'], 2)
        invalid = subprocess.run([sys.executable, str(script), '--operator-backend', 'cann',
                                  '--dry-run', '--m', '257'], capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 2)


class ExportTests(unittest.TestCase):
    def test_unsupported_counter_has_explicit_failure(self):
        fake_npu = SimpleNamespace(profiler=SimpleNamespace(AiCMetrics=SimpleNamespace()))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'unsupported'
            with patch.dict(sys.modules, {'torch_npu': fake_npu}), redirect_stdout(io.StringIO()):
                status = collect_trace(None, None, None, directory, 'MemoryAccess',
                                       SimpleNamespace(trace_iterations=5),
                                       {'case': 'replay', 'counts': [2, 2]}, 'warm')
            self.assertEqual(status['status'], 'failed')
            self.assertIn('does not expose', status['error'])
            self.assertEqual(json.loads((directory / 'collection.json').read_text()), status)
            self.assertNotIn('mte2_time', status)

    def test_keeps_units_per_core_and_filters_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'metrics.csv').write_text(
                'Name,core,aic_mte2_time(us),aic_mte2_wait_ratio\n'
                'GroupedMatmul,0,12.5,0.2\nGroupedMatmul,1,17.0,0.4\nZero,0,99,0\n')
            manifest = export_metric_rows(directory, r'grouped.*matmul')
            rows = [json.loads(line) for line in (directory / 'matched_rows.jsonl').read_text().splitlines()]
            self.assertEqual(manifest['matched_rows'], 2)
            self.assertEqual(rows[0]['fields']['aic_mte2_time(us)'], '12.5')
            self.assertEqual(rows[1]['fields']['core'], '1')
            self.assertEqual(rows[1]['source_row'], 3)

    def test_missing_metrics_are_not_reported_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export_metric_rows(Path(temporary), 'gmm')
            self.assertEqual(manifest['csv_files'], [])
            self.assertIn('does not mean zero', manifest['note'])


if __name__ == '__main__':
    unittest.main()
