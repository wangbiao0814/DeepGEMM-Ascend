#!/usr/bin/env python3
"""BF16 grouped GEMM experiment: DeepGEMM (950) or installed CANN (910C).

--dry-run needs only Python. Use --operator-backend cann on 910C.
"""

import argparse
import csv
import json
import os
import platform
from pathlib import Path


BLOCK_M = 256  # csrc/jit/jit.hpp: Runtime::kMkAlignment (not configurable).


def describe_case(experts, tokens, n, k, block_m=BLOCK_M):
    if min(experts, tokens, n, k, block_m) <= 0:
        raise ValueError('All dimensions must be positive')
    stride = (tokens + block_m - 1) // block_m * block_m
    copies = stride // block_m
    weights = experts * n * k * 2
    return dict(
        experts=experts, tokens_per_expert=tokens, n=n, k=k, block_m=block_m,
        padded_tokens_per_expert=stride,
        psum=[e * stride + tokens for e in range(experts)],
        m_tiles_per_expert=copies,
        unique_weight_bytes=weights,
        model_weight_gm_to_l1_bytes=weights * copies,
        useful_flops=2 * experts * tokens * n * k,
        padded_flops=2 * experts * stride * n * k,
    )


def capture_trace(torch, fn, flush, directory, metric, iterations):
    """Keep raw exports: PMU columns depend on CANN/device, so do not guess units."""
    from torch_npu import profiler as p

    if not hasattr(p.AiCMetrics, metric):
        raise RuntimeError(f'This torch_npu does not expose AiCMetrics.{metric}')
    directory.mkdir(parents=True, exist_ok=False)
    with p.profile(
        activities=[p.ProfilerActivity.NPU],
        on_trace_ready=p.tensorboard_trace_handler(str(directory)),
        experimental_config=p._ExperimentalConfig(
            profiler_level=p.ProfilerLevel.Level1,
            aic_metrics=getattr(p.AiCMetrics, metric),
            l2_cache=(metric == 'L2Cache'),
            data_simplification=False,
        ),
    ) as prof:
        for _ in range(iterations):
            if flush is not None:
                flush.zero_()
            fn()
        torch.npu.synchronize()
        prof.step()


def check_output(torch, x, weights, output, case):
    # Check first and last valid rows of EVERY expert; references run outside profiling.
    worst = 0.0
    stride = case['padded_tokens_per_expert']
    for e in range(case['experts']):
        count = case['tokens_per_expert']
        rows = sorted(set(range(min(4, count))) | set(range(max(0, count - 4), count)))
        offsets = torch.tensor([e * stride + r for r in rows], device=x.device)
        sample = x.index_select(0, offsets).cpu().float()
        reference = sample @ weights[e].cpu().float().T
        actual = output.index_select(0, offsets).cpu().float()
        if not torch.isfinite(actual).all():
            raise AssertionError(f'Non-finite output for expert {e}')
        denominator = (actual.square() + reference.square()).sum().item()
        diff = (actual - reference).square().sum().item() / max(denominator, 1e-30)
        worst = max(worst, diff)
        if diff > 1e-5:
            raise AssertionError(f'Expert {e}: normalized squared error {diff} > 1e-5')
    return worst


def run(args, cases):
    # Set before importing DeepGEMM, so the actual JIT configuration is logged.
    os.environ.setdefault('DG_PRINT_CONFIGS', '1')
    import torch
    import torch_npu
    import deep_gemm
    from deep_gemm.testing.bench import bench_msprof, close_persistent_profiler

    if not torch.npu.is_available():
        raise RuntimeError('An Ascend NPU and a working torch_npu installation are required')
    torch.npu.set_device(args.device)
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    if alignment != BLOCK_M:
        raise RuntimeError(f'Expected BLOCK_M={BLOCK_M}, runtime reports {alignment}; update model first')
    torch.manual_seed(args.seed)
    properties = torch.npu.get_device_properties(torch.npu.current_device())
    metadata = dict(
        python=platform.python_version(), torch=torch.__version__,
        torch_npu=getattr(torch_npu, '__version__', 'unknown'),
        device=str(properties), l2_cache_bytes=int(properties.L2_cache_size),
        args=vars(args), cases=cases,
        measurement_note='model_weight_gm_to_l1_bytes is NOT measured HBM traffic',
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    weights = torch.randn(args.experts, args.n, args.k, dtype=torch.bfloat16, device='npu')
    modes = ['warm', 'evict'] if args.cache == 'both' else [args.cache]
    records = []
    try:
        for case in cases:
            stride = case['padded_tokens_per_expert']
            x = torch.zeros(args.experts * stride, args.k, dtype=torch.bfloat16, device='npu')
            for e in range(args.experts):
                x[e * stride:e * stride + case['tokens_per_expert']].normal_()
            layout = torch.tensor(case['psum'], dtype=torch.int32, device='npu')
            output = torch.empty(args.experts * stride, args.n, dtype=torch.bfloat16, device='npu')

            def fn():
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                    x, weights, output, layout, use_psum_layout=True,
                    expected_m_for_psum_layout=stride,
                )

            fn()  # Compile JIT and validate before measurement.
            torch.npu.synchronize()
            diff = check_output(torch, x, weights, output, case)
            for mode in modes:
                profile = bench_msprof(
                    fn, kernel_names='bf16_gemm', num_warmups=args.warmups,
                    num_tests=args.iterations, flush_l2=(mode == 'evict'), backend=args.backend,
                )
                record = {key: value for key, value in case.items() if key != 'psum'}
                record.update(
                    cache_mode=mode, measured_kernel_us=profile.dur_us,
                    measured_aic_mte2_ratio=profile.aic_mte2,
                    measured_aic_mad_ratio=profile.aic_mad,
                    sampled_correctness_diff=diff,
                )
                records.append(record)
                # Save after every completed measurement, including if a later PMU pass fails.
                with (args.output / 'results.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(record))
                    writer.writeheader()
                    writer.writerows(records)
                print(json.dumps(record), flush=True)
                if args.trace_metrics:
                    close_persistent_profiler()
                    flush = None
                    if mode == 'evict':
                        flush = torch.empty(2 * int(properties.L2_cache_size), dtype=torch.uint8, device='npu')
                    for metric in args.trace_metrics:
                        # Prime identical buffers immediately before the counter pass.
                        for _ in range(args.warmups):
                            fn()
                        torch.npu.synchronize()
                        directory = args.output / f"m{case['tokens_per_expert']}_{mode}_{metric}"
                        capture_trace(torch, fn, flush, directory, metric, args.trace_iterations)
                    del flush
            del fn, x, output, layout
    finally:
        close_persistent_profiler()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-backend', choices=['deepgemm', 'cann'], default='deepgemm')
    parser.add_argument('--experts', type=int, default=None)
    parser.add_argument('--m', type=int, nargs='+', default=None, help='DeepGEMM: per-expert M scan')
    parser.add_argument('--n', type=int, default=2048)
    parser.add_argument('--k', type=int, default=2048)
    parser.add_argument('--cache', choices=['warm', 'evict', 'both'], default='both')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--backend', choices=['persistent', 'fast', 'slow'], default='persistent',
                        help='DeepGEMM only: profiler backend')
    parser.add_argument('--trace-metrics', nargs='+', choices=[
        'MemoryAccess', 'Memory', 'L2Cache', 'PipeUtilization', 'ResourceConflictRatio'])
    parser.add_argument('--trace-iterations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--output', type=Path, default=Path('results/grouped_weight_reload'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tokens', type=int, default=64, help='CANN: original token count')
    parser.add_argument('--topk', type=int, default=8, help='CANN: experts selected per token')
    parser.add_argument('--case', action='append', default=[], metavar='NAME:COUNTS',
                        help='CANN: repeatable case, e.g. balanced:128,128,128,128')
    parser.add_argument('--order', choices=['fixed', 'shuffle'], default='shuffle',
                        help='CANN: case order within each timing round')
    parser.add_argument('--flush-mib', type=float, default=None,
                        help='CANN: explicit eviction buffer MiB if L2 capacity is unavailable')
    parser.add_argument('--kernel-pattern', default=r'grouped.*(?:matmul|gemm)',
                        help='CANN: case-insensitive regex for exported profiler rows')
    args = parser.parse_args()
    if args.experts is None:
        args.experts = 16 if args.operator_backend == 'cann' else 4
    if min(args.experts, args.n, args.k, *(args.m or []), args.warmups, args.iterations, args.trace_iterations) <= 0:
        parser.error('dimensions, warmups and iteration counts must be positive')
    if args.n % 64 or args.k % 64:
        parser.error('This experiment requires N and K to be multiples of 64 to avoid sector/tail effects')
    if args.operator_backend == 'cann':
        if args.m is not None:
            parser.error('CANN uses --tokens/--topk and --case; --m is the DeepGEMM-only scan')
        from cann_experiment import build_cases, run_cann
        try:
            cases = build_cases(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.dry_run:
            print(json.dumps(cases, indent=2))
            return
        try:
            run_cann(args, cases)
        except ImportError as exc:
            raise SystemExit(f'CANN runtime import failed: {exc}. No DeepGEMM build is required.') from exc
        return
    if args.case:
        parser.error('--case is currently supported only with --operator-backend cann')
    if args.m is None:
        args.m = [64, 255, 256, 257, 512, 513, 1024]
    args.m = sorted(set(args.m))
    cases = [describe_case(args.experts, m, args.n, args.k) for m in args.m]
    if args.dry_run:
        print(json.dumps(cases, indent=2))
        return
    try:
        run(args, cases)
    except ImportError as exc:
        raise SystemExit(f'NPU runtime import failed: {exc}. Use --dry-run for the analytical model.') from exc


if __name__ == '__main__':
    main()
