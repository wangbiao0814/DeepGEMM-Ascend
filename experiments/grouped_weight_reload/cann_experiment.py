"""Standalone CANN experiment helpers. This module must not import deep_gemm.

Python-only planning/statistics/export parsing also work without torch installed.
"""

import csv
import json
import math
import platform
import random
import re
import statistics
import time
from pathlib import Path


def describe_counts(name, counts, n, k):
    ends, total = [], 0
    for count in counts:
        total += count
        ends.append(total)
    active = sum(count > 0 for count in counts)
    mean = total / len(counts)
    return dict(
        case=name, counts=counts, group_list=ends,
        experts=len(counts), active_experts=active, routed_rows=total,
        min_tokens=min(counts), max_tokens=max(counts),
        load_cv=statistics.pstdev(counts) / mean,
        n=n, k=k, input_bytes=total * k * 2, output_bytes=total * n * 2,
        active_weight_bytes=active * n * k * 2,
        allocated_weight_bytes=len(counts) * n * k * 2,
        useful_flops=2 * total * n * k,
        actual_block_m=None, actual_output_tiles=None,
        model_weight_gm_to_l1_bytes=None,
        model_note='CANN tiling is unknown: no 256-row padding or inferred MTE2/HBM traffic',
    )


def build_cases(args):
    experts, tokens, topk = args.experts, args.tokens, args.topk
    if min(experts, tokens, topk) <= 0 or topk > experts:
        raise ValueError('Require tokens > 0 and 1 <= topk <= experts')
    if args.flush_mib is not None and (not math.isfinite(args.flush_mib) or args.flush_mib <= 0):
        raise ValueError('--flush-mib must be finite and positive')
    try:
        re.compile(args.kernel_pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f'Invalid --kernel-pattern: {exc}') from exc
    total = tokens * topk
    named = []
    if args.case:
        for text in args.case:
            name, separator, values = text.partition(':')
            if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                raise ValueError('--case must be NAME:COUNT,COUNT,... with a safe, unique name')
            try:
                counts = [int(value) for value in values.split(',')]
            except ValueError as exc:
                raise ValueError(f'Invalid counts in --case {text}') from exc
            named.append((name, counts))
    else:
        if total < experts:
            raise ValueError('Default suite keeps all experts active; increase tokens or provide --case')
        balanced = [total // experts + int(i < total % experts) for i in range(experts)]
        # Keep every expert active, keep sum fixed, and never route one token twice
        # to the same expert: each expert receives at most `tokens` rows.
        skewed = [1] * experts
        remaining = total - experts
        for i in range(experts):
            extra = min(remaining, tokens - 1)
            skewed[i] += extra
            remaining -= extra
        assert remaining == 0
        named = [('balanced', balanced), ('skewed', skewed), ('skewed_reversed', skewed[::-1])]
    if len({name for name, _ in named}) != len(named):
        raise ValueError('Case names must be unique')
    for name, counts in named:
        if len(counts) != experts or any(count < 0 or count > tokens for count in counts):
            raise ValueError(f'{name}: require {experts} counts, each in [0, tokens={tokens}]')
        if sum(counts) != total:
            raise ValueError(f'{name}: sum(counts) must equal tokens * topk = {total}')
    return [describe_counts(name, counts, args.n, args.k) for name, counts in named]


def summarize(values):
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    std = statistics.pstdev(ordered)

    def percentile(q):
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return dict(samples=len(values), mean_us=mean, std_us=std,
                cv=std / mean if mean else 0.0, min_us=ordered[0],
                p50_us=percentile(0.5), p95_us=percentile(0.95), max_us=ordered[-1])


def export_metric_rows(directory, pattern):
    """Preserve PMU rows and original units; do not infer per-call/per-core totals."""
    matcher = re.compile(pattern, re.IGNORECASE)
    sources, found = [], []
    for path in sorted(directory.rglob('*.csv')):
        sources.append(str(path.relative_to(directory)))
        with path.open(encoding='utf-8-sig', errors='replace', newline='') as stream:
            reader = csv.DictReader(stream)
            for row_number, row in enumerate(reader, start=2):
                if matcher.search(' '.join(str(value) for value in row.values())):
                    found.append(dict(source_csv=str(path.relative_to(directory)),
                                      source_row=row_number, fields=row))
    with (directory / 'matched_rows.jsonl').open('w') as stream:
        for row in found:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = dict(csv_files=sources, matched_rows=len(found), kernel_pattern=pattern,
                    note='Rows may be kernels, calls or per-core records; retain original units. '
                         'No matching rows does not mean zero activity. Consult raw CSV/DB/trace.')
    (directory / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def make_call(torch_npu, x, weight, layout):
    def call():
        return torch_npu.npu_grouped_matmul(
            [x], [weight], group_list=layout,
            split_item=3, group_type=0, group_list_type=0,
        )[0]
    return call


def check_output(torch, x_cpu, w_cpu, output, case):
    worst, start = 0.0, 0
    if tuple(output.shape) != (case['routed_rows'], case['n']) or output.device.type != 'npu':
        raise AssertionError(f'Unexpected CANN output {tuple(output.shape)}, {output.device}')
    for expert, end in enumerate(case['group_list']):
        count = end - start
        if count:
            rows = sorted(set(range(min(4, count))) | set(range(max(0, count - 4), count)))
            indices = [start + row for row in rows]
            reference = x_cpu[indices].float() @ w_cpu[expert].float()
            # Transfer only the sampled rows. Indexing is outside timed/PMU windows.
            actual = torch.stack([output[i] for i in indices]).cpu().float()
            if not torch.isfinite(actual).all():
                raise AssertionError(f'{case["case"]}: non-finite output, expert {expert}')
            denominator = (actual.square() + reference.square()).sum().item()
            diff = (actual - reference).square().sum().item() / max(denominator, 1e-30)
            if diff > 1e-5:
                raise AssertionError(f'{case["case"]}, expert {expert}: diff={diff} > 1e-5')
            worst = max(worst, diff)
        start = end
    return worst


def collect_trace(torch, call, flush, directory, metric, args, case, mode):
    from torch_npu import profiler as p

    directory.mkdir(parents=True, exist_ok=False)
    description = dict(case=case['case'], counts=case['counts'], cache_mode=mode,
                       metric=metric, calls=args.trace_iterations, status='pending',
                       note='Separate counter pass; not the same invocations as samples.csv')
    status_file = directory / 'collection.json'
    try:
        if not hasattr(p.AiCMetrics, metric):
            raise RuntimeError(f'Installed torch_npu does not expose AiCMetrics.{metric}')
        for _ in range(args.warmups):
            output = call()
        torch.npu.synchronize()
        with p.profile(
            activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
            record_shapes=True,
            on_trace_ready=p.tensorboard_trace_handler(str(directory)),
            experimental_config=p._ExperimentalConfig(
                profiler_level=p.ProfilerLevel.Level1,
                aic_metrics=getattr(p.AiCMetrics, metric),
                l2_cache=(metric == 'L2Cache'), data_simplification=False,
            ),
        ) as prof:
            for iteration in range(args.trace_iterations):
                if flush is not None:
                    flush.zero_()
                    torch.npu.synchronize()
                with torch.autograd.profiler.record_function(
                    f'gmm/{case["case"]}/{mode}/{metric}/call_{iteration}'
                ):
                    output = call()
                    torch.npu.synchronize()
            prof.step()
        manifest = export_metric_rows(directory, args.kernel_pattern)
        description.update(status='exported', **manifest)
        if not manifest['matched_rows']:
            print(f'WARNING: no matching CSV rows in {directory}; inspect raw CSV/DB and kernel-pattern')
    except (RuntimeError, ValueError, AttributeError, OSError) as exc:
        description.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(f'WARNING: counter pass {case["case"]}/{mode}/{metric} failed: {exc}', flush=True)
    finally:
        status_file.write_text(json.dumps(description, indent=2) + '\n')
    return description


def run_cann(args, cases):
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError('No NPU available in the current torch_npu environment')
    if not hasattr(torch_npu, 'npu_grouped_matmul'):
        raise RuntimeError('Installed torch_npu does not expose npu_grouped_matmul')
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    properties = torch.npu.get_device_properties(torch.npu.current_device())
    l2_bytes = getattr(properties, 'L2_cache_size', None)
    l2_bytes = int(l2_bytes) if l2_bytes is not None and int(l2_bytes) > 0 else None
    modes = ['warm', 'evict'] if args.cache == 'both' else [args.cache]
    flush_bytes = None
    if 'evict' in modes:
        if args.flush_mib is not None:
            flush_bytes = int(args.flush_mib * 1024 * 1024)
            if flush_bytes < 1:
                raise ValueError('--flush-mib is too small')
        elif l2_bytes is not None:
            flush_bytes = 2 * l2_bytes
        else:
            raise RuntimeError('Device does not report L2 capacity; pass --flush-mib or --cache warm')
        if l2_bytes and flush_bytes < 2 * l2_bytes:
            print('WARNING: eviction buffer is less than 2x reported L2 capacity')

    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        operator_backend='cann', python=platform.python_version(),
        torch=torch.__version__, torch_npu=getattr(torch_npu, '__version__', 'unknown'),
        device=str(properties), l2_cache_bytes=l2_bytes, flush_bytes=flush_bytes,
        args=vars(args), cases=cases,
        measurements={
            'operator_event_us': 'One GMM call bracketed by NPU events; includes device idle gaps '
                                 'while host dispatches/allocates. NOT isolated MTE2 or kernel time.',
            'host_call_us': 'Host API call including output allocation/dispatch; excludes synchronize.',
            'counter_passes': 'Separate profiler invocations; no automatic per-sample PMU correlation.',
        },
        controls='Same X and weight storage across all cases, same routed rows/N/K. '
                 'Output is allocated by torch_npu per call; no fixed-output API is assumed. '
                 'Synthetic X_perm; routing, attention and communication are not executed.',
    )
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    summary_rows, samples, statuses = [], [], []

    def write_summary():
        for mode in modes:
            for case in cases:
                selected = [row['operator_event_us'] for row in samples
                            if row['case'] == case['case'] and row['cache_mode'] == mode]
                if selected:
                    stats = summarize(selected)
                    row = dict(case=case['case'], cache_mode=mode,
                               active_experts=case['active_experts'], routed_rows=case['routed_rows'],
                               load_cv=case['load_cv'], active_weight_bytes=case['active_weight_bytes'])
                    row.update({f'operator_event_{key}': value for key, value in stats.items()})
                    summary_rows.append(row)
        if summary_rows:
            with (args.output / 'results.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
                writer.writeheader()
                writer.writerows(summary_rows)

    # CANN consumes tightly concatenated input; no assumed 256-row alignment.
    x_cpu = torch.randn(cases[0]['routed_rows'], args.k, dtype=torch.bfloat16)
    w_cpu = torch.randn(args.experts, args.k, args.n, dtype=torch.bfloat16)
    x = x_cpu.to(args.device)
    weight = w_cpu.to(args.device)
    layouts = [torch.tensor(case['group_list'], dtype=torch.int64, device=args.device) for case in cases]
    calls = [make_call(torch_npu, x, weight, layout) for layout in layouts]
    outputs = [None] * len(cases)
    diffs = []
    flush = torch.empty(flush_bytes, dtype=torch.uint8, device=args.device) if flush_bytes else None
    # Pre-create timing events so event creation is outside the sampled intervals.
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    rng = random.Random(args.seed)
    with torch.inference_mode():
        # Compile/initialize and validate EVERY case before warmup or measurements.
        for i, case in enumerate(cases):
            outputs[i] = calls[i]()
            torch.npu.synchronize()
            diff = check_output(torch, x_cpu, w_cpu, outputs[i], case)
            diffs.append(diff)
            print(f'CHECK {case["case"]}: counts={case["counts"]}, sampled_diff={diff:.3e}', flush=True)
        del x_cpu, w_cpu
        metadata.update(input_address=x.data_ptr(), weight_address=weight.data_ptr(),
                        sampled_correctness=dict(zip([case['case'] for case in cases], diffs)))
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')

        with (args.output / 'samples.csv').open('w', newline='') as stream:
            fields = ['case', 'cache_mode', 'round', 'position_in_round', 'operator_event_us',
                      'host_call_us', 'output_address', 'sampled_correctness_diff']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            try:
                for mode in modes:
                    for i, call in enumerate(calls):
                        for _ in range(args.warmups):
                            outputs[i] = call()
                    torch.npu.synchronize()
                    for iteration in range(args.iterations):
                        order = list(range(len(cases)))
                        if args.order == 'shuffle':
                            rng.shuffle(order)
                        for position, i in enumerate(order):
                            if mode == 'evict':
                                flush.zero_()
                                torch.npu.synchronize()
                            begin.record()
                            host_begin = time.perf_counter_ns()
                            outputs[i] = calls[i]()
                            host_us = (time.perf_counter_ns() - host_begin) / 1000
                            end.record()
                            torch.npu.synchronize()
                            row = dict(case=cases[i]['case'], cache_mode=mode, round=iteration,
                                       position_in_round=position,
                                       operator_event_us=begin.elapsed_time(end) * 1000,
                                       host_call_us=host_us, output_address=outputs[i].data_ptr(),
                                       sampled_correctness_diff=diffs[i])
                            writer.writerow(row)
                            stream.flush()
                            samples.append(row)
                    print(f'TIMING {mode}: {args.iterations} rounds per case saved', flush=True)
            finally:
                write_summary()
        for row in summary_rows:
            print(json.dumps(row), flush=True)
        # All timing precedes profiling, so failed/invasive PMU passes cannot pollute it.
        for mode in modes:
            for i, case in enumerate(cases):
                for metric in args.trace_metrics or []:
                    directory = args.output / 'traces' / case['case'] / mode / metric
                    status = collect_trace(torch, calls[i], flush if mode == 'evict' else None,
                                           directory, metric, args, case, mode)
                    statuses.append(status)
                    (args.output / 'trace_status.json').write_text(json.dumps(statuses, indent=2) + '\n')
    print(f'Results: {args.output.resolve()}')
    print('Event timings alone cannot establish MTE2 stalls, L1 backpressure or HBM reloads.')
