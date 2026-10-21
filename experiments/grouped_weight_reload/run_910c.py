#!/usr/bin/env python3
"""Run the installed CANN grouped MatMul via torch_npu; no DeepGEMM import.

This exercises the vendor implementation, not this repository's 950 kernel.
Run in an existing, compatible CANN / PyTorch / torch_npu environment.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m', nargs='+', type=int, default=[100, 300, 600])
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args()
    if min(*args.m, args.k, args.n) <= 0:
        parser.error('This smoke test requires positive dimensions and nonempty experts')

    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise SystemExit('No available NPU; check the installed driver and torch_npu environment')
    if not hasattr(torch_npu, 'npu_grouped_matmul'):
        raise SystemExit('Installed torch_npu does not expose npu_grouped_matmul')
    torch.npu.set_device(args.device)
    torch.manual_seed(1234)
    print(f'torch={torch.__version__}, torch_npu={getattr(torch_npu, "__version__", "unknown")}')
    print(f'device={torch.npu.get_device_name(torch.npu.current_device())}')

    # Tight concatenation: no manually inserted 256-row padding.
    # CPU tensors also define the FP32 reference, outside the GMM invocation.
    x_cpu = torch.randn(sum(args.m), args.k, dtype=torch.bfloat16)
    w_cpu = torch.randn(len(args.m), args.k, args.n, dtype=torch.bfloat16)
    ends = []
    total = 0
    for count in args.m:
        total += count
        ends.append(total)
    x = x_cpu.to(args.device)
    weight = w_cpu.to(args.device)
    group_list = torch.tensor(ends, dtype=torch.int64, device=args.device)

    # One input tensor, one stacked weight tensor, one concatenated output.
    # group_list_type=0 uses cumulative valid row counts: [100, 400, 1000].
    with torch.inference_mode():
        outputs = torch_npu.npu_grouped_matmul(
            [x], [weight], group_list=group_list,
            split_item=3, group_type=0, group_list_type=0,
        )
        torch.npu.synchronize()
    if len(outputs) != 1:
        raise AssertionError(f'Expected one concatenated output, received {len(outputs)}')
    y = outputs[0]
    if tuple(y.shape) != (sum(args.m), args.n) or y.device.type != 'npu':
        raise AssertionError(f'Unexpected output: shape={tuple(y.shape)}, device={y.device}')
    actual = y.cpu().float()
    start = 0
    for expert, end in enumerate(ends):
        reference = x_cpu[start:end].float() @ w_cpu[expert].float()
        sample = actual[start:end]
        if not torch.isfinite(sample).all():
            raise AssertionError(f'Expert {expert}: nonfinite output')
        denominator = (sample.square() + reference.square()).sum().item()
        diff = (sample - reference).square().sum().item() / max(denominator, 1e-30)
        if diff > 1e-5:
            raise AssertionError(f'Expert {expert}: normalized squared error {diff} > 1e-5')
        print(f'expert={expert}, M={args.m[expert]}, normalized_squared_error={diff:.3e}')
        start = end
    print(f'PASS: X={tuple(x.shape)}, W={tuple(weight.shape)}, Y={tuple(y.shape)}')
    print('Backend: installed CANN GMM. This is not the DeepGEMM-Ascend 950 kernel.')


if __name__ == '__main__':
    main()
