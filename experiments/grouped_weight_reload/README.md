# MoE GroupedMatmul 专家权重重复加载实验

这个实验针对本仓库的 **BF16 M-grouped GEMM**，区分两件事：

1. 同一权重是否为不同 M tile 重复发起 GM→L1 搬运：由执行路径和分块数量推导。
2. 重复请求是否再次访问 HBM，还是被 L2 满足：由设备 PMU 数据判断。

脚本不会把推导流量填写成实测 HBM 流量，也不会根据耗时单独宣布存在 HBM 重读。只测试一个专家线性层，不包含路由、通信、激活、第二层和 combine。

## 实验依据

当前 `csrc/jit/jit.hpp` 把 M-grouped 对齐固定为 256；`csrc/jit_kernels/heuristics.hpp` 要求 M-grouped 的 `block_m` 等于该对齐值。因此这里固定 `BLOCK_M=256`，不修改 kernel 或启发式。

`deep_gemm/include/deep_gemm/bf16_gemm.hpp` 中，`kResidentB` 仅允许用于 Normal GEMM。M-grouped 每个输出 tile 都进入 K 循环执行 `copy_gm_to_l1`。对所有 N/K tile 求和，得到权重**有效元素的逻辑请求量**：

```text
weight_bytes = experts × N × K × 2
m_tiles      = ceil(tokens_per_expert / 256)
model_weight_gm_to_l1_bytes = weight_bytes × m_tiles
```

不计 cache sector、总线传输粒度等放大；不是 DMA 指令数，也不是 PMU 实测值。模型不应直接用于有驻留复用的 FP8 路径。

默认 E=4、N=K=2048，所有专家权重合计 32 MiB：

| 每专家有效 token | 实际占用行数 | 每专家 M tile | 模型权重请求量 |
|---:|---:|---:|---:|
| 64 / 255 / 256 | 256 | 1 | 32 MiB |
| 257 / 512 | 512 | 2 | 64 MiB |
| 513 | 768 | 3 | 96 MiB |
| 1024 | 1024 | 4 | 128 MiB |

每个专家的有效区间是 `[e * stride, e * stride + M)`，PSUM 为这些区间的末端，不是简单的 `cumsum(M)`。输入 padding 清零。所有 case 共享同一份权重；M 改变时输入重新生成。输出使用 BF16，每专家抽查首尾最多 8 个有效行，对比 CPU FP32 matmul，归一化平方误差必须不大于 `1e-5`。这只是抽样精度检查，不是全量验证。

## 运行

无需 torch/NPU，可先查看模型与 PSUM：

```bash
python3 experiments/grouped_weight_reload/run.py --dry-run
```

在已经能够运行本仓库 BF16 测试的 Ascend 950 环境中，从仓库根目录运行：

```bash
python experiments/grouped_weight_reload/run.py \
  --output results/reload_small
```

脚本打印实际 JIT 配置，先编译与抽查精度，再用仓库 `bench_msprof` 测量 GEMM kernel。`results.csv` 随每次测量写出，`metadata.json` 记录设备、L2 大小、版本、参数与 PSUM。输出目录必须不存在，以避免覆盖旧实验。保留终端配置日志可使用 shell 的 `tee`。

默认比较两种条件：

- `warm`：同一地址反复运行，不主动清理缓存。**不保证权重全在 L2**。
- `evict`：每次 GEMM 前写入 2 倍设备报告 L2 容量的独立 buffer，尝试驱逐旧缓存。flush kernel 不计入报告的 GEMM 耗时；该方法**不保证严格冷缓存**，需用 PMU 确认。

两者区分的是**调用之间**缓存状态；evict 不会禁止一次 GEMM 内部的 L2 复用。不要将 evict 情况解释为每次 load 都绕过 L2。

为了观察容量影响，可另外运行更大的权重工作集：

```bash
python experiments/grouped_weight_reload/run.py \
  --experts 16 --n 4096 --k 4096 --m 256 257 512 513 1024 \
  --output results/reload_large
```

此配置权重约 512 MiB，另需输入、输出、flush、profiler 及参考检查的内存空间。根据实际可用内存调整 E/N/K，并用 `metadata.json` 的设备 L2 大小确认大小关系。不同 E/N/K 的耗时不能直接归因于缓存：并行度和计算量也改变了。

## 采集硬件指标

建议先单 case 运行，确认当前 CANN/设备支持计数器：

```bash
python experiments/grouped_weight_reload/run.py \
  --m 256 257 512 513 --iterations 30 \
  --trace-metrics MemoryAccess L2Cache --trace-iterations 5 \
  --output results/reload_counters
```

每种指标分开采集，以免不同 PMU 配置互相覆盖。采集结果保存在例如 `m257_evict_MemoryAccess/` 的独立目录中，保留 profiler 原始文件和导出文件；时间测量与这些 counter pass 分开。查看 `ASCEND_PROFILER_OUTPUT` 下实际生成的 CSV/数据库，匹配 `bf16_gemm` kernel；排除 `zero_` 等缓存驱逐辅助 kernel，并按 GEMM 调用数归一化。若指标不受支持或导出没有相关列，本次数据不能回答对应问题；不要把缺失值当作 0。脚本会拒绝 torch_npu 未暴露的指标名称，但硬件支持情况仍须核对采集日志和输出。

优先核对：

- GM→L1 / MTE2 数据量（如果设备提供），是否随 M tile 数增加。
- L2 读命中/缺失及请求量，重复请求是否主要由 L2 满足。
- HBM/DDR 实际读字节数或读带宽；只有带宽时，须先核对其统计时间窗口和单位，不能直接乘另一次测量的 kernel 时间。

可用字段随 CANN 和设备改变；`MemoryAccess` 不保证一定导出 HBM 字节数。详细配置参考 [torch_npu profiler 官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_001391.html) 和 [Ascend AiCMetrics 文档](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-profiler/torch_npu-profiler-AiCMetrics.md)。如果只有 L2 命中率，没有 HBM 字节计数，结论只能是缓存复用的证据，不能声称测出了权重 HBM 重读倍数。

## 如何判断

1. **逻辑加载重复**：M 从 256 到 257，权重地址和大小相同，模型请求遍数从 1 到 2。该结论基于实际 BF16 kernel 路径；PMU 如提供相应字节数，可交叉验证。
2. **缓存掩盖重复请求**：逻辑请求量增加，而 HBM 读量增长很小、L2 命中较高，支持“重复请求由缓存满足”。
3. **HBM 重读证据**：evict 条件下，单次 GEMM 实际读量显著超过一次读取输入与全部权重所需字节，且随 M tile 数增加，支持 HBM 重读；仍须排查 A 的重复加载和计数范围。
4. **同 tile 数对照**：255 与 256、257 与 512 的实际分块工作量相同，可检查 padding 与计时稳定性。不同 tile 数时 padded FLOPs、并行度也增加，所以耗时台阶本身不能证明带宽瓶颈。

PMU 是整个 kernel 的统计，无法自动区分输入 A、权重 B 和输出 D。尤其 A 会沿 N tile 重复请求，HBM 总读量超过 `weight_bytes` 并不能单独证明权重重读。要报告精确的权重重读倍数，需要进一步用可归因的硬件计数或受控的 kernel 变体排除其他流量。CSV 的 `measured_aic_mte2_ratio` 是管线忙碌比例，**不是字节数或 HBM 利用率**。

所有预期都属于实验假设；真实结论必须附设备、配置日志与 PMU 数据。当前脚本不生成虚构的 NPU 性能结果。
