# MoE GroupedMatmul 专家权重重复加载实验

## 910C + CANN 9.0.0：先运行独立 smoke test

本仓库 README 的构建目标是 Ascend 950 + CANN 9.20。`run.py` 默认仍使用该后端；在 910C 上要显式选择 `--operator-backend cann`，不需要编译或导入 DeepGEMM。先用以下脚本检查已安装的 PyTorch/torch_npu/CANN 环境：

```bash
python experiments/grouped_weight_reload/run_910c.py --m 100 300 600 --k 512 --n 1024
```

`run_910c.py` 只导入 torch 与 torch_npu，调用 `npu_grouped_matmul`，无需安装或编译本仓库；默认使用 BF16，并对每个专家的完整输出执行 CPU FP32 参考校验。输入按有效 token 紧密拼接，`group_list=[100,400,1000]`，不手动插入 256 行 padding。该接口支持情况与实际使用的 torch_npu/CANN 版本有关，须使用适配 CANN 9.0.0 的软件组合。

这会调用 CANN 提供的另一份 GMM 实现，不能用来验证本仓库固定 BLOCK_M=256 的调度与加载量模型。本地仅检查了脚本语法与分组边界，尚未在 910C 上实测。参考 [torch_npu GMM 文档](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)；C++ 调用可参考 CANN 的 `aclnnGroupedMatmulV5` 接口，CANN 9.0.0 已将旧版本接口标记为废弃。

## 910C：比较负载不均与逐轮耗时方差

先预览默认负载（不需要 torch/NPU）：

```bash
python experiments/grouped_weight_reload/run.py --operator-backend cann --dry-run
```

默认原始 token=64、top-k=8、专家=16，总 routed rows=512，自动生成三组：

- `balanced`：每专家 32 行。
- `skewed`：`[64,64,64,64,64,64,64,56,1,1,1,1,1,1,1,1]`。
- `skewed_reversed`：将 skewed 的专家位置反转，负载大小的集合不变。

三组的有效专家数、总行数、有效 FLOPs、权重大小相同。每专家行数不超过原始 token 数，符合一个 token 不重复选择同一专家的 top-k 计数约束。输入是人工构造的 `X_perm`；脚本不实际执行路由，也不构造每个原始 token 的重复副本，不运行 Attention、通信或激活。目的在于控制 GMM 的输入布局与负载分布。

运行均衡/不均衡对照，并采集当前环境支持的管线、访存、资源冲突与 L2 指标：

```bash
python experiments/grouped_weight_reload/run.py \
  --operator-backend cann --experts 16 --tokens 64 --topk 8 \
  --n 2048 --k 2048 --iterations 100 \
  --trace-metrics PipeUtilization Memory ResourceConflictRatio L2Cache \
  --trace-iterations 20 --output results/cann_decode_small
```

要覆盖部分专家超过 256、部分不足 256 的分配，增加原始 token 数：

```bash
python experiments/grouped_weight_reload/run.py \
  --operator-backend cann --experts 16 --tokens 512 --topk 8 \
  --n 2048 --k 2048 --iterations 100 \
  --trace-metrics PipeUtilization Memory ResourceConflictRatio \
  --output results/cann_decode_large
```

此时 balanced 为每专家 256 行，skewed 同时有大、小专家。**256 只是我们选择的输入边界，不是已知的 CANN tile 大小**。CANN 的分块、驻留与尾块处理需看真实 kernel/tiling 和 profiler，脚本将 `actual_block_m`、`actual_output_tiles`、`model_weight_gm_to_l1_bytes` 留为 null。

默认比较 warm 与 evict：warm 不主动清理缓存；evict 在每次调用前写入 2 倍设备报告 L2 容量的 buffer 并同步，flush 在计时区间外。它只尝试驱逐跨调用缓存，不保证完全驱逐，也不禁止本次 GMM 内部的复用。若 torch_npu 未报告 L2 容量，必须手动指定驱逐 buffer，或选择 `--cache warm`。例如，**仅在你确认 L2 为 192 MiB 时**可以指定 `--flush-mib 384`。

### 自定义负载与固定重放

`--case NAME:COUNT,...` 可以重复传入，每组必须有 `--experts` 个数，和为 `--tokens × --topk`，每个数在 `[0, tokens]`。显式 case 会替代默认三组。零 token 专家允许输入，但应核对当前 CANN 支持情况；各 case 激活专家数不相同时，不能视作相同权重工作集对照。

例如总分配数均为 1024、同样激活 4 个专家：

```bash
python experiments/grouped_weight_reload/run.py \
  --operator-backend cann --experts 4 --tokens 512 --topk 2 \
  --case balanced:256,256,256,256 \
  --case skewed:512,256,128,128 \
  --iterations 100 --output results/cann_custom
```

只保留一个 case 就是固定输入/权重/分组重放，可排除轮间路由变化：

```bash
python experiments/grouped_weight_reload/run.py \
  --operator-backend cann --experts 4 --tokens 512 --topk 2 \
  --case replay:512,256,128,128 --cache warm \
  --iterations 100 --trace-metrics PipeUtilization ResourceConflictRatio \
  --output results/cann_replay
```

`--m` 仍是原 DeepGEMM 的均匀 M 扫描参数，不能在 CANN 路径使用。`--backend persistent/fast/slow` 也是原 DeepGEMM profiler 参数，CANN 使用独立 torch_npu profiler。

### 输出与判读

- `metadata.json`：所有 case 的计数/紧密 PSUM、字节数、设备/软件版本、输入/权重地址和抽样正确性结果。
- `samples.csv`：每个 case 每轮的 `operator_event_us`、`host_call_us`、输出地址和调用顺序。默认每轮随机打乱 case 顺序，种子固定；`--order fixed` 则按 case 列表顺序轮流调用。两个选项都没有 Attention/通信重叠，每次调用完成后同步。
- `results.csv`：各 case/cache 条件的均值、标准差、CV、p50/p95、最小/最大值。**operator_event_us 是 NPU event 包围一次调用的时间窗口，不是纯 kernel/MTE2 时间**；它可能包含 host 发射期间的设备空档。
- `trace_status.json`：每个独立 PMU pass 的成功/失败状态；不支持的指标会警告并记录错误，保留已完成的计时，不填成 0。
- `traces/<case>/<cache>/<metric>/`：原始 profiler 导出，`collection.json` 记录 case 和调用数；CPU trace 标记每个调用为 `gmm/<case>/<cache>/<metric>/call_N`。
- 同目录 `matched_rows.jsonl`：按 `--kernel-pattern` 匹配的 CSV 原始行，保留原字段名/单位、文件名和行号；不会擅自将 per-core 行相加、推断单次 load 延迟或转换缺失指标。`export_manifest.json` 列出导出 CSV。没有匹配行时检查原始 CSV/数据库与实际 kernel 名，必要时修改 regex。

同一次运行中所有 case 共用相同的输入和权重存储。输出由 torch_npu 每次调用分配，脚本不假设存在固定输出 buffer API，记录地址便于检查；host 调用时间也独立保存。每个 case 都先执行并抽查每专家首尾最多 8 个有效行，对比 CPU FP32，随后预热。计时与 PMU 分开采集，PMU pass 不可直接对应为 samples.csv 中某条慢样本，也不能用另一遍的耗时换算带宽。

建议先看以下证据，不自动宣布某个原因成立：

| 现象 | 下一步核对 |
|---|---|
| 固定 replay 仍波动 | 外部竞争、设备状态、host 发射开销、逐核表现；不能再归因于路由变化 |
| 负载对照的算子窗口时间不同 | 查实际 kernel/tiling、真实搬运量和 PMU；相同有效 FLOPs 不保证实际工作量相同 |
| MTE2 执行时间相近，但整体时间变长 | 查同步与下游管线；并非直接证明 L1 buffer 等待 |
| MTE2/队列等待相关计数增加 | 结合具体字段定义与 MTE1/Cube/Fixpipe 指标分析，队列拥塞不等于 L1 等待 |
| 同样流量但活跃带宽下降 | 查并发竞争、访存分布及测量口径 |
| 原始指标缺失或不受支持 | 本次采集不足以验证该假设，使用匹配 CANN 9.0.0 的 msopprof 工具进一步分析 |

各 CANN/芯片支持的指标不完全相同，不能用其他版本的字段定义代替当前导出。参考 [Profiler 配置说明](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_001391.html)。脚本没有修改设备 kernel，也没有直接测出单条 GM→L1 指令或 L1 buffer 等待时长。

CPU-only 验证：

```bash
python -B -m unittest discover -s experiments/grouped_weight_reload -p test_experiment.py -v
```

## 单专家：观察 M 增大时的权重重读证据

`--single-expert-m` 在**同一个进程**中扫描不同 M，自动选择 experts=1、topk=1，不与 `--case` 同用。所有 case 共用一份权重和最大输入的前 M 行视图，权重大小/地址、输入基地址保持一致；与负载对照不同，这个实验有意改变有效总行数。`--tokens` 不参与该扫描的计数生成。

先比较 100 与 600，权重取 K=7168、N=4096 的 BF16 矩阵，大小恰好 56 MiB：

```bash
python experiments/grouped_weight_reload/run.py \
  --operator-backend cann --single-expert-m 100 600 \
  --k 7168 --n 4096 --cache warm --iterations 100 \
  --trace-metrics PipeUtilization L2Cache Memory MemoryAccess \
  --kernel-pattern 'matmul|gemm' \
  --trace-iterations 20 --output results/one_expert_reload_probe
```

只有 PipeUtilization 不能判断 HBM 重读。L2Cache/Memory/MemoryAccess 是否支持、能否给出有效字节数，取决于设备与 torch_npu/CANN 版本；检查 `trace_status.json` 和原始导出，失败或缺失不视为 0。

如果 100/600 都没有明显差异，可以扩大扫描至 `--single-expert-m 100 600 2048 8192 16384`。以假设的 192 MiB L2 为例，56 MiB 权重配合 600 行输入/输出的唯一数据量约 69 MiB，不一定产生容量压力；16384 行时输入本身约 224 MiB，更适合探索压力下的行为。**唯一数据量超过 L2 仍不保证权重一定被淘汰并重读**，这还依赖访问顺序、替换策略与后续复用。当前代码不预设 CANN 的 M tile 大小，扩大 M 也可能改变 kernel/tiling。

为排除跨调用热缓存，另跑一次 `--cache evict`，指定新输出目录。设备能报告 L2 时自动用 2 倍容量的 buffer；否则使用你确认过的容量设置 `--flush-mib`。驱逐发生在调用之前，不能用它证明调用内部发生了淘汰，也不能与“禁用 L2”混为一谈。冷启动第一次加载是基线，不能统计成权重重读。

### 证据分别支持什么

| 需要验证的环节 | 可以观察的证据 | 不能单独当证据的现象 |
|---|---|---|
| 多个 M tile 重复请求权重 | 实际 tiling/反汇编确认同权重地址被多任务读取；或可归因的权重搬运字节数 | MTE2 时间增长、假定 BLOCK_M=256 |
| L2 复用变差 | L2 读命中/缺失与请求量随 M 变化 | 只有全算子的低命中率：首次输入读取也会 miss |
| HBM 重读 | 同次采集中确认的 HBM/GM 实际读字节数，排除输入重复读取及 workspace 后的权重读取放大 | 读带宽增长、总读量超过单份权重大小 |
| 因输入挤出权重而重读 | 在固定 tiling 下只改变 A/output 的缓存分配策略，B 策略不变，看到权重侧 miss/重读随之改变 | 同时改变 M、kernel、权重大小或关闭全部 L2 |

总读取量包含 A、B 和可能的内部 workspace，不能直接用 `(HBM_bytes - unique_input_bytes) / weight_bytes` 宣称权重重读倍数，因为 A 也可能沿 N tile 被重复读取。保留每次调用、每核和总量的不同统计口径；不要把 L2 命中百分比相加，也不要把另一遍计时乘到本遍带宽上。

### torch_npu 导出不够时

可进一步使用与本机 CANN 9.0.0 配套的 `msprof op`，先检查 `msprof op --help`。官方工具的 MemoryDetail 提供 GM 数据传输量与缓存相关视图，并支持 A3；具体功能/命令以本机版本为准。先从已有导出获取真实 kernel 名，再按该名字筛选，避免把启动时的数据拷贝当成 GMM。**单专家时不能假定底层 kernel 名一定含 GroupedMatmul**。

不要在 `msprof op` 外部采集时同时传入本脚本的 `--trace-metrics`，以免嵌套 profiler。也要核对 replay 模式：官方文档说明 kernel/range replay 会清 L2，application replay 不主动清 L2；它们得到的缓存初始条件不能混比。参考 [msOpProf 指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)。如果匹配版本的工具没有可归因的权重数据，结果应表述为“支持/不支持重读假设”，不能宣称完整证明淘汰因果链。

本地只运行了 Python-only 检查，没有在 910C 上实际采集。特别是旧版工具可能不提供指令级地址轨迹；不要将新版 950 专用指令流水功能假定为 910C 可用。

## Ascend 950 源码实验

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
