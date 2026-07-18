# MMRebuttal Supplement Plan

## 目标

本轮补实验服务 MMRebuttal，核心不是再证明方法能跑，而是补强审稿人/师兄关心的三个证据：

1. **Overlap 相关性**：证明 radius、displacement、fused/composite metric 和“DB 检索轨迹是否与 VLA/实际执行轨迹重叠”之间有定量关系。
2. **Case Study**：给出可视化案例，用橙色 VLA/实际执行轨迹和蓝色 DB 检索轨迹说明什么时候重叠、什么时候偏离，以及系统为什么选择 DB、Verify 或 fallback。
3. **各部分耗时**：按 reviewer 点名的模块拆开统计 ViT/视觉编码、LLM/AR、drafter、retrieval、verification 和必要的数据传输；不要把环境 step、视频保存、等待时间混进去。

一句话版本：我们不能只说“半径和位移可以判断轨迹是否稳定”，要拿数据证明 **DB trajectory 和 VLA/executed trajectory 重叠的片段** 与 **不重叠的片段** 在 radius / displacement / composite 分布上确实不同。

## 代码依据

HeiSD/SpecVLA 代码入口主要看：

```text
/path/to/SpecVLA/openvla/experiments/robot/libero
```

优先复用这些文件：

| 作用 | 文件 |
| --- | --- |
| 主方法入口，最近修改，包含 Block SD、2:1、composite 指标和 step 级日志 | `run_libero_block_sd.py` |
| 半径、位移、composite 指标定义 | `calc_r.py` |
| accept length / execution degree 分析 | `run_fenxi_accept_len.py`, `plot_accept_len.py` |
| top-k ambiguity / top-k retrieval 分析 | `run_fenxi_top5_ambiguity.py` |
| local Qdrant 检索耗时 | `local_qdrant_retrieval.py`, `test_qdrant_local_retrieval_speed_mix.py` |
| 参数消融脚本 | `run_alpha_ablation_*.sh`, `run_window_ablation_*.sh`, `run_libero_block_sd.sh` |

`run_libero_block_sd.py` 已经记录了很多需要的字段：`embedding_time`、`retrieval_time`、`generation_time`、`composite_metric`、`mode`、`accepted_tokens`、block verify 结果等。

但注意：这些字段只能做 accepted-length / verifier 一致性分析。要回应 reviewer 和 case study 图里的“trajectory overlap”，还必须补充记录：

```text
retrieved_action_trajectory      # DB 返回的 current_action + next_actions
executed_action_trajectory       # 同一 horizon 内 VLA/实际执行的 final action trajectory
trajectory_mean_deviation        # 两条轨迹逐点平均距离
trajectory_endpoint_error        # 两条轨迹终点距离
trajectory_overlap_ratio         # 误差低于阈值的点比例
```

第一版继续尽量少改主实验代码，但必须把整段 retrieval trajectory 存下来，不能只存第一个 action。

## 数据和数据库

使用当前正在重建的 RT-Cache/HeiSD mix-view Qdrant 数据库：

```text
dataset:  /data/zhihao/dataset/rtcache_libero
qdrant:   /data/zhihao/database/rtcache_mix_qdrant/storage
outputs:  /data/zhihao/outputs/mmrebuttal
```

mix collection 预期覆盖：

```text
libero_goal_mix:    10 collections
libero_10_mix:      10 collections
libero_object_mix:   9 collections
libero_spatial_mix: 10 collections
```

`libero_object=9` 是旧 RT-Cache mix 库也出现过的状态，先按数据实际 instruction hash 结果处理。

## 实验 1：Overlap / Non-overlap 与指标相关性

### Step-level 记录字段

每个推理 step 记录一行，输出为：

```text
outputs/mmrebuttal/overlap_correlation/step_records.jsonl
outputs/mmrebuttal/overlap_correlation/step_records.csv
```

字段：

```text
suite, task_id, task_description, episode_idx, step
success, mode
retrieval_success, top_k, top1_score
accepted_length 或 accepted_tokens
block_verified_count, block_ar_count
retrieval_action, target_or_generated_action
action_l2_error, action_pos_l2_error, token_error
raw_radius, raw_displacement
norm_radius, norm_displacement, composite_metric
embedding_time_ms, qdrant_search_time_ms
retrieval_time_ms, verify_time_ms, generation_time_ms
fallback_type
```

### Overlap 定义

主定义用同一动作空间里的 trajectory overlap，而不是 accepted length。

注意：RT-Cache payload 里的 action 前三维不能直接当作 EEF 米级 delta 去积分，否则会产生错误尺度。因此第一版使用和数据库 payload 同空间的 action xyz 子轨迹进行比较；这也对应图中 X/Y/Z 轨迹的可比动作维度。

```text
给定同一个 step t 和 horizon H：

orange trajectory = VLA/实际执行 action xyz 轨迹:
  A_exec = [a_t[:3], a_{t+1}[:3], ..., a_{t+H-1}[:3]]

blue trajectory = DB 检索 action xyz 轨迹:
  A_db = [a_db,t[:3], a_db,t+1[:3], ..., a_db,t+H-1[:3]]

mean_deviation = mean_i ||A_exec[i] - A_db[i]||
endpoint_error = ||A_exec[H-1] - A_db[H-1]||
overlap_ratio  = mean_i [ ||A_exec[i] - A_db[i]|| < eps ]
```

标签定义：

```text
overlap     = mean_deviation 小，并且 endpoint_error 小
non-overlap = mean_deviation 大，或者 endpoint_error 大
neutral     = 中间区域，不参与二分类显著性表
```

阈值建议第一版用 suite 内分位数，避免手调绝对阈值：

```text
overlap     = mean_deviation <= q25 且 endpoint_error <= q50
non-overlap = mean_deviation >= q75 或 endpoint_error >= q75
```

辅助定义保留 accepted length / action error，但只能作为交叉验证：

```text
verifier-overlap proxy     = accepted_length >= 5 或 accepted_tokens >= 5
verifier-non-overlap proxy = accepted_length <= 2 或 accepted_tokens <= 2
action-overlap proxy       = action_l2_error <= suite 内 q25
action-non-overlap proxy   = action_l2_error >= suite 内 q75
```

原因：accepted length 说明 verifier 对 DB token/action 的接受程度，但它不是图中“橙色轨迹和蓝色轨迹是否重叠”的定义。最终 rebuttal 主表应使用 trajectory-overlap 标签，accepted length 只作为补充相关性。

### 统计表

输出：

```text
outputs/mmrebuttal/overlap_correlation/summary_by_suite.csv
outputs/mmrebuttal/overlap_correlation/summary_by_task.csv
outputs/mmrebuttal/overlap_correlation/stat_tests.csv
outputs/mmrebuttal/overlap_correlation/plots/radius_distribution_by_overlap.png
outputs/mmrebuttal/overlap_correlation/plots/displacement_distribution_by_overlap.png
outputs/mmrebuttal/overlap_correlation/plots/composite_distribution_by_overlap.png
outputs/mmrebuttal/overlap_correlation/plots/radius_displacement_scatter_by_overlap.png
```

表格至少包含：

| group | radius mean | displacement mean | composite mean | mean deviation | endpoint error |
| --- | ---: | ---: | ---: | ---: | ---: |
| overlap | x | x | x | lower | lower |
| non-overlap | x | x | x | higher | higher |

显著性检验：

```text
Mann-Whitney U test
KS test
Cliff's delta 或 Cohen's d
Spearman correlation: metric vs mean_deviation
Spearman correlation: metric vs endpoint_error
Spearman correlation: metric vs accepted_length
ROC-AUC: radius/displacement/composite 预测 overlap
```

重点结论要能写成：

> Overlapping DB-VLA trajectory segments have significantly different radius/displacement/composite distributions from non-overlapping segments, indicating that the proposed kinematic signals are correlated with retrieval reliability.

### 分布图要求

这个图是回应 reviewer 质疑的主证据之一，不能省略。图里必须直接对比：

```text
overlap segments     vs. non-overlap segments
radius distribution
displacement distribution
composite distribution
```

建议出两版：

1. 主图：按所有 suite 汇总，画 violin/box + KDE/hist，直观看两组分布是否分开。
2. 附表/附图：按 suite 分面，避免某个 suite 的尺度或任务难度掩盖整体趋势。

图注和 rebuttal 文字不要写成“我们理论上认为 radius/displacement 能判断 overlap”，而要写成：

> We empirically divide trajectory segments into overlap and non-overlap groups based on DB-VLA trajectory deviation, and observe statistically different radius/displacement distributions between the two groups.

## 实验 2：Case Study

输出目录：

```text
outputs/mmrebuttal/case_study
```

选择 3-4 个 case：

1. **成功且高 overlap**：composite 高，DB/BlockSD 大量通过，accepted length 长。
2. **失败或低 overlap**：composite 低或波动大，accepted length 短，fallback 多。
3. **阈值切换案例**：同一个 episode 中从稳定段进入精细操作段，指标下降后切换到 verify/SD。
4. **top-k 歧义案例**：top-k actions 分散，ambiguity 高，检索不稳定。

每个 case 生成：

```text
case_{suite}_{task}_{episode}/timeline.csv
case_{suite}_{task}_{episode}/trajectory_3d.png
case_{suite}_{task}_{episode}/metric_timeline.png
case_{suite}_{task}_{episode}/accept_timeline.png
case_{suite}_{task}_{episode}/mode_timeline.png
case_{suite}_{task}_{episode}/frames/
case_{suite}_{task}_{episode}/summary.md
```

图里要放：

```text
3D trajectory:
  orange = VLA inference / actual executed EEF trajectory
  blue   = DB retrieval trajectory
  annotate = small/large trajectory deviation, endpoint match/mismatch

timeline:
  x-axis: step
  y-axis: radius / displacement / composite / trajectory deviation / accepted length
  background color: mode, e.g. DB / BlockSD / SD / AR fallback
```

最终论文/回复里用一张组合图：

```text
左：环境关键帧
中：EEF 轨迹 + 检索轨迹
右：metric timeline + accepted length / mode
```

## 实验 3：各部分耗时拆解

原则：不统计环境时间、视频保存、等待时间、日志写入、episode wall time。只统计模型和检索路径本身。

输出：

```text
outputs/mmrebuttal/timing_breakdown/timing_records.csv
outputs/mmrebuttal/timing_breakdown/summary_by_suite.csv
outputs/mmrebuttal/timing_breakdown/summary_by_mode.csv
outputs/mmrebuttal/timing_breakdown/timing_stacked_bar.png
```

拆分字段：

| component | 含义 |
| --- | --- |
| vit_or_embedding_time_ms | two-view OpenVLA mix embedding / ViT image encoding |
| qdrant_search_time_ms | Qdrant top-k search |
| retrieval_total_ms | embedding + search + payload decode |
| tokenization_time_ms | action-to-token / token-to-action |
| llm_forward_time_ms | AR / target model LLM forward |
| drafter_time_ms | drafter proposal generation |
| block_verify_time_ms | BlockSD verify forward |
| sd_time_ms | 原始 SD 推理 |
| ar_time_ms | AR fallback / target model generation |
| noverify_time_ms | 直接使用 DB action 的本地开销 |
| cpu_gpu_transfer_ms | 必要的数据搬运/embedding transfer；若未单独测量，不能混进主结论 |

需要给两类表：

1. **每种 mode 的平均耗时**

| mode | count | mean ms | median ms | p90 ms |
| --- | ---: | ---: | ---: | ---: |
| Retrieval_DB | x | x | x | x |
| Retrieval_BlockSD_fully_verified | x | x | x | x |
| Retrieval_BlockSD_partial_AR | x | x | x | x |
| SD | x | x | x | x |
| AR fallback | x | x | x | x |

2. **suite 级耗时占比**

| suite | embedding | qdrant | verify | SD | AR | noverify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_goal | x% | x% | x% | x% | x% | x% |

注意：如果用 HTTP retrieval server，会混入网络传输；MMRebuttal 主表优先用 local Qdrant / local embedding 或明确标注“server overhead included”。师兄要的“各部分耗时”建议主表不放网络通讯时间。

## 实验 4：轻量参数消融

如果时间允许，补一个小规模 ablation，说明 composite 不是拍脑袋：

```text
alpha: 0.3, 0.5, 0.7
window_size: 5, 10, 15
threshold: suite default ± small grid
```

复用：

```text
run_alpha_ablation_cuda0.sh
run_alpha_ablation_cuda1.sh
run_window_ablation_cuda0.sh
run_window_ablation_cuda1.sh
```

输出：

```text
outputs/mmrebuttal/ablation/alpha_summary.csv
outputs/mmrebuttal/ablation/window_summary.csv
```

这里不用跑很大，目标是证明选择的 alpha/window 不离谱，趋势稳定。

## 实验规模

### Pilot

先跑：

```text
4 suites x task 0-1 x 3 episodes
```

目的：

```text
确认 step_records 字段完整
确认 overlap/non-overlap 标签数量足够
确认 timing 没混入 env step
确认 case study 能自动画图
```

### 正式统计

建议：

```text
4 suites x all tasks x 10 episodes
```

这个规模足够做分布和显著性分析，不必先上 full30。若统计不稳定，再扩到 30 episodes。

## 实现步骤

### 1. 补 step-level 记录

在 MMRebuttal 下新增脚本，不直接大改 SpecVLA 原始代码：

```text
experiments/mmrebuttal/collect_heisd_step_records.py
experiments/mmrebuttal/analyze_overlap_metrics.py
experiments/mmrebuttal/analyze_timing_breakdown.py
experiments/mmrebuttal/plot_case_studies.py
experiments/mmrebuttal/run_mmrebuttal_pilot_tmux.sh
```

如果必须改 SpecVLA，只加一个开关：

```text
--mmrebuttal_record_step_metrics
--mmrebuttal_output_dir /data/zhihao/outputs/mmrebuttal/...
```

默认关闭，避免影响原 HeiSD 代码。

### 2. 从已有 JSON 兼容读取

先尝试直接读 `run_libero_block_sd.py` 输出的 `_block_sd.json`。如果字段不够，再补运行时记录：

```text
all_data[*].steps[*].composite_metric
all_data[*].steps[*].embedding_time
all_data[*].steps[*].retrieval_time
all_data[*].steps[*].generation_time
all_data[*].steps[*].mode
all_data[*].steps[*].accepted_tokens
all_data[*].steps[*].blocks
```

### 3. 补缺失字段

可能需要新增：

```text
raw_radius
raw_displacement
norm_radius
norm_displacement
top1_score
action_l2_error
token_error
qdrant_search_time_ms
```

这些都可以在 `run_libero_block_sd.py` 当前 step 里拿到或计算出来。

### 4. 分析与出图

分析脚本统一输出 CSV、PNG、Markdown：

```text
outputs/mmrebuttal/summary.md
outputs/mmrebuttal/overlap_correlation/*.csv
outputs/mmrebuttal/overlap_correlation/*.png
outputs/mmrebuttal/timing_breakdown/*.csv
outputs/mmrebuttal/timing_breakdown/*.png
outputs/mmrebuttal/case_study/*/*.png
```

## 最终交付

最终给论文/rebuttal 的材料：

1. **一张 overlap/non-overlap 统计表**：radius、displacement、composite、accepted length、retrieval error。
2. **一张分布图**：overlap vs non-overlap 的 composite/radius/displacement violin 或 box plot。
3. **一张 ROC 或 correlation 图**：说明 composite 对 overlap 有预测能力。
4. **一张 case study 图**：关键帧 + metric timeline + mode/accepted length。
5. **一张耗时 breakdown 表/堆叠柱状图**：证明加速来自哪些部分，瓶颈在哪里。

## 风险和处理

| 风险 | 处理 |
| --- | --- |
| overlap/non-overlap 样本不平衡 | 同时用 accepted length 和 action error 分位数定义，保证两组都有样本 |
| HTTP retrieval 混入网络时间 | 主表使用 local Qdrant 或拆出 server overhead |
| object suite 只有 9 个 collection | 按实际 instruction hash 结果记录并说明 |
| PDF 实验列表无法文本抽取 | 先按师兄口头需求和代码入口落地，后续如有截图/OCR内容再补 |
| 原始 HeiSD 代码改动过大 | 所有新增逻辑优先放 MMRebuttal，SpecVLA 只加默认关闭的记录开关 |

## 当前优先级

1. 等 RT-Cache mix Qdrant clean build 完成并通过 inspect。
2. 跑 pilot，生成 step_records。
3. 做 overlap/non-overlap 统计和耗时 breakdown。
4. 选 case study 并出图。
5. 再决定是否补 alpha/window ablation。
