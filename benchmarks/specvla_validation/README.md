# SpecVLA

本仓库是论文 **Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance (EMNLP 2025)** 的代码实现，基于 OpenVLA 派生。

除了 SpecVLA（SD + Relaxed Acceptance）主线实现外，本仓库还包含一套 **DB 检索（Qdrant + Retrieval Server）** 的实验代码，用于探索：

1) 交替执行：DB 取动作片段（action slice）与模型生成交替执行（已实现脚本入口）。
2) 下一步想做：**DB 检索 + Speculative Decoding 精确结合**（把检索到的 action slice 作为 SD 的 draft / prior，交给 target 模型精确验证并统计接受长度，属于进行中 idea）。

---

## 1. 项目目录（代码地图）

建议从下面几个目录理解整个工程：

```
SpecVLA/
├── openvla/
│   ├── experiments/robot/               # 机器人/仿真评测入口（LIBERO 等）
│   │   └── libero/                      # 你当前最主要关注的脚本目录（AR / Spec / Relaxed / DB 实验入口）
│   ├── specdecoding/                    # SpecVLA 核心：draft/verify、tree-attention、动态解码等
│   │   ├── model/                       # speculative decoding 组件（KV cache、EA model、utils 等）
│   │   ├── train-scripts/               # draft 模型训练脚本、ckpt、ds config
│   │   └── test-speed/                  # LIBERO 测速/日志输出目录（脚本会往这里落日志）
│   ├── prismatic/extern/hf/             # OpenVLA/SpecVLA HF wrapper（predict_action 等）
│   └── scripts/                         # OpenVLA 相关脚本（派生自上游）
├── LIBERO/                              # LIBERO 基准与环境依赖（子模块式结构）
├── backbone_models/                     # 微调后的 OpenVLA checkpoint（如 openvla-7b-finetuned-libero-goal）
├── dataset/                             # 训练/生成数据（可选）
├── exp/                                 # 实验产物：accept length 统计 npz、分析图等
└── rollouts/                            # rollout 结果（按日期分目录）
```

补充说明：

- `openvla/experiments/robot/libero/` 是 LIBERO 评测的“入口脚本层”。
- `openvla/specdecoding/` 是 SpecVLA（speculative decoding）核心实现。
- `openvla/prismatic/extern/hf/` 是模型封装与推理逻辑入口（`predict_action` 等）。

---

## 2. 环境与安装

已测试组合（来自仓库现有说明）：

- Python >= 3.10
- PyTorch == 2.2.0（CUDA 12.1 测试过）
- LIBERO == 0.1.0

基础安装：

```bash
pip install -r requirements-min.txt
cd openvla
pip install -e .
```

运行 LIBERO 时常用环境变量（按需调整 GPU）：

```bash
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO
export MUJOCO_GL=egl
export ROBOSUITE_LOG_FILE=$PWD/robosuite.log
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
```

---

## 3. LIBERO 评测脚本入口（重点）

目录：`openvla/experiments/robot/libero/`

### 3.1 基线：AR / Spec / Relaxed

- **AR（OpenVLA 原始自回归）**：`run_libero_goal_AR.py`
- **Speculative Decoding（严格接受）**：`run_libero_goal_Spec.py`
- **Speculative Decoding + Relaxed Acceptance**：`run_libero_goal_Spec_Relaxed.py`

示例（libero_goal）：

```bash
python openvla/experiments/robot/libero/run_libero_goal_AR.py \
  --model_family openvla \
  --pretrained_checkpoint $PWD/backbone_models/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True
```

Spec / Relaxed 需要额外提供 draft checkpoint（`--spec_checkpoint`）以及阈值（`--accept_threshold`）。

### 3.2 DB 检索相关：交替执行（已实现）

当前仓库里“能跑通”的 DB 相关模式主要是 **交替执行**：

- AR + DB 交替：`run_libero_goal_AR_DB.py`
- Spec/Relaxed + DB 交替：`run_libero_goal_Spec_Relaxed_DB.py`

这类脚本的核心参数是：

- `--db_steps N`：连续执行 N 个 DB 检索动作
- `--model_steps M`：连续执行 M 个模型生成动作

这条路径用于快速评估“检索动作片段”对速度/成功率/时延的影响，但它并不是严格意义上的“DB 作为 SD draft”的精确结合。

---

## 4. DB 检索服务（Qdrant + Retrieval Server）

DB 检索依赖外部组件（你本机路径里常见是 `rtcache`）：

1) 启动 Qdrant 数据库
2) 启动 embedding server（默认 9020）
3) 启动 retrieval server（默认 5002，提供 `/pipeline`）

仓库内有更详细的追踪说明：

- `openvla/experiments/robot/libero/README_DB_TRACKING.md`

该文档描述了如何记录“DB 检索动作片段的接受长度”与如何在 `exp/` 下落盘 npz 与可视化结果。

---

## 5. 当前要实现的 Idea：DB 检索 + SD 精确结合（整理版）

### 5.1 目标

把 DB 检索得到的 **action slice** 当成 Speculative Decoding 的“外部 draft”（或强 prior），由 target 模型进行 **精确验证**，并输出：

- 实际接受了 action slice 的前多少步（accept length）
- 未接受部分如何回退到模型生成（fallback）
- 对速度与成功率的影响（与 AR、Spec、Relaxed、交替模式对比）

### 5.2 与“交替执行”的区别

- 交替执行：DB 给 1~K 步动作，直接执行；然后再用模型生成若干步。
- 精确结合（想做的）：DB 给出一段候选动作序列，**不直接执行**，而是作为 SD 的 draft，让 target 模型逐步验证并决定接受长度；这样 acceptance 统计在机制上更可解释，也更贴近 SD 框架。

### 5.3 建议的数据流（最小闭环）

1) 输入：当前观测（图像）+ 指令
2) Retrieval：返回候选轨迹 `db_action_slice`（shape: [L, 7] 或等价格式）
3) Draft 生成：将 `db_action_slice` 转换为 action tokens（或作为 draft token 序列）
4) Verify：target 模型验证，得到 `accept_lengths` 与最终动作
5) 执行：执行最终动作（或连续动作）
6) 记录：保存 `db_accept_lengths`、`db_action_slices`、整体耗时等

### 5.4 代码落点（你后续要改/要合并的关键文件）

当前仓库里已经出现了一些“DB + SD 精确结合”的实验性代码形态，但它们主要以 `* copy.py` 的形式存在：

- `openvla/experiments/robot/openvla_utils copy.py`：在 `get_vla_action(...)` 增加 `track_accept_length/db_action_slice/use_db_action_slice`，并解析 `predict_action` 的扩展返回值。
- `openvla/prismatic/extern/hf/modeling_speculation copy.py`：在 `predict_action(...)` 中接收 `db_action_slice/use_db_action_slice`，并从 `eagenerate` 返回 DB tracking 数据。
- `openvla/experiments/robot/libero/run_libero_goal_Spec_Relaxed copy.py`：尝试把 retrieval 得到的 slice 传入推理并统计接受长度。

建议后续整理方向：

1) **把 `* copy.py` 的改动合并回主文件**（去掉“copy 分叉”带来的路径/接口不一致），优先合并到：
   - `openvla/experiments/robot/openvla_utils.py`
   - `openvla/prismatic/extern/hf/modeling_speculation.py`
   - `openvla/experiments/robot/robot_utils.py`
2) 让 `run_libero_goal_Spec_Relaxed.py` 成为唯一主入口，通过 flag 开关 DB+SD 逻辑（避免多个脚本重复演进）。

---

## 6. 结果与日志位置

- 评测日志通常会写入：`openvla/specdecoding/test-speed/`（按脚本配置目录命名）
- DB 接受长度追踪（npz / png）会写入：`exp/`（参考 `openvla/experiments/robot/libero/README_DB_TRACKING.md`）

---

## 7. Draft Model Checkpoints (LIBERO)

我们提供了四个 LIBERO suite 的 draft model checkpoint，可直接用于 SpecVLA speculative decoding：

| LIBERO Task Suite | Draft Model Checkpoint |
|-------------------|------------------------|
| **LIBERO Goal**   | [Download](https://drive.google.com/drive/folders/1W7nBHM9-bf9tq4NQDUfles583OUfrtLv?usp=share_link) |
| **LIBERO Object** | [Download](https://drive.google.com/drive/folders/1HHQv5iRMXRSfajjIgB62h_jMGboXu250?usp=share_link) |
| **LIBERO Spatial**| [Download](https://drive.google.com/drive/folders/1Het7jUEiWSObG8Tn7H2CjmqdH_XdJr5x?usp=share_link) |
| **LIBERO 10**     | [Download](https://drive.google.com/drive/folders/1LhV2bAzdivbaz6MM1Owl_jNCCdnNs-8W?usp=share_link) |

---

## Citing

如果你使用了本仓库代码，请引用：

```bibtex
@article{wang2025spec,
  title={Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance},
  author={Wang, Songsheng and Yu, Rucheng and Yuan, Zhihang and Yu, Chao and Gao, Feng and Wang, Yu and Wong, Derek F},
  journal={arXiv preprint arXiv:2507.22424},
  year={2025}
}
```
