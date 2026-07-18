# SpecVLA 项目记录

## 2025-12-21: 纯数据库检索基准实现 (无模型模式)

### 目标
创建一个绕过繁重 VLA 模型推理的基准测试，改为直接通过 API 从向量数据库检索动作轨迹。这允许更快的执行速度，并能单独测试检索管道的有效性。

### 创建的文件
*   `openvla/experiments/robot/libero/run_libero_goal_AR_DB.py`

### 主要更改

1.  **移除模型加载**:
    *   注释掉了 `get_model(cfg)` 和 `get_processor(cfg)`。
    *   这消除了加载 7B 参数 OpenVLA 模型的需求，显著降低了 GPU 显存占用（避免在较小 GPU 上出现 OOM 错误）并减少了启动时间。

2.  **添加 API 检索逻辑**:
    *   导入了 `requests`, `PIL.Image`, `io.BytesIO`。
    *   定义了 `RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"`。

3.  **实现动作队列机制**:
    *   在每个 episode 开始时引入 `action_queue = []`。
    *   **逻辑流程**:
        1.  检查 `action_queue` 是否有待执行的动作。
        2.  **如果队列为空**:
            *   获取当前观测图像。
            *   将图像和任务描述发送到检索 API。
            *   从 API 响应中接收轨迹（例如：动作序列 `rtcache_trajectory` 或 `averaged_trajectory`）。
            *   将这些检索到的动作填充到 `action_queue` 中。
        3.  **如果队列有动作**:
            *   从 `action_queue` 中弹出下一个动作并立即执行（模拟 0 推理时间）。

4.  **回退机制**:
    *   如果 API 调用失败或未返回轨迹，系统默认执行安全的“张开夹爪”动作（`action[-1] = -1.0`），以防止仿真程序崩溃。

5.  **动作后处理**:
    *   保留了 `normalize_gripper_action` 和 `invert_gripper_action`，以确保检索到的原始动作能正确映射到 Libero 环境预期的动作空间。

### 运行命令
```bash
conda activate specvla && \
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO && \
export MUJOCO_GL=egl && \
export ROBOSUITE_LOG_FILE=/path/to/SpecVLA/robosuite.log && \
export CUDA_VISIBLE_DEVICES=1 && \
export MUJOCO_EGL_DEVICE_ID=1 && \
python openvla/experiments/robot/libero/run_libero_goal_AR_DB.py \
  --pretrained_checkpoint /path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal \
  --model_family openvla \
  --task_suite_name libero_goal \
  --use_spec False \
  --center_crop True
```
