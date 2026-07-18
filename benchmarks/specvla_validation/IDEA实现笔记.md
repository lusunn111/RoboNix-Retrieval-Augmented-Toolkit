Idea

# SD部分复现记录（SpecVLA）

```Bash
# 1. 创建 Python 3.10 环境 (README 要求 Python >= 3.10)
conda create -n specvla python=3.10 -y

# 2. 激活环境
conda activate specvla

# 3. [关键] 安装编译工具和图形库 (替代系统 apt-get/sudo 安装)
# mesalib/glew/glfw -> 解决 MuJoCo/Libero 渲染问题
# gxx/gcc/make -> 解决 flash-attn 和其他包的编译问题
conda install -y -c conda-forge \
    mesalib \
    glew \
    glfw \
    patchelf \
    gxx_linux-64 \
    gcc_linux-64 \
    sysroot_linux-64 \
    make \
    unzip
    
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# 4. [关键] 配置环境变量，确保系统使用 Conda 的库而不是系统的库
# 建议将这几行也添加到您的 ~/.bashrc 文件末尾，避免每次都要输
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$CONDA_PREFIX/include:$CPATH
export MUJOCO_GL=egl   # 强制使用 EGL 后端，适用于服务器无显示器环境

# 安装 PyTorch 2.2.0 + CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# 1. 安装项目最小依赖 (README: pip install -r requirements-min.txt)
pip install -r requirements-min.txt

# 2. 安装 Libero 仿真环境 (README: Libero == 0.1.0)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .


cd openvla
pip install -e .

# 3. 安装 Libero 实验所需的额外依赖 (这一步很容易漏，但在运行实验时是必须的)
pip install -r openvla/experiments/robot/libero/libero_requirements.txt

# 4. (强烈推荐) 安装 Flash Attention 2
# 因为第一步我们配置好了 gxx_linux-64，这里应该能顺利编译安装
pip install flash-attn==2.5.5 --no-build-isolation
conda activate specvla && \
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO && \
export MUJOCO_GL=egl && \
export ROBOSUITE_LOG_FILE=/path/to/SpecVLA/robosuite.log && \
export CUDA_VISIBLE_DEVICES=1 && \
export MUJOCO_EGL_DEVICE_ID=1 && \

python openvla/experiments/robot/libero/run_libero_goal_AR.py \
  --pretrained_checkpoint /path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal \
  --model_family openvla \
  --task_suite_name libero_goal\
  --use_spec False \
 --center_crop True
 
 
conda activate specvla && \
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO && \
export MUJOCO_GL=egl && \
export ROBOSUITE_LOG_FILE=/path/to/SpecVLA/robosuite.log && \
export CUDA_VISIBLE_DEVICES=1 && \
export MUJOCO_EGL_DEVICE_ID=1 && \

python openvla/experiments/robot/libero/run_libero_goal_Spec.py \
  --pretrained_checkpoint /path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal \
  --spec_checkpoint /path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/state_190 \
  --model_family openvla \
  --task_suite_name libero_goal \
  --center_crop True
```

# 检索部分复现记录（RT-Cache）

```Bash
# 检索服务器
export CUDA_VISIBLE_DEVICES=1
python scripts/retrieval/retrieval_server.py

# embedding服务器
export CUDA_VISIBLE_DEVICES=1
python scripts/embedding/embedding_server.py

# 数据处理脚本
export CUDA_VISIBLE_DEVICES=1
source /path/to/miniconda3/bin/activate rt-mzh && python scripts/data_processing/process_datasets.py --datasets libero_goal_no_noops --max_episodes 1000

# 数据库清空脚本
python /path/to/rtcache/scripts/data_processing/clear_databases.py
```

# LIBERO仿真环境上的RT-Cache迁移

## LIBERO-Goal仿真环境的数据结构

```Python
# 只能使用gpu1
import os
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 指向具体的数据集版本目录，例如 libero_spatial_no_noops/1.0.0
db_dir = '/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_goal_no_noops/1.0.0'
builder = tfds.builder_from_directory(db_dir)
# 2. 获取数据集的元数据 (Metadata)
print("=== 数据集信息 ===")
print(f"数据集名称: {builder.name}")
print(f"版本: {builder.version}")
print(f"包含的 Split (划分): {builder.info.splits.keys()}")
# print(f"总样本数 (Test): {builder.info.splits['test'].num_examples}")
print(f"总样本数 (Train): {builder.info.splits['train'].num_examples}")
print("-" * 30)
ds = builder.as_dataset(split='train[:10]')
for episode in ds.take(1):
    print("\n=== Episode (轨迹) 结构 ===")
    print("Episode Keys:", episode.keys())
    # 常见的 Episode key 有: 'steps', 'episode_metadata' (如 task_id, file_path)
    
    if 'episode_metadata' in episode:
            print("Episode Metadata:", episode['episode_metadata'])

    # 5. 查看轨迹中的具体步骤 (Steps)
    # 'steps' 是一个嵌套的 Dataset，包含了每一步的 (Obs, Action, Reward)
    steps = episode['steps']
    
    print("\n=== Step (单步) 内容示例 ===")
    # 取出这一条轨迹中的第一帧/第一步
    for step in steps.take(1):
        # 递归打印字典结构和 Tensor 形状
        for key, value in step.items():
            if isinstance(value, dict):
                print(f"Key: '{key}'")
                for sub_key, sub_val in value.items():
                    print(f"  - {sub_key}: shape={sub_val.shape}, dtype={sub_val.dtype}")
            else:
                print(f"Key: '{key}': shape={value.shape}, dtype={value.dtype}")
                
        # 如果你想看具体的 Action 值
        print("\n示例 Action 数据:", step['action'].numpy())
        # 如果你想看具体的 Reward
        print("示例 Reward 数据:", step['reward'].numpy())
```

结果如下

```Bash
=== 数据集信息 ===
数据集名称: libero_goal
版本: 1.0.0
包含的 Split (划分): dict_keys(['train'])
总样本数 (Train): 428
------------------------------
=== Episode (轨迹) 结构 ===
Episode Keys: dict_keys(['episode_metadata', 'steps'])
Episode Metadata: {'file_path': <tf.Tensor: shape=(), dtype=string, numpy=b'/iris/u/moojink/prismatic-dev/LIBERO/libero/datasets/regenerated--no_noops/libero_goal/put_the_bowl_on_the_plate_demo.hdf5'>}

=== Step (单步) 内容示例 ===
Key: 'action': shape=(7,), dtype=<dtype: 'float32'>
Key: 'discount': shape=(), dtype=<dtype: 'float32'>
Key: 'is_first': shape=(), dtype=<dtype: 'bool'>
Key: 'is_last': shape=(), dtype=<dtype: 'bool'>
Key: 'is_terminal': shape=(), dtype=<dtype: 'bool'>
Key: 'language_instruction': shape=(), dtype=<dtype: 'string'>
Key: 'observation'
  - image: shape=(256, 256, 3), dtype=<dtype: 'uint8'>
  - joint_state: shape=(7,), dtype=<dtype: 'float32'>
  - state: shape=(8,), dtype=<dtype: 'float32'>
  - wrist_image: shape=(256, 256, 3), dtype=<dtype: 'uint8'>
Key: 'reward': shape=(), dtype=<dtype: 'float32'>

示例 Action 数据: [ 0.07232143  0.         -0.         -0.01178571  0.04392857  0.03535714
 -1.        ]
示例 Reward 数据: 0.0
```

## LIBERO-Goal仿真环境下的轨迹数据库构建

```Bash
# 插入一条轨迹到数据库 (修复版)
import os
import sys
import time
import uuid
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from io import BytesIO
import requests
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from pathlib import Path
import base64
import torch

# 确保使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 配置
MONGO_URL = "mongodb://localhost:27017/"
MONGO_DB_NAME = "OpenVLACollection"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_SERVER_URL = "http://127.0.0.1:9020/predict"
IMAGE_STORAGE_PATH = "./data/images"

# 向量维度配置 (需要与 Embedding Server 输出一致)
OPENVLA_DIM = 2176
CLIP_DIM = 512

def insert_single_trajectory():
    # 1. 连接数据库
    print("Connecting to databases...")
    mongo_client = MongoClient(MONGO_URL)
    mongo_db = mongo_client[MONGO_DB_NAME]
    mongo_collection = mongo_db["trajectories"]
    
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # --- 修复 3: 确保 Qdrant Collection 存在 ---
    collections = [
        ("image_collection", OPENVLA_DIM),
        ("clip_image_collection", CLIP_DIM)
    ]
    
    for name, dim in collections:
        if not qdrant_client.collection_exists(name):
            print(f"Creating collection {name} with dim={dim}...")
            qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance="Cosine")
            )
        else:
            print(f"Collection {name} already exists.")
    
    dataset_name = "libero_goal_no_noops"
    
    # 3. 处理并插入
    for episode_idx, episode in enumerate(ds.take(1)):
        print(f"Processing episode {episode_idx}...")
        steps = list(episode['steps'].as_numpy_iterator())
        total_steps = len(steps)
        
        # --- 修复 1: 正确提取文本 ---
        # 文本通常在 Step 的顶层，而不是 observation 里面
        # 我们取第一步的 language_instruction 作为整个 episode 的文本
        first_step = steps[0]
        if 'language_instruction' in first_step:
            episode_text = first_step['language_instruction'].decode('utf-8')
        else:
            episode_text = ""
        print(f"  Instruction: {episode_text}")
        
        batch_points_image = []
        batch_points_clip = []
        mongo_docs = []
        
        for step_idx, step in enumerate(steps):
            # --- 修复 2: 正确提取 Action ---
            # Libero 的 action 直接就是 (7,) 的 numpy array
            raw_action = step['action']
            
            # 简单的归一化 (示例逻辑，根据需要调整)
            normalized_action = raw_action.copy()
            normalized_action[:3] = np.clip(normalized_action[:3], -0.1, 0.1)
            normalized_action[3:6] = np.clip(normalized_action[3:6], -0.5, 0.5)
            normalized_action[6] = 1.0 if normalized_action[6] > 0 else 0.0
            
            # --- 提取图像 ---
            image_data = step['observation']['image']
            image = Image.fromarray(image_data)
            
            # 保存图像
            doc_id = f"{dataset_name}_test_{episode_idx}_{step_idx}"
            image_dir = Path(IMAGE_STORAGE_PATH)
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{doc_id}.png"
            image.save(image_path)
            
            # --- 生成 Embedding ---
            buf = BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            files = {"file": ("image.png", buf, "image/png")}
            data = {"instruction": episode_text, "option": "image"} 
            
            try:
                resp = requests.post(EMBEDDING_SERVER_URL, files=files, data=data, timeout=30)
                resp.raise_for_status()
                emb_result = resp.json()
            except Exception as e:
                print(f"  Error getting embedding for step {step_idx}: {e}")
                continue

            # --- 准备数据库记录 ---
            
            # MongoDB 文档
            mongo_doc = {
                'id': doc_id,
                'dataset_name': dataset_name,
                'episode_idx': episode_idx,
                'step_idx': step_idx,
                'total_steps': total_steps,
                'raw_action': raw_action.tolist(), # 存入 list
                'normalized_action': normalized_action.tolist(),
                'text': episode_text, # 存入正确的文本
                'image_path': str(image_path)
            }
            mongo_docs.append(mongo_doc)
            
            # Qdrant Points
            if "image_features" in emb_result:
                b64 = emb_result["image_features"]
                tensor = torch.load(BytesIO(base64.b64decode(b64)), map_location="cpu")
                vector = tensor.squeeze(0).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        'logical_id': doc_id,
                        'dataset_name': dataset_name,
                        'episode_idx': episode_idx,
                        'step_idx': step_idx,
                        'text': episode_text
                    }
                )
                batch_points_image.append(point)

            if "clip_image_features" in emb_result:
                b64 = emb_result["clip_image_features"]
                tensor = torch.load(BytesIO(base64.b64decode(b64)), map_location="cpu")
                vector = tensor.squeeze(0).tolist()
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        'logical_id': doc_id,
                        'dataset_name': dataset_name,
                        'episode_idx': episode_idx,
                        'step_idx': step_idx,
                        'text': episode_text
                    }
                )
                batch_points_clip.append(point)
                
            if step_idx % 10 == 0:
                print(f"  Processed step {step_idx}/{total_steps}")

        # --- 批量插入 ---
        if mongo_docs:
            print(f"Inserting {len(mongo_docs)} documents to MongoDB...")
            mongo_collection.insert_many(mongo_docs)
            
        if batch_points_image:
            print(f"Inserting {len(batch_points_image)} points to Qdrant (image_collection)...")
            qdrant_client.upsert(
                collection_name="image_collection",
                points=batch_points_image
            )
            
        if batch_points_clip:
            print(f"Inserting {len(batch_points_clip)} points to Qdrant (clip_image_collection)...")
            qdrant_client.upsert(
                collection_name="clip_image_collection",
                points=batch_points_clip
            )
            
    print("Done! Trajectory inserted with correct Action and Text.")


# 运行插入函数
insert_single_trajectory()
Connecting to databases...
Creating collection image_collection with dim=2176...
Creating collection clip_image_collection with dim=512...
Processing episode 0...
  Instruction: put the bowl on the plate
  Processed step 0/112
  Processed step 10/112
  Processed step 20/112
  Processed step 30/112
  Processed step 40/112
  Processed step 50/112
  Processed step 60/112
  Processed step 70/112
  Processed step 80/112
  Processed step 90/112
  Processed step 100/112
  Processed step 110/112
Inserting 112 documents to MongoDB...
Inserting 112 points to Qdrant (image_collection)...
Inserting 112 points to Qdrant (clip_image_collection)...
Done! Trajectory inserted with correct Action and Text.
```

## LIBERO-Goal仿真环境下的轨迹数据库检索

```Python
# 单元格 1: 直接向量数据库检索测试
import time
import random
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from pymongo import MongoClient
from qdrant_client import QdrantClient
import base64

# 配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MONGO_URL = "mongodb://localhost:27017/"
MONGO_DB_NAME = "OpenVLACollection"
EMBEDDING_SERVER_URL = "http://127.0.0.1:9020/predict"
COLLECTION_NAME = "image_collection" 

def test_direct_db_retrieval():
    print("=== 直接数据库检索测试 (n=1) ===")
    
    # 1. 随机采样
    # 假设 ds 已经加载 (如果未加载，请取消下面注释)
    # builder = tfds.builder_from_directory(db_dir)
    # ds = builder.as_dataset(split='train[:100]') 
    
    # 从前 10 个 Episode 中随机选一个
    episode = next(iter(ds.shuffle(10).take(1)))
    steps = list(episode['steps'].as_numpy_iterator())
    
    # 随机选一步
    step_idx = random.randint(0, len(steps)-1)
    step = steps[step_idx]
    
    # 获取数据
    image_data = step['observation']['image']
    image = Image.fromarray(image_data)
    gt_action = step['action']
    
    if 'language_instruction' in steps[0]:
        instruction = steps[0]['language_instruction'].decode('utf-8')
    else:
        instruction = ""
        
    print(f"Selected: Episode ?, Step {step_idx}")
    print(f"Instruction: {instruction}")
    print(f"GT Action: {gt_action}")
    
    # 2. 生成 Embedding
    t0 = time.time()
    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    files = {"file": ("image.png", buf, "image/png")}
    data = {"instruction": instruction, "option": "image"}
    
    resp = requests.post(EMBEDDING_SERVER_URL, files=files, data=data)
    emb_result = resp.json()
    
    if COLLECTION_NAME == "image_collection":
        b64_feat = emb_result["image_features"]
    else:
        b64_feat = emb_result["clip_image_features"]
        
    tensor = torch.load(BytesIO(base64.b64decode(b64_feat)), map_location="cpu")
    query_vector = tensor.squeeze(0).tolist()
    t1 = time.time()
    print(f"Embedding Time: {(t1-t0)*1000:.2f} ms")
    
    # 3. Qdrant 查询
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    t2 = time.time()
    if hasattr(client, 'search'):
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1,
            with_payload=True
        )
    else:
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=1,
            with_payload=True
        ).points
    t3 = time.time()
    print(f"Vector DB Query Time: {(t3-t2)*1000:.2f} ms")
    
    if not search_result:
        print("No results found in Qdrant.")
        return

    top1 = search_result[0]
    logical_id = top1.payload.get('logical_id')
    print(f"Top 1 Score: {top1.score:.4f}")
    print(f"Top 1 ID: {logical_id}")
    
    # 4. MongoDB 查询
    t4 = time.time()
    mongo_client = MongoClient(MONGO_URL)
    mongo_coll = mongo_client[MONGO_DB_NAME]["trajectories"]
    
    # 注意：数据库中的 id 字段存储的是 logical_id 字符串
    doc = mongo_coll.find_one({"id": logical_id})
    t5 = time.time()
    print(f"MongoDB Query Time: {(t5-t4)*1000:.2f} ms")
    
    if doc:
        retrieved_action = np.array(doc['raw_action'])
        print(f"Retrieved Action (Top 1): {retrieved_action}")
        
        mse = np.mean((gt_action - retrieved_action)**2)
        print(f"MSE (GT vs Retrieved): {mse:.6f}")
    else:
        print("Document not found in MongoDB.")

test_direct_db_retrieval()
```

结果如下：

```Bash
=== 直接数据库检索测试 (n=1) ===
Selected: Episode ?, Step 25
Instruction: open the top drawer and put the bowl inside
GT Action: [-0.26785713 -0.20892857 -0.77678573  0.045      -0.1575     -0.
 -1.        ]
Embedding Time: 47.26 ms
Vector DB Query Time: 11.12 ms
Top 1 Score: 1.0000
Top 1 ID: libero_goal_no_noops_3_25
MongoDB Query Time: 4.24 ms
Retrieved Action (Top 1): [-0.26785713 -0.20892857 -0.77678573  0.045      -0.1575     -0.
 -1.        ]
MSE (GT vs Retrieved): 0.000000
```

测试检索服务器，并分析返回结果：

```Bash
# 单元格 2: API 检索测试
import time
import random
import requests
import numpy as np
from PIL import Image
from io import BytesIO

# 配置
RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"

def test_api_retrieval():
    print("\n=== API 检索测试 (n=1) ===")
    
    # 1. 随机采样 (复用 ds)
    episode = next(iter(ds.shuffle(10).take(1)))
    steps = list(episode['steps'].as_numpy_iterator())
    step_idx = random.randint(0, len(steps)-1)
    step = steps[step_idx]
    
    image_data = step['observation']['image']
    image = Image.fromarray(image_data)
    gt_action = step['action']
    
    if 'language_instruction' in steps[0]:
        instruction = steps[0]['language_instruction'].decode('utf-8')
    else:
        instruction = ""
        
    print(f"Selected: Episode ?, Step {step_idx}")
    print(f"Instruction: {instruction}")
    print(f"GT Action: {gt_action}")
    
    # 2. 调用 API
    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    
    files = {"file": ("image.png", buf, "image/png")}
    data = {
        "instruction": instruction,
        "option": "both"
    }
    
    t0 = time.time()
    try:
        response = requests.post(RETRIEVAL_URL, files=files, data=data)
        t1 = time.time()
        latency = (t1 - t0) * 1000
        print(f"API Total Time: {latency:.2f} ms")
        
        if response.status_code == 200:
            result = response.json()
            # print(f"API Result: {result}") # 调试用
            
            # 提取 Action
            pred_action = None
            
            # 检查 rtcache_trajectory
            if 'rtcache_trajectory' in result and result['rtcache_trajectory']:
                traj = np.array(result['rtcache_trajectory'])
                print(f"Retrieved Trajectory Shape: {traj.shape}")
                print(f"Full Retrieved Trajectory (All Steps):\n{traj}")
                
                # 如果是列表的列表，取第一个
                if traj.ndim > 1:
                    pred_action = traj[0]
                    print(f"Using first step of trajectory for MSE calculation.")
                else:
                    pred_action = traj
            # 检查 averaged_trajectory
            elif 'averaged_trajectory' in result and result['averaged_trajectory']:
                traj = np.array(result['averaged_trajectory'])
                print(f"Retrieved Trajectory Shape: {traj.shape}")
                print(f"Full Retrieved Trajectory (All Steps):\n{traj}")
                
                if traj.ndim > 1:
                    pred_action = traj[0]
                    print(f"Using first step of trajectory for MSE calculation.")
                else:
                    pred_action = traj
            
            if pred_action is not None:
                print(f"Retrieved Action (API, Step 0): {pred_action}")
                mse = np.mean((gt_action - pred_action)**2)
                print(f"MSE (GT vs API Step 0): {mse:.6f}")
            else:
                print("No action found in API response.")
                print(f"Response keys: {result.keys()}")
                
        else:
            print(f"API Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"API Error: {e}")

test_api_retrieval()
```

结果如下：

```Bash
=== API 检索测试 (n=1) ===
Selected: Episode ?, Step 195
Instruction: open the top drawer and put the bowl inside
GT Action: [ 0.         0.        -0.4794643  0.         0.0975    -0.
  1.       ]
API Total Time: 79.47 ms
API Result: {'averaged_trajectory': [[0.19285714477300644, 0.0042857140302658085, -0.31982142329216, 0.0, 0.006428571976721286, -0.010071428678929805, -0.6], [0.14892857596278192, 0.002142857201397419, -0.33107143044471743, 0.0, 0.01307142823934555, -0.012000000453554094, -0.6], [0.11196428686380386, -0.001607142947614193, -0.3380357205867767, 0.0, 0.016499999910593033, -0.016071428637951614, -0.6]], 'br_trajectory': [], 'closest_dataset': 'libero_goal_no_noops', 'enabled_models': ['rtcache', 'vinn', 'br'], 'episode': 1, 'rtcache_trajectory': [[0.19285714477300644, 0.0042857140302658085, -0.31982142329216, 0.0, 0.006428571976721286, -0.010071428678929805, -0.6], [0.14892857596278192, 0.002142857201397419, -0.33107143044471743, 0.0, 0.01307142823934555, -0.012000000453554094, -0.6], [0.11196428686380386, -0.001607142947614193, -0.3380357205867767, 0.0, 0.016499999910593033, -0.016071428637951614, -0.6]], 'save_results': False, 'step': 1, 'vinn_trajectory': []}
Retrieved Trajectory Shape: (3, 7)
Full Retrieved Trajectory (All Steps):
[[ 0.19285714  0.00428571 -0.31982142  0.          0.00642857 -0.01007143
  -0.6       ]
 [ 0.14892858  0.00214286 -0.33107143  0.          0.01307143 -0.012
  -0.6       ]
 [ 0.11196429 -0.00160714 -0.33803572  0.          0.0165     -0.01607143
  -0.6       ]]
Using first step of trajectory for MSE calculation.
Retrieved Action (API, Step 0): [ 0.19285714  0.00428571 -0.31982142  0.          0.00642857 -0.01007143
 -0.6       ]
MSE (GT vs API Step 0): 0.375871
```

也就是说，只需要将每次检索到的三次的结果去执行即可。

# 推理过程中引入检索

## 主要更改

- 主文件：`openvla/experiments/robot/libero/run_libero_goal_AR_DB.py`
- **移除模型加载**:
  - 注释掉了 `get_model(cfg)` 和 `get_processor(cfg)`。
  - 这消除了加载 7B 参数 OpenVLA 模型的需求，显著降低了 GPU 显存占用（避免在较小 GPU 上出现 OOM 错误）并减少了启动时间。
- **添加 API 检索逻辑**:
  - 导入了 `requests`, `PIL.Image`, `io.BytesIO`。
  - 定义了 `RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"`。
- **实现动作队列机制**:
  - 在每个 episode 开始时引入 `action_queue = []`。
  - **逻辑流程**:
    - 检查 `action_queue` 是否有待执行的动作。
      1. **如果队列为空**:
         1. 获取当前观测图像。
         2. 将图像和任务描述发送到检索 API。
         3. 从 API 响应中接收轨迹（例如：动作序列 `rtcache_trajectory` 或 `averaged_trajectory`）。
         4. 将这些检索到的动作填充到 `action_queue` 中。
      2. **如果队列有动作**:
         1. 从 `action_queue` 中弹出下一个动作并立即执行（模拟 0 推理时间）。
- **回退机制**:
  - 如果 API 调用失败或未返回轨迹，系统默认执行安全的“张开夹爪”动作（`action[-1] = -1.0`），以防止仿真程序崩溃。
- **动作后处理**:
  - 保留了 `normalize_gripper_action` 和 `invert_gripper_action`，以确保检索到的原始动作能正确映射到 Libero 环境预期的动作空间。

## 运行命令

```Bash
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

## 数据库结构重构及测试（Demo）

根据对数据库进行查看和调研，发现这样几个点

- Qdrant这个向量数据库，本身是的结构是，集合为单位的，集合内每个元素以，map<vector,json>的样式存储。
- LIBERO-Goal数据集，整体存下来，不超过5.2w条数据。Qdrant天然支持内存加载，而mongodb商业版才支持。内存加载避免了硬盘访问，整体速度会变快不少。
- 故而设计存储结构为，以DatasetName_TaskId为集合名，集合内存一个任务的，向量-json对应，json内主要存储，当前帧对应的action slice+未来三帧的action slice。
- 在构建TaskId时，为方便构建，使用了字符串hash，对hash值mod1001作为ID。

以下是向量数据库的LIBERO-Goal所涉及的task：共10个

```Bash
collections=[CollectionDescription(name='libero_goal_task_362'), CollectionDescription(name='libero_goal_task_237'), CollectionDescription(name='libero_goal_task_775'), CollectionDescription(name='libero_goal_task_632'), CollectionDescription(name='libero_goal_task_128'), CollectionDescription(name='libero_goal_task_847'), CollectionDescription(name='libero_goal_task_806'), CollectionDescription(name='libero_goal_task_319'), CollectionDescription(name='libero_goal_task_548'), CollectionDescription(name='libero_goal_task_93')]
```

以下是向量数据库中一个集合的存储结构：

```Bash
| id                                   | dataset_name         |   episode_idx |   step_idx | current_action                                                                                                                           | next_actions                                                                                                                                                                                                                                                                                                                                                                                                               | language_instruction                     |
|:-------------------------------------|:---------------------|--------------:|-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------|
| 00042193-8f11-40f9-882a-17801fbd20cf | libero_goal_no_noops |            32 |        133 | [0.5544642806053162, -0.10982143133878708, 0.13124999403953552, 0.009642857126891611, -0.2378571480512619, -0.03321428596973419, -1.0]   | [[0.4285714328289032, -0.3187499940395355, -0.0, 0.06750000268220901, -0.2346428632736206, -0.08142857253551483, -1.0], [0.31607142090797424, -0.4124999940395355, -0.0, 0.07928571105003357, -0.22499999403953552, -0.12857143580913544, -1.0], [0.20357142388820648, -0.42053571343421936, -0.0, 0.07607142627239227, -0.2282142788171768, -0.14892856776714325, -1.0]]                                                  | push the plate to the front of the stove |
| 00368df3-3c93-410b-8600-427b21c35023 | libero_goal_no_noops |            82 |          8 | [0.9375, 0.0026785715017467737, -0.06428571790456772, -0.04392857104539871, 0.01607142947614193, -0.0, -1.0]                             | [[0.9375, 0.0053571430034935474, -0.06964285671710968, -0.04607142880558968, 0.018214285373687744, -0.0, -1.0], [0.9375, 0.0, -0.0803571417927742, -0.04285714402794838, 0.02142857201397419, -0.0, -1.0], [0.9375, 0.0, -0.09910714626312256, -0.034285712987184525, 0.02464285679161549, -0.0, -1.0]]                                                                                                                    | push the plate to the front of the stove |
| 004e6a53-5f7c-43f8-914a-675b617790a6 | libero_goal_no_noops |           156 |        160 | [-0.2276785671710968, 0.8464285731315613, -0.4312500059604645, 0.040714286267757416, 0.09642857313156128, -0.0, -1.0]                    | [[-0.22232143580913544, 0.8410714268684387, -0.43392857909202576, 0.040714286267757416, 0.1103571429848671, -0.0, -1.0], [-0.26249998807907104, 0.8357142806053162, -0.4151785671710968, 0.03750000149011612, 0.12214285880327223, -0.0, -1.0], [-0.30267858505249023, 0.8169642686843872, -0.3857142925262451, 0.03750000149011612, 0.1339285671710968, -0.0, -1.0]]                                                      | push the plate to the front of the stove |
| 00620bac-01e7-4877-a39e-5a89d6c12bb3 | libero_goal_no_noops |           268 |          0 | [0.23035714030265808, 0.0026785715017467737, -0.06160714104771614, -0.013928571715950966, -0.03214285895228386, -0.0, -1.0]              | [[0.2410714328289032, 0.0, -0.0535714291036129, -0.012857142835855484, -0.027857143431901932, -0.0, -1.0], [0.34017857909202576, 0.0, -0.02678571455180645, -0.02142857201397419, -0.019285714253783223, -0.0, -1.0], [0.5035714507102966, 0.0, -0.010714286006987097, -0.027857143431901932, -0.008571428246796131, -0.0, -1.0]]                                                                                          | push the plate to the front of the stove |
| 006e5a15-30c2-4869-8411-815b08a33be4 | libero_goal_no_noops |            32 |        164 | [-0.04553571343421936, 0.39375001192092896, -0.09910714626312256, -0.10499999672174454, 0.11464285850524902, -0.12857143580913544, -1.0] | [[-0.06428571790456772, 0.2973214387893677, -0.0535714291036129, -0.1071428582072258, 0.1403571367263794, -0.12214285880327223, -1.0], [-0.13660714030265808, 0.21964286267757416, -0.04553571343421936, -0.1071428582072258, 0.1574999988079071, -0.10928571224212646, -1.0], [-0.24642856419086456, 0.16875000298023224, -0.11249999701976776, -0.1039285734295845, 0.16928571462631226, -0.09214285761117937, -1.0]]    | push the plate to the front of the stove |
```

## 新格式数据库的构建与检索

针对LIBERO数据集，简单重构了process_datasets.py，如下

```Python
import os
import sys
import argparse
import logging
import time
import uuid
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from io import BytesIO
from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
import requests

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.rt_cache_config import get_config

# Try to import LIBERO benchmark
try:
    from libero.libero import benchmark
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("Warning: LIBERO benchmark not available. Task matching will use language_instruction only.")

def setup_logging(level="INFO"):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@dataclass
class ProcessingConfig:
    """Configuration for LIBERO-Goal dataset processing"""
    
    # Dataset path
    dataset_path: str = "/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_goal_no_noops/1.0.0"
    dataset_name: str = "libero_goal_no_noops"
    
    # Server URLs
    embedding_server_url: str = "http://127.0.0.1:9020/predict"
    
    # Database settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Processing parameters
    batch_size: int = 50
    max_episodes: int = -1  # -1 means all episodes
    min_episode_length: int = 5
    
    # Embedding dimensions
    openvla_image_dim: int = 2176
    
    # Task suite name for LIBERO benchmark
    task_suite_name: str = "libero_goal"

class LiberoGoalProcessor:
    """
    Processor for LIBERO-Goal dataset.
    
    Creates separate Qdrant collections for each task and stores:
    - Image embeddings
    - Action sequences (current + next 3 steps)
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the processor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self._init_storage()
        
        # Initialize task mapping
        self._init_task_mapping()
        
        # Statistics
        self.stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "skipped_episodes": 0,
            "failed_embeddings": 0,
            "task_distribution": defaultdict(int)
        }
        
        # Track which instructions have been seen (to reduce log noise)
        self.seen_instructions = set()
        
    def _init_storage(self):
        """Initialize Qdrant connection"""
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            timeout=60.0
        )
        self.logger.info(f"Connected to Qdrant at {self.config.qdrant_host}:{self.config.qdrant_port}")
        
    def _init_task_mapping(self):
        """Initialize task mapping from LIBERO benchmark"""
        self.task_descriptions = {}
        self.task_id_to_collection = {}
        self.instruction_to_task_id = {}  # Cache for hash-based assignment
        
        # Check if LIBERO is available
        try:
            from libero.libero import benchmark
            libero_available = True
        except ImportError:
            libero_available = False
            
        if libero_available:
            try:
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict[self.config.task_suite_name]()
                num_tasks = task_suite.n_tasks
                
                self.logger.info(f"Found {num_tasks} tasks in {self.config.task_suite_name}")
                
                for task_id in range(num_tasks):
                    task = task_suite.get_task(task_id)
                    task_description = task.language
                    self.task_descriptions[task_description.lower().strip()] = task_id
                    
                    # Create collection name: libero_goal_task_0, libero_goal_task_1, etc.
                    collection_name = f"libero_goal_task_{task_id}"
                    self.task_id_to_collection[task_id] = collection_name
                    
                    # Create collection if it doesn't exist
                    if not self.qdrant_client.collection_exists(collection_name):
                        self.qdrant_client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=self.config.openvla_image_dim,
                                distance="Cosine"
                            )
                        )
                        self.logger.info(f"Created collection: {collection_name}")
                    else:
                        self.logger.info(f"Collection {collection_name} already exists")
                        
                self.logger.info(f"Initialized {len(self.task_descriptions)} task mappings")
                
            except Exception as e:
                self.logger.warning(f"Failed to initialize LIBERO benchmark: {e}")
                self.logger.warning("Will use language_instruction matching only")
                libero_available = False
        else:
            self.logger.warning("LIBERO benchmark not available. Will create collections on-the-fly.")
            
    def _match_task_id(self, language_instruction: str) -> Optional[int]:
        """
        Match language instruction to task_id.
        
        Args:
            language_instruction: The language instruction from the dataset
            
        Returns:
            task_id if matched, None otherwise
        """
        if not language_instruction:
            return None
            
        instruction_lower = language_instruction.lower().strip()
        
        # Direct match
        if instruction_lower in self.task_descriptions:
            return self.task_descriptions[instruction_lower]
        
        # Fuzzy match: check if any task description is contained in the instruction
        for task_desc, task_id in self.task_descriptions.items():
            if task_desc in instruction_lower or instruction_lower in task_desc:
                return task_id
                
        return None
        
    def _get_or_create_collection(self, task_id: int) -> str:
        """
        Get or create collection name for a task_id.
        
        Args:
            task_id: Task ID
            
        Returns:
            Collection name
        """
        if task_id in self.task_id_to_collection:
            return self.task_id_to_collection[task_id]
        
        # Create collection name on-the-fly
        collection_name = f"libero_goal_task_{task_id}"
        
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.openvla_image_dim,
                    distance="Cosine"
                )
            )
            self.logger.info(f"Created collection on-the-fly: {collection_name}")
        
        self.task_id_to_collection[task_id] = collection_name
        return collection_name
        
    def process_dataset(self):
        """Process the entire LIBERO-Goal dataset"""
        self.logger.info(f"Loading dataset from: {self.config.dataset_path}")
        
        # Load dataset
        builder = tfds.builder_from_directory(builder_dir=self.config.dataset_path)
        ds = builder.as_dataset(split='train', shuffle_files=False)
        
        # Get total number of episodes
        total_episodes = builder.info.splits['train'].num_examples
        self.logger.info(f"Total episodes in dataset: {total_episodes}")
        
        # Process episodes
        batch_buffers = defaultdict(lambda: {
            'points': []
        })
        
        episode_count = 0
        for episode_idx, episode in enumerate(tqdm(ds, desc="Processing episodes")):
            if self.config.max_episodes > 0 and episode_idx >= self.config.max_episodes:
                break
                
            try:
                # Process episode
                task_id = self._process_episode(
                    episode, episode_idx, batch_buffers
                )
                
                if task_id is not None:
                    episode_count += 1
                    self.stats['task_distribution'][task_id] += 1
                    
                    # Flush batch if needed (check only the current task's buffer)
                    if len(batch_buffers[task_id]['points']) >= self.config.batch_size:
                        self._flush_batch(task_id, batch_buffers[task_id])
                            
            except Exception as e:
                self.logger.error(f"Error processing episode {episode_idx}: {e}")
                self.stats['skipped_episodes'] += 1
                continue
                
        # Final flush for all tasks
        for task_id, buffer in batch_buffers.items():
            if buffer['points']:
                self._flush_batch(task_id, buffer)
                
        # Print statistics
        self._print_statistics()
        
    def _process_episode(self, episode: Dict, episode_idx: int, 
                        batch_buffers: Dict) -> Optional[int]:
        """
        Process a single episode.
        
        Args:
            episode: Episode data
            episode_idx: Episode index
            batch_buffers: Batch buffers for each task
            
        Returns:
            task_id if successful, None otherwise
        """
        # Convert steps to list
        steps_dataset = episode['steps']
        steps_list = list(steps_dataset.as_numpy_iterator())
        
        if len(steps_list) < self.config.min_episode_length:
            self.stats['skipped_episodes'] += 1
            return None
            
        self.stats['total_episodes'] += 1
        
        # Extract language instruction from first step
        first_step = steps_list[0]
        language_instruction = None
        
        if 'language_instruction' in first_step:
            lang_data = first_step['language_instruction']
            if isinstance(lang_data, bytes):
                language_instruction = lang_data.decode('utf-8')
            elif isinstance(lang_data, str):
                language_instruction = lang_data
            elif hasattr(lang_data, 'numpy'):
                language_instruction = lang_data.numpy().decode('utf-8')
                
        # Match to task_id
        task_id = self._match_task_id(language_instruction)
        
        if task_id is None:
            # If no match, use hash-based assignment to ensure same instruction always gets same task_id
            # This is a fallback for when LIBERO benchmark is not available
            if language_instruction:
                # Check if we've already assigned this instruction
                if language_instruction in self.instruction_to_task_id:
                    task_id = self.instruction_to_task_id[language_instruction]
                else:
                    # Use hash of instruction to consistently assign task_id (0-1000)
                    import hashlib
                    instruction_hash = int(hashlib.md5(language_instruction.encode('utf-8')).hexdigest(), 16)
                    task_id = instruction_hash % 1001  # Use 0-1000 as task_id range
                    self.instruction_to_task_id[language_instruction] = task_id
                    
                    # Only log once per unique instruction to reduce log noise
                    if language_instruction not in self.seen_instructions:
                        self.seen_instructions.add(language_instruction)
                        self.logger.info(f"Using hash-based task_id={task_id} for instruction: '{language_instruction}' (LIBERO benchmark not available)")
            else:
                # Fallback to episode_idx if no instruction
                task_id = episode_idx % 10
            
        # Get or create collection
        collection_name = self._get_or_create_collection(task_id)
        
        # Process each step
        total_steps = len(steps_list)
        for step_idx, step in enumerate(steps_list):
            self.stats['total_steps'] += 1
            
            # Extract action
            action = step['action']
            if isinstance(action, tf.Tensor):
                action = action.numpy()
            action = np.array(action, dtype=np.float32)
            
            # Get next 3 actions (if available)
            next_actions = []
            for offset in range(1, 4):  # next 1, 2, 3 steps
                next_idx = step_idx + offset
                if next_idx < total_steps:
                    next_action = steps_list[next_idx]['action']
                    if isinstance(next_action, tf.Tensor):
                        next_action = next_action.numpy()
                    next_actions.append(next_action.tolist())
                else:
                    # Pad with zeros if not available
                    next_actions.append([0.0] * 7)
                    
            # Extract image
            image_data = step['observation']['image']
            if isinstance(image_data, tf.Tensor):
                image_data = image_data.numpy()
            image = Image.fromarray(image_data)
            
            # Generate embedding
            embedding = self._generate_embedding(image, language_instruction)
            
            if embedding is None:
                self.stats['failed_embeddings'] += 1
                continue
                
            # Create Qdrant point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'dataset_name': self.config.dataset_name,
                    'episode_idx': episode_idx,
                    'step_idx': step_idx,
                    'current_action': action.tolist(),
                    'next_actions': next_actions,  # List of 3 actions
                    'language_instruction': language_instruction or ""
                }
            )
            
            batch_buffers[task_id]['points'].append(point)
            
        return task_id
        
    def _generate_embedding(self, image: Image.Image, instruction: Optional[str]) -> Optional[List[float]]:
        """
        Generate image embedding via embedding server.
        
        Args:
            image: PIL Image
            instruction: Optional text instruction
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Prepare request
            buf = BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            
            files = {"file": ("image.png", buf, "image/png")}
            data = {
                "instruction": instruction if instruction else "",
                "option": "image"  # Get image embeddings only
            }
            
            # Send request
            response = requests.post(
                self.config.embedding_server_url,
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            
            # Decode embedding
            result = response.json()
            
            if "image_features" in result:
                b64_string = result["image_features"]
                binary_data = base64.b64decode(b64_string)
                buffer = BytesIO(binary_data)
                tensor = torch.load(buffer, map_location="cpu")
                return tensor.squeeze(0).tolist()
            else:
                self.logger.warning("No image_features in embedding response")
                return None
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None
            
    def _flush_batch(self, task_id: int, buffer: Dict):
        """
        Flush batch buffer to Qdrant.
        
        Args:
            task_id: Task ID
            buffer: Batch buffer containing points
        """
        if not buffer['points']:
            return
            
        collection_name = self._get_or_create_collection(task_id)
        
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=buffer['points']
            )
            self.logger.debug(f"Inserted {len(buffer['points'])} points to {collection_name}")
            buffer['points'].clear()
        except Exception as e:
            self.logger.error(f"Failed to insert batch to {collection_name}: {e}")
            
    def _print_statistics(self):
        """Print processing statistics"""
        self.logger.info("=" * 60)
        self.logger.info("Processing Statistics:")
        self.logger.info(f"  Total episodes: {self.stats['total_episodes']}")
        self.logger.info(f"  Total steps: {self.stats['total_steps']}")
        self.logger.info(f"  Skipped episodes: {self.stats['skipped_episodes']}")
        self.logger.info(f"  Failed embeddings: {self.stats['failed_embeddings']}")
        self.logger.info("  Task distribution:")
        for task_id, count in sorted(self.stats['task_distribution'].items()):
            self.logger.info(f"    Task {task_id}: {count} episodes")
        self.logger.info("=" * 60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process LIBERO-Goal dataset for RT-Cache"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/path/to/rtcache/libero/datasets--openvla--modified_libero_rlds/snapshots/6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551/libero_goal_no_noops/1.0.0",
        help="Path to LIBERO-Goal dataset"
    )
    parser.add_argument(
        "--embedding_server_url",
        type=str,
        default="http://127.0.0.1:9020/predict",
        help="URL of embedding server"
    )
    parser.add_argument(
        "--qdrant_host",
        type=str,
        default="localhost",
        help="Qdrant host"
    )
    parser.add_argument(
        "--qdrant_port",
        type=int,
        default=6333,
        help="Qdrant port"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch size for insertion"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="Maximum episodes to process (-1 for all)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create configuration
    config = ProcessingConfig(
        dataset_path=args.dataset_path,
        embedding_server_url=args.embedding_server_url,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes
    )
    
    # Override with config file if available
    try:
        rt_config = get_config()
        config.embedding_server_url = rt_config.server.embedding_url
        config.qdrant_host = rt_config.database.qdrant_host
        config.qdrant_port = rt_config.database.qdrant_port
        config.openvla_image_dim = rt_config.retrieval.openvla_dim
    except Exception as e:
        logger.warning(f"Could not load config file: {e}. Using defaults.")
    
    logger.info(f"Processing LIBERO-Goal dataset")
    logger.info(f"  Dataset path: {config.dataset_path}")
    logger.info(f"  Embedding server: {config.embedding_server_url}")
    logger.info(f"  Qdrant: {config.qdrant_host}:{config.qdrant_port}")
    logger.info(f"  Batch size: {config.batch_size}")
    
    # Create processor and run
    processor = LiberoGoalProcessor(config)
    processor.process_dataset()
    
    logger.info("Dataset processing complete!")

if __name__ == "__main__":
    main()
```

针对检索，基于原来的retrieval_server.py，将两重数据库，改为纯向量数据库，沿用了flask api接口的方式，后续可以作为类导入，可能更快一点，暂时不修改。

整体上沿用了rtcache对topk轨迹的处理。代码如下

```Python
#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import requests
from PIL import Image
from flask import Flask, request, jsonify

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import LIBERO benchmark for task matching
try:
    from libero.libero import benchmark
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("Warning: LIBERO benchmark not available. Will use fuzzy matching.")

###############################################################################
# Configuration
###############################################################################
class RetrievalConfig:
    """Configuration for retrieval server"""
    
    # Server settings
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5002
    
    # Embedding server
    EMBEDDING_URL = "http://127.0.0.1:9020/predict"
    
    # Qdrant settings
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    
    # LIBERO-Goal specific
    TASK_SUITE_NAME = "libero_goal"
    NUM_TASKS = 10  # libero_goal has 10 tasks
    COLLECTION_PREFIX = "libero_goal_task_"
    
    # Retrieval parameters
    TOP_K = 10  # Number of similar samples to retrieve
    NUM_ACTIONS = 3  # Number of next actions to return (current + next 3)
    SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    LOG_LEVEL = "INFO"

###############################################################################
# Task Matching
###############################################################################
class TaskMatcher:
    """Match language instructions to task IDs using hash-based mapping"""
    
    def __init__(self, task_suite_name: str = "libero_goal"):
        self.task_suite_name = task_suite_name
        # Cache for already computed hashes
        self.instruction_cache = {}
    
    def match(self, instruction: str) -> Optional[int]:
        """
        Match instruction to task_id using hash-based mapping.
        This matches the logic used in process_libero_goal.py
        
        Args:
            instruction: Language instruction
            
        Returns:
            task_id (0-9) based on MD5 hash
        """
        if not instruction:
            return None
        
        instruction_lower = instruction.lower().strip()
        
        # Check cache first
        if instruction_lower in self.instruction_cache:
            return self.instruction_cache[instruction_lower]
        
        # Use MD5 hash to consistently assign task_id (0-1000)
        # This matches the current logic in process_libero_goal.py
        import hashlib
        instruction_hash = int(hashlib.md5(instruction_lower.encode('utf-8')).hexdigest(), 16)
        task_id = instruction_hash % 1001
        
        # Cache the result
        self.instruction_cache[instruction_lower] = task_id
        
        logging.info(f"Hash-based mapping: '{instruction}' -> task_id={task_id}")
        
        return task_id

###############################################################################
# Payload Cache
###############################################################################
class PayloadCache:
    """In-memory cache for all collection payloads"""
    
    def __init__(self, qdrant_client: QdrantClient, collection_names: List[str]):
        self.qdrant_client = qdrant_client
        self.collection_names = collection_names
        
        # Cache: collection_name -> {point_id: payload}
        self.cache = {}
        
        # Statistics
        self.stats = {
            "total_points": 0,
            "collections": {}
        }
        
        # Load all payloads for given collections
        self._load_all_payloads(collection_names)
    
    def _load_all_payloads(self, collection_names: List[str]):
        """Load all payloads from specified collections into memory"""
        logging.info("Loading payloads into memory for specified collections...")
        for collection_name in collection_names:
            self._load_collection(collection_name)
        logging.info(f"Payload cache loaded: {self.stats['total_points']} total points across {len(self.cache)} collections")

    def _load_collection(self, collection_name: str):
        """Load a single collection's payloads into memory"""
        try:
            # Check if already loaded
            if collection_name in self.cache:
                return
            # Check if collection exists
            if not self.qdrant_client.collection_exists(collection_name):
                logging.warning(f"Collection {collection_name} does not exist, skipping load")
                return
            # Get collection info
            collection_info = self.qdrant_client.get_collection(collection_name)
            point_count = getattr(collection_info, 'points_count', None)
            if point_count is not None:
                logging.info(f"Loading {point_count} points from {collection_name}...")
            else:
                logging.info(f"Loading points from {collection_name}...")
            # Scroll through all points
            points_dict = {}
            offset = None
            batch_size = 100
            while True:
                records, offset = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                if not records:
                    break
                for record in records:
                    points_dict[str(record.id)] = record.payload
                if offset is None:
                    break
            self.cache[collection_name] = points_dict
            self.stats["collections"][collection_name] = len(points_dict)
            self.stats["total_points"] += len(points_dict)
            logging.info(f"Loaded {len(points_dict)} points from {collection_name}")
        except Exception as e:
            logging.error(f"Error loading payloads from {collection_name}: {e}")

    def ensure_collection_loaded(self, collection_name: str):
        """Ensure a collection is loaded into cache, load on demand if needed"""
        if collection_name not in self.cache:
            self._load_collection(collection_name)
    
    def get_payload(self, collection_name: str, point_id: str) -> Optional[Dict]:
        """Get payload for a point"""
        if collection_name not in self.cache:
            return None
        
        return self.cache[collection_name].get(point_id)
    
    def get_all_payloads(self, collection_name: str) -> Dict[str, Dict]:
        """Get all payloads for a collection"""
        return self.cache.get(collection_name, {})

###############################################################################
# Embedding Generator
###############################################################################
class EmbeddingGenerator:
    """Generate embeddings via remote server"""
    
    def __init__(self, embedding_url: str):
        self.embedding_url = embedding_url
    
    def generate(self, pil_image: Image.Image, instruction: str = "") -> torch.Tensor:
        """
        Generate image embedding
        
        Args:
            pil_image: Input image
            instruction: Optional text instruction
            
        Returns:
            Embedding tensor
        """
        try:
            # Prepare request
            buf = BytesIO()
            pil_image.save(buf, format='PNG')
            buf.seek(0)
            
            files = {"file": ("image.png", buf, "image/png")}
            data = {
                "instruction": instruction,
                "option": "image"  # Get image embeddings only
            }
            
            # Send request
            response = requests.post(
                self.embedding_url,
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            
            # Decode embedding
            result = response.json()
            
            if "image_features" in result:
                b64_string = result["image_features"]
                binary_data = base64.b64decode(b64_string)
                buffer = BytesIO(binary_data)
                tensor = torch.load(buffer, map_location="cpu")
                return tensor.squeeze(0)
            else:
                logging.error("No image_features in embedding response")
                return None
                
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}")
            return None

###############################################################################
# Retrieval Engine
###############################################################################
class RetrievalEngine:
    """Main retrieval engine"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
        # Initialize components
        self.qdrant_client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            timeout=60.0
        )
        
        # Initialize task matcher
        self.task_matcher = TaskMatcher(config.TASK_SUITE_NAME)
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(config.EMBEDDING_URL)
        
        # Discover all collections from Qdrant matching the prefix
        self.collection_names = self._discover_collections(config.COLLECTION_PREFIX)
        
        # Initialize payload cache with discovered collections
        self.payload_cache = PayloadCache(self.qdrant_client, self.collection_names)
        
        logging.info("Retrieval engine initialized")

    def _discover_collections(self, prefix: str) -> List[str]:
        """Discover existing Qdrant collections with given prefix"""
        names: List[str] = []
        try:
            resp = self.qdrant_client.get_collections()
            # resp.collections is a list of CollectionDescription
            coll_list = getattr(resp, 'collections', [])
            for c in coll_list:
                # Support both attr and dict-like
                cname = getattr(c, 'name', None) or (c.get('name') if isinstance(c, dict) else None)
                if cname and cname.startswith(prefix):
                    names.append(cname)
        except Exception as e:
            logging.warning(f"Could not list collections from Qdrant: {e}. Falling back to default range.")
        # If none found, fall back to an empty list (lazy loading will handle on-demand)
        logging.info(f"Discovered {len(names)} collections with prefix '{prefix}'")
        return names

    def _search_points(self, collection_name: str, query_vector: List[float], limit: int = 10):
        """Version-agnostic Qdrant search with fallback to query_points.

        Returns a list of ScoredPoint-like objects with .id and .score.
        """
        # Try legacy / common API first
        if hasattr(self.qdrant_client, "search"):
            try:
                return self.qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
            except Exception as e:
                logging.warning(f"Legacy search failed, will try query_points: {e}")

        # Fallback: use query_points (newer API)
        try:
            # For single-vector collections, pass raw vector directly as `query`
            result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            # Some versions return object with `.points`, others return list
            if hasattr(result, "points"):
                return result.points
            return result
        except Exception as e:
            logging.error(f"query_points fallback failed: {e}")
            raise
    
    def retrieve(
        self,
        pil_image: Image.Image,
        instruction: str,
        top_k: int = None
    ) -> Dict:
        """
        Retrieve similar samples and return action trajectory
        
        Args:
            pil_image: Input image
            instruction: Language instruction (used to hash to task_id)
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with retrieval results
        """
        if top_k is None:
            top_k = self.config.TOP_K
        
        # Match instruction to task using hash-based mapping
        if not instruction:
            return {
                "success": False,
                "error": "Instruction must be provided",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        task_id = self.task_matcher.match(instruction)
        
        if task_id is None:
            return {
                "success": False,
                "error": "Failed to map instruction to task_id",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        collection_name = f"{self.config.COLLECTION_PREFIX}{task_id}"
        
        # Ensure collection is loaded (on-demand)
        if collection_name not in self.payload_cache.cache:
            self.payload_cache.ensure_collection_loaded(collection_name)
            if collection_name not in self.payload_cache.cache:
                return {
                    "success": False,
                    "error": f"Collection {collection_name} not found",
                    "rtcache_trajectory": None,
                    "averaged_trajectory": None
                }
        
        # Generate embedding
        embedding = self.embedding_generator.generate(pil_image, instruction)
        
        if embedding is None:
            return {
                "success": False,
                "error": "Failed to generate embedding",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        # Search in Qdrant
        try:
            search_results = self._search_points(
                collection_name=collection_name,
                query_vector=embedding.tolist(),
                limit=top_k,
            )
        except Exception as e:
            logging.error(f"Search failed: {e}")
            return {
                "success": False,
                "error": f"Search failed: {e}",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        if not search_results:
            return {
                "success": False,
                "error": "No results found",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        # Get payloads from cache
        results = []
        for result in search_results:
            payload = self.payload_cache.get_payload(collection_name, str(result.id))
            if payload:
                results.append({
                    "id": str(result.id),
                    "score": result.score,
                    "payload": payload
                })
        
        if not results:
            return {
                "success": False,
                "error": "No payloads found in cache",
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }
        
        # Extract actions
        # Use the top result's action sequence
        top_result = results[0]
        current_action = top_result["payload"].get("current_action", [])
        next_actions = top_result["payload"].get("next_actions", [])
        
        # Construct trajectory: [current_action] + next_actions (typically 3 next steps)
        trajectory = [current_action] + next_actions
        
        # Also compute averaged trajectory from top-k results
        all_trajectories = []
        for r in results[:min(1, len(results))]:  # Average top 5
            traj = [r["payload"].get("current_action", [])] + r["payload"].get("next_actions", [])
            all_trajectories.append(traj)
        
        # Average trajectories
        if all_trajectories:
            # Convert to numpy for averaging
            try:
                all_traj_array = np.array(all_trajectories)
                averaged_traj = np.mean(all_traj_array, axis=0)
                averaged_trajectory = averaged_traj.tolist()
            except:
                averaged_trajectory = trajectory
        else:
            averaged_trajectory = trajectory
        
        return {
            "success": True,
            "task_id": task_id,
            "collection_name": collection_name,
            "top_score": results[0]["score"],
            "num_results": len(results),
            "rtcache_trajectory": trajectory,
            "averaged_trajectory": averaged_trajectory,
            "metadata": {
                "episode_idx": top_result["payload"].get("episode_idx"),
                "step_idx": top_result["payload"].get("step_idx"),
                "dataset_name": top_result["payload"].get("dataset_name"),
                "language_instruction": top_result["payload"].get("language_instruction")
            }
        }

###############################################################################
# Flask Server
###############################################################################
def create_app(config: RetrievalConfig) -> Flask:
    """Create Flask application"""
    
    app = Flask(__name__)
    
    # Initialize retrieval engine
    engine = RetrievalEngine(config)
    
    @app.route("/pipeline", methods=["POST"])
    def pipeline():
        """Main retrieval endpoint"""
        try:
            # Get image from request
            if 'file' not in request.files:
                return jsonify({
                    "success": False,
                    "error": "No file provided"
                }), 400
            
            file = request.files['file']
            pil_image = Image.open(file.stream).convert('RGB')
            
            # Get instruction (required for hash-based task mapping)
            instruction = request.form.get('instruction', '')
            
            if not instruction:
                return jsonify({
                    "success": False,
                    "error": "Instruction must be provided"
                }), 400
            
            # Retrieve (instruction will be hashed to determine collection)
            result = engine.retrieve(pil_image, instruction)
            
            return jsonify(result)
            
        except Exception as e:
            logging.error(f"Error in pipeline: {e}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e),
                "rtcache_trajectory": None,
                "averaged_trajectory": None
            }), 500
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "collections": len(engine.payload_cache.cache),
            "total_points": engine.payload_cache.stats["total_points"]
        })
    
    @app.route("/stats", methods=["GET"])
    def stats():
        """Statistics endpoint"""
        return jsonify(engine.payload_cache.stats)
    
    return app

###############################################################################
# Main
###############################################################################
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RT-Cache Retrieval Server for LIBERO-Goal"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=RetrievalConfig.SERVER_HOST,
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=RetrievalConfig.SERVER_PORT,
        help="Server port"
    )
    parser.add_argument(
        "--embedding-url",
        type=str,
        default=RetrievalConfig.EMBEDDING_URL,
        help="Embedding server URL"
    )
    parser.add_argument(
        "--qdrant-host",
        type=str,
        default=RetrievalConfig.QDRANT_HOST,
        help="Qdrant host"
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=RetrievalConfig.QDRANT_PORT,
        help="Qdrant port"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=RetrievalConfig.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Update config
    config = RetrievalConfig()
    config.SERVER_HOST = args.host
    config.SERVER_PORT = args.port
    config.EMBEDDING_URL = args.embedding_url
    config.QDRANT_HOST = args.qdrant_host
    config.QDRANT_PORT = args.qdrant_port
    
    logging.info("=" * 60)
    logging.info("RT-Cache Retrieval Server for LIBERO-Goal")
    logging.info("=" * 60)
    logging.info(f"Server: {config.SERVER_HOST}:{config.SERVER_PORT}")
    logging.info(f"Embedding URL: {config.EMBEDDING_URL}")
    logging.info(f"Qdrant: {config.QDRANT_HOST}:{config.QDRANT_PORT}")
    logging.info(f"Task Suite: {config.TASK_SUITE_NAME}")
    logging.info(f"Device: {config.DEVICE}")
    logging.info("=" * 60)
    
    # Create and run app
    app = create_app(config)
    app.run(
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()
```

返回结构如下：

```JSON
{'averaged_trajectory': [
        [
            0.5705357193946838,
            0.02142857201397419,
            0.0,
            -0.040714286267757416,
            0.14571428298950195,
            0.14142857491970062,
            -1.0
        ],
        [
            0.5625,
            0.02142857201397419,
            0.0,
            -0.04607142880558968,
            0.12857143580913544,
            0.1510714292526245,
            -1.0
        ],
        [
            0.5544642806053162,
            0.010714286006987097,
            0.0,
            -0.05678571388125419,
            0.10607142746448515,
            0.1607142835855484,
            -1.0
        ],
        [
            0.5491071343421936,
            0.0,
            0.0,
            -0.06535714119672775,
            0.0803571417927742,
            0.1735714226961136,
            -1.0
        ]
    ], 'collection_name': 'libero_goal_task_632', 'metadata': {'dataset_name': 'libero_goal_no_noops', 'episode_idx': 26, 'language_instruction': 'open the middle drawer of the cabinet', 'step_idx': 5
    }, 'num_results': 10, 'rtcache_trajectory': [
        [
            0.5705357193946838,
            0.02142857201397419,
            -0.0,
            -0.040714286267757416,
            0.14571428298950195,
            0.14142857491970062,
            -1.0
        ],
        [
            0.5625,
            0.02142857201397419,
            -0.0,
            -0.04607142880558968,
            0.12857143580913544,
            0.1510714292526245,
            -1.0
        ],
        [
            0.5544642806053162,
            0.010714286006987097,
            -0.0,
            -0.05678571388125419,
            0.10607142746448515,
            0.1607142835855484,
            -1.0
        ],
        [
            0.5491071343421936,
            0.0,
            -0.0,
            -0.06535714119672775,
            0.0803571417927742,
            0.1735714226961136,
            -1.0
        ]
    ], 'success': True, 'task_id': 632, 'top_score': 0.9820276
}
```

## 基于检索的动作控制测试

### 验证实验1: 任务精度测试

在run_libero_goal_AR.py基础上，将get_action替换为，当队列为空时，数据库检索，检索一次，将所有action slices入队列，依次执行队列内action slices。

替换后代码如下：

```Python
"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb
import json
import requests
from io import BytesIO
from PIL import Image
import time as time_module

RETRIEVAL_URL = "http://127.0.0.1:5002/pipeline"

# Append current directory so that interpreter can find experiments.robot
#sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    parallel_draft: bool = False
    accept_threshold: int = None
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    use_spec: bool = True
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    spec_checkpoint: Union[str, Path] = "/path/to/SpecVLA/openvla/specdecoding/train-scripts/ckpt_libero_goal_debug_ckpt/STATE_ID"
    task_suite_name: str = "libero_goal"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    # model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    # if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        # if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        #     cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        # assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    # if cfg.model_family == "openvla":
    #     processor = get_processor(cfg)

    # Initialize local logging
    target_dir = "/path/to/SpecVLA/openvla/specdecoding/test-speed/libero_goal_AR"
    os.makedirs(target_dir,exist_ok=True)
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(target_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    local_log_timefilepath = os.path.join(target_dir, run_id + "libero_Goal_AR.json")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        total_episode_time = []
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            action_queue = []
            total_time = []
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    # action,time = get_action(
                    #     cfg,
                    #     model,
                    #     observation,
                    #     task_description,
                    #     processor=processor,
                    #     return_time=True,
                    #     generate_mode = 'AR'
                    # )
                    
                    model = None # Placeholder for model since we are not using it

                    if len(action_queue) > 0:
                        action = action_queue.pop(0)
                        time = 0.0
                    else:
                        # Call Retrieval API
                        try:
                            pil_img = Image.fromarray(img)
                            buf = BytesIO()
                            pil_img.save(buf, format='PNG')
                            buf.seek(0)
                            
                            files = {"file": ("image.png", buf, "image/png")}
                            data = {
                                "instruction": task_description
                            }
                            
                            t0_req = time_module.time()
                            response = requests.post(RETRIEVAL_URL, files=files, data=data, timeout=30)
                            t1_req = time_module.time()
                            time = t1_req - t0_req
                            
                            if response.status_code == 200:
                                result = response.json()
                                print(result)
                                # Check if retrieval was successful
                                if not result.get('success', False):
                                    print(f"Retrieval failed: {result.get('error', 'Unknown error')}")
                                    action = np.zeros(7) 
                                    action[-1] = -1.0
                                else:
                                    # Try to get trajectory from result
                                    retrieved_traj = None
                                    if 'rtcache_trajectory' in result and result['rtcache_trajectory']:
                                        retrieved_traj = np.array(result['rtcache_trajectory'])
                                    elif 'averaged_trajectory' in result and result['averaged_trajectory']:
                                        retrieved_traj = np.array(result['averaged_trajectory'])
                                    
                                    if retrieved_traj is not None and len(retrieved_traj) > 0:
                                        # Store trajectory in queue
                                        if retrieved_traj.ndim == 1:
                                            action_queue = [retrieved_traj]
                                        else:
                                            action_queue = [a for a in retrieved_traj]
                                        
                                        # Pop first action
                                        action = action_queue.pop(0)
                                        
                                        # Suppress verbose retrieval source logging
                                    else:
                                        print("No trajectory found in API response.")
                                        action = np.zeros(7) 
                                        action[-1] = -1.0
                            else:
                                print(f"API Failed with status {response.status_code}: {response.text}")
                                action = np.zeros(7)
                                action[-1] = -1.0
                        except Exception as e:
                            print(f"API Error: {e}")
                            action = np.zeros(7)
                            action[-1] = -1.0
                            time = 0.0

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                    total_time.append(time)

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            total_episode_time.append(total_time)

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
    with open(local_log_timefilepath,mode='w') as f:
        json.dump(total_episode_time,f)
    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

if __name__ == "__main__":
    eval_libero()
```

效果如下：

<video data-lark-video-uri="drivetoken://KsfQbE8QRoG7MfxEKuRcUbFAnAe" data-lark-video-mime="video/mp4" data-lark-video-size="108123" data-lark-video-duration="0" data-lark-video-name="2c80e036b212017b15d92f9d5e3f102c.mp4" data-lark-video-width="224" data-lark-video-height="224"></video>

<video data-lark-video-uri="drivetoken://XE5wbCNIroTAZjxiU8Ucg7dXnZn" data-lark-video-mime="video/mp4" data-lark-video-size="95632" data-lark-video-duration="0" data-lark-video-name="9f1c41c3a7ef4cd0ad6446762c48b909.mp4" data-lark-video-width="224" data-lark-video-height="224"></video>

### 验证试验2: 接受长度测试

目前想基于run_libero_goal_Spec_Relaxed.py去做改进。

目前在tree decoding位置，如果是第一次，使用DB当前帧结果进行验证，否则使用SD。

目前测试得到，阈值为9的时候

```Bash
接受长度：0.8386229155459924
分布(1-5)：[630, 996, 157, 56, 19, 1]
```

阈值为15的时候

```Bash
接受长度：1.6126126126126126
分布(1-5)：[220, 488, 213, 174, 97, 29]
```

## 四个仿真环境下的扩展（建立数据库）

修正了/path/to/rtcache/scripts/data_processing/process_libero_goal.py

如下执行：

```Bash
# 处理 LIBERO-Goal
python process_libero_goal.py --dataset_type goal

# 处理 LIBERO-10
python process_libero_goal.py --dataset_type 10

# 处理 LIBERO-Object
python process_libero_goal.py --dataset_type object

# 处理 LIBERO-Spatial
python process_libero_goal.py --dataset_type spatial

# 指定其他参数
python process_libero_goal.py --dataset_type goal --batch_size 100 --max_episodes 50
```

重构了检索脚本，启动如下

```Bash
# 启动仅 goal
./start_libero_goal_retrieval.sh --dataset-types goal

# 启动 goal + object + spatial
./start_libero_goal_retrieval.sh --dataset-types "goal,object,spatial"

# 启动全部（goal/object/spatial/10/90）
./start_libero_goal_retrieval.sh --dataset-types all
```

启动纯DB测试命令

```Bash
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
  
conda activate specvla && \
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO && \
export MUJOCO_GL=egl && \
export ROBOSUITE_LOG_FILE=/path/to/SpecVLA/robosuite.log && \
export CUDA_VISIBLE_DEVICES=1 && \
export MUJOCO_EGL_DEVICE_ID=1 && \
python openvla/experiments/robot/libero/run_libero_goal_AR_DB.py \
  --pretrained_checkpoint /path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal \
  --model_family openvla \
  --task_suite_name libero_spatial \
  --use_spec False \
  --center_crop True
  
  
conda activate specvla && \
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO && \
export MUJOCO_GL=egl && \
export ROBOSUITE_LOG_FILE=/path/to/SpecVLA/robosuite.log && \
export CUDA_VISIBLE_DEVICES=1 && \
export MUJOCO_EGL_DEVICE_ID=1 && \
python openvla/experiments/robot/libero/run_libero_goal_AR_DB.py \
  --pretrained_checkpoint /path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal \
  --model_family openvla \
  --task_suite_name libero_object \
  --use_spec False \
  --center_crop True
  
  
conda activate specvla && \
export PYTHONPATH=$PWD:$PWD/openvla:$PWD/LIBERO && \
export MUJOCO_GL=egl && \
export ROBOSUITE_LOG_FILE=/path/to/SpecVLA/robosuite.log && \
export CUDA_VISIBLE_DEVICES=1 && \
export MUJOCO_EGL_DEVICE_ID=1 && \
python openvla/experiments/robot/libero/run_libero_goal_AR_DB.py \
  --pretrained_checkpoint /path/to/SpecVLA/backbone_models/openvla-7b-finetuned-libero-goal \
  --model_family openvla \
  --task_suite_name libero_10 \
  --use_spec False \
  --center_crop True
```

执行结果

```Bash
# 10
# episodes completed so far: 500
# successes: 22 (4.4%)
Current task success rate: 0.08
Current total success rate: 0.044

# goal
# episodes completed so far: 500
# successes: 150 (30.0%)
Current task success rate: 0.24
Current total success rate: 0.3


# object
# episodes completed so far: 500
# successes: 170 (34.0%)
Current task success rate: 0.6
Current total success rate: 0.34

# spatial
# episodes completed so far: 500
# successes: 122 (24.4%)
Current task success rate: 0.2
Current total success rate: 0.244
```

## Naive 检索+SD

数据集-SD(每轮执行次数)-DB(每轮执行次数)-Relax(放松阈值)

GOAL-SD(1)-DB(1)-Relax(9)执行结果

```Bash
Current task success rate: 0.4
Current total success rate: 0.59
================================================================================
Timing Summary:
================================================================================
[Timing] task 0: DB retrieval: 1074 calls, avg 0.047765s | Model (Spec): 1076 calls, avg 0.178512s | Total steps: 2145 (10 episodes)
[Timing] task 1: DB retrieval: 691 calls, avg 0.047112s | Model (Spec): 695 calls, avg 0.160325s | Total steps: 1378 (10 episodes)
[Timing] task 2: DB retrieval: 492 calls, avg 0.046976s | Model (Spec): 499 calls, avg 0.160109s | Total steps: 981 (10 episodes)
[Timing] task 3: DB retrieval: 1449 calls, avg 0.048251s | Model (Spec): 1450 calls, avg 0.164839s | Total steps: 2898 (10 episodes)
[Timing] task 4: DB retrieval: 889 calls, avg 0.046696s | Model (Spec): 890 calls, avg 0.157011s | Total steps: 1773 (10 episodes)
[Timing] task 5: DB retrieval: 1125 calls, avg 0.048306s | Model (Spec): 1127 calls, avg 0.168052s | Total steps: 2247 (10 episodes)
[Timing] task 6: DB retrieval: 956 calls, avg 0.047671s | Model (Spec): 959 calls, avg 0.154245s | Total steps: 1909 (10 episodes)
[Timing] task 7: DB retrieval: 512 calls, avg 0.047445s | Model (Spec): 515 calls, avg 0.152451s | Total steps: 1018 (10 episodes)
[Timing] task 8: DB retrieval: 1060 calls, avg 0.047188s | Model (Spec): 1063 calls, avg 0.150622s | Total steps: 2118 (10 episodes)
[Timing] task 9: DB retrieval: 1171 calls, avg 0.047745s | Model (Spec): 1173 calls, avg 0.166218s | Total steps: 2340 (10 episodes)
================================================================================
Overall Statistics:
  DB Retrieval: 9419 calls, average time: 0.047620s
  Model (Spec): 9447 calls, average time: 0.162281s
  Total steps: 18807
  Total episodes: 100
================================================================================
```

GOAL-SD(1)-DB(0)-Relax(9)

```Bash
Current task success rate: 0.3
Current total success rate: 0.75

================================================================================
Timing Summary:
================================================================================
[Timing] task 0: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 1953 calls, avg 0.176323s | Total steps: 1947 (10 episodes)
[Timing] task 1: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 948 calls, avg 0.158390s | Total steps: 938 (10 episodes)
[Timing] task 2: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 1199 calls, avg 0.153864s | Total steps: 1190 (10 episodes)
[Timing] task 3: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 2498 calls, avg 0.159118s | Total steps: 2492 (10 episodes)
[Timing] task 4: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 1223 calls, avg 0.156555s | Total steps: 1214 (10 episodes)
[Timing] task 5: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 1855 calls, avg 0.164980s | Total steps: 1847 (10 episodes)
[Timing] task 6: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 1964 calls, avg 0.148958s | Total steps: 1959 (10 episodes)
[Timing] task 7: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 953 calls, avg 0.152236s | Total steps: 943 (10 episodes)
[Timing] task 8: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 984 calls, avg 0.151297s | Total steps: 975 (10 episodes)
[Timing] task 9: DB retrieval: 0 calls, avg 0.000000s | Model (Spec): 2541 calls, avg 0.157718s | Total steps: 2538 (10 episodes)
================================================================================
Overall Statistics:
  DB Retrieval: 0 calls, average time: 0.000000s
  Model (Spec): 16118 calls, average time: 0.158906s
  Total steps: 16043
  Total episodes: 100
================================================================================
```

## 实验结果整理及分析

### 实验结果1: 纯检索-快速验证

|                      | **Long** | **Goal** | **Object** | **Spatial** |
| -------------------- | -------- | -------- | ---------- | ----------- |
| **S****uccess Rate** | 4.4%     | 30.0%    | 34.0%      | 24.4%       |

结论：纯检索的方案，精度差但速度快

### 实验结果2: naive 检索+SD

|                          | **纯检索** | **Naive 检索+SD（1:1交替）** | **纯SD**  |
| ------------------------ | ---------- | ---------------------------- | --------- |
| **检索****次数**         | 24875      | 9419                         | --        |
| **检索****平均每步时间** | 0.0523     | 0.047620s                    | --        |
| **检索总时间**           | 1300       | 448.533s                     | --        |
| **SD执行次数**           | --         | 9447                         | 16118     |
| **SD平均每步时间**       | --         | 0.153534s                    | 0.158906s |
| **SD总时间**             | --         | 1450.436s                    | 2561.247s |
| **SR**                   | 30.0%      | 59.0%                        | 75.0%     |
| **总加速比**             | 1.97 x     | 1.35 x                       | 1x        |

结论：

1. 结合检索和SD，有希望实现精度和速度的最优折衷。
2. naive结合的效果不好，需要有指标指导，什么时候用检索，什么时候用SD。
3. 指标可以从运动学引申出来，比如轨迹一致性、动作突变率之类的。