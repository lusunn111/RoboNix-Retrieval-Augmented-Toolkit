<div align="center">

# RoboNix Experience Memory and Reuse Skill

**A system-level experience memory, retrieval, and action-reuse Skill for embodied models**

[中文文档](README-CN.md) · [🚀 Quick Start](#quick-start) · [📦 Dataset](#datasets) · [🗄️ Build Database](#build-index) · [📝 Citation](#citation)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.16-DC244C)
![Status](https://img.shields.io/badge/two--view_pipeline-verified-1f9d72)
[![License](https://img.shields.io/badge/license-MulanPSL--2.0-red)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/lusunn111/RoboNix-Retrieval-Augmented-Toolkit?style=flat&logo=github)](https://github.com/lusunn111/RoboNix-Retrieval-Augmented-Toolkit/stargazers)

</div>

The **RoboNix Experience Memory and Reuse Skill** lets existing embodied models
use historical execution experience during online decision-making instead of
leaving it as offline data. It retrieves candidate actions that match the
current scene and instruction, then uses verification and policy fallback to
contain retrieval errors, reducing repeated inference without replacing the
original policy. The Skill currently supports OpenVLA and π0, single-view and
two-view Mix retrieval, and chunk-level hybrid verification.

<a id="performance-snapshot"></a>
## 📊 Performance Snapshot

The experience-memory path has been evaluated as part of verified hybrid
execution on both token-based and diffusion-based VLA models. Results report
task success rate (SR) and end-to-end speedup over the original model.

| Model | LIBERO suite | SR | Speedup |
| --- | --- | ---: | ---: |
| OpenVLA | Goal | 73.0% | 2.38× |
| OpenVLA | Object | 71.0% | **2.45×** |
| OpenVLA | Spatial | 78.0% | 1.90× |
| OpenVLA | Long | 47.0% | 1.79× |
| π0 | Goal | 93.33% | 2.97× |
| π0 | Object | **98.33%** | 2.21× |
| π0 | Spatial | 94.67% | 2.47× |
| π0 | Long | 78.33% | **3.01×** |

## 📚 Table of Contents

- [📊 Performance Snapshot](#performance-snapshot)
- [📰 News](#news)
- [⚡ System Capability and Results](#system-results)
- [🧠 Architecture Overview](#architecture)
- [🔌 RoboNix Integration and Outlook](#robonix-integration)
- [🧪 Validated Release](#validated-release)
- [⚙️ Requirements](#requirements)
- [🚀 Quick Start](#quick-start)
- [📦 Dataset and Checkpoint Sources](#datasets)
- [🗄️ Build the Retrieval Index](#build-index)
- [🌐 Start the Retrieval Service](#retrieval-service)
- [🗺️ Roadmap](#roadmap)
- [📝 Citation](#citation)
- [🤝 Contributors](#contributors)
- [📄 License](#license)

<a id="news"></a>
## 📰 News

- **2026-07-19**: 🆕 Released the system-level experience memory and reuse Skill
  with capability results, model support, and bilingual documentation.
- **2026-07-18**: 🔥 Validated the complete two-view image → embedding → Qdrant
  → 4×7 action-trajectory request path from the independent repository root.
- **2026-07-18**: 🗄️ Strictly checked 39 Mix collections containing 273,465
  experience points with 4,352-dimensional cosine vectors and action payloads.

<a id="system-results"></a>
## ⚡ System Capability and Results

From the RoboNix runtime perspective, this Skill is a stateful experience-memory
provider. It converts historical observations and actions into reusable memory,
retrieves candidate trajectories for the current task, and keeps verification
and fallback between retrieval and physical execution.

| System-level result | Current capability |
| --- | --- |
| Experience memory | **273,465** indexed robot-execution points |
| Online retrieval scale | **39** Qdrant collections with **4,352D** two-view vectors |
| Verified service response | Two images + instruction → **4×7** candidate action trajectory |
| Hybrid embodied execution | More than **2×** acceleration on OpenVLA and nearly **3×** on π0 |

### Supported models

| Model family | Status | Scope |
| --- | --- | --- |
| OpenVLA | ✅ Completed | Scene encoding, Qdrant retrieval, hybrid candidate generation, and action response |
| π0 | ✅ Completed | Chunk-level candidate verification in the preserved research implementation |
| π0.5 / π0-FAST | ⏳ In progress | Public end-to-end workflows are not yet completed |

<a id="architecture"></a>
## 🧠 Architecture Overview

<!--
IMAGEGEN ASSET
Active asset: docs/assets/retrieval-memory-overview-v2.png
Regeneration prompt: docs/assets/IMAGEGEN_PROMPTS.md
The original SVG is retained as an editable fallback.
-->

<div align="center">
  <img width="96%" alt="RoboNix retrieval-augmented memory architecture" src="docs/assets/retrieval-memory-overview-v2.png" />
  <p><b>Figure 1.</b> Offline experience-memory construction and online two-view retrieval with policy fallback and continuous memory updates.</p>
</div>

The validated two-view Mix path stores both searchable vectors and action
payloads in Qdrant; it does not require MongoDB. MongoDB remains available for
legacy collection workflows. The FastAPI embedding service generates OpenVLA
vision features, while the Flask retrieval service selects a task-specific
collection and returns a retrieved and optionally averaged action trajectory.

The single-view pipeline uses a third-person image. The Mix pipeline concatenates features from third-person and wrist views into a 4,352-dimensional representation:

| Feature                 |       Dimension |
| ----------------------- | --------------: |
| Third-person DINOv2     |           1,024 |
| Third-person SigLIP     |           1,152 |
| Wrist-view DINOv2       |           1,024 |
| Wrist-view SigLIP       |           1,152 |
| **Mix embedding** | **4,352** |

<a id="robonix-integration"></a>
## 🔌 RoboNix Integration and Outlook

This Skill is an independently deployable RoboNix provider that maintains experience memory for embodied execution. Scene observations and task context form a structured retrieval request; Atlas discovers the provider, Nexus carries multimodal references, and Pilot consumes the retrieved trajectories without embedding database logic into the RoboNix core.

<div align="center">
  <img width="96%" alt="RoboNix system architecture" src="docs/assets/robonix-system-architecture.png" />
  <p><b>Figure 2.</b> System-level integration points for reusable memory services, custom services, and VLA-based user skills.</p>
</div>

Looking forward, the service can evolve toward continuously updated robot memory with pluggable encoders, hierarchical indexes, redundancy compression, expiration policies, and safety-aware trajectory reuse across tasks and robot platforms.

<a id="validated-release"></a>
## 🧪 Validated Release

The release was validated on an NVIDIA A100 40GB server using an existing
modified LIBERO RLDS dataset and a prebuilt Qdrant Mix index.

| Check | Result |
| --- | --- |
| Package and independent-root CLI tests | 5 passed |
| Embedding service | OpenVLA loaded; `GET /health` healthy |
| Qdrant schema | 39 collections, 4,352D cosine vectors, payload checks passed |
| Database contents | 273,465 experience points |
| End-to-end request | Two images + instruction returned a 4×7 action trajectory |

<a id="quick-start"></a>
## 🚀 Quick Start

```bash
conda create -n robonix-retrieval python=3.10 -y
conda activate robonix-retrieval
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python -m pytest -q tests
python -m scripts.run --help
```

Prepare a large data root and download the pinned modified LIBERO RLDS dataset:

```bash
export DATA_ROOT=/data/robonix-retrieval
DOWNLOAD_FULL_DATASET=1 \
  scripts/data/download_libero_rlds.sh "$DATA_ROOT/datasets/libero_rlds"
```

Start Qdrant, then run the two-view embedding and retrieval services in separate
terminals or tmux sessions. The detailed commands are provided below. The
repository does not ship models, datasets, populated databases, or outputs.

<a id="datasets"></a>
## 📦 Dataset and Checkpoint Sources

| Asset | Source | Default placement |
| --- | --- | --- |
| Base OpenVLA | `openvla/openvla-7b` on Hugging Face | `$HF_HOME/hub` or a local model directory |
| Modified LIBERO RLDS | `openvla/modified_libero_rlds` | `$DATA_ROOT/datasets/libero_rlds` |
| Dataset revision | `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551` | Pinned by the download script |
| Qdrant index | Built from the RLDS dataset with `process_libero_goal_mix.py` | `$DATA_ROOT/databases/rtcache_mix_qdrant` |

The database is derived from third-person images, wrist images, language
instructions, the current 7D action, and the next three 7D actions. A public
release may publish a prebuilt index separately, but the repository itself
contains only the reproducible builder.

<a id="requirements"></a>
## ⚙️ Requirements

| Component        | Requirement                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| Operating system | Linux recommended for CUDA, Docker, LIBERO, and robot integration                                |
| Python           | 3.10 or later                                                                                    |
| PyTorch          | 2.2.0                                                                                            |
| CUDA             | Required for practical OpenVLA embedding throughput; match driver, toolkit, and PyTorch versions |
| Databases        | Qdrant for the validated Mix path; MongoDB only for legacy workflows                            |
| Models           | OpenVLA checkpoint; CLIP is used by applicable embedding modes                                   |
| Simulation       | LIBERO and its dataset assets for simulation experiments                                         |

The repository does not include model weights, datasets, populated databases, or robot-control software. Prepare the following before running the complete pipeline:

- an OpenVLA checkpoint and sufficient GPU memory;
- a Qdrant instance with persistent storage; MongoDB only for legacy workflows;
- robot demonstrations or LIBERO RLDS datasets;
- writable data, image, cache, log, and Qdrant backup directories;
- robot-side controllers when running real-hardware experiments.

## 🧰 Step 1: Installation

Clone the project and run all commands from the repository root:

```bash
git clone https://github.com/lusunn111/RoboNix-Retrieval-Augmented-Toolkit.git
cd RoboNix-Retrieval-Augmented-Toolkit

conda create -n rt-cache python=3.10 -y
conda activate rt-cache

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The root `requirements.txt` is the reproducible installation entry and delegates
to `requirements/requirements.txt`. Flash Attention is CUDA- and compiler-sensitive;
if installation fails, install the compatible PyTorch build first and then build
Flash Attention separately:

```bash
python -m pip install packaging ninja
python -m pip install "flash-attn==2.5.5" --no-build-isolation
```

Run lightweight smoke checks after installation:

```bash
python -c "import service_bootstrap as s; print(s.activate_vendor())"
python -m scripts.run --help
```

## 🗄️ Step 2: Start Qdrant and Optional MongoDB

The validated two-view Mix path requires only Qdrant. Start it with persistent
storage:

```bash
docker run -d \
  --name rtcache-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v rtcache_qdrant:/qdrant/storage \
  qdrant/qdrant
```

Start MongoDB only when using a preserved legacy workflow:

```bash
docker run -d \
  --name rtcache-mongo \
  -p 27017:27017 \
  -v rtcache_mongo:/data/db \
  mongo:6
```

Check that both services are reachable:

```bash
curl http://localhost:6333/healthz
python -c "from pymongo import MongoClient; print(MongoClient('mongodb://localhost:27017/').admin.command('ping'))"
```

Use authenticated connections, private networks, access controls, backups, and externally managed secrets for shared or production deployments. The example configuration is intended for local development.

## 🔧 Step 3: Configuration

Copy the environment template and customize it:

```bash
cp configs/.env.example .env
```

Important settings include:

```dotenv
# Databases
MONGO_URL=mongodb://localhost:27017/
MONGO_DB_NAME=OpenVLACollection
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Services
EMBEDDING_SERVER_HOST=0.0.0.0
EMBEDDING_SERVER_PORT=9020
EMBEDDING_SERVER_URL=http://127.0.0.1:9020/predict
RETRIEVAL_SERVER_HOST=0.0.0.0
RETRIEVAL_SERVER_PORT=5002

# Models and computation
OPENVLA_MODEL_PATH=/path/to/openvla
DEVICE=cuda:0
CUDA_VISIBLE_DEVICES=0

# Data
DATA_ROOT=/path/to/robot-datasets
LIBERO_DATASET_ROOT=/path/to/libero-rlds
QDRANT_BACKUP_ROOT=/path/to/qdrant-backups
```

The centralized configuration implementation is in `configs/rtcache/rt_cache_config.py`. CLI arguments may override some values. Prefer absolute paths for datasets, model checkpoints, and backups, and verify that no preserved upstream script still contains a source-machine path:

```bash
grep -R "/home/\|PATH_TO" scripts vendor/rtcache/scripts configs
```

## 🧠 Step 4: Start the Embedding Service

### Single-view service

The standard FastAPI service accepts one image and an optional instruction. It generates OpenVLA and/or CLIP features and listens on port 9020 by default:

```bash
python -m scripts.run \
  scripts/embedding/embedding_server.py \
  --host 0.0.0.0 \
  --port 9020 \
  --device cuda:0 \
  --workers 1
```

Verify the service:

```bash
curl http://localhost:9020/health

curl -X POST http://localhost:9020/predict \
  -F "file=@/path/to/observation.png" \
  -F "instruction=pick up the red object" \
  -F "option=both"
```

Interactive FastAPI documentation is available at `http://localhost:9020/docs`.

### Two-view Mix service

The Mix service requires third-person and wrist images and listens on port 9021 by default:

```bash
python -m scripts.run \
  scripts/embedding/embedding_server_mix.py \
  --host 0.0.0.0 \
  --port 9021 \
  --device cuda:0 \
  --workers 1
```

```bash
curl -X POST http://localhost:9021/predict \
  -F "third_person_image=@/path/to/third_person.png" \
  -F "wrist_image=@/path/to/wrist.png" \
  -F "instruction=place the object in the bowl" \
  -F "return_individual=false"
```

Do not run multiple model workers unless enough GPU memory is available. Each worker initializes its own model state.

<a id="build-index"></a>
## 🗄️ Step 5: Build the Retrieval Index

Start the embedding service before processing a dataset. The following example embeds LIBERO-Goal observations and inserts them into task-specific Qdrant collections:

```bash
python -m scripts.run \
  scripts/data_processing/process_libero_goal.py \
  --dataset_type goal \
  --base_dataset_path /path/to/libero-rlds \
  --embedding_server_url http://127.0.0.1:9020/predict \
  --qdrant_host localhost \
  --qdrant_port 6333 \
  --batch_size 50 \
  --max_episodes -1
```

Supported single-view dataset types are `goal`, `10`, `object`, and `spatial`. Collections follow names such as `libero_goal_task_0`. Use `--dataset_path` to override the path derived from the dataset type and `--use_benchmark` when LIBERO benchmark task IDs should replace hash-based mapping.

For Mix indexing:

```bash
python -m scripts.run \
  scripts/data_processing/process_libero_goal_mix.py \
  --dataset_type goal \
  --base_dataset_path /path/to/libero-rlds \
  --embedding_server_url http://127.0.0.1:9021/predict \
  --qdrant_host localhost \
  --qdrant_port 6333 \
  --batch_size 50 \
  --max_episodes -1 \
  --backup \
  --backup_name mix_base
```

Mix collections use names such as `libero_goal_mix_task_0`. Options including `--clear_db` and `--clear_all` delete existing collections and are destructive; verify the Qdrant target and backup required data before using them.

<a id="retrieval-service"></a>
## 🌐 Step 6: Start the Retrieval Service

### Single-view LIBERO retrieval

```bash
python -m scripts.run \
  scripts/retrieval/retrieval_libero_goal.py \
  --host 0.0.0.0 \
  --port 5002 \
  --embedding-url http://127.0.0.1:9020/predict \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --dataset-types goal
```

Send a retrieval request:

```bash
curl -X POST http://localhost:5002/pipeline \
  -F "file=@/path/to/observation.png" \
  -F "instruction=put the butter in the bowl" \
  -F "dataset_type=goal"
```

### Two-view Mix retrieval

```bash
python -m scripts.run \
  scripts/retrieval/retrieval_libero_goal_mix.py \
  --host 0.0.0.0 \
  --port 5003 \
  --embedding-url http://127.0.0.1:9021/predict \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --dataset-types goal
```

```bash
curl -X POST http://localhost:5003/pipeline \
  -F "third_person_image=@/path/to/third_person.png" \
  -F "wrist_image=@/path/to/wrist.png" \
  -F "instruction=put the butter in the bowl" \
  -F "dataset_type=goal"
```

Both variants expose `GET /health`, `GET /stats`, and `POST /pipeline`. A successful pipeline response includes retrieval metadata and fields such as `rtcache_trajectory` and `averaged_trajectory`. Treat returned actions as untrusted candidates: validate dimensions, limits, freshness, robot state, and collision constraints before execution on physical hardware.

## 🔌 Service Ports

| Service               | Default port | Purpose                                                                       |
| --------------------- | -----------: | ----------------------------------------------------------------------------- |
| MongoDB               |        27017 | Trajectory records and metadata                                               |
| Qdrant HTTP           |         6333 | Vector storage and similarity search                                          |
| Qdrant gRPC           |         6334 | Optional high-throughput vector API                                           |
| Single-view embedding |         9020 | OpenVLA/CLIP embedding API                                                    |
| Mix embedding         |         9021 | Two-view 4,352-dimensional embedding API                                      |
| Data collection       |         5002 | Robot demonstration collection when enabled                                   |
| Single-view retrieval |         5002 | Standard retrieval API; do not colocate with data collection on the same port |
| Mix retrieval         |         5003 | Two-view retrieval API                                                        |

Port 5002 is used by more than one preserved workflow. Assign distinct ports when data collection and retrieval services run concurrently.

## 📊 Benchmarking Guidelines

Compare RT-Cache with VINN, BehaviorRetrieval, and non-retrieval baselines using the same observations, task split, control frequency, and robot or simulator configuration. At minimum, report:

- task success rate and completed episodes;
- embedding, vector-search, and end-to-end latency;
- p50, p95, and p99 online latency after warm-up;
- retrieval top score, top-k setting, and similarity threshold;
- database collection count, point count, and embedding dimension;
- GPU memory, embedding throughput, and database resource usage;
- action-horizon length and fallback or rejection count.

Measure cold start separately from steady-state performance. The retrieval services preload collection payloads into memory, so startup time and host-memory usage scale with the database. Network transfer, image encoding, database placement, and robot control-loop latency must be included in end-to-end measurements.

## 🗂️ Repository Layout

```text
.
├── modules/                       # Lazy module catalogs
│   ├── database/                  # MongoDB and Qdrant backends
│   ├── scene_encoding/            # OpenVLA/CLIP feature generation
│   ├── indexing/                  # Vector indexing views
│   ├── retrieval/                 # Online similarity retrieval
│   ├── memory_update/             # Backup, restore, and cleanup
│   └── verified_execution/        # SpecVLA and rebuttal implementations
├── scripts/
│   ├── data/                      # Dataset processing and acquisition
│   ├── serve/                     # Embedding and retrieval service views
│   ├── maintenance/               # Qdrant backup and restore utilities
│   └── run.py                     # Stable RT-Cache script runner
├── benchmarks/
│   ├── behavior_retrieval/        # BehaviorRetrieval baseline
│   ├── vinn/                      # VINN baseline
│   ├── specvla_validation/        # SpecVLA validation snapshot
│   └── rebuttal/                  # FLASH/OpenPI rebuttal snapshot
├── configs/                       # Environment, database, and RT-Cache config
├── requirements.txt               # Reproducible installation entry
├── requirements/                  # Python dependency pins
├── tests/                         # Layout and lazy-import tests
├── docs/assets/                   # Architecture assets and the web ImageGen prompt
├── utils/                         # Database, embedding, and image utilities
├── vendor/rtcache/                # Canonical RT-Cache source tree
└── service_bootstrap.py           # Vendor activation and guarded runner
```

`vendor/rtcache/` is the canonical import-compatible retrieval implementation. Top-level directories provide an engineering-oriented view of the data, service, maintenance, and benchmark workflows. Validation and rebuttal sources are canonical under their respective benchmark directories.

<a id="roadmap"></a>
## 🗺️ Roadmap

- [x] Publish an independently runnable source-only repository.
- [x] Validate OpenVLA embedding, Qdrant retrieval, and a 4×7 action response.
- [x] Preserve the π0 chunk-level retrieval and verification research path.
- [x] Adopt the RoboNix Mulan PSL v2 license and remove citation placeholders.
- [ ] Publish a small public two-view example dataset and prebuilt Qdrant index.
- [ ] Add container images and authenticated production API examples.
- [ ] Add incremental memory deduplication, compression, and expiration policies.
- [ ] Complete public end-to-end workflows for π0.5 and π0-FAST.
- [ ] Provide a versioned RoboNix service adapter.

<a id="citation"></a>
## 📝 Citation

If this Skill supports your research, please consider giving the repository a
star ⭐ and citing this software repository:

```bibtex
@software{mao2026robonix_experience_memory_reuse_skill,
  author  = {Mao, Zhihao and He, Huiru and Zheng, Zihao},
  title   = {RoboNix Experience Memory and Reuse Skill},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/lusunn111/RoboNix-Retrieval-Augmented-Toolkit}
}
```

<a id="contributors"></a>
## 🤝 Contributors

We thank [HuiruHe](https://github.com/HuiruHe) and
[zhengzihaoPKU](https://github.com/zhengzihaoPKU) for their contributions to
the Skill. See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the contributor policy.

<a id="license"></a>
## 📄 License

The project is licensed under the Mulan Permissive Software License, Version 2
(Mulan PSL v2); see [LICENSE](LICENSE). Vendored components retain their included
licenses.
