# Oil-Gas VLM — 油气地球物理多模态Agent工作流

基于 **VLM** 串联 11 个下游模型，实现地震图像与测井曲线的多模态特征融合与有利目标识别。

VLM 支持两种后端：
- **本地**：本地部署的 Qwen3-VL（默认，需要 GPU + 模型权重）
- **API**：任意 OpenAI 兼容多模态端点（DashScope / OpenAI / vLLM / 代理 等）

两种后端共用同一套 Prompt、Schema、下游模型与报告结构，便于 A/B 对比。

## 架构

```
                           ┌──────────────────────────────────┐
                           │            VLM (大脑)              │
                           │  本地 Qwen3-VL  |  OpenAI 兼容 API  │
                           │                                  │
                           │  ① 分析图像 → 理解地质场景        │
                           │  ② 规划 → 自主选择下游模型+参数   │
                           │  ③ 验证 → 下游结果+原图→判断真伪  │
                           │  ④ 迭代 → 调整参数重试至收敛      │
                           └──────┬──────────────┬────────────┘
                                  │ ② plan       │ ③ results
                                  ▼              ▼
                    ┌──────────────────────────────────────────┐
                    │          11 个下游模型 (手)                │
                    ├────────────────┬─────────────────────────┤
                    │ 带权重(5)       │ 领域算法(6)              │
                    │ cig_fault      │ seismic_domain_model    │
                    │ cig_channel    │ horizon_tracker         │
                    │ seismic_       │ facies_classifier       │
                    │   foundation   │ well_log_analyzer       │
                    │ well_log_ml    │ attribute_extractor     │
                    │                │ traditional_code        │
                    │                │ sam (轻量分割)           │
                    └────────────────┴─────────────────────────┘
```

**核心设计**：VLM 做"大脑"（场景理解+规划+验证），下游模型做"手"（精确执行）。VLM 不是一次性发指令，而是**闭环迭代**：规划→执行→验证→调整→收敛。

## 数据流

```
raw .sgy/.las/.csv
     │
     ▼  geo_adapter (前置处理，独立模块)
     │
runs/<sample_id>/
  ├── assets/seismic/{inline,crossline,slice,patch}_model.png
  ├── assets/well_logs/well_log_panel.png
  ├── prompts/{system_prompt,user_prompt}.txt
  ├── manifest.json
  └── schemas/expected_model_output.schema.json
     │
     ▼  python -m pipeline --run-dir ...
     │
Phase 1: VLM 分视图规划 (每个物理视图单独分析→合并证据与可执行步骤)
Phase 2: 执行下游 (遍历workflow_steps, 调对应模型)
Phase 3: VLM 验证 (下游结果+原图→逐条判断真伪)
Phase 4: 迭代 (need_retry→调整参数→回到Phase 2, 最多N轮)
Phase 5: 聚合输出 (归一化→去重→3D SEG-Y + 标注PNG + report.json)
```

## 快速开始

### 1) 选择后端 + 装依赖

**本地模式**（默认；需要 GPU + Qwen3-VL 权重）：

```bash
pip install -r requirements.txt
export VLM_BACKEND=local
export QWEN_VL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

**API 模式**（无 GPU，远程多模态 API）：

```bash
pip install -r requirements-api.txt      # 不包含 torch / transformers
export VLM_BACKEND=api
export VLM_API_KEY=...                   # 或 DASHSCOPE_API_KEY=...
export VLM_BASE_URL=https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
export VLM_MODEL=qwen3-vl-plus
```

> `.env` 文件也可（参考 `.env.example`；`.env` 已在 `.gitignore` 中，**绝不**提交）。

### 2) 跑 Pipeline

两个后端**调用方式完全一致**——只需要切换 `VLM_BACKEND`：

```bash
# 任意后端都这样跑
python -m pipeline --run-dir runs/<sample_id> --output-dir out/

# 推荐 A/B 第一轮：关验证、只跑 1 轮迭代（降低费用 / 时间）
python -m pipeline --run-dir runs/<sample_id> --no-verify --max-iter 1

# 想从 CLI 覆盖后端
python -m pipeline --run-dir runs/<sample_id> --vlm-backend api --vlm-model qwen3-vl-plus
```

**Windows PowerShell**：

```powershell
$env:VLM_BACKEND = "api"
$env:VLM_API_KEY  = "<填写在本机，不要提交>"
$env:VLM_BASE_URL = "https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
$env:VLM_MODEL    = "qwen3-vl-plus"

python -m pipeline `
  --run-dir runs/<sample_id> `
  --output-dir out_api `
  --no-verify `
  --max-iter 1
```

### 3) 关键说明

1. **API 模式不需要本地 GPU**——只装 `requirements-api.txt` 就能在笔记本上跑。
2. **API 模式不加载本地模型权重**——torch / transformers 都不会被 import。
3. **API 调用可能产生费用**——规划阶段按物理视图逐图调用，验证阶段也按图调用。**第一轮 A/B 务必加 `--no-verify --max-iter 1`** 控制成本。
4. **本地与 API 用相同 Prompt、Schema、下游模型**——便于公平比较。
5. **地震 PNG 用于构造和位置证据，不用于恢复原始振幅精度**——VLM 使用带坐标、色标和原生网格尺寸的分析图；下游仍使用无标注的模型图和 NPY 数组。
6. **`.env` 已在 `.gitignore` 里**——`.env.example` 可以提交作为模板。
7. **测井精确数值来自结构化摘要**——曲线统计与代表性采样点来自 CSV/数组；PNG 只用于趋势判断，禁止靠像素估读精确值。
8. **目标类别只是候选，不是图中存在的证据**——没有可定位的像素证据时允许输出 `absent` / `insufficient` 和空工作流；运行前还会过滤当前机器不可执行的下游模型。

## 输出

```
out/
├── vlm_plan.json              # VLM 工作流规划
├── report.json                # 汇总报告
├── fault_attribute.sgy        # 每个类别一个 3D 属性 SEG-Y
├── channel_attribute.sgy
├── *_inline.png               # 标注 PNG (bbox 叠加原图)
└── *_crossline.png
```

## 下游模型

| 模型 | 类型 | 说明 |
|------|------|------|
| **cig_fault** | 开源权重 | CIG-Bench HRNet 断层检测 (~40MB, 自动下载) |
| **cig_channel** | 开源权重 | CIG-Bench HRNet 河道检测 (~40MB, 自动下载) |
| **seismic_foundation** | 开源权重 | SFM seismic预训练ViT (~85MB) |
| **well_log_ml** | 开源权重 | RandomForest 岩石物理模型 (~1MB, 缓存) |
| **seismic_domain_model** | 领域算法 | 相干体+结构张量+梯度断层检测 |
| **horizon_tracker** | 领域算法 | 互相关层位自动追踪 |
| **facies_classifier** | 领域算法 | PCA+GMM 多属性沉积相分类 |
| **well_log_analyzer** | 领域算法 | changepoint分割+交会图岩性/流体识别 |
| **attribute_extractor** | 领域算法 | Hilbert 瞬时属性+GLCM纹理+谱分解 |
| **traditional_code** | 规则引擎 | VLM 指定阈值规则的精确执行 |
| **sam** | 轻量分割 | Otsu/flood fill, seismic 优化 |

## RAG 知识库

5 篇领域知识文档，TF-IDF + 关键词混合检索。VLM 规划时自动注入相关知识：

| 文档 | 内容 |
|------|------|
| `01_fault_detection.md` | 断层识别标志、模型选择、假阳性来源 |
| `02_horizon_tracking.md` | 层位识别、horizon_tracker 参数建议 |
| `03_facies_analysis.md` | 反射构型→沉积相对应表、聚类参数 |
| `04_well_log_analysis.md` | GR/DEN/RT岩性表、流体识别阈值 |
| `05_seismic_attributes.md` | 属性分类、地质含义、裂缝检测参数 |

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `VLM_BACKEND` | 否 | `local`（默认）或 `api`。未配置时按以下规则选：若 `VLM_API_KEY` 或 `DASHSCOPE_API_KEY` 存在且**显式** `VLM_BACKEND` 没设，仍默认 `local`（安全默认：避免产生意外费用）；要 API 必须显式设 `VLM_BACKEND=api`。 |
| **本地后端** | | |
| `QWEN_VL_PATH` | 本地必填 | Qwen3-VL 模型路径 |
| `VLM_LOCAL_DTYPE` | 否 | `bfloat16`（默认）/`float16`/`float32` |
| `VLM_LOCAL_DEVICE_MAP` | 否 | `auto`（默认） |
| **API 后端** | | |
| `VLM_API_KEY` / `DASHSCOPE_API_KEY` | API 必填 | 优先级：`VLM_API_KEY` > `DASHSCOPE_API_KEY` |
| `VLM_BASE_URL` | API 推荐 | 兼容模式端点，如 `https://...compatible-mode/v1` |
| `VLM_MODEL` | 否 | 默认 `qwen3-vl-plus` |
| `VLM_TIMEOUT` | 否 | 单次 HTTP 超时秒数，默认 180 |
| `VLM_MAX_TOKENS` | 否 | 默认 6144 |
| `VLM_TEMPERATURE` | 否 | 默认 0.1 |
| `VLM_API_MAX_RETRIES` | 否 | 网络错误有限重试次数（与 schema 重试独立），默认 2 |
| `VLM_API_JSON_MODE` | 否 | `true` 时传 `response_format={"type":"json_object"}`，默认 `false` |
| `VLM_API_MAX_IMAGE_EDGE` | 否 | 长边缩放像素；`0`（默认）=不缩放 |
| `VLM_API_IMAGE_FORMAT` | 否 | `PNG`（默认）/`JPEG`/`WEBP` |
| `VLM_API_JPEG_QUALITY` | 否 | 默认 95 |
| `OIL_GAS_LOG_LEVEL` | 否 | pipeline 日志级别，默认 `INFO` |

> 安全原则：错误日志中**绝不**打印 API key；图片 base64 **绝不**入日志（只记尺寸 / 字节数 / 格式）。

## CLI 参数

| 参数 | 说明 |
|------|------|
| `--vlm-backend {local,api}` | 覆盖 `VLM_BACKEND` env。CLI 显式 > env > 默认 local |
| `--vlm-model NAME` | 覆盖 `VLM_MODEL` |
| `--vlm-base-url URL` | 覆盖 `VLM_BASE_URL` |
| `--vlm-api-key` | **故意不暴露**——避免密钥进 shell 历史 / 进程列表，请用 env / `.env` |

## 测试

```bash
# 单元测试（不加载 VLM，秒级；API 测试全部 mock，不产生费用）
python test/test_pipeline_unit.py      # 19: schema/registry/JSON
python test/test_io_unit.py            # 12: SEG-Y/几何/exporter
python test/test_downstream_unit.py    # 34: 下游模型+输出格式
python test/test_loop_unit.py          # 22: fake-VLM 闭环/假阳性过滤/仅重跑目标 step
python test/test_vlm_api_unit.py       # 26: API 后端全套（mock）

# 真实 API 冒烟（**会**产生费用！需先设 VLM_API_KEY）
$env:VLM_API_KEY = "..."; python scripts/test_vlm_api.py --max-images 2

# A/B 对比（默认 dry-run；显式 --run-both 才真实调用两边）
python scripts/compare_vlm_backends.py --run-dir runs/<sample>
python scripts/compare_vlm_backends.py --run-dir runs/<sample> --run-both --no-verify --max-iter 1
```

## 项目结构

```
oil-gas-llm/
├── pipeline/
│   ├── orchestrator.py       # 统一闭环编排
│   ├── vlm.py                # VLMClient 门面：后端切换 + JSON 提取 + Schema 重试
│   ├── vlm_backends/         # 后端实现
│   │   ├── base.py           #   抽象基类
│   │   ├── local_qwen.py     #   本地 Qwen3-VL（惰性加载 torch）
│   │   └── openai_compatible.py # OpenAI 兼容 API（不加载 torch）
│   ├── agents.py             # LoopAgent + SingleShotAgent
│   ├── prompts.py            # Prompt 模板 (planning/verification/TASK_MAP)
│   ├── adapter.py            # geo_adapter RunPackage 载入
│   ├── tasks.py              # 地质任务注册表 + CLASS_ALIASES
│   ├── exporter.py           # 归一化→聚合→PNG/JSON/SEG-Y导出
│   ├── downstream/           # 11 个下游模型
│   ├── rag/                  # RAG 知识库 + TF-IDF 检索
│   └── io/                   # SEG-Y读写 + 几何 + 渲染
├── schemas/output_schemas.py # 8套 JSON Schema + validate
├── scripts/                  # 冒烟测试 + A/B 对比
│   ├── test_vlm_api.py
│   └── compare_vlm_backends.py
├── test/                     # 单元测试（纯 mock，不产生费用）
├── docs/                     # 技术文档
├── requirements.txt          # 完整依赖（本地 + API）
├── requirements-api.txt      # 仅 API 模式（无 torch）
├── .env.example              # 环境变量模板（提交）
└── .env                      # 本机真实密钥（**绝不**提交，已在 .gitignore）
```
