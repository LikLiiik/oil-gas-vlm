# Oil-Gas VLM — 油气地球物理多模态Agent工作流

基于本地部署 VLM (Qwen3-VL-8B) 串联 11 个下游模型，实现地震图像与测井曲线的多模态特征融合与有利目标识别。

## 架构

```
                           ┌──────────────────────────────────┐
                           │       Qwen3-VL-8B (大脑)          │
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
Phase 1: VLM 规划 (看5张图+任务+11个模型清单→自主选模型+参数)
Phase 2: 执行下游 (遍历workflow_steps, 调对应模型)
Phase 3: VLM 验证 (下游结果+原图→逐条判断真伪)
Phase 4: 迭代 (need_retry→调整参数→回到Phase 2, 最多3轮)
Phase 5: 聚合输出 (归一化→去重→3D SEG-Y + 标注PNG + report.json)
```

## 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt
#    geo_adapter 是独立子包，需单独安装（提供 geo-adapter CLI）
pip install -e 多模态接口/
#    可选：CIG-Bench 预训练断层/河道检测（cig_fault / cig_channel，需 GPU + 自动下载权重）
pip install cig-bench

# 2) geo_adapter 前置处理
geo-adapter prepare --config sample.yaml
# → runs/<sample_id>/

# 3) 本 pipeline
CUDA_VISIBLE_DEVICES=1 python -m pipeline \
    --run-dir runs/<sample_id> \
    --output-dir out/

# 关闭验证回环（快一倍）
python -m pipeline --run-dir runs/<sample> --no-verify
```

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

| 变量 | 说明 |
|------|------|
| `QWEN_VL_PATH` | Qwen3-VL 模型路径（必填，未配置会启动报错） |
| `OIL_GAS_LOG_LEVEL` | pipeline 日志级别（DEBUG/INFO/WARNING，默认 INFO） |

## 测试

```bash
# 93 个单元测试（不加载 VLM，秒级）
python test/test_pipeline_unit.py      # 19: schema/registry/JSON
python test/test_io_unit.py            # 12: SEG-Y/几何/exporter
python test/test_adapter_unit.py       # 14: adapter/tasks/aggregate
python test/test_downstream_unit.py    # 34: 下游模型+输出格式
python test/test_loop_unit.py          # 14: fake-VLM 闭环/假阳性过滤/仅重跑目标 step

# 端到端
CUDA_VISIBLE_DEVICES=1 python -m pipeline --run-dir runs/<sample>
```

## 项目结构

```
oil-gas-llm/
├── pipeline/
│   ├── orchestrator.py       # 统一闭环编排
│   ├── vlm.py                # VLMClient + JSON提取 + Schema重试
│   ├── agents.py             # LoopAgent + SingleShotAgent
│   ├── prompts.py            # Prompt 模板 (planning/verification/TASK_MAP)
│   ├── adapter.py            # geo_adapter RunPackage 载入
│   ├── tasks.py              # 地质任务注册表 + CLASS_ALIASES
│   ├── exporter.py           # 归一化→聚合→PNG/JSON/SEG-Y导出
│   ├── downstream/           # 11 个下游模型
│   ├── rag/                  # RAG 知识库 + TF-IDF 检索
│   └── io/                   # SEG-Y读写 + 几何 + 渲染
├── schemas/output_schemas.py # 8套 JSON Schema + validate
├── test/                     # 79 个单元测试
├── docs/                     # 技术文档
└── requirements.txt
```
