# Oil-Gas VLM — 油气地球物理多模态Agent工作流

基于本地部署 VLM (Qwen3-VL-8B) 串联纯推理下游模块，实现地震图像与测井曲线的多模态特征融合与有利目标识别。

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
                    │          9 个纯推理模块 (手)               │
                    ├────────────────┬─────────────────────────┤
                    │ 带权重(5)       │ 领域算法(6)              │
                    │ cig_fault      │ seismic_domain_model    │
                    │ cig_channel    │ horizon_tracker         │
                    │ sam            │ facies_classifier       │
                    │                │ well_log_analyzer       │
                    │                │ attribute_extractor     │
                    │                │ traditional_code        │
                    │                │ sam (轻量分割)           │
                    └────────────────┴─────────────────────────┘
```

**核心设计**：VLM 做"大脑"（场景理解+规划+验证），下游模型做"手"（精确执行）。VLM 不是一次性发指令，而是**闭环迭代**：规划→执行→验证→调整→收敛。

## `star` 分支本次更新

本分支保持原有 `geo_adapter → Qwen3-VL 规划 → 下游纯推理 → PNG/SEG-Y` 技术路线，未引入训练或微调，主要针对真实数据测试中发现的下游可靠性问题进行修正。

### 1. 下游执行约束

- 将原始数组形状、有效 trace/sample 索引范围写入规划上下文，减少 VLM 生成越界参数。
- 执行前校正层位追踪种子点，并在 `execution_adjustments` 中记录校正前后坐标。
- 同时存在 inline 和 crossline 时，断层任务自动补充另一方向的检测步骤，用正交剖面提供一致性证据。
- 为每个下游结果生成包含图像名和 workflow step 的唯一 ID，避免不同方向结果使用相同 ID 而串用验证结论。

### 2. 断层重试与误报控制

- 当 VLM 验证结果中误报多于真阳性时，重试只能收紧断层参数，不能降低置信度阈值或最小区域面积。
- 误报占优时同步约束 inline/crossline：`confidence_threshold >= 0.55`、`min_region_area_pixels >= 1000`。
- 每个方向最多保留 8 个高分断层候选，防止宽松参数产生大量连通域并挤占验证上下文。
- VLM 未审查或判为误报的稠密结果保留为数值证据，但输出类别降级为 `fault_candidate`；只有明确验证通过的结果才能使用 `fault`。
- `report.json` 新增 `downstream.verification_coverage`，分别报告候选总数、已审查、已验证、已拒绝和未审查数量。

### 3. 沉积相与测井解释

- GMM 聚类结果统一命名为 `attribute_cluster_0...N`，不在缺少地质标定时直接宣称为具体沉积相。
- 测井解释支持读取邻近 formation tops，并在结果中保留层位上下文。
- 高电阻率不再自动解释为水层，改用 `high_resistivity_anomaly`、`hydrocarbon_candidate` 和 `fluid_ambiguous` 等保守类别。

### 4. 输出与真实数据支持

- 属性 SEG-Y 优先复制参考 SEG-Y 的 trace header、inline/crossline 和采样轴，保持原始三维几何。
- 最终 `vlm_plan.json` 保存实际执行的重试参数和全部调整记录，而不是只保存初始规划。
- 新增 `scripts/prepare_teapot_real.py`，用于 Teapot/RMOTC 数据的 API 编号归一化、LAS 单位转换、井轨迹与层位表整理，以及非标准 SEG-Y inline/crossline 字段映射。
- 新增合成批量测试脚本和断层 F1 阈值评测，均只评估推理结果，不更新模型参数。

> 注意：`fault_candidate`、`attribute_cluster_*` 和 `fluid_ambiguous` 是待解释或待验证结果，不能在报告中直接表述为已确认断层、地震相或油气层。

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
Phase 1: VLM 规划 (看5张图+任务+已注册模型清单→自主选模型+参数)
Phase 2: 执行下游 (遍历workflow_steps, 调对应模型)
Phase 3: VLM 验证 (下游结果+原图→逐条判断真伪)
Phase 4: 迭代 (need_retry→调整参数→回到Phase 2, 最多3轮)
Phase 5: 聚合输出 (归一化→去重→3D SEG-Y + 标注PNG + report.json)
```

## 快速开始

```bash
# 1) 安装依赖
pip install -r requirements.txt

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
| `QWEN_VL_PATH` | Qwen3-VL 模型路径 |
| `YOLO_WORLD_PATH` | YOLO-World 权重路径（可选，未启用） |

## 测试

```bash
# 核心单元测试
python test/test_pipeline_unit.py      # 19: schema/registry/JSON
python test/test_io_unit.py            # 12: SEG-Y/几何/exporter
python test/test_adapter_unit.py       # 14: adapter/tasks/aggregate
python test/test_downstream_unit.py    # 34: 下游模型+输出格式

# 端到端
CUDA_VISIBLE_DEVICES=1 python -m pipeline --run-dir runs/<sample>
```

### 分割 F1 / Dice 评测（不训练模型）

```bash
# 固定阈值
python downstream/scripts/evaluate_segmentation_f1.py \
  --prediction out/fault_attribute.sgy \
  --target labels/fault.sgy \
  --threshold 0.5

# 仅扫描推理阈值，不更新任何模型参数
python downstream/scripts/evaluate_segmentation_f1.py \
  --prediction out/fault_attribute.sgy \
  --target labels/fault.sgy \
  --sweep 0.1,0.9,0.05 \
  --save out/fault_metrics.json
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
│   ├── downstream/           # 9 个默认纯推理模块
│   ├── rag/                  # RAG 知识库 + TF-IDF 检索
│   └── io/                   # SEG-Y读写 + 几何 + 渲染
├── schemas/output_schemas.py # 8套 JSON Schema + validate
├── test/                     # 单元测试与纯推理回归测试
├── docs/                     # 技术文档
└── requirements.txt
```
