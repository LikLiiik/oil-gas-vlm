# Oil-Gas VLM — 油气地球物理多模态Agent工作流

基于本地部署 VLM (Qwen3-VL-8B) 串联下游专家模型（YOLO-World / SAM / 传统岩石物理代码），
实现地震图像与测井曲线的多模态特征融合与有利目标识别。

**赛题数据流**：`.sgy + .las + .csv` → **geo_adapter**（前置数据处理） → 标准 run 目录 → **本 pipeline**（VLM + YOLO + 验证回环） → JSON + 标注 PNG + 属性 SEG-Y。

```
[前置：多模态接口 / geo_adapter]                    [本 pipeline]
raw SEG-Y   ┐                                       ┌──▶ 标注 PNG (bbox 叠原图)
raw LAS      ├──▶ geo-adapter prepare               │
well_loc.csv ┘        │                              ├──▶ 属性 3D SEG-Y (OpendTect/Petrel 可视化)
                      ▼                              │
              runs/<sample_id>/                     ├──▶ vlm_output.json (符合契约 schema)
                assets/*.png                        │
                prompts/*.txt        ────────────▶  ├──▶ vlm_output_refined.json (验证回环)
                manifest.json                       │
                schemas/expected_*.json             └──▶ report.json (总)
```

**两条命令跑通端到端**：

```bash
# 1) 前置：把原始数据处理成标准 run 包（用另一个 repo）
cd /path/to/多模态接口 && geo-adapter prepare --config path/to/sample.yaml
# → runs/<sample_id>/ 生成

# 2) 本 pipeline：消费 run 包
CUDA_VISIBLE_DEVICES=1 python -m pipeline \
    --run-dir /path/to/多模态接口/runs/<sample_id> \
    --output-dir out/
```

## 技术路线

### 核心思路：VLM 做"大脑"，下游模型做"手"，结果回环验证

VLM 不是一次性发指令，而是**闭环迭代**：

```
                       ┌──────────────────────────────┐
                       │        Qwen3-VL-8B (大脑)      │
                       │                               │
                       │  ① 分析图像 → 生成下游指令      │
                       │  ④ 接收下游结果 → 验证/拒绝     │
                       │  ⑤ 地质合理性判断 → 调整指令    │
                       │  ⑥ 迭代至收敛 → 输出最终结论    │
                       └──────┬────────────┬───────────┘
                              │            │
              ① downstream   │            │  ④ results
              prompts         │            │  come back
                              ▼            ▼
                    ┌──────────────┐  ┌──────────────┐
                    │  YOLO-World  │  │   SAM + 代码  │
                    │  精确检测     │  │  精确分割/计算 │
                    │  (bbox)      │  │  (mask/数值)  │
                    └──────────────┘  └──────────────┘
```

**单次 VLM → 下游 是不够的**。VLM 需要看到下游的检测结果才能做两件关键的事：

1. **地质合理性验证**：YOLO 检测到 3 个"断层"候选 → VLM 逐一判断是真断层还是假阳性
2. **迭代优化**：如果检测结果不满意 → VLM 调整下游参数（扩大搜索范围、降低阈值、增加新类别）

### 闭环示例

```
Round 1:
  VLM → YOLO: "detect fault plane in CDP 60-100"
  YOLO → VLM: bbox1(CDP 78-85, conf=0.45), bbox2(CDP 195-202, conf=0.32)

Round 2 (VLM 验证):
  VLM: "bbox1: 同相轴明显错断 ✓ 真断层"
       "bbox2: 只是河道边缘的反射终止 ✗ 假阳性，排除"
       "CDP 115-125 附近似乎遗漏了一条小断层 → 追加检测"
  VLM → YOLO: "add detection in CDP 110-130, lower threshold to 0.25"

Round 3 (收敛):
  YOLO → VLM: bbox3(CDP 118-123, conf=0.28)
  VLM: "bbox3: 微小错断，可能是伴生断层 ✓ 保留，标注low confidence"
  VLM → 最终输出: faults=[bbox1(high), bbox3(low)]
```

### 为什么这样设计

| 模型层 | 擅长 | 不擅长 | 在流程中的角色 |
|--------|------|--------|---------------|
| **VLM** | 场景理解、语义识别 | 空间精度（像素偏差大）、数值读取不准 | 看图像 → 决定需要检测什么 → 生成下游指令 |
| **YOLO-World** | 开放词汇目标检测、bbox精确 | 需要明确的类别描述 | 接收 `class_name` + `expected_range` → 输出精确bbox |
| **SAM** | 像素级分割、point/bbox引导 | 不知道"该分割什么" | 接收 VLM 给的 point/bbox → 输出精确mask |
| **传统代码** | 数值精度(±0.1m)、可复现 | 无法理解图像 | 接收 VLM 给的阈值规则 → 在原始numpy数据上精确定位 |

---

## 四个Agent的数据流

### Agent 1: SeismicInterpAgent — 地震剖面解释

```
SEG-Y 3D体 → 提取2D切片 → PNG图像
    │
    ▼
┌─ Qwen3-VL-8B ──────────────────────────────────────┐
│                                                     │
│  ① 分析: 识别褶皱地层、可能断层区、亮点异常区            │
│     ↓                                               │
│  ② 规划: 生成 YOLO-World + SAM 检测指令               │
│     ↓                                               │
│  [下游执行: YOLO检测 → SAM分割]                       │
│     ↓                                               │
│  ③ 验证回环: 接收 bbox/mask 结果                       │
│     - bbox1(CDP 78-85): 同相轴错断 ✓ 真断层           │
│     - bbox2(CDP 195): 河道边缘 ✗ 假阳性，排除          │
│     - 追加 CDP 115-125 区域检测（遗漏的小断层）         │
│     ↓                                               │
│  ④ 迭代: 调整阈值 → 重新检测 → 再次验证                 │
│     ↓                                               │
│  ⑤ 收敛: 输出最终 faults + anomalies + 解释报告        │
└─────────────────────────────────────────────────────┘
```

### Agent 2: LogAnalysisAgent — 测井曲线分析

```
LAS文件 → 6道综合图 → PNG图像
    │
    ▼
┌─ Qwen3-VL-8B ──────────────────────────────────────┐
│                                                     │
│  ① 分析: GR低值段=砂岩, RT尖峰=含气段                  │
│     ↓                                               │
│  ② 规划: 生成传统代码的阈值指令                         │
│     "GR<50 in 1200-1260m, 1550-1620m"               │
│     ↓                                               │
│  [代码执行: GR阈值扫描 → 精确边界(±0.1m)]               │
│     ↓                                               │
│  ③ 验证回环: 接收精确定位结果                           │
│     - 1247.6-1248.1m: 仅0.5m间隙 → 噪声，合并          │
│     - 1600.3-1600.8m: 薄泥岩夹层？→ 保留              │
│     ↓                                               │
│  ⑤ 收敛: 合并/调整层段 → 输出最终岩性+流体分层          │
└─────────────────────────────────────────────────────┘
```

### Agent 3: WellSeismicFusionAgent — 井震多模态融合

```
SEG-Y + LAS + Checkshot → 井旁地震道 + GR曲线 → 并排对比图
    │
    ▼
Qwen3-VL-8B 看图像，输出:
    │
    ├─▶ 标定指令:
    │     时深控制点: [(800ms,950m), (1000ms,1220m), ...]
    │     关键界面: Top_Reservoir=1230m/1005ms
    │   → 标定代码建立时深转换函数
    │
    └─▶ analysis: 标定质量评价、井震关联分析
```

### Agent 4: ProspectEvaluationAgent — 目标综合评价

```
前序Agent输出JSON → 文本摘要
    │
    ▼
Qwen3-VL-8B 看文本，输出:
    │
    ├─▶ 决策指令:
    │     T1背斜: trap=2, reservoir=2, seal=2, charge=2 → Pg=65%
    │     → drill_ready, 建议在inline 350/crossline 180部署探井
    │     T2断块: trap=4, seal=4 → Pg=45%
    │     → inventory, 需补充高精度地震
    │
    └─▶ analysis: 勘探成功率预估、推荐钻探顺序
```

---

## 目录结构

```
oil-gas-llm/
├── pipeline/                             # 生产逻辑（可 import 的 Python 包）
│   ├── __init__.py                       # 公共 API: Pipeline, VLMClient, AgentResult...
│   ├── __main__.py                       # CLI: python -m pipeline ...
│   ├── vlm.py                            # VLMClient: 加载 Qwen3-VL + schema 校验重试
│   ├── prompts.py                        # workflow_planning + verification + 4 agent prompt
│   ├── agents.py                         # LoopAgent（闭环）+ SingleShotAgent + AgentResult
│   ├── orchestrator.py                   # Pipeline: run_from_adapter (赛题主) + run_all + run_volume
│   ├── adapter.py                        # RunPackage: 载入 geo_adapter 的 run 目录
│   ├── tasks.py                          # 地质任务注册表 (fault/horizon/facies/fracture) + CLASS_ALIASES
│   ├── exporter.py                       # AgentResult → PNG / JSON / 属性 SEG-Y
│   ├── downstream/                       # 下游模型注册表
│   │   ├── base.py                       # DownstreamModel Protocol + register/get
│   │   ├── yolo_world.py                 # 真实 YOLO-World + mock fallback
│   │   ├── sam.py                        # SAM (mock，可替换为 SAM-2)
│   │   └── traditional_code.py           # 规则引擎，兼容多种 VLM 字段名
│   ├── io/                               # SEG-Y I/O + 几何 + 渲染
│   │   ├── segy.py                       # read_segy, write_attribute_segy, extract_*
│   │   ├── geometry.py                   # SliceGeometry + pixel↔data 双向映射
│   │   └── render.py                     # 数组切片 → PIL + geometry
│   └── *.md                              # 迭代文档
├── prompts/                              # Agent Prompt模板
├── schemas/output_schemas.py             # 6 套 JSON Schema + validate_output
├── weights/yolov8s-world.pt              # YOLO-World checkpoint（首次下载）
├── test/
│   ├── data.py                           # 合成数据 fixture
│   ├── test_pipeline_unit.py             # pipeline 内部单测（19 个）
│   ├── test_io_unit.py                   # I/O + geometry + exporter 单测（12 个）
│   ├── test_adapter_unit.py              # geo_adapter 对接单测（14 个）
│   ├── test_loop.py                      # 端到端集成测试
│   └── test_live.py / test_accuracy.py / test_batch.py / test_two_stage.py  # legacy
└── README.md
```

---

## 快速开始

### 环境

```bash
conda activate qwen35grpo
pip install matplotlib Pillow scipy segyio jsonschema ultralytics
```

### 模型权重

```bash
# Qwen3-VL 8B（主 VLM）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir /data/models/qwen3-vl-8b

# YOLO-World checkpoint
mkdir -p weights && curl -L -o weights/yolov8s-world.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-world.pt
```

可选环境变量:
- `QWEN_VL_PATH`：Qwen3-VL 权重路径
- `YOLO_WORLD_PATH`：YOLO checkpoint 路径（默认 `weights/yolov8s-world.pt`）
- `USE_MOCK_DOWNSTREAM=1`：强制所有下游模型走 mock（不加载 YOLO/SAM）

---

## 赛题工作流（推荐入口）

### 步骤 1：用 geo_adapter 处理原始数据

前置模块地址：`/home/newdisk/yxjiang/new_8T/oil-gas-llm/多模态接口/`

它把 `.sgy/.segy/.npz` 地震 + `.las/.csv` 测井（+ 可选井位/轨迹/时深）转成标准 run 目录 `runs/<sample_id>/`。目录里已经准备好：
- 归一化后的切片 PNG（`assets/seismic/{inline,crossline,slice,local_patch}_model.png`）
- 测井曲线综合图（`assets/well_logs/well_log_panel.png`）
- 定制好的 system/user prompt
- 元数据 `manifest.json`（含 shape、CRS、view.source_indices、pixel_to_physical、alignment 等）
- VLM 输出契约 `schemas/expected_model_output.schema.json`

```bash
cd /home/newdisk/yxjiang/new_8T/oil-gas-llm/多模态接口
pip install -e '.[all]'    # 首次
geo-adapter prepare --config examples/sample_config.yaml
```

### 步骤 2：本 pipeline 消费 run 目录

```bash
CUDA_VISIBLE_DEVICES=1 python -m pipeline \
    --run-dir /home/newdisk/yxjiang/new_8T/oil-gas-llm/多模态接口/runs/demo_sample_001 \
    --output-dir out/
```

pipeline 内部完成:
1. 载入 run 包（prompt / 图像 / manifest / expected_schema）
2. 首次 VLM 分析 → 输出符合 `expected_model_output.schema.json` 的 JSON（含 `downstream_plan`）
3. YOLO-World 用 `downstream_plan.class_prompts` + `regions_of_interest[bbox_xyxy_norm]` 在指定图像上跑真实检测
4. **默认**：把 YOLO 结果回喂 VLM 做一次验证回环（`--no-verify` 关闭）
5. 用 `manifest.seismic.views[*].source_indices` 反推 3D 坐标，把归一化 bbox 聚合成属性体
6. 每个 target_class 输出一个 `.sgy` 属性体 + 每张图一份标注 PNG

输出目录：

```
out/
├── report.json                       # 汇总（包）
├── vlm_output.json                   # 首轮 VLM 输出（含 seismic/well_log/cross_modal_analysis, downstream_plan, uncertainty）
├── vlm_output_refined.json           # 验证回环后的最终输出
├── fault_plane_attribute.sgy         # 每个类别一个 3D 属性 SEG-Y
├── channel_attribute.sgy
├── reservoir_candidate_attribute.sgy
├── fault_seismic_inline.png          # 每张有检测的图一份标注 PNG
├── fault_seismic_crossline.png
└── ...
```

### 步骤 3：编程接口

```python
from pipeline import Pipeline, load_run

pkg = load_run("runs/demo_sample_001")
print(pkg.to_summary())      # {sample_id, task_type, target_classes, n_images, ...}

p = Pipeline()
report = p.run_from_adapter(
    run_dir="runs/demo_sample_001",
    out_dir="out/",
    verify=True,              # 默认开启 VLM 二次验证
    yolo_conf=0.25,           # VLM 未指定时的置信度阈值
)
```

### 支持的地质任务类别

`target_classes` 从 `manifest.task.target_classes` 读，由 geo_adapter 的 `input_config.yaml` 决定。本 pipeline 内置这些别名映射（`pipeline/tasks.py`）：

| target_class | canonical | 内置地质描述 |
|---|---|---|
| `fault` / `断层` | fault | 同相轴垂直/倾斜错断、反射终止、断面波 |
| `horizon` / `层位` | horizon | 横向连续强反射轴，地层界面/不整合面 |
| `facies` / `沉积相` / `channel` / `reservoir_candidate` | facies | 反射构型（平行/S 形前积/杂乱/丘状/透镜/河道充填） |
| `fracture` / `裂缝` | fracture | 高密度不连续反射带、相干性异常 |

不在这张表里的类别，pipeline 会透传原始类别名给 VLM 和 YOLO（无中文提示）。要新增内置类别，在 `pipeline/tasks.py::TASKS` 里加一个 `GeologicalTask`，在 `CLASS_ALIASES` 里加别名。

---

## 其它入口（非赛题主流程）

### Fallback：不经 geo_adapter 直读 SEG-Y

如果 geo_adapter 前置没跑，可以让本 pipeline 自己读 SEG-Y 并切片。功能弱于 geo_adapter 流程（没有多视图、没有标定信息），仅用于调试。

```bash
python -m pipeline --input path/to/volume.sgy \
    --tasks fault,horizon,facies,fracture \
    --slice-axis inline --slice-stride 5 \
    --output-dir out/
```

### 4-Agent 语义解释流水线

跟赛题无关的独立能力：分别对地震剖面、测井曲线、井震融合图、勘探目标做**语义分析**，出 4 份 JSON 报告（无 SEG-Y 输出）。

```bash
python -m pipeline --agent all \
    --seismic-image section.png --log-image log.png \
    --output out.json
```

编程接口：`Pipeline().run_all(...)`。

---

## 测试

```bash
# 秒级单测（无需 VLM，共 45 个）
python test/test_pipeline_unit.py        # 19: schema/registry/JSON 解析
python test/test_io_unit.py              # 12: SEG-Y 读写/几何/exporter
python test/test_adapter_unit.py         # 14: adapter/tasks/aggregate

# 端到端集成
CUDA_VISIBLE_DEVICES=1 python test/test_loop.py       # 合成数据（fallback 路径）
CUDA_VISIBLE_DEVICES=1 python -m pipeline --run-dir /path/to/runs/demo_sample_001 \
    --output-dir /tmp/adapter_smoke                    # 赛题主流程

# 老评测脚本（未迁移，legacy）
CUDA_VISIBLE_DEVICES=1 python test/test_accuracy.py
```

## VLM 推理参数

Planning + Verification 阶段用 `temperature=0`（决定性输出）。自定义 agent 时可通过 `VLMClient.call_json(temperature=T)` 覆盖。

---

## 评测结果

### 模型对比

| 指标 | Qwen3.5-9B (thinking) | Qwen3-VL-8B |
|------|----------------------|-------------|
| Think浪费 | 57-81% | **0%** |
| 单次推理 | 141-164s | **10-17s** |
| JSON完整率 | 30% | **100%** |

### 两阶段 vs 纯VLM（LogAnalysisAgent，5次批量）

| 指标 | 纯VLM | 两阶段（VLM+代码） |
|------|-------|-------------------|
| Sand召回率 | 50% | **100% ± 0%** |
| Fluid准确率 | 33-50% | **100% ± 0%** |
| 深度精度 | ±50-200m | **±0.1m** |
| 批量稳定性 | 不稳定 | **CV=0%** |

### SeismicInterpAgent（真实模拟数据）

| 目标 | Ground Truth | VLM检测 | 偏差 |
|------|-------------|---------|------|
| 亮点 | CDP 180-210, 1200-1215ms | CDP 170, 1300ms | CDP~25, 时间~85ms |
| 河道 | CDP 50-80, 1500-1540ms | CDP 60, 1600ms | 接近 |
| 断层 | CDP 120, 200 | 未检测 | VLM视觉识别瓶颈 |

> 注：本地未找到SEG-Y真实数据（比赛数据需从云端下载），以上用 Ricker 子波褶积生成的逼真模拟数据测试。
> 断层检测建议在下游用 YOLO-World 在 VLM 指定的范围内做精确检测，弥补 VLM 的空间精度不足。

---

### Agent 5（隐式）: 验证回环

每个 Agent 的 VLM 在下游返回结果后，都会被再次调用进行验证：

```
下游结果 + 原始图像 → VLM 验证 prompt:

"YOLO-World 在 CDP 78-85, time 980-1020ms 处检测到置信度0.45的断层候选。
 请重新查看原始地震剖面中该位置的图像：
 1. 确认这是真实断层还是假阳性（如河道边缘、处理噪声）
 2. 如果真实，评估断距和置信度
 3. 如果虚假，说明原因并建议修改检测参数
 4. 检查是否有遗漏的断层（YOLO未检测到但剖面中可见）"
```

这个验证回环是 VLM 发挥最大价值的地方——它不需要做精确的空间定位（下游已经做了），
但它的地质知识可以判断检测结果的**真实性**和**地质意义**。

---

## 设计原则

1. **VLM = 大脑，下游 = 手**：VLM 规划+验证，下游精确执行，结果必须回环
2. **VLM 回答 "What + Where 大概"**：类别名 + 描述 + 搜索范围 → 下游模型精确执行
3. **VLM 做地质合理性判断**：下游返回候选后，VLM 逐条验证真伪（这是 VLM 的核心价值）
4. **Few-shot prompt**：给出一个完整 JSON 示例，格式准确率远高于纯文字描述
5. **数值精度交给代码**：VLM 不输出精确数字，只输出阈值规则和大致区间
6. **分流到最合适的下游模型**：检测→YOLO、分割→SAM、数值→代码、决策→规则引擎
7. **迭代至收敛**：VLM 验证→拒绝假阳性→追加新检测→重新验证，直到结果稳定
