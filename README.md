# Oil-Gas VLM — 油气地球物理多模态Agent工作流

基于本地部署 VLM (Qwen3-VL-8B) 串联下游专家模型（YOLO-World / SAM / 传统岩石物理代码），
实现地震图像与测井曲线的多模态特征融合与有利目标识别。

## 技术路线

### 核心思路：VLM 做"任务规划"，下游模型做"精确执行"

VLMs 擅长理解图像语义（"这是什么？大概在哪？"），但空间精度差（像素偏移大、数值读取不准）。
YOLO-World / SAM / 传统算法正好相反——空间精度高但缺乏语义理解。

**把两者级联：VLM 看图像 → 生成下游可执行的检测指令 → 下游专家模型精确执行。**

```
                     ┌──────────────┐
                     │  Qwen3-VL-8B  │  理解场景，生成下游指令
                     └──────┬───────┘
                            │  downstream_prompts
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  YOLO-World  │  │     SAM      │  │  传统代码     │
   │  目标检测     │  │  像素级分割   │  │  数值计算     │
   │  (bbox)      │  │  (mask)      │  │  (±0.1m)     │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                 │                  │
          └─────────────────┼──────────────────┘
                            ▼
                     ┌──────────────┐
                     │  结果融合     │  综合评估 + 目标排序
                     └──────────────┘
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
Qwen3-VL-8B 看图像，输出:
    │
    ├─▶ YOLO-World prompts:
    │     "detect fault plane in CDP 60-100, time 800-1600ms"
    │     "detect bright spot in CDP 160-220, time 1100-1400ms"
    │     "detect channel in CDP 40-90, time 1400-1700ms"
    │   → YOLO-World 返回精确 bbox
    │
    ├─▶ SAM prompts:
    │     {type: "point", label: "horizon_H1", point: [100, 500]}
    │     {type: "bbox", label: "bright_spot_region", bbox: [...]}
    │   → SAM 返回精确 mask
    │
    └─▶ analysis: 构造背景、层位描述、异常体解释
```

### Agent 2: LogAnalysisAgent — 测井曲线分析

```
LAS文件 → 6道综合图(GR/SP/RT/AC/DEN/CNL) → PNG图像
    │
    ▼
Qwen3-VL-8B 看图像，输出:
    │
    ├─▶ 传统代码 thresholds:
    │     GR < 50 → 砂岩段 (区间1200-1260m, 1550-1620m)
    │     RT > 20 + DEN < 2.35 + CNL < 0.20 → 含气段 (区间1550-1600m)
    │   → 代码在numpy数据上精确定位 (±0.1m)
    │
    └─▶ analysis: 岩性总结、沉积相解释
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
oil-gas-vlm/
├── prompts/                              # Agent Prompt模板 + 设计文档
│   ├── README.md                         # Prompt设计文档（架构详解）
│   ├── seismic_interp_agent.md           # → YOLO-World + SAM
│   ├── log_analysis_agent.md             # → 传统代码阈值
│   ├── well_seismic_fusion_agent.md      # → 标定代码
│   └── prospect_evaluation_agent.md      # → 决策模型
├── schemas/
│   └── output_schemas.py                 # JSON Schema + 校验函数
├── pipeline/
│   ├── data_processing.md                # 数据处理Pipeline详解
│   ├── iteration_notes.md                # Prompt迭代记录
│   └── accuracy_report.md                # 准确性评测报告
├── test/
│   ├── test_live.py                      # 单次功能测试
│   ├── test_accuracy.py                  # 准确性评测（ground truth对比）
│   ├── test_batch.py                     # 批量统计测试 (N=5)
│   └── test_two_stage.py                 # 两阶段策略验证
├── .gitignore
└── README.md
```

---

## 快速开始

### 环境

```bash
conda activate qwen35grpo
pip install matplotlib Pillow scipy segyio
```

### 部署模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir /data/models/qwen3-vl-8b
```

### 测试

```bash
# 单个Agent功能测试
CUDA_VISIBLE_DEVICES=1 python test/test_live.py

# 准确性评测（含ground truth对比）
CUDA_VISIBLE_DEVICES=1 python test/test_accuracy.py

# 批量统计测试
CUDA_VISIBLE_DEVICES=1 python test/test_batch.py

# 两阶段策略验证（VLM粗分 + 代码精确）
CUDA_VISIBLE_DEVICES=1 python test/test_two_stage.py
```

### VLM 推理参数

```python
Qwen3VLForConditionalGeneration.from_pretrained(model_path, ...)
model.generate(
    max_new_tokens=4096,
    do_sample=True,
    temperature=0.3,
    repetition_penalty=1.1,
    top_p=0.95,
)
```

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

## 设计原则

1. **VLM 回答 "What + Where 大概"**：类别名 + 描述 + 搜索范围 → 下游模型精确执行
2. **Few-shot prompt**：给出一个完整 JSON 示例，格式准确率远高于纯文字描述
3. **数值精度交给代码**：VLM 不输出精确数字，只输出阈值规则和大致区间
4. **分流到最合适的下游模型**：检测→YOLO、分割→SAM、数值→代码、决策→规则引擎
