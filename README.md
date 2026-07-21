# Oil-Gas VLM — 油气地球物理多模态Agent工作流

基于本地部署 VLM (Qwen3-VL-8B) 串联下游专家模型（YOLO-World / SAM / 传统岩石物理代码），
实现地震图像与测井曲线的多模态特征融合与有利目标识别。

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
