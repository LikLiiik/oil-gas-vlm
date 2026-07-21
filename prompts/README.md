# Agent Prompt 设计文档

## 概述

4个Agent共享同一套VLM调用模式：`System Prompt（角色+领域知识+输出格式） + User Message（图像+任务指令）`。

所有Agent使用 Qwen3-VL-8B，推理参数：
```python
max_new_tokens=4096      # 足够大部分JSON输出
do_sample=True            # 避免重复循环
temperature=0.3           # 低温度保证一致性
repetition_penalty=1.1    # 防重复
top_p=0.95
```

## Agent 1: SeismicInterpAgent — 地震剖面解释

### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| 图像 | PNG (base64) | 地震剖面图，建议用`cmap='seismic'`色标，dpi≥120 |
| 坐标 | 隐含 | CDP (横轴) 和 Two-Way Time ms (纵轴) 从图像坐标轴读取 |

### System Prompt 设计要点

```
角色: 地球物理专家
色标说明: 红色/暖色=波峰(正振幅)，蓝色/冷色=波谷(负振幅)
识别目标: 断层(同相轴错断) + 亮点(强负振幅异常) + 层位(连续条带)
输出约束: 仅JSON，格式参考示例
```

### 输出 JSON Schema

```json
{
  "faults": [{
    "id": "F1",
    "type": "normal|reverse|strike-slip",
    "positions": [[x1, y1], [x2, y2]],
    "throw_ms": 10,
    "confidence": 0.9,
    "evidence": "同相轴错断约10ms"
  }],
  "anomalies": [{
    "id": "A1",
    "type": "bright_spot|flat_spot|gas_chimney",
    "position": [x, y],
    "depth_ms": 1200,
    "confidence": 0.9,
    "description": "强负振幅异常，疑似含气亮点"
  }],
  "horizons": [{
    "id": "H1",
    "name": "H1",
    "depth_range_ms": [1000, 1100],
    "amplitude": "strong|medium|weak",
    "confidence": 0.85
  }],
  "summary": "综合描述"
}
```

### 数据处理 Pipeline

```
SEG-Y 3D体 → segyio读取 → 提取inline/crossline切片
→ matplotlib绘制（seismic色标, dpi=150）
→ PIL Image → base64 → VLM
→ 解析JSON → 像素坐标→实际inline/xl/time坐标转换
```

---

## Agent 2: LogAnalysisAgent — 测井曲线分析

### 两阶段架构

```
Stage 1 (VLM): 看测井曲线图 → 识别岩性类型和大致区间
Stage 2 (代码): GR<50阈值 → 精确定位砂层边界(±0.1m)
              RT>20+DEN<2.35 → 精确判断流体类型
```

### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| 图像 | PNG (base64) | 6道测井综合图 (GR/SP/RT/AC/DEN/CNL)，dpi≥120 |
| 深度轴 | 隐含 | 纵轴为深度(m)，从坐标轴读取 |

### System Prompt 设计要点

```
角色: 测井解释专家
识别规则: GR<50=砂岩, GR>75=泥岩, RT>20+低DEN+低CNL=含气层
输出约束: 仅JSON。深度值不需要精确（精确提取由后续代码完成）
```

### 输出 JSON Schema (Stage 1 VLM)

```json
{
  "zones": [{
    "depth_approx": "1200-1260",
    "lithology": "sandstone|shale|limestone",
    "fluid": "gas|oil|water|null"
  }],
  "summary": "识别出3套砂岩..."
}
```

### Stage 2 代码精确定位

```python
# GR阈值提取砂岩段
sand_intervals = []
for i in range(1, len(gr)):
    if gr[i-1] >= 50 and gr[i] < 50:
        sand_start = depth[i]
    elif gr[i-1] < 50 and gr[i] >= 50:
        sand_intervals.append((sand_start, depth[i]))  # ±0.1m精度

# RT+DEN判断流体
avg_rt = rt[mask].mean()
avg_den = den[mask].mean()
if avg_rt > 20 and avg_den < 2.35:
    fluid = "gas"
elif avg_rt < 5:
    fluid = "water"
```

### 数据处理 Pipeline

```
LAS文件 → lasio读取 → 提取GR/SP/RT/AC/DEN/CNL曲线
→ matplotlib 6道图 (dpi=120) → base64 → VLM (Stage 1)
→ VLM输出zones → GR阈值精确定位 (Stage 2)
→ RT+DEN阈值判断流体 (Stage 2)
→ 合并输出精确结果
```

---

## Agent 3: WellSeismicFusionAgent — 井震多模态融合

### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| 图像 | PNG (base64) | 左侧井旁地震道 + 右侧测井曲线(GR) 并排图 |
| 文本 | string | 井名 + 时深对照表 |

### System Prompt 设计要点

```
角色: 多模态地球物理融合专家
任务: 井震标定分析、时深关系建立、地质界面对应
输出约束: 仅JSON
```

### 输出 JSON Schema

```json
{
  "well_seismic_calibration": {
    "well_name": "Well-A",
    "correlation_coefficient": 0.85,
    "time_shift_ms": 5.0,
    "calibration_quality": "good"
  },
  "time_depth_table": [
    {"time_ms": 800, "depth_m": 950, "velocity_ms": 2800}
  ],
  "key_geological_interfaces": [{
    "id": "I1",
    "name": "Top_Reservoir",
    "depth_m": 1230,
    "time_ms": 1005,
    "seismic_character": "强波峰",
    "log_character": "GR突变"
  }],
  "fusion_summary": "融合分析总结"
}
```

### 数据处理 Pipeline

```
SEG-Y + LAS + Checkshot → 井震标定（时深转换）
→ 提取井旁地震道 + GR曲线
→ 并排绘制对比图 (dpi=100) → base64 → VLM
→ 输出标定结果 + 界面列表
```

---

## Agent 4: ProspectEvaluationAgent — 目标综合评价

### 输入

| 参数 | 类型 | 说明 |
|------|------|------|
| 文本 | string | 前3个Agent输出的JSON摘要 |

### System Prompt 设计要点

```
角色: 勘探决策专家
风险评分: 1=极低 2=低 3=中 4=高 5=极高
决策分类: drill_ready (建议钻探) / data_gap (需补充资料) / inventory (列入储备) / drop (放弃)
输出约束: 仅JSON
```

### 输出 JSON Schema

```json
{
  "targets": [{
    "id": "P1",
    "name": "T1背斜",
    "priority_rank": 1,
    "risk_assessment": {
      "trap_risk": 2,
      "reservoir_risk": 2,
      "seal_risk": 2,
      "charge_risk": 2,
      "geological_success_probability_pct": 65
    },
    "decision": {
      "category": "drill_ready",
      "rationale": "构造落实，储层好，已证实含气",
      "next_steps": ["部署探井"]
    }
  }],
  "risk_summary": {
    "total_prospects": 3,
    "drill_ready": 1,
    "data_gap": 1,
    "inventory": 1
  },
  "overall_summary": "综合评价总结"
}
```

### 数据处理 Pipeline

```
前序JSON → 提取关键信息（断层数/层位数/砂层数/含气段数/井震相关系数）
→ 构造结构化文本摘要 → VLM (纯文本，无图像)
→ 输出目标列表 + 风险评估
```

---

## 通用设计原则

1. **System Prompt 用Few-shot** — 给出1个完整JSON示例，模型输出格式更准确
2. **User Message 要短** — "仅输出JSON。" 比长篇指令更有效
3. **颜色/坐标明确说明** — VLM需要知道色标含义（红=正振幅，蓝=负振幅）
4. **数值精度交给代码** — VLM不需要输出精确深度，Stage 2代码完成
5. **图像质量影响大** — dpi≥120, 坐标轴清晰, 色标合理
