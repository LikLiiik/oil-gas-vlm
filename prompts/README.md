# Agent Prompt 设计文档 — 下游模型级联架构

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: VLM 分析理解（Qwen3-VL-8B）                            │
│  → 看图像 → 理解场景 → 决定需要检测什么                           │
│  → 输出: 下游模型可执行的检测指令 (YOLO-World / SAM / ...)        │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: 下游专家模型执行                                        │
│  → YOLO-World: 开放词汇目标检测 (fault, bright_spot, channel)     │
│  → SAM/Grounded-SAM: 分割 (horizon, sand_body, salt_dome)        │
│  → 传统算法: GR阈值 / RT阈值 (±0.1m精度)                          │
├─────────────────────────────────────────────────────────────────┤
│  Stage 3: VLM 综合评估                                           │
│  → 收集下游结果 → 综合分析 → 生成勘探建议                          │
└─────────────────────────────────────────────────────────────────┘
```

## VLM 输出的下游 Prompt 格式

每个 Agent 的 VLM 输出包含两部分：
1. **`downstream_prompts`**: 传给下游模型的检测指令
2. **`analysis`**: VLM 自身的理解和后续 Agent 需要的信息

---

## Agent 1: SeismicInterpAgent

### VLM → YOLO-World 的 Prompt

```json
{
  "downstream_prompts": {
    "yolo_world": {
      "task": "open_vocabulary_detection",
      "categories": [
        {
          "class_name": "fault plane",
          "description": "同相轴垂直错断、反射终止、断面波的位置",
          "expected_cdp_range": [60, 100],
          "expected_time_range_ms": [800, 1600],
          "confidence_threshold": 0.3,
          "max_detections": 5
        },
        {
          "class_name": "bright spot anomaly",
          "description": "局部强蓝/冷色区域（强负振幅），椭圆形，下方可能有低频阴影",
          "expected_cdp_range": [160, 220],
          "expected_time_range_ms": [1100, 1400],
          "confidence_threshold": 0.5,
          "max_detections": 3
        },
        {
          "class_name": "channel",  
          "description": "透镜状、丘状反射构型，内部反射杂乱或空白",
          "expected_cdp_range": [40, 90],
          "expected_time_range_ms": [1400, 1700],
          "confidence_threshold": 0.3,
          "max_detections": 3
        },
        {
          "class_name": "anticline structure",
          "description": "向上凸起的反射层，两侧倾角对称或不对称",
          "expected_time_range_ms": [500, 2000],
          "confidence_threshold": 0.4,
          "max_detections": 2
        }
      ],
      "image_preprocessing": {
        "color_map": "seismic",
        "normalize": "clip_98_percentile"
      }
    },
    "sam": {
      "task": "segment_anything",
      "prompts": [
        {
          "type": "point",
          "label": "horizon_H1_top",
          "point": [100, 500],
          "description": "第一个强连续反射波峰"
        },
        {
          "type": "point", 
          "label": "horizon_H2_top",
          "point": [100, 750],
          "description": "第二个强连续反射波峰"
        },
        {
          "type": "bbox",
          "label": "bright_spot_region",
          "bbox": [160, 220, 1100, 1400],
          "description": "亮点异常区域"
        }
      ]
    }
  },
  "analysis": {
    "summary": "该剖面显示褶皱地层，CDP 60-100间存在疑似断层（同相轴垂直错断），CDP 160-220/1100-1400ms处存在疑似含气亮点（强负振幅异常），CDP 40-90/1400-1700ms处存在透镜状河道",
    "structural_context": "整体为背斜构造，NW-SE走向",
    "key_observations": [
      "层位连续性中等，存在3处可能的不连续面",
      "下部振幅较弱，可能为泥岩段",
      "上部平行层理发育，为稳定陆棚沉积"
    ]
  }
}
```

### 为什么这样设计

| 下游模型 | VLM给什么 | 下游做什么 |
|----------|----------|-----------|
| **YOLO-World** | `class_name` + `description` + `expected_range` | 在图像中精确检测目标的bbox |
| **SAM** | `point` 或 `bbox` prompt | 精确分割层位/异常体边界 |
| VLM的作用 | 提供语义理解：这是什么？大概在哪？ | — |
| 下游的作用 | 提供空间精度：精确的xy坐标和像素级分割 | — |

---

## Agent 2: LogAnalysisAgent

### VLM → 下游模型的 Prompt

```json
{
  "downstream_prompts": {
    "segmentation_model": {
      "task": "curve_segmentation",
      "curves": {
        "GR": {
          "class_name": "low gamma ray sandstone",
          "description": "GR<50 API的连续低值段，代表砂岩或碳酸盐岩",
          "expected_depth_ranges": [
            {"top_m": 1150, "bottom_m": 1300},
            {"top_m": 1350, "bottom_m": 1480},
            {"top_m": 1520, "bottom_m": 1650}
          ],
          "threshold": 50,
          "mode": "below_threshold"
        },
        "RT": {
          "class_name": "high resistivity pay zone",
          "description": "RT>20 Ohm.m的高阻异常段，代表含油气层",
          "expected_depth_ranges": [
            {"top_m": 1520, "bottom_m": 1620}
          ],
          "threshold": 20,
          "mode": "above_threshold"
        }
      },
      "fluid_indicators": {
        "gas_zone": {
          "conditions": "RT>20 AND DEN<2.35 AND CNL<0.20",
          "expected_depth_range": {"top_m": 1520, "bottom_m": 1620}
        },
        "water_zone": {
          "conditions": "RT<5 AND GR<50",
          "expected_depth_range": {"top_m": 1700, "bottom_m": 1820}
        }
      }
    },
    "traditional_code": {
      "sand_detection": "GR < 50 → 精确砂体顶底 (±0.1m)",
      "fluid_detection": "RT>20 + DEN<2.35 → 含气段, RT<5 → 水层",
      "porosity_calculation": "DEN+CNL交会法或AC计算公式"
    }
  },
  "analysis": {
    "lithology_summary": "4套砂层: L1(1200-1255m细砂岩), L2(1400-1435m粉砂岩), L3(1550-1625m中砂岩/含气), L4(1750-1805m粗砂岩/含水)",
    "fluid_summary": "L3层段高RT(平均85Ω·m)、低DEN(2.28)、低CNL(0.16)，含气特征明显",
    "sedimentary_context": "整体为正旋回序列，L1-L2为三角洲前缘，L3-L4为河道砂"
  }
}
```

---

## Agent 3: WellSeismicFusionAgent

### VLM → 下游模型的 Prompt

```json
{
  "downstream_prompts": {
    "time_depth_registration": {
      "task": "well_seismic_tie",
      "control_points": [
        {"time_ms": 800, "depth_m": 950},
        {"time_ms": 1000, "depth_m": 1220}, 
        {"time_ms": 1200, "depth_m": 1530},
        {"time_ms": 1500, "depth_m": 2000}
      ],
      "key_interfaces": [
        {
          "name": "Top_Reservoir",
          "time_ms": 1005,
          "depth_m": 1230,
          "seismic_polarity": "positive",
          "log_marker": "GR突变增大，DEN减小"
        },
        {
          "name": "Base_Reservoir", 
          "time_ms": 1050,
          "depth_m": 1290,
          "seismic_polarity": "negative",
          "log_marker": "GR突变减小，RT降低"
        }
      ]
    },
    "cross_well_correlation": {
      "task": "stratigraphic_correlation",
      "wells": ["Well-A", "Well-B"],
      "marker_beds": [
        {"name": "MFS1", "depth_well_a": 1260, "depth_well_b": 1280},
        {"name": "Top_Sand_L3", "depth_well_a": 1550, "depth_well_b": 1570}
      ]
    }
  },
  "analysis": {
    "calibration_quality": "good (r=0.85)",
    "fusion_summary": "井震标定可靠，含气砂岩对应地震强波谷（亮点），横向可追踪约2km"
  }
}
```

---

## Agent 4: ProspectEvaluationAgent

### VLM → 下游决策模型的 Prompt

```json
{
  "downstream_prompts": {
    "risk_assessment_model": {
      "task": "prospect_ranking",
      "prospects": [
        {
          "id": "P1",
          "name": "T1背斜-亮点复合圈闭",
          "priority": 1,
          "risk_scores": {
            "trap": {"score": 2, "rationale": "背斜形态清楚，闭合幅度50ms"},
            "reservoir": {"score": 2, "rationale": "测井证实砂岩厚25m，por=18.5%"},
            "seal": {"score": 2, "rationale": "上覆厚层泥岩，侧向断层封堵"},
            "charge": {"score": 2, "rationale": "近源，断层作为运移通道"}
          },
          "expected_value": {"Pg_pct": 65, "recoverable_mmboe": 7.5},
          "drill_ready": true
        }
      ]
    }
  },
  "analysis": {
    "overall_summary": "T1具备钻探条件(Pg=65%)，建议优先部署探井",
    "recommended_actions": [
      "在T1高点部署探井 (inline 350, crossline 180)",
      "T2需补做高精度地震落实断层封堵性"
    ]
  }
}
```

---

## 设计原则

### 1. VLM 负责"What"和"Where大概"，下游负责"Where精确"

```
VLM:  "CDP 60-100之间有断层，YOLO去精确找" 
      → YOLO-World prompt: "detect fault plane in CDP 60-100"

VLM:  "1550-1620m有砂岩，代码去提取精确边界"
      → GR threshold: 找GR<50的精确深度 (±0.1m)
```

### 2. 下游 prompt 要具体、可执行

| 好 | 不好 |
|----|------|
| `"fault plane with vertical offset in CDP 60-100"` | `"look for faults"` |
| `"GR<50 sandstone intervals"` | `"find reservoir"` |
| `bbox: [160, 220, 1100, 1400]` in pixel coords | `"around the middle"` |

### 3. 包含置信度和阈值

每个检测类别附带：
- `confidence_threshold`: 下游模型的最低置信度
- `max_detections`: 防止过度检测
- `expected_range`: 缩小搜索范围，减少误报

### 4. 分流到最合适的下游模型

| 任务类型 | 合适的下游模型 | VLM提供的prompt |
|----------|---------------|----------------|
| 目标检测 (断层/亮点) | YOLO-World / DINOv2 | 类别名 + 描述 + 范围 |
| 分割 (层位/盐体) | SAM / Grounded-SAM | point/bbox prompts |
| 数值回归 (深度/孔隙度) | 传统算法/代码 | 阈值 + 计算公式 |
| 分类 (岩性/流体) | 轻量CNN / 代码 | 判别规则 + 特征描述 |
