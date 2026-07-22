# Verification Prompt — 下游结果回环验证

每个 Agent 在下游模型返回结果后，VLM 被再次调用进行地质合理性验证。

## 通用 Verification Prompt 模板

```
你是地球物理专家。下游模型（YOLO-World/SAM/代码）已完成初次检测，现在需要你验证结果。

原始图像: [附上原始地震剖面/测井曲线图]
原始分析任务: {planning阶段的context}
下游检测结果: {YOLO bbox列表 / SAM mask / 代码数值}

请逐一验证每条检测结果:
1. 是否符合地质规律？（断层=同相轴错断？亮点=强振幅异常？砂岩=低GR高RT？）
2. 是否可能是假阳性？（处理噪声？河道边缘？泥岩薄互层？井壁垮塌？）
3. 是否有遗漏？（图像中可见但下游没检测到的目标）
4. 如有问题，给出修正建议（合并区间/调整阈值/追加检测类别）

输出JSON:
{
  "verified": [{
    "id": "原始检测ID",
    "is_real": true/false,
    "adjusted_confidence": 0.9,
    "geological_reason": "支撑或否定这条检测的地质依据",
    "rejection_reason": "如果虚假，说明原因",
    "correction": "如果真实但边界需调整，给出修正值"
  }],
  "false_positives_removed": 2,
  "missed_targets": [{
    "class_name": "额外需要检测的类别",
    "search_region": "CDP 110-130 / depth 1200-1300m",
    "reason": "剖面中可见但下游遗漏"
  }],
  "interval_adjustments": [{
    "original_range": [1200.1, 1255.1],
    "adjusted_range": [1200.1, 1255.1],
    "reason": "合并0.5m噪音间隙"
  }],
  "refined_prompts": {
    "yolo_world": {"categories": [...], "confidence_threshold": 0.25},
    "traditional_code": {"curves": {...}}
  },
  "convergence_status": "converged|need_another_round"
}
```

## Seismic 验证专用 Prompt

```
原始地震剖面: [图像]
我的初次分析: {analysis中的summary}

YOLO-World在以下位置检测到目标:
- fault plane: bbox1(CDP 78-85, 980-1020ms, conf=0.45)
- fault plane: bbox2(CDP 195-202, 1500-1530ms, conf=0.32)
- bright spot: bbox3(CDP 160-200, 1180-1320ms, conf=0.88)

SAM分割了以下层位:
- horizon_H1: mask覆盖CDP 1-300, time 480-520ms

请重新查看原始剖面：
1. bbox1区域：同相轴是否有垂直错断？断距多大？是否伴随牵引构造？
2. bbox2区域：该处反射特征是什么？是断层还是河道边缘/沉积尖灭？
3. bbox3区域：是否确实是局部强负振幅？下方有无低频阴影？
4. horizon_H1: SAM的分割是否准确跟随了该反射层位？
5. 有没有遗漏？CDP 110-130附近是否有微小错断？
```

## Log 验证专用 Prompt

```
原始测井曲线: [图像]
我的初次分析: {analysis中的lithology_summary}

传统代码根据GR<50阈值提取了以下砂岩段:
- low_GR_sandstone: 1200.1-1247.6m (厚47.5m)
- low_GR_sandstone: 1248.1-1255.1m (厚7.0m)
- low_GR_sandstone: 1400.2-1425.2m (厚25.0m)
- ...

请重新查看原始图像:
1. 1247.6-1248.1m仅0.5m的间隔：这是真实的泥岩夹层还是GR曲线的短暂随机波动？应该合并为单一砂层1200.1-1255.1m吗？
2. 1425.2m处的GR回升：是真实的岩性界面还是薄层泥岩？对应的RT/AC/DEN有变化吗？
3. RT>20的高阻段(1555.3-1595.3m)：是否确实对应低GR+低DEN？含气证据充分吗？
4. 1750.4-1799.9m段的RT仅3 Ω·m：这个"水层"判断正确吗？SP的负异常是否持续？
5. 有没有遗漏的薄砂层（如GR短暂下探到50以下的指状尖峰）？
```

## 设计要点

1. **验证必须有原始图像** — VLM 需要重看图来验证，不能只看下游返回的数字
2. **明确地质判断标准** — 断层=错断+牵引，含气=高RT+低DEN+低CNL，不是单一指标
3. **允许修正** — 不仅是真/假，还能建议合并区间、调整边界、改变置信度
4. **补充遗漏** — VLM 的全局视野可以补下游的漏（下游模型可能miss小目标）
5. **明确收敛条件** — `convergence_status` 告诉上层是否需要继续迭代
