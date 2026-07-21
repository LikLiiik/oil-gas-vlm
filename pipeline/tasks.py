"""地质任务注册表：断层 / 层位 / 沉积相 / 裂缝。

每个任务定义:
- yolo_classes: 建议给 VLM/YOLO 的类别名
- description: 地质特征描述（用于 VLM prompt 提示）
- downstream_hint: 首选下游模型
- recommended_model: VLM 规划时的推荐模型（可被 VLM 根据实际情况调整）
- overlay_color: 可视化叠加颜色
- attribute_default: 属性体默认值（未检测区域）
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeologicalTask:
    name: str
    yolo_classes: list[str]
    description: str
    downstream_hint: str            # sam | traditional_code | seismic_domain_model | ...
    recommended_model: str          # 在 workflow_planning 时的首选模型
    overlay_color: str              # matplotlib color name / hex
    attribute_default: float = 0.0  # 属性体未检测区域的取值

    def prompt_hint(self) -> str:
        cls = ", ".join(self.yolo_classes)
        return (f"[任务: {self.name}] {self.description} "
                f"目标类别: {cls}。推荐模型: {self.recommended_model}。"
                f"备用: {self.downstream_hint}。")


TASKS: dict[str, GeologicalTask] = {
    "fault": GeologicalTask(
        name="fault",
        yolo_classes=["fault plane", "fault trace", "normal fault", "reverse fault"],
        description=(
            "断层：同相轴垂直/倾斜错断、反射终止、断面波、"
            "两侧同相轴倾角突变、断层拖曳构造。"
        ),
        downstream_hint="seismic_domain_model",
        recommended_model="cig_fault",
        overlay_color="#ff2d55",
    ),
    "horizon": GeologicalTask(
        name="horizon",
        yolo_classes=["strong continuous reflection", "seismic horizon"],
        description=(
            "层位：横向连续的强反射轴，代表地层界面/不整合面/最大湖泛面。"
            "适合用 horizon_tracker 互相关自动追踪。"
        ),
        downstream_hint="sam",
        recommended_model="horizon_tracker",
        overlay_color="#00d1b2",
    ),
    "facies": GeologicalTask(
        name="facies",
        yolo_classes=[
            "parallel reflection facies", "sigmoid reflection",
            "chaotic reflection", "mounded reflection",
            "channel fill", "prograding clinoform",
        ],
        description=(
            "沉积相：由反射构型定义。平行/亚平行=陆棚稳定沉积；"
            "S 形前积=三角洲；杂乱/丘状=生物礁或碎屑流；"
            "透镜/丘状+侧向尖灭=河道充填。先提取 envelope+frequency+sweetness 属性，"
            "再用 facies_classifier 做多属性聚类分相。"
        ),
        downstream_hint="attribute_extractor",
        recommended_model="facies_classifier",
        overlay_color="#ffcc00",
    ),
    "fracture": GeologicalTask(
        name="fracture",
        yolo_classes=[
            "fracture zone", "fracture swarm", "high-density fracture",
        ],
        description=(
            "裂缝：高密度不连续反射带、相干性异常、局部振幅衰减、"
            "断层伴生的破碎带。用 seismic_domain_model 计算相干/方差属性检测，"
            "宽度小于分辨率的用属性体标记。"
        ),
        downstream_hint="traditional_code",
        recommended_model="seismic_domain_model",
        overlay_color="#a463f2",
    ),
}


def get(name: str) -> GeologicalTask:
    if name not in TASKS:
        raise KeyError(f"unknown task '{name}'. 可用: {list(TASKS)}")
    return TASKS[name]


def get_optional(name: str) -> GeologicalTask | None:
    return TASKS.get(name)


def all_names() -> list[str]:
    return list(TASKS)


def tasks_prompt_hint(names: list[str]) -> str:
    """把多个任务打包成一段 prompt 前缀，让 VLM 有针对性规划。"""
    lines = ["本次识别任务列表:"]
    for i, n in enumerate(names, 1):
        lines.append(f"  {i}. {get(n).prompt_hint()}")
    lines.append("请针对以上每项任务在 workflow_steps 中至少给出一步。"
                 "优先使用各任务推荐的模型。")
    return "\n".join(lines)


# 别名：geo_adapter 的 target_classes 里可能出现的名字 → 内置 TASKS 里的 canonical 名
CLASS_ALIASES = {
    # 内置
    "fault": "fault", "horizon": "horizon", "facies": "facies", "fracture": "fracture",
    # geo_adapter demo 用到 / 常见同义
    "channel": "facies",                  # 河道 → 沉积相
    "reservoir_candidate": "facies",
    "断层": "fault", "层位": "horizon",
    "沉积相": "facies", "裂缝": "fracture",
}


def hint_for_target_classes(target_classes: list[str]) -> str:
    """给 geo_adapter 的 target_classes 生成一段中文地质描述提示。
    识别不到别名就退化为原样透传。"""
    lines = ["目标类别地质描述（补充给 VLM 参考）:"]
    for cls in target_classes:
        alias = CLASS_ALIASES.get(cls)
        if alias and alias in TASKS:
            t = TASKS[alias]
            lines.append(
                f"  - {cls} (canonical={alias}): {t.description} "
                f"推荐下游模型: {t.recommended_model}"
            )
        else:
            lines.append(f"  - {cls}: (无内置描述，请自行判断特征)")
    return "\n".join(lines)
