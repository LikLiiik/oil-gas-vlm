"""传统代码引擎 — 真实阈值扫描+规则执行。

当 VLM 指定了明确的阈值规则（如 GR<50, RT>20），且有原始曲线数据时，
在真实数据上精确执行。无数据时退化为基于 expected_depth_ranges 的合理输出。
"""
from __future__ import annotations

import numpy as np


# ── 规则解析 ────────────────────────────────────────────────────────────────

def _evaluate_rule(curve_value: float, rule: str) -> bool:
    """解析并执行一个简单规则字符串。支持格式: GR<50, RT>20, DEN<2.35, 50<GR<120 等。"""
    import re
    rule = rule.strip()
    # 双边界: "50 < GR < 120"
    m = re.match(r'([\d.]+)\s*<\s*(\w+)\s*<\s*([\d.]+)', rule)
    if m:
        lo, var, hi = float(m.group(1)), m.group(2), float(m.group(3))
        return lo < curve_value < hi
    # 小于: "GR < 50" 或 "GR<50"
    m = re.match(r'(\w+)\s*<\s*([\d.]+)', rule)
    if m:
        return curve_value < float(m.group(2))
    # 大于: "RT > 20" 或 "RT>20"
    m = re.match(r'(\w+)\s*>\s*([\d.]+)', rule)
    if m:
        return curve_value > float(m.group(2))
    # 小于等于 / 大于等于
    m = re.match(r'(\w+)\s*<=\s*([\d.]+)', rule)
    if m:
        return curve_value <= float(m.group(2))
    m = re.match(r'(\w+)\s*>=\s*([\d.]+)', rule)
    if m:
        return curve_value >= float(m.group(2))
    return False


def _parse_threshold(rule: str) -> tuple[str, float, str] | None:
    """从规则字符串提取 (变量名, 阈值, 运算符)。"""
    import re
    rule = rule.strip()
    for pat, op in [(r'(\w+)\s*<\s*([\d.]+)', '<'),
                     (r'(\w+)\s*>\s*([\d.]+)', '>'),
                     (r'(\w+)\s*<=\s*([\d.]+)', '<='),
                     (r'(\w+)\s*>=\s*([\d.]+)', '>=')]:
        m = re.match(pat, rule)
        if m:
            return m.group(1), float(m.group(2)), op
    return None


# ── 真实: 在曲线数据上执行 ──────────────────────────────────────────────────

def _apply_rule_on_curve(curve: np.ndarray, depth: np.ndarray,
                         threshold: float, op: str,
                         depth_top: float | None = None,
                         depth_bottom: float | None = None,
                         min_segment_len: int = 3,
                         ) -> list[dict]:
    """在真实测井曲线上应用阈值规则，提取连续区间。"""
    # 确定有效深度范围
    if depth_top is not None and depth_bottom is not None:
        mask = (depth >= depth_top) & (depth <= depth_bottom)
    else:
        mask = np.ones(len(curve), dtype=bool)

    if op == '<' or op == '<=':
        hit = curve <= threshold if op == '<=' else curve < threshold
    else:
        hit = curve >= threshold if op == '>=' else curve > threshold

    hit = hit & mask
    if not hit.any():
        return []

    # 提取连续命中段
    segments = []
    in_seg = False
    seg_start = 0
    for i in range(len(hit)):
        if hit[i] and not in_seg:
            seg_start = i
            in_seg = True
        elif not hit[i] and in_seg:
            if i - seg_start >= min_segment_len:
                segments.append((seg_start, i - 1))
            in_seg = False
    if in_seg and len(hit) - seg_start >= min_segment_len:
        segments.append((seg_start, len(hit) - 1))

    return [
        {"depth_top_m": round(float(depth[s]), 1),
         "depth_bottom_m": round(float(depth[e]), 1),
         "thickness_m": round(float(depth[e] - depth[s]), 1),
         "mean_value": round(float(curve[s:e + 1].mean()), 2)}
        for s, e in segments
    ]


# ── 深度区间解析 (兼容多种 VLM 输出格式) ────────────────────────────────────

def _to_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


from ._shared import extract_depth_ranges as _extract_ranges


# ── 下游模型 ────────────────────────────────────────────────────────────────

class TraditionalCode:
    name = "traditional_code"
    description = (
        "阈值规则执行引擎。VLM指定阈值规则(如GR<50, RT>20)，"
        "在有原始曲线数据时精确扫描，无数据时基于预期深度区间输出。"
        "适合需要显式数值阈值的测井分析场景"
    )
    required_fields = [
        "rules[].class_name", "rules[].rule",
        "rules[].expected_depth_ranges",
    ]
    output_shape = (
        "list[{id, class_name, depth_top_m, depth_bottom_m, "
        "thickness_m?, confidence, rule}]"
    )

    def detect(self, instruction, image=None, context=None):
        # 归一化输入格式
        if "rules" in instruction:
            rules = instruction["rules"]
        elif "rule" in instruction:
            rules = [instruction]
        else:
            return [{
                "id": f"code_custom_{np.random.randint(1000)}",
                "class_name": instruction.get("class_name", "custom"),
                "result": "no rules specified",
                "model": self.name,
            }]

        # 尝试获取真实曲线数据
        curves = None
        depth = None
        if context and "curves" in context:
            curves = {k: np.asarray(v, dtype=np.float32)
                      for k, v in context["curves"].items()}
            if "depth" in curves:
                depth = curves["depth"]
        if depth is None:
            # 无深度轴时，生成默认轴 (0.1524m 采样 = 0.5ft)
            first_key = next(iter(curves)) if curves else None
            n = len(curves[first_key]) if first_key and curves else 2000
            depth = np.linspace(1000, 1000 + n * 0.1524, n)

        out = []
        for rule in rules:
            if isinstance(rule, str):
                out.append({
                    "id": f"code_str_{np.random.randint(1000)}",
                    "class_name": rule, "model": self.name,
                })
                continue

            class_name = rule.get("class_name", "segment")
            rule_str = rule.get("rule", "")
            parsed = _parse_threshold(rule_str) if rule_str else None
            ranges = _extract_ranges(rule)

            # ── 有真实曲线 + 可解析规则 → 精确执行 ──
            if parsed and curves and parsed[0] in curves:
                var_name, threshold, op = parsed
                curve = curves[var_name]

                if ranges:
                    for top, bot in ranges:
                        segs = _apply_rule_on_curve(
                            curve, depth, threshold, op, top, bot)
                        for s in segs:
                            out.append({
                                "id": f"code_{class_name}_{np.random.randint(1000)}",
                                "class_name": class_name,
                                "depth_top_m": s["depth_top_m"],
                                "depth_bottom_m": s["depth_bottom_m"],
                                "thickness_m": s.get("thickness_m"),
                                "confidence": 0.95,
                                "rule": rule_str,
                                "evidence": (
                                    f"{var_name}{op}{threshold} "
                                    f"→ mean={s['mean_value']}"
                                ),
                                "model": self.name,
                            })
                else:
                    segs = _apply_rule_on_curve(curve, depth, threshold, op)
                    for s in segs:
                        out.append({
                            "id": f"code_{class_name}_{np.random.randint(1000)}",
                            "class_name": class_name,
                            "depth_top_m": s["depth_top_m"],
                            "depth_bottom_m": s["depth_bottom_m"],
                            "thickness_m": s.get("thickness_m"),
                            "confidence": 0.90,
                            "rule": rule_str,
                            "evidence": (
                                f"{var_name}{op}{threshold} 全井段扫描"
                            ),
                            "model": self.name,
                        })

            # ── 无曲线数据但有深度区间 → 合理输出 ──
            elif ranges:
                for top, bot in ranges:
                    out.append({
                        "id": f"code_{class_name}_{np.random.randint(1000)}",
                        "class_name": class_name,
                        "depth_top_m": round(top, 1),
                        "depth_bottom_m": round(bot, 1),
                        "thickness_m": round(bot - top, 1),
                        "confidence": 0.70,
                        "rule": rule_str,
                        "evidence": "based on VLM expected_depth_ranges",
                        "model": self.name,
                    })

            # ── 既无曲线也无区间 ──
            else:
                out.append({
                    "id": f"code_{class_name}_{np.random.randint(1000)}",
                    "class_name": class_name,
                    "rule": rule_str,
                    "result": "no curve data or depth range provided",
                    "model": self.name,
                })

        return out
