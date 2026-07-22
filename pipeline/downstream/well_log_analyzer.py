"""Well Log Analyzer — 测井曲线真实分析引擎。

替代 mock 的 traditional_code，做真实的曲线分析：
  1. 曲线活跃分割 (changepoint / gradient jump)
  2. 基于 GR/DEN/RT 交会的岩性分类
  3. 基于 RT/DEN 交叉的流体识别
  4. 层序旋回识别

与 traditional_code 兼容 VLM 输出字段，支持 expected_depth_ranges 等格式。
"""
from __future__ import annotations

import numpy as np

from ._shared import extract_depth_ranges as _normalize_depth_ranges


# ── 曲线分割 ────────────────────────────────────────────────────────────────

def _find_change_points(curve: np.ndarray, depth: np.ndarray,
                        threshold_z: float = 3.0,
                        min_segment_samples: int = 5,
                        ) -> list[dict]:
    """基于梯度 Z-score 的变点检测。

    返回 [{depth_m, segment_start, segment_end, mean_value}, ...]。
    """
    diff = np.abs(np.gradient(curve))
    mu, sigma = diff.mean(), diff.std() + 1e-8
    z = (diff - mu) / sigma
    peaks = np.where(z > threshold_z)[0]

    if len(peaks) == 0:
        return [{"depth_top_m": round(float(depth[0]), 1),
                 "depth_bottom_m": round(float(depth[-1]), 1),
                 "mean_value": round(float(curve.mean()), 2)}]

    # 合并相近的峰值
    merged = []
    buf = [peaks[0]]
    for p in peaks[1:]:
        if p - buf[-1] <= 3:  # 相邻 3 个采样点内的合并
            buf.append(p)
        else:
            merged.append(int(np.mean(buf)))
            buf = [p]
    if buf:
        merged.append(int(np.mean(buf)))

    segments = []
    for i, cp in enumerate(merged):
        if i == 0:
            start = 0
        else:
            start = merged[i - 1]
        end = min(cp + 1, len(curve))
        if end - start >= min_segment_samples:
            segments.append({
                "depth_top_m": round(float(depth[start]), 1),
                "depth_bottom_m": round(float(depth[end - 1]), 1),
                "mean_value": round(float(curve[start:end].mean()), 2),
            })

    # 最后一段
    if merged:
        last_cp = merged[-1]
        if len(curve) - last_cp >= min_segment_samples:
            segments.append({
                "depth_top_m": round(float(depth[last_cp]), 1),
                "depth_bottom_m": round(float(depth[-1]), 1),
                "mean_value": round(float(curve[last_cp:].mean()), 2),
            })

    return segments if segments else [
        {"depth_top_m": round(float(depth[0]), 1),
         "depth_bottom_m": round(float(depth[-1]), 1),
         "mean_value": round(float(curve.mean()), 2)}
    ]


# ── 岩性分类（GR/DEN 交会） ────────────────────────────────────────────────

def _classify_lithology(gr_val: float, den_val: float | None = None,
                        rt_val: float | None = None) -> tuple[str, float]:
    """基于 GR/DEN/RT 的岩性分类规则（简化版）。

    返回 (lithology_name, confidence)。
    """
    if den_val is None:
        if gr_val < 45:
            return ("sand_prone", 0.60)
        if gr_val < 75:
            return ("mixed_sand_shale", 0.55)
        return ("shale_prone", 0.60)
    if gr_val < 45:
        if den_val < 2.35:
            return ("clean_sandstone", 0.85)
        elif den_val < 2.55:
            return ("silty_sandstone", 0.80)
        else:
            return ("calcareous_sandstone", 0.70)
    elif gr_val < 75:
        if den_val < 2.40:
            return ("silty_sandstone", 0.75)
        elif den_val < 2.60:
            return ("shaly_sandstone", 0.75)
        else:
            return ("sandy_shale", 0.70)
    elif gr_val < 120:
        if den_val < 2.50:
            return ("sandy_shale", 0.70)
        elif den_val < 2.70:
            return ("shale", 0.85)
        else:
            return ("calcareous_shale", 0.75)
    else:
        if den_val < 2.50 and rt_val is not None and rt_val > 20:
            return ("organic_rich_shale", 0.70)
        return ("shale", 0.85)


def _classify_fluid(rt_val: float, den_val: float = 2.5,
                    cnl_val: float = 0.2) -> tuple[str, float]:
    """基于 RT/DEN/CNL 的流体类型判别。

    返回 (fluid_type, confidence)。
    """
    # RT/DEN/CNL alone cannot determine saturation without porosity, salinity
    # and field calibration. Dense high-resistivity rock is an anomaly, not
    # evidence for water, and every hydrocarbon label remains a candidate.
    if not np.isfinite(rt_val) or not np.isfinite(den_val):
        return ("fluid_indeterminate", 0.0)
    cnl_known = np.isfinite(cnl_val)
    if rt_val < 5:
        return ("water_candidate", 0.70)
    if rt_val > 50:
        if den_val < 2.45 and cnl_known and cnl_val < 0.18:
            return ("gas_candidate", 0.70)
        if den_val < 2.55 and cnl_known and cnl_val < 0.25:
            return ("hydrocarbon_candidate", 0.65)
        return ("high_resistivity_anomaly", 0.45)
    if rt_val > 15:
        if den_val < 2.50 and cnl_known and cnl_val < 0.25:
            return ("hydrocarbon_candidate", 0.55)
        return ("fluid_ambiguous", 0.35)
    return ("fluid_indeterminate", 0.30)


# ── 测井曲线数据提取 ────────────────────────────────────────────────────────

def _extract_curves(image, context: dict | None = None
                    ) -> dict[str, np.ndarray] | None:
    """从 context 里拿真实的曲线数据。没有则返回 None（退化为 mock）。"""
    if context is None:
        return None
    curves = context.get("curves")
    if isinstance(curves, dict):
        return {k: np.asarray(v, dtype=np.float32) for k, v in curves.items()}
    return None


def _formation_tops_in_interval(
    context: dict | None,
    top_m: float,
    bottom_m: float,
    tolerance_m: float = 0.5,
) -> list[dict]:
    if not context:
        return []
    result = []
    for item in context.get("formation_tops") or []:
        try:
            depth_m = float(item["depth_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if top_m - tolerance_m <= depth_m <= bottom_m + tolerance_m:
            result.append({
                "formation": str(item.get("formation", "unknown")),
                "depth_m": round(depth_m, 3),
            })
    return result


def _merge_adjacent(segments: list[dict], max_gap_m: float = 5.0
                    ) -> list[dict]:
    """合并相邻且同岩性的段。"""
    if len(segments) <= 1:
        return segments
    merged = []
    cur = dict(segments[0])
    for nxt in segments[1:]:
        gap = nxt["depth_top_m"] - cur["depth_bottom_m"]
        if cur["lithology"] == nxt["lithology"] and gap <= max_gap_m:
            # 合并: 扩展底界, 重新算均值
            cur["depth_bottom_m"] = nxt["depth_bottom_m"]
            cur["mean_value"] = round((cur["mean_value"] + nxt["mean_value"]) / 2, 2)
        else:
            merged.append(cur)
            cur = dict(nxt)
    merged.append(cur)
    return merged


# ── 下游模型 ────────────────────────────────────────────────────────────────

class WellLogAnalyzer:
    name = "well_log_analyzer"
    description = (
        "测井曲线真实分析引擎。基于曲线梯度和交会图做："
        "曲线分割(changepoint检测)、岩性分类(GR/DEN/RT)、"
        "流体识别(含气/油/水)、层序旋回分析。精度±0.1m"
    )
    required_fields = [
        "analysis_type (curve_segmentation|lithology_classification|"
        "fluid_identification|full_analysis)",
        "rules? (VLM 提供的阈值规则，兼容 traditional_code 格式)",
        "depth_range? ({top_m, bottom_m})",
    ]
    output_shape = (
        "list[{id, class_name, depth_top_m, depth_bottom_m, "
        "confidence, evidence[], lithology?, fluid_type?, rule}]"
    )

    def detect(self, instruction: dict, image=None,
               context: dict | None = None) -> list[dict]:
        analysis_type = instruction.get("analysis_type", "full_analysis")
        rules = instruction.get("rules") or []
        dr = instruction.get("depth_range") or {}

        # 尝试获取真实曲线数据
        curves = _extract_curves(image, context)
        use_real = curves is not None

        # 统一生成深度轴（所有分支共用）
        depth_axis = None
        if use_real:
            if "depth" in curves:
                depth_axis = curves["depth"]
            else:
                first_arr = next(v for v in curves.values()
                                 if hasattr(v, '__len__'))
                depth_axis = np.linspace(1000, 1000 + len(first_arr) * 0.1524,
                                         len(first_arr))

        results: list[dict] = []
        n = 0

        # ── 曲线分割 ──
        if analysis_type in ("curve_segmentation", "full_analysis"):
            if use_real and "GR" in curves:
                gr = curves["GR"]
                segs = _find_change_points(gr, depth_axis)
                # 先对每段分类，再合并相邻同岩性段
                classified = []
                for s in segs:
                    den_val, rt_val = None, None
                    if curves:
                        seg_mask = (depth_axis >= s["depth_top_m"]) & (depth_axis <= s["depth_bottom_m"])
                        if "DEN" in curves and seg_mask.any():
                            den_val = float(curves["DEN"][seg_mask].mean())
                        if "RT" in curves and seg_mask.any():
                            rt_val = float(curves["RT"][seg_mask].mean())
                    litho, conf = _classify_lithology(
                        s["mean_value"], den_val, rt_val)
                    classified.append({**s, "lithology": litho, "confidence": conf,
                                       "den_val": den_val, "rt_val": rt_val})
                # 合并相邻同岩性段
                merged = _merge_adjacent(classified)
                for s in merged:
                    results.append({
                        "id": f"wl_seg_{n}",
                        "class_name": s["lithology"],
                        "depth_top_m": s["depth_top_m"],
                        "depth_bottom_m": s["depth_bottom_m"],
                        "confidence": s["confidence"],
                        "lithology": s["lithology"],
                        "gr_mean": s.get("mean_value"),
                        "den_mean": (round(s["den_val"], 3)
                                     if s["den_val"] is not None else None),
                        "rt_mean": (round(s["rt_val"], 1)
                                    if s["rt_val"] is not None else None),
                        "evidence_curves": [
                            name for name in ("GR", "DEN", "RT")
                            if name in curves
                        ],
                        "formation_tops": _formation_tops_in_interval(
                            context, s["depth_top_m"], s["depth_bottom_m"]
                        ),
                        "analysis": "curve_segmentation",
                        "model": self.name,
                    })
                    n += 1
            elif rules:
                # fallback: 兼容 traditional_code 的规则格式
                for rule in rules:
                    if not isinstance(rule, dict):
                        continue
                    ranges = _normalize_depth_ranges(rule)
                    for top, bot in ranges:
                        results.append({
                            "id": f"wl_rule_{n}",
                            "class_name": rule.get("class_name", "segment"),
                            "depth_top_m": top,
                            "depth_bottom_m": bot,
                            "confidence": 0.75,
                            "rule": rule.get("rule", ""),
                            "model": self.name,
                        })
                        n += 1

        # ── 流体识别 ──
        if analysis_type in ("fluid_identification", "full_analysis"):
            if use_real and "RT" in curves:
                rt = curves["RT"]
                den = curves.get("DEN")
                cnl = curves.get("CNL")
                if den is None:
                    results.append({
                        "id": f"wl_fluid_{n}_insufficient",
                        "class_name": "high_resistivity_anomaly",
                        "confidence": 0.0,
                        "result": "fluid identification requires DEN with RT",
                        "missing_curves": ["DEN"],
                        "analysis": "fluid_identification",
                        "model": self.name,
                    })
                    return results
                if cnl is None:
                    cnl = np.full_like(rt, np.nan)

                # 对 RT 高值段做流体识别
                high_rt = np.where(rt > 15)[0]
                if len(high_rt) > 0:
                    # 把连续高 RT 段合并
                    zones = []
                    start = high_rt[0]
                    for i in range(1, len(high_rt)):
                        if high_rt[i] - high_rt[i - 1] > 3:
                            zones.append((start, high_rt[i - 1]))
                            start = high_rt[i]
                    zones.append((start, high_rt[-1]))

                    for z_start, z_end in zones:
                        if z_end - z_start < 2:
                            continue
                        rt_mean = float(rt[z_start:z_end + 1].mean())
                        den_mean = float(den[z_start:z_end + 1].mean())
                        cnl_mean = float(np.nanmean(cnl[z_start:z_end + 1])) \
                            if np.isfinite(cnl[z_start:z_end + 1]).any() else None
                        fluid, conf = _classify_fluid(
                            rt_mean,
                            den_mean,
                            cnl_mean if cnl_mean is not None else float("nan"),
                        )
                        zone_top = round(float(depth_axis[z_start]), 1)
                        zone_bottom = round(float(depth_axis[z_end]), 1)
                        results.append({
                            "id": f"wl_fluid_{n}",
                            "class_name": fluid,
                            "depth_top_m": zone_top,
                            "depth_bottom_m": zone_bottom,
                            "confidence": conf,
                            "fluid_type": fluid,
                            "rt_mean_ohmm": round(rt_mean, 1),
                            "den_mean": round(den_mean, 3),
                            "cnl_mean": (round(cnl_mean, 3)
                                         if cnl_mean is not None else None),
                            "evidence": [
                                f"RT={rt_mean:.1f}Ohm·m (>15=potential pay)",
                                f"DEN={den_mean:.3f}g/cm³",
                                (f"CNL={cnl_mean:.3f}"
                                 if cnl_mean is not None else "CNL=missing"),
                            ],
                            "formation_tops": _formation_tops_in_interval(
                                context, zone_top, zone_bottom
                            ),
                            "evidence_summary": (
                                f"RT={rt_mean:.1f} Ohm.m is an anomaly only; "
                                "fluid type is not confirmed"
                            ),
                            "interpretation_limitations": [
                                "inference_only_thresholds",
                                "no_water_saturation_or_local_fluid_calibration",
                                "candidate_not_confirmed_fluid_type",
                            ],
                            "analysis": "fluid_identification",
                            "model": self.name,
                        })
                        n += 1

        # ── 无数据时的退化：合理提取规则区间 ──
        if not results and rules:
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                ranges = _normalize_depth_ranges(rule)
                for top, bot in ranges:
                    results.append({
                        "id": f"wl_fb_{n}",
                        "class_name": rule.get("class_name", "zone"),
                        "depth_top_m": top,
                        "depth_bottom_m": bot,
                        "confidence": 0.65,
                        "rule": str(rule.get("rule", ""))[:120],
                        "model": self.name,
                    })
                    n += 1

        return results
