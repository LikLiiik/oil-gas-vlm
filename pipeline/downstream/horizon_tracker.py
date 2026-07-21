"""Horizon Tracker — 基于互相关的层位自动追踪。

VLM 在图像上标记种子点（cdp / time_ms），tracker 沿层位横向追踪，
输出连续层位点序列。

支持 tracking_mode:
  - peak:   追踪局部波峰（正相位极大值）
  - trough: 追踪局部波谷（负相位极小值）
  - zero_crossing_pos2neg: 正→负零交叉
  - zero_crossing_neg2pos: 负→正零交叉
  - correlation: 模板互相关追踪（最鲁棒）
"""
from __future__ import annotations

import numpy as np

from ._shared import image_to_array as _get_array


def _pick_peak(arr_col: np.ndarray, mode: str,
               window_half: int = 8) -> int:
    """在 1D 数组（某一列）上按 mode 定位目标样本索引。"""
    n = len(arr_col)
    if n < 2:
        return 0
    if mode == "peak":
        # 取窗口内的 argmax
        return int(np.argmax(arr_col[:min(window_half * 3, n)]))
    elif mode == "trough":
        return int(np.argmin(arr_col[:min(window_half * 3, n)]))
    elif "zero_crossing" in mode:
        # 从中间位置开始找最近零交叉
        mid = n // 2
        for i in range(mid, n - 1):
            if mode.endswith("pos2neg") and arr_col[i] >= 0 > arr_col[i + 1]:
                return i
            if mode.endswith("neg2pos") and arr_col[i] <= 0 < arr_col[i + 1]:
                return i
        return mid
    else:  # correlation / 默认取峰
        return int(np.argmax(np.abs(arr_col[:min(window_half * 3, n)])))


def _trace_correlation(arr: np.ndarray, start_trace: int,
                        start_sample: int,
                        window_half: int = 25,
                        search_half: int = 15,
                        max_traces: int | None = None,
                        min_confidence: float = 0.4,
                        ) -> list[dict]:
    """从种子点出发，逐道用互相关追踪层位。

    使用 numpy.correlate 一次性计算全部时移的互相关，
    鲁棒性比逐点滑动模板更好，不受窄波峰影响。
    """
    ns, nt = arr.shape
    max_t = nt if max_traces is None else max_traces

    # 提取模板：种子点附近的局部波形
    t0 = max(0, start_sample - window_half)
    t1 = min(ns, start_sample + window_half)
    template = arr[t0:t1, start_trace].copy()
    tpl_len = len(template)
    if template.std() < 1e-8:
        return [{"trace_idx": start_trace, "sample_idx": start_sample,
                 "confidence": 0.0}]

    # 归一化模板
    template = (template - template.mean()) / (template.std() + 1e-8)

    points: list[dict] = []

    def _track_one_way(step: int):
        nonlocal start_sample
        prev_sample = start_sample
        tr = start_trace + step
        while 0 <= tr < nt and len(points) < max_t * 2:
            # 提取当前道比模板更长的区段
            s_lo = max(0, prev_sample - search_half - window_half)
            s_hi = min(ns, prev_sample + search_half + window_half)
            if s_hi - s_lo < tpl_len + 4:
                break
            trace_col = arr[s_lo:s_hi, tr].astype(np.float64).copy()
            if trace_col.std() < 1e-8:
                tr += step
                continue

            # 互相关 → 找最佳时移
            trace_col = (trace_col - trace_col.mean()) / (trace_col.std() + 1e-8)
            xcorr = np.correlate(trace_col, template, mode="valid")
            best_offset = int(np.argmax(xcorr))
            best_corr = min(float(xcorr[best_offset]) / tpl_len, 1.0)

            # offset → 绝对 sample 位置
            # xcorr[0] 对应 template 左端在 trace_col[0]
            best_sample = s_lo + best_offset + tpl_len // 2

            if best_corr < min_confidence:
                break

            # clamp 到有效范围
            best_sample = max(0, min(ns - 1, best_sample))

            points.append({
                "trace_idx": tr,
                "sample_idx": int(best_sample),
                "confidence": round(float(best_corr), 3),
            })
            prev_sample = best_sample
            tr += step

    # 先向左，再向右
    _track_one_way(-1)
    points.append({"trace_idx": start_trace, "sample_idx": start_sample,
                   "confidence": 1.0})
    _track_one_way(1)

    points.sort(key=lambda p: p["trace_idx"])
    return points


# ── 下游模型 ────────────────────────────────────────────────────────────────

class HorizonTracker:
    name = "horizon_tracker"
    description = (
        "层位自动追踪。从VLM标记的种子点沿地震反射同相轴横向追踪，"
        "输出连续层位点序列。适合层位解释和构造图制作"
    )
    required_fields = [
        "seed_points: [{trace_idx, sample_idx|time_ms}, ...]",
        "tracking_mode: peak|trough|zero_crossing|correlation",
        "search_window_samples? (搜索窗口半径，默认10)",
    ]
    output_shape = (
        "list[{id, horizon_name, points:[{trace_idx, sample_idx, time_ms, confidence}]"
        ", continuity_score, average_confidence}]"
    )

    def detect(self, instruction: dict, image=None,
               context: dict | None = None) -> list[dict]:
        arr = _get_array(image, context)
        if arr is None:
            return []

        seeds = instruction.get("seed_points") or []
        if not seeds:
            return []

        mode = instruction.get("tracking_mode", "correlation")
        search_half = int(instruction.get("search_window_samples", 10))
        horizon_name = instruction.get("horizon_name", "H")

        ns, nt = arr.shape
        results: list[dict] = []
        for si, seed in enumerate(seeds):
            # 推导 trace_idx + sample_idx
            tr = int(seed.get("trace_idx", seed.get("cdp", 0)))
            if "time_ms" in seed and context:
                # 用 context 里的时间轴信息换算样本索引
                t_axis = context.get("time_axis_ms")
                if t_axis is not None:
                    t_axis = np.asarray(t_axis)
                    trg_t = float(seed["time_ms"])
                    trg_s = int(np.argmin(np.abs(t_axis - trg_t)))
                else:
                    trg_s = int(seed.get("sample_idx",
                               ns // 2))
            elif "sample_idx" in seed:
                trg_s = int(seed["sample_idx"])
            else:
                trg_s = ns // 2

            tr = max(0, min(nt - 1, tr))
            trg_s = max(0, min(ns - 1, trg_s))

            if mode in ("peak", "trough") or "zero_crossing" in mode:
                col = arr[:, tr]
                s_pick = _pick_peak(col[max(0, trg_s - 4):trg_s + 5], mode)
                trg_s = max(0, trg_s - 4) + s_pick

            points = _trace_correlation(
                arr, tr, trg_s,
                search_half=search_half,
                max_traces=nt,
            )
            if not points:
                continue

            confs = [p["confidence"] for p in points]
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            gaps = [
                abs(points[i]["trace_idx"] - points[i - 1]["trace_idx"])
                for i in range(1, len(points))
            ]
            continuity = 1.0 / (1.0 + sum(g > 1 for g in gaps) / max(1, len(gaps)))

            results.append({
                "id": f"htrack_{horizon_name}_{si}",
                "horizon_name": f"{horizon_name}_{si}",
                "seed_trace_idx": tr,
                "seed_sample_idx": trg_s,
                "tracking_mode": mode,
                "points": points,
                "continuity_score": round(continuity, 3),
                "average_confidence": round(avg_conf, 3),
                "n_points": len(points),
                "model": self.name,
            })

        return results
