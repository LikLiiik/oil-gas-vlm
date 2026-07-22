"""Experimental prebuilt well-log classifier (not registered by default).

The inference pipeline never trains this model. It can only load an existing
compatible artifact; normal runs use deterministic log analysis and rules.

输入: GR, RT, DEN, CNL, SP 曲线段统计值
输出: 岩性分类 + 流体类型 + 置信度
"""
from __future__ import annotations

import os
import pickle

import numpy as np

CACHE_DIR = os.path.expanduser("~/.cache/oil_gas_models")
CACHE_PATH = os.path.join(CACHE_DIR, "well_log_lithology_model.pkl")


def _generate_training_data() -> tuple[np.ndarray, np.ndarray]:
    """基于岩石物理知识生成合成训练数据。

    特征: [GR_mean, GR_std, RT_mean, DEN_mean, CNL_mean,
           depth_gradient, thickness_m, rt_den_ratio]
    标签: 0=shale, 1=silty_sand, 2=clean_sand, 3=carbonate, 4=coal, 5=organic_shale
    """
    rng = np.random.default_rng(42)
    n_per_class = 500
    X_list, y_list = [], []

    # 0: Shale — 高GR, 中等DEN, 低RT
    gr = rng.normal(105, 15, n_per_class)
    rt = np.clip(rng.normal(4, 2, n_per_class), 0.1, None)
    den = rng.normal(2.58, 0.08, n_per_class)
    cnl = rng.normal(0.28, 0.06, n_per_class)
    thick = rng.exponential(30, n_per_class)
    for i in range(n_per_class):
        X_list.append([
            gr[i], abs(rng.normal(2, 1)), rt[i], den[i], cnl[i],
            abs(rng.normal(2, 1.5)), thick[i],
            rt[i] / max(den[i], 1e-8),
        ])
        y_list.append(0)

    # 1: Silty sandstone — 中等GR, 中等DEN, 中等RT
    gr = rng.normal(60, 12, n_per_class)
    rt = np.clip(rng.normal(8, 4, n_per_class), 0.1, None)
    den = rng.normal(2.45, 0.06, n_per_class)
    cnl = rng.normal(0.20, 0.05, n_per_class)
    thick = rng.exponential(15, n_per_class)
    for i in range(n_per_class):
        X_list.append([
            gr[i], abs(rng.normal(5, 2)), rt[i], den[i], cnl[i],
            abs(rng.normal(2, 1.5)), thick[i],
            rt[i] / max(den[i], 1e-8),
        ])
        y_list.append(1)

    # 2: Clean sandstone — 低GR, 低DEN, 高RT
    gr = rng.normal(35, 8, n_per_class)
    rt = np.clip(rng.normal(40, 20, n_per_class), 0.1, None)
    den = rng.normal(2.30, 0.05, n_per_class)
    cnl = rng.normal(0.15, 0.04, n_per_class)
    thick = rng.exponential(20, n_per_class)
    for i in range(n_per_class):
        X_list.append([
            gr[i], abs(rng.normal(4, 2)), rt[i], den[i], cnl[i],
            abs(rng.normal(2, 1.5)), thick[i],
            rt[i] / max(den[i], 1e-8),
        ])
        y_list.append(2)

    # 3: Carbonate — 低GR, 高DEN, 高RT
    gr = rng.normal(25, 10, n_per_class)
    rt = np.clip(rng.normal(100, 50, n_per_class), 0.1, None)
    den = rng.normal(2.65, 0.05, n_per_class)
    cnl = rng.normal(0.05, 0.03, n_per_class)
    thick = rng.exponential(10, n_per_class)
    for i in range(n_per_class):
        X_list.append([
            gr[i], abs(rng.normal(5, 2)), rt[i], den[i], cnl[i],
            abs(rng.normal(3, 2)), thick[i],
            rt[i] / max(den[i], 1e-8),
        ])
        y_list.append(3)

    # 4: Coal — 极低GR, 极低DEN, 高CNL
    gr = rng.normal(20, 8, n_per_class)
    rt = np.clip(rng.normal(500, 200, n_per_class), 0.1, None)
    den = rng.normal(1.65, 0.10, n_per_class)
    cnl = rng.normal(0.40, 0.05, n_per_class)
    thick = rng.exponential(3, n_per_class)
    for i in range(n_per_class):
        X_list.append([
            gr[i], abs(rng.normal(3, 1)), rt[i], den[i], cnl[i],
            abs(rng.normal(3, 2)), thick[i],
            rt[i] / max(den[i], 1e-8),
        ])
        y_list.append(4)

    # 5: Organic-rich shale — 高GR, 高RT, 中等DEN
    gr = rng.normal(130, 20, n_per_class)
    rt = np.clip(rng.normal(25, 10, n_per_class), 0.1, None)
    den = rng.normal(2.50, 0.06, n_per_class)
    cnl = rng.normal(0.25, 0.06, n_per_class)
    thick = rng.exponential(25, n_per_class)
    for i in range(n_per_class):
        X_list.append([
            gr[i], abs(rng.normal(10, 4)), rt[i], den[i], cnl[i],
            abs(rng.normal(3, 2.5)), thick[i],
            rt[i] / max(den[i], 1e-8),
        ])
        y_list.append(5)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


CLASS_NAMES = [
    "shale", "silty_sandstone", "clean_sandstone",
    "carbonate", "coal", "organic_rich_shale",
]

FLUID_CLASSES = {
    "clean_sandstone": {
        "gas": {"rt_min": 50, "den_max": 2.35, "cnl_max": 0.20},
        "oil": {"rt_min": 20, "den_max": 2.42},
        "water": {"rt_max": 8},
    },
    "silty_sandstone": {
        "oil": {"rt_min": 15, "den_max": 2.45},
        "water": {"rt_max": 10},
    },
    "carbonate": {
        "gas": {"rt_min": 80, "den_max": 2.55},
        "oil": {"rt_min": 30, "den_max": 2.60},
    },
}


class WellLogML:
    name = "well_log_ml"
    description = (
        "测井岩性ML分类器。基于岩石物理知识训练的RandomForest，"
        "6类岩性(shale/silty_sand/clean_sand/carbonate/coal/organic_shale)"
        "+ 流体识别(gas/oil/water)。模型缓存到~/.cache"
    )
    required_fields = [
        "analysis_type: lithology|fluid|full",
        "depth_range?",
    ]
    output_shape = (
        "list[{id, class_name, depth_top_m, depth_bottom_m, "
        "lithology, lithology_proba:{}, fluid_type?, confidence}]"
    )

    def __init__(self):
        self._model = None
        self._scaler = None

    def _load(self):
        if self._model is not None:
            return True

        # Inference-only policy: load an existing artifact, never fit a model.
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                self._model = data["model"]
                self._scaler = data["scaler"]
                print(f"[well_log_ml] loaded cached model from {CACHE_PATH}")
                return True
            except Exception:
                pass
        print("[well_log_ml] disabled: no prebuilt inference artifact")
        return False

    def detect(self, instruction, image=None, context=None):
        if not self._load():
            return []

        curves = None
        depth_axis = None
        if context and "curves" in context:
            curves = {k: np.asarray(v, dtype=np.float32)
                      for k, v in context["curves"].items()}
            if "depth" in curves:
                depth_axis = curves["depth"]

        if curves is None or depth_axis is None:
            return [{"id": "wlml_no_data",
                     "result": "need curves in context", "model": self.name}]

        analysis_type = instruction.get("analysis_type", "full")
        dr = instruction.get("depth_range") or {}

        # 滑动窗口分段
        window = 20  # 每20个采样点(约3m)一个预测窗口
        results = []
        gr = curves.get("GR", np.full_like(depth_axis, 95.0))
        rt = curves.get("RT", np.full_like(depth_axis, 3.0))
        den = curves.get("DEN", np.full_like(depth_axis, 2.55))
        cnl = curves.get("CNL", np.full_like(depth_axis, 0.25))

        for i in range(0, len(depth_axis) - window, window // 2):
            seg = slice(i, min(i + window, len(depth_axis)))
            d_top = float(depth_axis[seg][0])
            d_bot = float(depth_axis[seg][-1])

            # 深度范围过滤
            if dr:
                try:
                    dr_top = float(dr.get("top_m", -999) or -999)
                    dr_bot = float(dr.get("bottom_m", 99999) or 99999)
                except (ValueError, TypeError):
                    dr_top, dr_bot = -999, 99999
                if d_bot < dr_top:
                    continue
                if d_top > dr_bot:
                    break

            seg_gr = gr[seg]
            seg_rt = rt[seg]
            seg_den = den[seg]
            seg_cnl = cnl[seg]
            thickness = d_bot - d_top

            X = np.array([[
                float(seg_gr.mean()), float(seg_gr.std()),
                float(seg_rt.mean()), float(seg_den.mean()),
                float(seg_cnl.mean()),
                float(abs(np.gradient(seg_gr)).mean()) if len(seg_gr) > 1 else 0,
                thickness,
                float(seg_rt.mean()) / max(float(seg_den.mean()), 1e-8),
            ]], dtype=np.float32)

            X_scaled = self._scaler.transform(X)
            proba = self._model.predict_proba(X_scaled)[0]
            best_cls = int(np.argmax(proba))
            best_conf = float(proba[best_cls])
            litho = CLASS_NAMES[best_cls]

            result = {
                "id": f"wlml_{i}",
                "class_name": litho,
                "depth_top_m": round(d_top, 1),
                "depth_bottom_m": round(d_bot, 1),
                "lithology": litho,
                "lithology_confidence": round(best_conf, 3),
                "model": self.name,
            }

            # 流体识别
            if analysis_type in ("fluid", "full") and litho in FLUID_CLASSES:
                rt_m = float(seg_rt.mean())
                den_m = float(seg_den.mean())
                cnl_m = float(seg_cnl.mean())
                for fluid, rules in FLUID_CLASSES[litho].items():
                    ok = True
                    if "rt_min" in rules and rt_m < rules["rt_min"]:
                        ok = False
                    if "rt_max" in rules and rt_m > rules["rt_max"]:
                        ok = False
                    if "den_max" in rules and den_m > rules["den_max"]:
                        ok = False
                    if "cnl_max" in rules and cnl_m > rules["cnl_max"]:
                        ok = False
                    if ok:
                        result["fluid_type"] = fluid
                        result["fluid_confidence"] = round(best_conf * 0.85, 3)
                        result["rt_mean"] = round(rt_m, 1)
                        result["den_mean"] = round(den_m, 3)
                        break

            results.append(result)

        return self._merge_adjacent(results)

    @staticmethod
    def _merge_adjacent(segments: list[dict]) -> list[dict]:
        if len(segments) <= 1:
            return segments
        merged = []
        cur = dict(segments[0])
        for nxt in segments[1:]:
            gap = nxt["depth_top_m"] - cur["depth_bottom_m"]
            if (cur["lithology"] == nxt["lithology"]
                    and gap < 3.0
                    and cur.get("fluid_type") == nxt.get("fluid_type")):
                cur["depth_bottom_m"] = nxt["depth_bottom_m"]
                cur["lithology_confidence"] = round(
                    max(cur["lithology_confidence"],
                        nxt["lithology_confidence"]), 3)
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)
        return merged
