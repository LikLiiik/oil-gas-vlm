"""闭环单元测试（不加载 VLM，秒级跑完）。

注入 FakeVLM + stub 下游模型，验证两条关键修复：
  #1  VLM 验证判假的检测会被剔除，不再进最终输出
  #3  重试时只重跑被调整的那一个 step，其余 step 不重复执行

    python test/test_loop_unit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from pipeline import Pipeline, downstream
from pipeline.agents import LoopAgent, _dedup_by_det_id
from pipeline.downstream.base import _REGISTRY
from pipeline.loop_core import (
    apply_retry,
    bbox_iou,
    dedup_by_iou,
    ensure_bbox_norm,
    match_false_positives,
    tag_detection,
)
from pipeline.vlm import VLMResponse

# ---- 工具 -----------------------------------------------------------------

class FakeVLM:
    """按脚本顺序返回 VLMResponse，记录每次调用。不加载任何模型。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call_json(self, system_prompt, images, user_text, schema=None,
                  max_new_tokens=4096, temperature=0.0):
        self.calls.append({
            "system": system_prompt, "user": user_text,
            "n_images": len(images) if images else 0,
        })
        data = self.responses.pop(0)
        return VLMResponse(
            text=json.dumps(data, ensure_ascii=False), data=data,
            elapsed_s=0.0, attempts=1, schema_valid=True, schema_errors=[],
        )


class StubModel:
    """可脚本化的下游模型：按 factory(instruction) 返回固定检测。"""

    def __init__(self, name, factory):
        self.name = name
        self.description = "stub"
        self.required_fields = []
        self.output_shape = ""
        self._factory = factory
        self.detect_calls = 0
        self.instructions = []

    def detect(self, instruction, image=None, context=None):
        self.detect_calls += 1
        self.instructions.append(dict(instruction))
        return self._factory(instruction)


def _img():
    import numpy as np
    arr = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _det(det_id, conf, class_name="fault", xyxy=(10, 10, 20, 20)):
    return {"id": det_id, "det_id": det_id, "class_name": class_name,
            "bbox_pixel": list(xyxy), "confidence": conf}


def _detn(det_id, conf, bbox_norm, model="m", class_name="fault"):
    """带 bbox_norm 的检测，用于 match_false_positives 的纯函数测试。"""
    return {"id": det_id, "det_id": det_id, "class_name": class_name,
            "model": model, "bbox_norm": list(bbox_norm), "confidence": conf}


# ---- loop_core 纯函数 -----------------------------------------------------

def test_tag_detection_synthesizes_id():
    d = tag_detection({"class_name": "fault"}, step=1, image_name="img",
                     model_name="sam", index=0)
    assert d["det_id"] == "sam_s1_i0"
    assert d["id"] == "sam_s1_i0"
    assert d["step"] == 1 and d["image_name"] == "img" and d["model"] == "sam"


def test_tag_detection_reuses_model_id():
    d = tag_detection({"id": "real_1", "class_name": "fault"}, step=2,
                     image_name="img", model_name="sam", index=3)
    assert d["det_id"] == "real_1"   # 复用模型自带的 id
    assert d["det_id"] == d["id"]


def test_tag_detection_keeps_existing_det_id():
    d = tag_detection({"det_id": "keep_me"}, step=1, image_name="img",
                     model_name="sam", index=0)
    assert d["det_id"] == "keep_me"


# ---- match_false_positives: A(bbox 兜底) + B(置信度门槛/存疑侧栏) ---------

def test_match_fp_by_result_id_high_conf_drops():
    dets = [_detn("a", 0.8, [0.1, 0.1, 0.3, 0.3]),
            _detn("b", 0.4, [0.5, 0.5, 0.7, 0.7])]
    ver = {"verified": [
        {"result_id": "a", "is_real": True},
        {"result_id": "b", "is_real": False, "confidence": 0.9,
         "rejection_reason": "假阳性"},
    ]}
    drop, dropped, review = match_false_positives(ver, dets)
    assert drop == {"b"}
    assert [d["det_id"] for d in dropped] == ["b"]
    assert review == []


def test_match_fp_bbox_fallback_when_no_result_id():
    """A: VLM 没回填 result_id，靠 bbox_xyxy_norm + 同 model 做几何匹配命中。"""
    dets = [_detn("a", 0.8, [0.10, 0.10, 0.30, 0.30], model="sam"),
            _detn("b", 0.5, [0.55, 0.55, 0.75, 0.75], model="sam")]
    ver = {"verified": [
        {"model": "sam", "bbox_xyxy_norm": [0.12, 0.11, 0.31, 0.29],
         "is_real": False, "confidence": 0.8, "rejection_reason": "无错断"},
    ]}
    drop, dropped, review = match_false_positives(ver, dets)
    assert drop == {"a"}   # 几何上最接近 a
    assert review == []


def test_match_fp_low_confidence_goes_to_review_not_dropped():
    """B: 判假但 confidence<0.5 -> 不删，进 review 保留。"""
    dets = [_detn("a", 0.8, [0.1, 0.1, 0.3, 0.3])]
    ver = {"verified": [
        {"result_id": "a", "is_real": False, "confidence": 0.3,
         "rejection_reason": "不确定"},
    ]}
    drop, dropped, review = match_false_positives(ver, dets)
    assert drop == set()                  # 不删
    assert dropped == []
    assert len(review) == 1
    assert review[0]["det_id"] == "a"
    assert review[0]["reason"] == "low_confidence"


def test_match_fp_no_confidence_goes_to_review():
    """confidence 缺失 -> 视为低置信，进 review。"""
    dets = [_detn("a", 0.8, [0.1, 0.1, 0.3, 0.3])]
    ver = {"verified": [{"result_id": "a", "is_real": False}]}
    drop, dropped, review = match_false_positives(ver, dets)
    assert drop == set()
    assert review[0]["reason"] == "no_confidence"


def test_match_fp_unmatched_goes_to_review():
    """VLM 给的 result_id 对不上、也没 bbox -> 匹配不上，进 review 不删。"""
    dets = [_detn("a", 0.8, [0.1, 0.1, 0.3, 0.3])]
    ver = {"verified": [
        {"result_id": "does_not_exist", "is_real": False, "confidence": 0.9},
    ]}
    drop, dropped, review = match_false_positives(ver, dets)
    assert drop == set()
    assert len(review) == 1
    assert review[0]["reason"] == "unmatched"
    assert review[0]["det_id"] is None


def test_match_fp_cross_model_bbox_not_matched():
    """bbox 接近但 model 不同 -> 不算同一条（避免误删别的模型的检测）。"""
    dets = [_detn("a", 0.8, [0.1, 0.1, 0.3, 0.3], model="sam"),
            _detn("b", 0.5, [0.11, 0.11, 0.29, 0.29], model="cig_fault")]
    ver = {"verified": [
        {"model": "sam", "bbox_xyxy_norm": [0.12, 0.12, 0.28, 0.28],
         "is_real": False, "confidence": 0.9},
    ]}
    drop, dropped, review = match_false_positives(ver, dets)
    assert drop == {"a"}            # 只删 sam 的 a，不碰 cig_fault 的 b
    assert "b" not in drop


def test_ensure_bbox_norm_from_pixel():
    d = {"bbox_pixel": [10, 20, 30, 40]}
    ensure_bbox_norm(d, 100, 200)
    assert d["bbox_norm"] == [0.1, 0.1, 0.3, 0.2]


def test_ensure_bbox_norm_keeps_existing():
    d = {"bbox_norm": [0.5, 0.5, 0.6, 0.6], "bbox_pixel": [1, 2, 3, 4]}
    ensure_bbox_norm(d, 100, 100)
    assert d["bbox_norm"] == [0.5, 0.5, 0.6, 0.6]   # 不覆盖已有


def test_apply_retry_merges_into_target():
    steps = [{"step": 1, "instruction": {"threshold": 0.3}},
             {"step": 2, "instruction": {"n_clusters": 5}}]
    target = apply_retry(steps, {"step": 2, "adjusted_params": {"n_clusters": 8}})
    assert target == 2
    assert steps[1]["instruction"] == {"n_clusters": 8}   # 合并(覆盖同名字段)


def test_apply_retry_returns_none_when_nothing_to_apply():
    steps = [{"step": 1, "instruction": {}}]
    assert apply_retry(steps, None) is None
    assert apply_retry(steps, {}) is None
    assert apply_retry(steps, {"step": 1}) is None          # 无 adjusted
    assert apply_retry(steps, {"step": 9, "adjusted_params": {"x": 1}}) is None


def test_bbox_iou_basic():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) > 0.999
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert 0.0 < bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]) < 0.5


def test_dedup_by_iou_keeps_max_conf():
    dets = [
        _det("a", 0.9, xyxy=(0, 0, 10, 10)),
        _det("b", 0.4, xyxy=(1, 1, 11, 11)),   # 与 a 重叠 -> 丢
        _det("c", 0.7, xyxy=(50, 50, 60, 60)),  # 不重叠 -> 留
    ]
    kept = dedup_by_iou(dets, iou_thr=0.5)
    ids = {d["det_id"] for d in kept}
    assert ids == {"a", "c"}


# ---- LoopAgent: 假阳性过滤 (#1) -------------------------------------------

def test_loopagent_filters_false_positive():
    """VLM 判假的检测不应出现在最终 results 里。"""
    stub = StubModel("stub_loop", lambda instr: [
        _det("real1", 0.9, xyxy=(10, 10, 20, 20)),
        _det("fp1", 0.4, xyxy=(50, 50, 60, 60)),
    ])
    downstream.register(stub)
    try:
        vlm = FakeVLM([
            # Phase 1 规划
            {"scene_understanding": "test", "max_iterations": 2,
             "workflow_steps": [{"step": 1, "model": "stub_loop",
                                  "instruction": {}}]},
            # Phase 3 验证：fp1 判假(confidence>=0.5 才删)，不重试
            {"verified": [
                {"result_id": "real1", "is_real": True},
                {"result_id": "fp1", "is_real": False, "confidence": 0.9,
                 "rejection_reason": "仅振幅渐变无错断"},
            ], "need_retry": False},
        ])
        agent = LoopAgent(vlm, verbose=False)
        r = agent.run(_img(), agent_name="t")
        assert r.ok
        ids = {d.get("det_id") for d in r.results}
        assert "real1" in ids
        assert "fp1" not in ids, f"假阳性未被过滤: {ids}"
        assert r.verifications[0]["filtered_ids"] == ["fp1"]
        dropped = r.verifications[0]["filtered"]["dropped"]
        assert [d["det_id"] for d in dropped] == ["fp1"]   # B: 被删的留侧栏可恢复
    finally:
        _REGISTRY.pop("stub_loop", None)


def test_loopagent_dedup_across_iterations():
    """两轮都返回同一 det_id，最终只保留一条（高置信）。"""
    call = {"n": 0}

    def factory(instr):
        call["n"] += 1
        return [_det("dup", 0.5 if call["n"] == 1 else 0.8,
                     xyxy=(10, 10, 20, 20))]

    stub = StubModel("stub_dup", factory)
    downstream.register(stub)
    try:
        vlm = FakeVLM([
            {"scene_understanding": "test", "max_iterations": 2,
             "workflow_steps": [{"step": 1, "model": "stub_dup",
                                  "instruction": {}}]},
            {"verified": [{"result_id": "dup", "is_real": True}],
             "need_retry": True,
             "retry_instructions": {"step": 1, "adjusted_params": {"x": 1}}},
            {"verified": [{"result_id": "dup", "is_real": True}],
             "need_retry": False},
        ])
        agent = LoopAgent(vlm, verbose=False)
        r = agent.run(_img(), agent_name="t")
        assert r.ok
        assert len(r.results) == 1, f"未按 det_id 去重: {r.results}"
        assert r.results[0]["confidence"] == 0.8   # 保留高置信
    finally:
        _REGISTRY.pop("stub_dup", None)


# ---- Pipeline._run_closed_loop: #1 过滤 + #3 仅重跑目标 step ---------------

def test_run_closed_loop_filters_fp_and_reruns_only_target():
    """iter1: step2 的 b1 被判假 -> 过滤；need_retry -> 只重跑 step2。
    iter2: step2 产出 b2（真），step1 不重跑。最终 = {a1, b2}。"""
    stub_a = StubModel("stub_a", lambda instr: [_det("a1", 0.8)])
    b_state = {"n": 0}

    def b_factory(instr):
        b_state["n"] += 1
        return [_det("b1", 0.5)] if b_state["n"] == 1 else [_det("b2", 0.6)]

    stub_b = StubModel("stub_b", b_factory)
    downstream.register(stub_a)
    downstream.register(stub_b)
    try:
        vlm = FakeVLM([
            # iter1 验证：b1 判假(confidence>=0.5 才删)，要求重试 step2
            {"verified": [
                {"result_id": "a1", "is_real": True},
                {"result_id": "b1", "is_real": False, "confidence": 0.85,
                 "rejection_reason": "假阳性"},
            ], "need_retry": True,
             "retry_instructions": {"step": 2, "adjusted_params": {"v": 1}}},
            # iter2 验证：b2 真，收敛
            {"verified": [{"result_id": "b2", "is_real": True}],
             "need_retry": False},
        ])
        p = Pipeline(vlm=vlm, verbose=False)
        steps = [
            {"step": 1, "model": "stub_a", "instruction": {}, "image_name": "img1"},
            {"step": 2, "model": "stub_b", "instruction": {"v": 0}, "image_name": "img1"},
        ]
        image_by_name = {"img1": SimpleNamespace(
            name="img1", pil=_img(), physical_view="inline")}
        step_results, verifs, _ = p._run_closed_loop(
            steps, image_by_name, pkg=None, verify=True, max_iter=3)

        # #3: step1 只跑 1 次，step2 跑 2 次（重试只重跑目标）
        assert stub_a.detect_calls == 1, f"step1 不应重跑: {stub_a.detect_calls}"
        assert stub_b.detect_calls == 2, f"step2 应重试一次: {stub_b.detect_calls}"

        # #1: 最终只含 a1 + b2，b1 被过滤
        final_ids = {d.get("det_id")
                     for dets in step_results.values() for d in dets}
        assert final_ids == {"a1", "b2"}, f"假阳性或旧结果残留: {final_ids}"
        assert verifs[0]["filtered_ids"] == ["b1"]
        assert [d["det_id"] for d in verifs[0]["filtered"]["dropped"]] == ["b1"]
        assert len(verifs) == 2
    finally:
        _REGISTRY.pop("stub_a", None)
        _REGISTRY.pop("stub_b", None)


def test_run_closed_loop_no_verify_keeps_all():
    """关闭验证时不过滤、不调用 VLM。"""
    stub = StubModel("stub_nv", lambda instr: [_det("x1", 0.5)])
    downstream.register(stub)
    try:
        vlm = FakeVLM([])   # 不应被调用
        p = Pipeline(vlm=vlm, verbose=False)
        steps = [{"step": 1, "model": "stub_nv", "instruction": {},
                  "image_name": "img1"}]
        image_by_name = {"img1": SimpleNamespace(
            name="img1", pil=_img(), physical_view="inline")}
        step_results, verifs, _ = p._run_closed_loop(
            steps, image_by_name, pkg=None, verify=False, max_iter=3)
        assert vlm.calls == []
        assert verifs == []
        ids = {d.get("det_id") for d in step_results[1]}
        assert ids == {"x1"}
    finally:
        _REGISTRY.pop("stub_nv", None)


def test_loopagent_low_confidence_fp_is_kept_as_review():
    """B: 判假但 confidence<0.5 -> 不删，保留在 results，进 filtered.review。"""
    stub = StubModel("stub_lc", lambda instr: [
        _det("a", 0.9, xyxy=(10, 10, 20, 20)),
        _det("b", 0.4, xyxy=(50, 50, 60, 60)),
    ])
    downstream.register(stub)
    try:
        vlm = FakeVLM([
            {"scene_understanding": "test", "max_iterations": 1,
             "workflow_steps": [{"step": 1, "model": "stub_lc", "instruction": {}}]},
            {"verified": [
                {"result_id": "a", "is_real": True},
                {"result_id": "b", "is_real": False, "confidence": 0.3,
                 "rejection_reason": "不确定"},
            ], "need_retry": False},
        ])
        r = LoopAgent(vlm, verbose=False).run(_img(), agent_name="t")
        ids = {d.get("det_id") for d in r.results}
        assert ids == {"a", "b"}, f"低置信不应被删: {ids}"
        rev = r.verifications[0]["filtered"]["review"]
        assert [x["det_id"] for x in rev] == ["b"]
        assert rev[0]["reason"] == "low_confidence"
    finally:
        _REGISTRY.pop("stub_lc", None)


def test_run_closed_loop_bbox_fallback_drops_without_result_id():
    """A: VLM 不回填 result_id，靠 bbox_xyxy_norm 几何匹配命中并删除假阳性。"""
    # stub 产出 bbox_pixel=[10,10,30,30]，image 64x64 -> bbox_norm≈[0.156,0.156,0.469,0.469]
    stub = StubModel("stub_bb", lambda instr: [_det("t1", 0.8, xyxy=(10, 10, 30, 30))])
    downstream.register(stub)
    try:
        vlm = FakeVLM([
            # 验证：不给 result_id，只给接近的 bbox_xyxy_norm，判假且高置信
            {"verified": [
                {"model": "stub_bb",
                 "bbox_xyxy_norm": [0.16, 0.16, 0.46, 0.46],
                 "is_real": False, "confidence": 0.85,
                 "rejection_reason": "无错断"},
            ], "need_retry": False},
        ])
        p = Pipeline(vlm=vlm, verbose=False)
        steps = [{"step": 1, "model": "stub_bb", "instruction": {},
                  "image_name": "img1"}]
        image_by_name = {"img1": SimpleNamespace(
            name="img1", pil=_img(), physical_view="inline")}
        step_results, verifs, _ = p._run_closed_loop(
            steps, image_by_name, pkg=None, verify=True, max_iter=2)
        final_ids = {d.get("det_id") for d in step_results[1]}
        assert final_ids == set(), f"bbox 兜底应删掉 t1: {final_ids}"
        assert verifs[0]["filtered_ids"] == ["t1"]
    finally:
        _REGISTRY.pop("stub_bb", None)


# ---- _dedup_by_det_id ----------------------------------------------------

def test_dedup_by_det_id_keeps_max_conf():
    dets = [_det("a", 0.5), _det("a", 0.9), _det("b", 0.3), {"det_id": None}]
    out = _dedup_by_det_id(dets)
    by = {d.get("det_id"): d for d in out if d.get("det_id")}
    assert by["a"]["confidence"] == 0.9
    assert "b" in by
    assert sum(1 for d in out if d.get("det_id") is None) == 1


# ---- 手动 runner（不装 pytest 也能跑） -----------------------------------

if __name__ == "__main__":
    import inspect
    fns = [(n, f) for n, f in globals().items()
           if n.startswith("test_") and callable(f)
           and inspect.signature(f).parameters == {}]
    passed, failed = 0, []
    for n, f in fns:
        try:
            f()
            passed += 1
            print(f"  ✓ {n}")
        except AssertionError as e:
            failed.append((n, str(e) or "assertion failed"))
            print(f"  ✗ {n}: {e}")
        except Exception as e:
            failed.append((n, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {n}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if not failed else 1)
