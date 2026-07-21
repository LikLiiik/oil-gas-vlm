"""闭环共享逻辑：det 标签、假阳性过滤、重试指令应用、bbox IoU。

LoopAgent 与 Pipeline.run_from_adapter 共用这一段，避免两套
plan->exec->verify->retry 实现里重复（且曾各自漏掉假阳性过滤）的同一段逻辑。
"""
from __future__ import annotations


def bbox_iou(a: list[float], b: list[float]) -> float:
    """两个 [x1,y1,x2,y2] bbox 的 IoU。"""
    xo = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    yo = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = xo * yo
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def tag_detection(det: dict, *, step: int, image_name: str,
                  model_name: str, index: int) -> dict:
    """给一条原始检测打上闭环需要的稳定标签。

    - det_id: 全流程唯一标识。喂回 VLM 验证时让它原样回填到 result_id，
      这样验证的 is_real=false 才能精确对应到具体检测。
      模型自带 id/det_id 就复用，否则合成 {model}_s{step}_i{index}。
    - step / image_name / model: 方便后续按步重试、按图聚合。
    """
    out = dict(det)
    existing = out.get("det_id") or out.get("id")
    if not existing:
        existing = f"{model_name}_s{step}_i{index}"
    out["det_id"] = existing
    out.setdefault("id", existing)
    out.setdefault("model", model_name)
    out.setdefault("step", step)
    out.setdefault("image_name", image_name)
    return out


# 假阳性过滤的默认门槛（见 match_false_positives）
DEFAULT_FP_CONF_THRESHOLD = 0.5
DEFAULT_FP_IOU_THRESHOLD = 0.5


def ensure_bbox_norm(det: dict, img_w: int, img_h: int) -> dict:
    """若 det 只有 bbox_pixel 没有 bbox_norm，按图像尺寸补上归一化 bbox。

    让 VLM 看到统一坐标，也让假阳性 IoU 匹配在归一化空间进行（否则 pixel vs norm
    的 bbox 算 IoU 会得到 ~0，bbox 兜底失效）。det 已是 tag_detection 的拷贝，原地改。
    """
    if det.get("bbox_norm") or not det.get("bbox_pixel"):
        return det
    try:
        x1, y1, x2, y2 = det["bbox_pixel"]
    except (TypeError, ValueError, KeyError):
        return det
    w = max(img_w, 1)
    h = max(img_h, 1)
    det["bbox_norm"] = [x1 / w, y1 / h, x2 / w, y2 / h]
    return det


def match_false_positives(ver_data: dict, detections: list[dict], *,
                          conf_threshold: float = DEFAULT_FP_CONF_THRESHOLD,
                          iou_thr: float = DEFAULT_FP_IOU_THRESHOLD,
                          ) -> tuple[set[str], list[dict], list[dict]]:
    """把 VLM 验证判假的条目匹配回具体检测，决定哪些删、哪些存疑。

    返回 (drop_ids, dropped, review):
      drop_ids : 高置信假阳性的 det_id 集合，应从输出剔除。
      dropped  : 被剔除检测的完整 dict 列表，进 report filtered.dropped 供人工复核/恢复。
      review   : 存疑条目(低置信/匹配不上/重复)的摘要列表，保留在输出，进 filtered.review。

    匹配优先级:
      1. result_id 精确命中已知 det_id
      2. 同 model + bbox IoU > iou_thr（VLM 没回填 id 时的兜底，几何定位更可靠）
    删除门槛:
      匹配上 且 verification.confidence >= conf_threshold 才删；
      否则(低置信/无置信)进 review，保留在输出。匹配不上也进 review。
    """
    by_id = {str(d.get("det_id")): d for d in detections
             if d.get("det_id") is not None}
    drop_ids: set[str] = set()
    dropped: list[dict] = []
    review: list[dict] = []
    for v in (ver_data.get("verified") or []):
        if v.get("is_real"):
            continue
        det = _match_verification(v, detections, by_id, iou_thr)
        conf = v.get("confidence")
        conf_ok = conf is not None and float(conf) >= conf_threshold
        if det is None:
            review.append(_review_item(v, None, "unmatched"))
            continue
        did = det.get("det_id")
        if conf_ok and did is not None and did not in drop_ids:
            drop_ids.add(did)
            dropped.append(det)
        else:
            reason = ("duplicate" if did in drop_ids
                      else "low_confidence" if conf is not None
                      else "no_confidence")
            review.append(_review_item(v, det, reason))
    return drop_ids, dropped, review


def _match_verification(v: dict, detections: list[dict],
                        by_id: dict[str, dict], iou_thr: float) -> dict | None:
    """单条验证 -> 对应的检测 dict。优先 result_id，回退 bbox IoU。"""
    rid = v.get("result_id")
    if rid is not None:
        det = by_id.get(str(rid))
        if det is not None:
            return det
    vb = v.get("bbox_xyxy_norm") or v.get("bbox_norm")
    if not vb or len(vb) != 4:
        return None
    vmodel = v.get("model")
    best, best_iou = None, iou_thr
    for d in detections:
        if vmodel and d.get("model") and d.get("model") != vmodel:
            continue
        db = d.get("bbox_norm") or d.get("bbox_pixel")
        if not db or len(db) != 4:
            continue
        iou = bbox_iou(list(vb), list(db))
        if iou > best_iou:
            best, best_iou = d, iou
    return best


def _review_item(v: dict, det: dict | None, reason: str) -> dict:
    return {
        "det_id": det.get("det_id") if det else None,
        "class_name": det.get("class_name") if det else None,
        "bbox_norm": det.get("bbox_norm") if det else None,
        "model": det.get("model") if det else None,
        "rejection_reason": v.get("rejection_reason"),
        "confidence": v.get("confidence"),
        "reason": reason,
    }


def apply_retry(steps: list[dict], retry: dict | None) -> int | None:
    """把验证给出的 retry_instructions 应用到对应 step，返回被调整的 step 号。

    合并而非替换 instruction（保留 VLM 上一轮指定的其他参数）。
    找不到目标 step 或没有可应用的调整时返回 None。
    """
    if not retry:
        return None
    target = retry.get("step")
    adjusted = retry.get("adjusted_params") or retry.get("adjusted_instruction")
    if target is None or not adjusted:
        return None
    for s in steps:
        if s.get("step") == target:
            if isinstance(adjusted, dict):
                s["instruction"] = {**s.get("instruction", {}), **adjusted}
            else:
                s["instruction"] = adjusted
            return target
    return None


def dedup_by_iou(dets: list[dict], *, iou_thr: float = 0.5,
                  key: str = "bbox_norm") -> list[dict]:
    """同一 class_name + 相近 bbox 的检测去重，保留 confidence 最高者。

    bbox 取 det[key]（归一化）或回退 bbox_pixel。
    """
    by_class: dict[str, list[dict]] = {}
    for d in dets:
        by_class.setdefault(d.get("class_name", "unknown"), []).append(d)
    kept: list[dict] = []
    for cdets in by_class.values():
        cdets.sort(key=lambda d: -float(d.get("confidence", 0.0) or 0.0))
        local: list[dict] = []
        for d in cdets:
            b1 = d.get(key) or d.get("bbox_pixel")
            is_dup = False
            if b1:
                for k in local:
                    b2 = k.get(key) or k.get("bbox_pixel")
                    if b2 and bbox_iou(b1, b2) > iou_thr:
                        is_dup = True
                        break
            if not is_dup:
                local.append(d)
        kept.extend(local)
    return kept
