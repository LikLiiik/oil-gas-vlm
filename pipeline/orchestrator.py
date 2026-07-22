"""Pipeline 编排：统一闭环——VLM自主选模型 → 执行 → 验证 → 迭代 → 输出。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from schemas.output_schemas import (
    FUSION_OUTPUT_SCHEMA,
    PROSPECT_OUTPUT_SCHEMA,
)

from ._logging import get_logger
from .agents import AgentResult, LoopAgent, SingleShotAgent
from .loop_core import (
    apply_retry,
    bbox_iou,
    ensure_bbox_norm,
    match_false_positives,
    tag_detection,
)
from .prompts import (
    TASK_MODEL_MAP,
    VERIFICATION_PROMPT,
    prospect_evaluation_prompt,
    well_seismic_fusion_prompt,
    workflow_planning_prompt,
)
from .vlm import VLMClient


@dataclass
class PipelineOutput:
    seismic: AgentResult | None = None
    log: AgentResult | None = None
    fusion: AgentResult | None = None
    prospect: AgentResult | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seismic": self.seismic.to_dict() if self.seismic else None,
            "log": self.log.to_dict() if self.log else None,
            "fusion": self.fusion.to_dict() if self.fusion else None,
            "prospect": self.prospect.to_dict() if self.prospect else None,
            "meta": self.meta,
        }


class Pipeline:
    """统一编排。赛题主流程 + Fallback SEG-Y + 4-Agent 语义解释。"""

    def __init__(self, vlm: VLMClient | None = None, verbose: bool = True):
        self.vlm = vlm or VLMClient()
        self.verbose = verbose
        self._logger = get_logger("orchestrator")
        self.loop_agent = LoopAgent(self.vlm, verbose=verbose)
        self.fusion_agent = SingleShotAgent(
            self.vlm,
            "well_seismic_fusion",
            well_seismic_fusion_prompt(),
            FUSION_OUTPUT_SCHEMA,
            verbose=verbose,
        )
        self.prospect_agent = SingleShotAgent(
            self.vlm,
            "prospect_evaluation",
            prospect_evaluation_prompt(),
            PROSPECT_OUTPUT_SCHEMA,
            verbose=verbose,
        )

    def _log(self, msg: str):
        if self.verbose:
            self._logger.info(msg)

    def run_from_adapter(
        self, run_dir, out_dir=None, verify: bool = True, max_iterations: int = 3
    ) -> dict:
        """赛题主入口：吃 geo_adapter 的 runs/<sample_id>/ 目录。

        统一闭环:
          Phase 1: VLM 看到全部图像 + 任务描述 + 可用模型清单
                   → 自主规划 workflow_steps（选模型+参数+目标图像）
          Phase 2: 逐个执行 workflow_steps（调用对应的下游模型）
          Phase 3: VLM 验证——把下游结果+原图喂回 VLM，逐条判断真伪
          Phase 4: 如果 need_retry → 调整参数回到 Phase 2 仅重跑被调整的那一个 step（最多 max_iterations 轮）
          Phase 5: 收敛后 → bbox 坐标反变换 → 3D 属性 SEG-Y + 标注 PNG
        """
        import numpy as np

        from schemas.output_schemas import (
            WORKFLOW_PLAN_SCHEMA,
        )

        from . import downstream
        from . import exporter as exporter_mod
        from . import tasks as tasks_mod
        from .adapter import load_run
        from .exporter import (
            aggregate_adapter_detections,
            export_annotated_png,
            summary_report,
        )
        from .io.segy import SegyVolume, write_attribute_segy

        pkg = load_run(run_dir)
        if not pkg.images:
            raise RuntimeError(f"no model images found under {pkg.run_dir}/assets/*")

        out_dir = Path(out_dir) if out_dir else pkg.run_dir / "model_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        image_by_name = {im.name: im for im in pkg.images}
        self._log(
            f"[adapter] loaded {pkg.sample_id}: "
            f"{len(pkg.images)} images, "
            f"target_classes={pkg.target_classes}"
        )

        # ── Phase 1: VLM 工作流规划 ──────────────────────────────────
        # 每个物理视图单独调用，避免多图注意力稀释和漏分析。
        hint = tasks_mod.hint_for_target_classes(pkg.target_classes)
        self._log(
            f"\n[adapter] Phase 1: VLM 分视图工作流规划 "
            f"(可执行模型: {len(downstream.runnable_names())})"
        )
        view_plans: list[dict] = []
        planning_failures: list[dict] = []
        planning_elapsed = 0.0
        planning_attempts = 0
        for im in pkg.images:
            plan_user_text = self._build_competition_plan_text(pkg, hint, [im])
            plan_resp = self.vlm.call_json(
                workflow_planning_prompt(),
                [im.vlm_pil],
                plan_user_text,
                schema=WORKFLOW_PLAN_SCHEMA,
                max_new_tokens=4096,
                temperature=0.0,
            )
            planning_elapsed += plan_resp.elapsed_s
            planning_attempts += plan_resp.attempts
            if plan_resp.data is None or not plan_resp.schema_valid:
                failure_path = out_dir / f"vlm_plan_failed_{im.name}.txt"
                failure_path.write_text(plan_resp.text, encoding="utf-8")
                planning_failures.append(
                    {
                        "image_name": im.name,
                        "errors": plan_resp.schema_errors,
                        "raw_output": str(failure_path),
                    }
                )
                self._log(f"  ❌ {im.name}: schema failed {plan_resp.schema_errors}")
                continue
            normalized = self._normalize_competition_plan(
                plan_resp.data,
                pkg,
                allowed_image_names={im.name},
            )
            normalized["source_image"] = im.name
            view_plans.append(normalized)
            self._log(
                f"  ✅ {im.name}: {normalized.get('analysis_status')} "
                f"steps={len(normalized.get('workflow_steps') or [])} "
                f"({plan_resp.elapsed_s:.0f}s)"
            )

        if not view_plans:
            report = {
                "sample_id": pkg.sample_id,
                "ok": False,
                "phase": "vlm_planning_failed",
                "errors": planning_failures,
            }
            summary_report(report, out_dir)
            return report

        plan = self._merge_view_plans(view_plans)
        if planning_failures:
            plan["planning_failures"] = planning_failures
        steps = plan.get("workflow_steps", [])
        (out_dir / "vlm_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._log(
            f"  ✅ VLM 分视图规划汇总 ({planning_elapsed:.0f}s, "
            f"attempts={planning_attempts})"
        )
        self._log(f"  Scene: {plan.get('scene_understanding', '')[:150]}")
        if self.verbose:
            for s in steps:
                img = s.get("image_name", "auto")
                self._log(
                    f"    Step{s.get('step')}: {s.get('model')} "
                    f"on {img} → {s.get('reason', '')[:80]}"
                )

        # ── Phase 2+3+4: 执行 → 验证 → 迭代 ──────────────────────────
        max_iter = min(plan.get("max_iterations", max_iterations), max_iterations)
        step_results, verifications, _ = self._run_closed_loop(
            steps,
            image_by_name,
            pkg,
            verify=verify,
            max_iter=max_iter,
        )

        # 重建 all_detections：按 image_name 聚合 step_results（已剔除假阳性）
        all_detections: dict[str, list[dict]] = {}
        for dets in step_results.values():
            for d in dets:
                all_detections.setdefault(d.get("image_name", ""), []).append(d)

        # ── Phase 5: 聚合输出 ────────────────────────────────────────
        self._log("\n[adapter] Phase 5: 聚合输出")

        # 5a: 归一化——各模型不同输出格式 → 统一 {class_name, bbox_norm, confidence}
        n_raw = sum(len(v) for v in all_detections.values())
        normalized = exporter_mod.normalize_detection_format(
            all_detections,
            pkg.manifest,
            image_by_name,
        )
        # 5b: 去重——归一化后 class_name 统一，去重准确
        n_before_dedup = sum(len(v) for v in normalized.values())
        normalized = self._dedup_detections(normalized)
        n_total_dets = sum(len(v) for v in normalized.values())
        if self.verbose and n_before_dedup != n_total_dets:
            self._logger.info(f"  dedup: {n_before_dedup} → {n_total_dets} detections")
        self._log(f"  normalized+dedup: {n_raw} raw → {n_total_dets} detections")

        # 5c: 坐标反变换 → 3D 属性 SEG-Y
        seismic_meta = pkg.manifest.get("seismic") or {}
        shape = seismic_meta.get("shape")
        attr_sgy_paths: dict[str, str] = {}
        if shape and len(shape) == 3:
            per_class = aggregate_adapter_detections(
                normalized,
                pkg.manifest,
                tuple(shape),
            )
            interval_ms = 4.0
            for cls, cube in per_class.items():
                fake_vol = SegyVolume(
                    cube=np.zeros(shape, dtype=np.float32),
                    inlines=np.arange(shape[0], dtype=np.int32),
                    xlines=np.arange(shape[1], dtype=np.int32),
                    sample_interval_ms=interval_ms,
                    n_samples=shape[2],
                )
                out_path = out_dir / f"{cls.replace(' ', '_')}_attribute.sgy"
                try:
                    write_attribute_segy(fake_vol, cube, str(out_path))
                    attr_sgy_paths[cls] = str(out_path)
                except Exception as e:
                    self._logger.warning(f"  [SEG-Y write failed for {cls}: {e}]")

        # 5d: 标注 PNG —— 用归一化后的 dets (已有正确的 class_name)
        png_paths: list[str] = []
        for im in pkg.images:
            dets = normalized.get(im.name, [])
            if not dets:
                continue
            class_to_task = {}
            for d in dets:
                cname = d.get("class_name", "")
                canonical = tasks_mod.CLASS_ALIASES.get(cname)
                if canonical:
                    class_to_task[cname] = tasks_mod.get_optional(canonical)
            default_task = tasks_mod.get("fault")
            first_task = next((t for t in class_to_task.values() if t), default_task)
            # normalized dets 没有 bbox_pixel，用 image 尺寸从 bbox_norm 反算
            w, h = im.pil.size
            results_for_png = []
            for d in dets:
                bn = d.get("bbox_norm")
                if bn and len(bn) == 4:
                    px = [bn[0] * w, bn[1] * h, bn[2] * w, bn[3] * h]
                else:
                    continue
                results_for_png.append({**d, "bbox_pixel": px})
            if not results_for_png:
                continue
            fake = AgentResult(agent=im.name, ok=True, results=results_for_png)
            p = export_annotated_png(fake, im.pil, None, first_task, out_dir)
            png_paths.append(str(p))

        # ── 汇总报告 ──────────────────────────────────────────────────
        models_used = sorted(
            set(d.get("model", "?") for v in all_detections.values() for d in v)
        )
        report = {
            "sample_id": pkg.sample_id,
            "ok": True,
            "package": pkg.to_summary(),
            "vlm_plan": plan,
            "iterations": len(verifications),
            "verifications": verifications,
            "downstream": {
                "models_used": models_used,
                "n_detections": n_total_dets,
                "detections_by_image": all_detections,
            },
            "outputs": {
                "annotated_pngs": png_paths,
                "attribute_sgy": attr_sgy_paths,
                "vlm_plan_json": str(out_dir / "vlm_plan.json"),
            },
        }
        summary_report(report, out_dir)
        self._log(
            f"[adapter] wrote {out_dir} (models={models_used}, dets={n_total_dets})"
        )
        return report

    # ── 竞赛流程辅助方法 ────────────────────────────────────────────────

    @staticmethod
    def _normalize_competition_plan(
        plan: dict,
        pkg,
        allowed_image_names: set[str] | None = None,
    ) -> dict:
        """Constrain a VLM plan to manifest ranges and runnable local assets."""
        from . import downstream

        adjustments: list[str] = []
        evidence = list(plan.get("visual_evidence") or [])
        if allowed_image_names is not None:
            evidence = [
                item
                for item in evidence
                if item.get("image_name") in allowed_image_names
            ]

        def _valid_bbox(value) -> bool:
            return (
                isinstance(value, list)
                and len(value) == 4
                and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in value)
                and value[0] <= value[2]
                and value[1] <= value[3]
            )

        image_meta = {
            image.name: image for image in (getattr(pkg, "images", None) or [])
        }
        for item in evidence:
            image_name = item.get("image_name")
            status = item.get("status")
            bbox = item.get("bbox_xyxy_norm")
            physical_view = getattr(image_meta.get(image_name), "physical_view", "")
            class_name = item.get("class_name")
            invalid_spatial_claim = (
                physical_view == "well_log_panel" and class_name in {"fault", "channel"}
            )
            if status in {"present", "suspected"} and (
                not _valid_bbox(bbox) or invalid_spatial_claim
            ):
                item["status"] = "insufficient"
                item["bbox_xyxy_norm"] = None
                item["confidence"] = min(float(item.get("confidence") or 0.0), 0.49)
                limitations = list(item.get("limitations") or [])
                if invalid_spatial_claim:
                    reason = (
                        "single-well panel has no lateral evidence for fault/channel"
                    )
                else:
                    reason = "candidate downgraded because no valid localized bbox was supplied"
                if reason not in limitations:
                    limitations.append(reason)
                item["limitations"] = limitations
                adjustments.append(
                    f"{image_name}/{class_name}: downgraded {status} to insufficient"
                )
        plan["visual_evidence"] = evidence

        def _candidate_items(image_name: str, classes: set[str] | None = None):
            native_shape = getattr(image_meta.get(image_name), "native_shape", None)
            min_confidence = (
                0.6
                if native_shape and len(native_shape) == 2 and min(native_shape) < 32
                else 0.0
            )
            return [
                item
                for item in evidence
                if item.get("image_name") == image_name
                and item.get("status") in {"present", "suspected"}
                and _valid_bbox(item.get("bbox_xyxy_norm"))
                and float(item.get("confidence") or 0.0) >= min_confidence
                and (classes is None or item.get("class_name") in classes)
            ]

        def _step_target_classes(step: dict) -> tuple[bool, set[str] | None]:
            model_name = step.get("model")
            instruction = step.get("instruction") or {}
            if model_name == "cig_fault" or (
                model_name == "seismic_domain_model"
                and instruction.get("task") == "fault_detection"
            ):
                return True, {"fault"}
            if model_name == "cig_channel":
                return True, {"channel"}
            if model_name in {
                "attribute_extractor",
                "facies_classifier",
                "sam",
                "seismic_foundation",
            }:
                return True, None
            return False, None

        runnable = set(downstream.runnable_names())
        normalized_steps: list[dict] = []
        for step in plan.get("workflow_steps") or []:
            model_name = step.get("model")
            image_name = step.get("image_name")
            if model_name not in runnable:
                adjustments.append(
                    f"step {step.get('step')}: removed unavailable model {model_name!r}"
                )
                continue
            if (
                allowed_image_names is not None
                and image_name not in allowed_image_names
            ):
                adjustments.append(
                    f"step {step.get('step')}: removed invalid image_name "
                    f"{image_name!r}"
                )
                continue
            evidence_gated, target_classes = _step_target_classes(step)
            if evidence_gated and not _candidate_items(image_name, target_classes):
                adjustments.append(
                    f"step {step.get('step')}: removed because no localized, "
                    "sufficient-confidence evidence supports it"
                )
                continue
            normalized_steps.append(step)
        plan["workflow_steps"] = normalized_steps

        wl = pkg.manifest.get("well_logs") or {}
        depth_range = wl.get("depth_range")
        if isinstance(depth_range, dict):
            known_top = depth_range.get("top_m", depth_range.get("top"))
            known_bottom = depth_range.get("bottom_m", depth_range.get("bottom"))
        elif isinstance(depth_range, (list, tuple)) and len(depth_range) >= 2:
            known_top, known_bottom = depth_range[:2]
        else:
            known_top = known_bottom = None

        try:
            known_top = float(known_top)
            known_bottom = float(known_bottom)
        except (TypeError, ValueError):
            known_top = known_bottom = None

        for step in plan.get("workflow_steps") or []:
            if step.get("model") not in {"well_log_analyzer", "well_log_ml"}:
                continue
            if known_top is None or known_bottom is None:
                continue
            instruction = step.get("instruction") or {}
            requested = instruction.get("depth_range")
            normalized = {"top_m": known_top, "bottom_m": known_bottom}
            if normalized != requested:
                instruction["depth_range"] = normalized
                step["instruction"] = instruction
                adjustments.append(
                    f"step {step.get('step')}: depth_range expanded to authoritative "
                    f"manifest range [{known_top}, {known_bottom}]"
                )
            if (
                step.get("model") == "well_log_analyzer"
                and instruction.pop("rules", None) is not None
            ):
                step["instruction"] = instruction
                adjustments.append(
                    f"step {step.get('step')}: removed VLM-authored numeric rules; "
                    "structured curves remain authoritative"
                )
        if adjustments:
            plan["plan_adjustments"] = [
                *(plan.get("plan_adjustments") or []),
                *adjustments,
            ]
        return plan

    @staticmethod
    def _merge_view_plans(view_plans: list[dict]) -> dict:
        """Merge independently audited view plans into one executable workflow."""
        status_rank = {
            "no_target_visible": 0,
            "insufficient": 1,
            "suspected": 2,
            "evidence_present": 3,
        }
        merged_steps: list[dict] = []
        evidence: list[dict] = []
        scenes: list[str] = []
        adjustments: list[str] = []
        max_iterations = 1
        overall_status = "no_target_visible"
        for view_plan in view_plans:
            source_image = view_plan.get("source_image", "unknown")
            scenes.append(
                f"[{source_image}] {view_plan.get('scene_understanding', '')}"
            )
            evidence.extend(view_plan.get("visual_evidence") or [])
            adjustments.extend(view_plan.get("plan_adjustments") or [])
            max_iterations = max(
                max_iterations,
                int(view_plan.get("max_iterations") or 1),
            )
            candidate_status = view_plan.get("analysis_status", "insufficient")
            if status_rank.get(candidate_status, 1) > status_rank[overall_status]:
                overall_status = candidate_status
            for step in view_plan.get("workflow_steps") or []:
                copied = dict(step)
                copied["step"] = len(merged_steps) + 1
                merged_steps.append(copied)
        merged = {
            "scene_understanding": "\n".join(scenes),
            "analysis_status": overall_status,
            "visual_evidence": evidence,
            "workflow_steps": merged_steps,
            "verification_strategy": "per_step" if merged_steps else "none",
            "max_iterations": max_iterations,
            "per_view_plans": view_plans,
        }
        if adjustments:
            merged["plan_adjustments"] = adjustments
        return merged

    def _build_competition_plan_text(self, pkg, task_hint: str, images: list) -> str:
        """构建给 VLM 的竞赛规划 user text。
        包含: 任务描述 + 图像清单 + 推荐模型 + 约束说明。
        """
        from . import downstream
        from .adapter import _slim_manifest

        img_lines = []
        for i, im in enumerate(images, 1):
            view = im.physical_view
            vm = pkg.view_meta(view) or {}
            shape = im.native_shape or vm.get("array_shape")
            seed_bounds = ""
            if (
                isinstance(shape, (list, tuple))
                and len(shape) == 2
                and view in {"inline", "crossline", "user_provided_2d_patch"}
            ):
                seed_bounds = (
                    f"  horizon_seed_bounds=trace_idx[0,{int(shape[0]) - 1}],"
                    f"sample_idx[0,{int(shape[1]) - 1}]"
                )
            img_lines.append(
                f'  {i}. image_name="{im.name}"  view={view}  '
                f"vlm_size_px={im.vlm_pil.size}  native_shape={shape}  "
                f"axis_labels={im.axis_labels or vm.get('axis_labels')}  "
                f"source_indices={im.source_indices or vm.get('source_indices')}"
                f"{seed_bounds}"
            )
        manifest_context = json.dumps(
            _slim_manifest(pkg.manifest),
            ensure_ascii=False,
            indent=2,
        )
        plan_text = (
            f"你是地球物理AI工作流规划器。以下是 geo_adapter 预处理好的 "
            f"当前只分析 {len(images)} 个物理视图。\n\n"
            f"=== 赛题任务 ===\n"
            f"sample_id: {pkg.sample_id}\n"
            f"target_classes_to_check: {pkg.target_classes}\n"
            f"注意：待检查类别不是目标存在性证据。\n"
            f"{task_hint}\n\n"
            f"=== 可用图像 (用 image_name 指定在哪个图上跑) ===\n"
            + "\n".join(img_lines)
            + "\n\n"
            f"=== manifest 权威上下文（不得臆造缺失曲线、坐标或范围）===\n"
            f"{manifest_context}\n\n"
            f"=== 测井结构化数值摘要（若非空则是数值权威来源）===\n"
            f"{json.dumps(pkg.numeric_summary, ensure_ascii=False, indent=2)}\n\n"
            f"=== 可用下游模型 ===\n"
            f"{downstream.available_models_desc()}\n\n"
            f"=== 任务→模型推荐 ===\n"
            f"{TASK_MODEL_MAP}\n\n"
            "请先给出 analysis_status 和 visual_evidence，再规划 workflow_steps。"
            "证据不足且不需要系统扫描时，workflow_steps 可以为空。每步必须指定:"
            "step(序号)、model(模型名)、image_name(要处理的图像名)、"
            "reason(选择原因)、instruction(模型参数)。\n"
            "不同图像适合不同模型，例如:\n"
            "- inline/crossline 剖面 → seismic_domain_model(断层) 或 "
            "horizon_tracker(层位)\n"
            "- 所有地震图像 → attribute_extractor + facies_classifier(沉积相)\n"
            "- well_log_panel → well_log_ml 或 well_log_analyzer(测井分析)\n\n"
            "硬性约束：image_name 必须来自上面的清单；horizon_tracker 的种子点必须落在"
            "对应 horizon_seed_bounds 内；测井分析只能使用 manifest 中 curves_present 列出的"
            "曲线，depth_range 必须落在 manifest 的 depth_range 内。测井数值只能引用上面的"
            "结构化摘要，不能从曲线 PNG 估读。图像像素尺寸不是地震道数或测井深度，不得据此"
            "虚构 CDP、样点或深度。\n\n"
            "仅输出JSON。"
        )
        # 注入 RAG 知识
        from pipeline.rag import retrieve_for_task

        views = [im.physical_view for im in images]
        rag_knowledge = retrieve_for_task(pkg.target_classes, views)
        if rag_knowledge:
            plan_text += rag_knowledge
        return plan_text

    def _run_closed_loop(
        self, steps, image_by_name, pkg, *, verify: bool, max_iter: int
    ):
        """Phase 2+3+4: 执行 -> 验证 -> 过滤假阳性 -> 仅重跑被调整的 step -> 收敛。

        返回 (step_results, verifications, total_elapsed):
          step_results: {step_num: [det,...]} 每条 det 已打 det_id 且已剔除假阳性。
          verifications: 每轮验证摘要（含被过滤的 det_id 列表）。
        """
        from schemas.output_schemas import WORKFLOW_VERIFICATION_SCHEMA

        from . import downstream

        step_results: dict[int, list[dict]] = {}
        verifications: list[dict] = []
        total_elapsed = 0.0
        run_queue: list[dict] | None = None  # None => 首轮跑全部 step

        for iteration in range(max_iter):
            current_steps = run_queue if run_queue is not None else steps
            self._log(
                f"\n[adapter] Phase 2: 执行下游模型 "
                f"(iter {iteration + 1}/{max_iter}, "
                f"{len(current_steps)} step)"
            )

            # ── Phase 2: 执行本轮 step（替换该 step 的历史结果）──
            for step in current_steps:
                step_num = step.get("step")
                model_name = step.get("model")
                image_name = step.get("image_name", "")
                im = image_by_name.get(image_name)
                if im is None and pkg is None:
                    im = next(iter(image_by_name.values()), None)
                if im is None:
                    self._log(
                        f"  ⚠️ Step{step_num}: unknown image_name '{image_name}', skip"
                    )
                    step_results[step_num] = []
                    continue
                model = downstream.get(model_name)
                if model is None:
                    self._log(f"  ⚠️ Step{step_num}: unknown '{model_name}', skip")
                    step_results[step_num] = []
                    continue
                self._log(f"  Step{step_num}: {model_name} on {im.name} ...")
                ctx = self._build_step_context(im, pkg) if pkg is not None else None
                try:
                    out = model.detect(
                        step.get("instruction") or {}, image=im.pil, context=ctx
                    )
                except Exception as e:
                    self._log(f"    ❌ {model_name} failed: {e}")
                    out = []
                w, h = im.pil.size
                tagged = []
                for i, d in enumerate(out):
                    t = tag_detection(
                        d,
                        step=step_num,
                        image_name=im.name,
                        model_name=model_name,
                        index=i,
                    )
                    ensure_bbox_norm(t, w, h)
                    tagged.append(t)
                step_results[step_num] = tagged
                self._log(f"    -> {len(tagged)} results")

            # 本轮刚跑出的检测（用于验证）
            round_dets = [
                d for s in current_steps for d in step_results.get(s.get("step"), [])
            ]
            n_dets = len(round_dets)
            if self.verbose:
                models_used = set(d.get("model", "?") for d in round_dets)
                self._log(f"  {n_dets} detections from models: {models_used}")
            if not verify or n_dets == 0:
                break

            # ── Phase 3: VLM 验证（喂回本轮检测，含 det_id）──
            self._log(
                f"\n[adapter] Phase 3: VLM 验证 (iter {iteration + 1}/{max_iter})"
            )
            detections_by_image: dict[str, list[dict]] = {}
            for detection in round_dets:
                detections_by_image.setdefault(
                    detection.get("image_name", ""), []
                ).append(detection)

            retry_data: dict | None = None
            for image_name, image_dets in detections_by_image.items():
                im = image_by_name.get(image_name)
                if im is None:
                    continue
                image_steps = [
                    step for step in steps if step.get("image_name") == image_name
                ]
                ver_text = (
                    f"当前只验证 image_name={image_name}。待检查类别不是目标存在性证据。\n\n"
                    f"该图工作流计划:\n"
                    f"{json.dumps(image_steps, ensure_ascii=False)[:3000]}\n\n"
                    f"该图下游检测结果（每条含 det_id 字段）:\n"
                    f"{json.dumps(image_dets, ensure_ascii=False)[:4000]}\n\n"
                    "请逐条对照当前这一张原图验证。verified[].result_id 必须填"
                    "检测的 det_id 原值；证据不足时不要把任务类别当作阳性证据。仅输出JSON。"
                )
                ver_resp = self.vlm.call_json(
                    VERIFICATION_PROMPT,
                    [getattr(im, "vlm_pil", im.pil)],
                    ver_text,
                    schema=WORKFLOW_VERIFICATION_SCHEMA,
                    max_new_tokens=4096,
                    temperature=0.0,
                )
                total_elapsed += ver_resp.elapsed_s
                if ver_resp.data is None:
                    self._log(f"  ❌ {image_name} 验证失败: {ver_resp.schema_errors}")
                    continue
                ver_data = ver_resp.data
                verified = ver_data.get("verified", [])
                real_n = sum(1 for item in verified if item.get("is_real"))
                fp_n = sum(1 for item in verified if not item.get("is_real"))

                drop_ids, dropped, review = match_false_positives(
                    ver_data,
                    image_dets,
                )
                if drop_ids:
                    for step in current_steps:
                        step_num = step.get("step")
                        step_results[step_num] = [
                            detection
                            for detection in step_results.get(step_num, [])
                            if detection.get("det_id") not in drop_ids
                        ]
                    self._log(
                        f"  🗑 {image_name} 过滤假阳性 {len(drop_ids)} 条: "
                        f"{sorted(drop_ids)}"
                    )
                if review:
                    self._log(
                        f"  ⚠️ {image_name} 存疑 {len(review)} 条(保留): "
                        f"{[item['det_id'] for item in review]}"
                    )

                verifications.append(
                    {
                        "iteration": iteration + 1,
                        "image_name": image_name,
                        "real": real_n,
                        "false_positive": fp_n,
                        "filtered": {"dropped": dropped, "review": review},
                        "filtered_ids": sorted(drop_ids),
                        "verification": ver_data,
                    }
                )
                self._log(
                    f"  ✅ {image_name}: {real_n} real, {fp_n} fp"
                    f"(过滤 {len(drop_ids)}), need_retry="
                    f"{ver_data.get('need_retry')} ({ver_resp.elapsed_s:.0f}s)"
                )
                if ver_data.get("need_retry") and retry_data is None:
                    retry_data = ver_data

            if retry_data is None:
                break

            # ── Phase 4: 应用首个有效重试指令，只重跑目标 step ──
            target = apply_retry(steps, retry_data.get("retry_instructions") or {})
            if target is None:
                break
            run_queue = [s for s in steps if s.get("step") == target]

        return step_results, verifications, total_elapsed

    @staticmethod
    def _build_step_context(image, pkg) -> dict | None:
        """从 RunPackage 构建下游模型的 context dict。

        如果 manifest 中有 views 且能定位到 npy 数组，则传入 array 数据。
        这样 seismic_domain_model、attribute_extractor 等可以直接在 raw 数据上计算。
        """
        ctx: dict = {}
        vm = pkg.view_meta(image.physical_view) or {}
        if vm:
            ctx["view_meta"] = vm
            # 优先加载适配器输出的处理后数组；路径相对于 run_dir。
            arrays_dir = pkg.run_dir / "arrays"
            arr_path = next(
                (
                    vm.get(key)
                    for key in (
                        "processed_array_path",
                        "raw_array_path",
                        "array_path",
                        "source_array",
                    )
                    if vm.get(key)
                ),
                None,
            )
            if arr_path:
                import numpy as np

                raw_path = Path(arr_path)
                candidates = (
                    [raw_path]
                    if raw_path.is_absolute()
                    else [
                        pkg.run_dir / raw_path,
                        arrays_dir / raw_path,
                        arrays_dir / raw_path.name,
                    ]
                )
                arr_file = next((p for p in candidates if p.is_file()), None)
                if arr_file is not None:
                    try:
                        arr = np.load(arr_file, allow_pickle=False)
                        # 适配器渲染 profile 时会转置；下游数组也保持同一坐标方向。
                        if image.physical_view in {
                            "inline",
                            "crossline",
                            "user_provided_2d_patch",
                        }:
                            arr = arr.T
                        ctx["array"] = arr
                    except (OSError, ValueError):
                        pass

        if image.physical_view == "well_log" or image.name == "well_log_panel":
            curves = Pipeline._load_well_curves(pkg.run_dir)
            if curves:
                ctx["curves"] = curves
        return ctx if ctx else None

    @staticmethod
    def _load_well_curves(run_dir: Path) -> dict | None:
        """无 pandas 依赖地读取适配器标准测井表，并补齐下游使用的 RT 别名。"""
        import numpy as np

        table = Path(run_dir) / "tables" / "well_logs_clean.csv"
        if not table.is_file():
            return None
        try:
            data = np.genfromtxt(
                table,
                delimiter=",",
                names=True,
                dtype=np.float32,
                encoding="utf-8",
            )
        except (OSError, ValueError):
            return None
        names = list(data.dtype.names or ())
        if not names:
            return None
        curves = {
            name: np.atleast_1d(np.asarray(data[name], dtype=np.float32))
            for name in names
        }
        depth_name = next(
            (name for name in names if name.upper() in {"DEPTH", "MD", "TVD", "TVDSS"}),
            names[0],
        )
        curves["depth"] = curves[depth_name]
        if "RT" not in curves and "RES_DEEP" in curves:
            curves["RT"] = curves["RES_DEEP"]
        return curves

    @staticmethod
    def _dedup_detections(
        detections_by_image: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        """对每张图的检测去重：同一 class_name + 相近 bbox → 取 max confidence。"""
        result: dict[str, list[dict]] = {}
        for img_name, dets in detections_by_image.items():
            if not dets:
                continue
            # 按 class_name 分组
            by_class: dict[str, list[dict]] = {}
            for d in dets:
                cname = d.get("class_name", "unknown")
                by_class.setdefault(cname, []).append(d)
            deduped: list[dict] = []
            for _cname, cdets in by_class.items():
                # 按 confidence 降序
                cdets.sort(key=lambda d: -d.get("confidence", 0))
                kept = []
                for d in cdets:
                    b1 = d.get("bbox_norm") or d.get("bbox_pixel")
                    if not b1:
                        kept.append(d)
                        continue
                    # 检查是否与已保留的重复（IoU > 0.5）
                    is_dup = False
                    for k in kept:
                        b2 = k.get("bbox_norm") or k.get("bbox_pixel")
                        if not b2:
                            continue
                        iou = bbox_iou(b1, b2)
                        if iou > 0.5:
                            is_dup = True
                            break
                    if not is_dup:
                        kept.append(d)
                deduped.extend(kept)
            result[img_name] = deduped
        return result

    # ============================================================
    # Fallback: 不经 geo_adapter，直读 SEG-Y
    # ============================================================

    def run_slice_for_tasks(self, image, geom, tasks: list[str], out_dir=None) -> dict:
        from . import tasks as tasks_mod
        from .exporter import export_annotated_png, export_json

        out: dict[str, dict] = {}
        for tname in tasks:
            spec = tasks_mod.get(tname)
            hint = tasks_mod.tasks_prompt_hint([tname])
            r = self.loop_agent.run(
                image,
                agent_name=f"{tname}_slice",
                task_hint=hint,
            )
            entry: dict = {"result": r}
            if out_dir is not None:
                entry["png"] = str(export_annotated_png(r, image, geom, spec, out_dir))
                entry["json"] = str(export_json(r, geom, tname, out_dir))
            out[tname] = entry
        return out

    def run_volume(
        self,
        volume,
        tasks: list[str],
        slice_axis: str = "inline",
        slice_stride: int = 5,
        out_dir=None,
    ) -> dict:
        import numpy as np

        from . import tasks as tasks_mod
        from .exporter import (
            build_slice_mask,
            export_annotated_png,
            export_json,
            export_volume_attribute,
            summary_report,
        )
        from .io.render import render_slice
        from .io.segy import extract_inline_slice, extract_xline_slice

        if slice_axis == "inline":
            n_slices = volume.cube.shape[0]
            coord_arr = volume.inlines
            other_arr = volume.xlines
            extract_fn = extract_inline_slice
            axis_x_name = "crossline"
        elif slice_axis == "crossline":
            n_slices = volume.cube.shape[1]
            coord_arr = volume.xlines
            other_arr = volume.inlines
            extract_fn = extract_xline_slice
            axis_x_name = "inline"
        else:
            raise ValueError(f"slice_axis must be inline|crossline, got {slice_axis}")

        picked = list(range(0, n_slices, slice_stride))
        report: dict = {
            "volume_meta": volume.to_meta(),
            "slice_axis": slice_axis,
            "picked_indices": picked,
            "tasks": {},
        }

        for tname in tasks:
            spec = tasks_mod.get(tname)
            hint = tasks_mod.tasks_prompt_hint([tname])
            per_slice_masks: dict[int, np.ndarray] = {}
            per_slice_report: list = []

            for s_idx in picked:
                arr2d = extract_fn(volume, s_idx)
                img, geom = render_slice(
                    arr2d,
                    x_min=float(other_arr.min()),
                    x_max=float(other_arr.max()),
                    y_top=0.0,
                    y_bottom=float(volume.time_axis_ms.max()),
                    axis_x_name=axis_x_name,
                    axis_y_name="time_ms",
                    slice_kind=slice_axis,
                    slice_index=int(coord_arr[s_idx]),
                    title=f"{slice_axis} {int(coord_arr[s_idx])} — {tname}",
                )
                r = self.loop_agent.run(
                    img,
                    agent_name=f"{tname}_{slice_axis}{int(coord_arr[s_idx])}",
                    task_hint=hint,
                )
                per_slice_masks[s_idx] = build_slice_mask(r, geom, arr2d.shape, spec)
                slice_entry = {
                    "slice_index": int(coord_arr[s_idx]),
                    "geometry": geom.to_dict(),
                    "n_detections": len(r.results),
                    "ok": r.ok,
                }
                if out_dir is not None:
                    slice_entry["png"] = str(
                        export_annotated_png(r, img, geom, spec, out_dir)
                    )
                    slice_entry["json"] = str(export_json(r, geom, tname, out_dir))
                per_slice_report.append(slice_entry)

            attr_path = None
            if out_dir is not None and slice_axis == "inline":
                try:
                    attr_path = str(
                        export_volume_attribute(
                            volume,
                            per_slice_masks,
                            spec,
                            out_dir,
                            slice_axis=slice_axis,
                        )
                    )
                except Exception as e:
                    self._logger.warning(f"  [export_volume_attribute failed: {e}]")
            report["tasks"][tname] = {
                "slices": per_slice_report,
                "attribute_sgy": attr_path,
            }

        if out_dir is not None:
            summary_report(report, out_dir)
        return report

    # ============================================================
    # 4-Agent 语义解释（非赛题独立能力）
    # ============================================================

    def run_seismic(self, image, task_hint: str | None = None) -> AgentResult:
        return self.loop_agent.run(image, agent_name="seismic", task_hint=task_hint)

    def run_log(self, image, task_hint: str | None = None) -> AgentResult:
        return self.loop_agent.run(image, agent_name="log", task_hint=task_hint)

    def run_fusion(
        self, image, time_depth_pairs: list | None = None, well_info: dict | None = None
    ) -> AgentResult:
        parts = []
        if well_info:
            parts.append(f"井信息: {json.dumps(well_info, ensure_ascii=False)}")
        if time_depth_pairs:
            parts.append(f"时深关系: {time_depth_pairs}")
        parts.append("分析井震对比图，仅输出JSON。")
        return self.fusion_agent.run([image], "\n".join(parts))

    def run_prospect(
        self,
        seismic: AgentResult | None,
        log: AgentResult | None,
        fusion: AgentResult | None = None,
        extra_image=None,
    ) -> AgentResult:
        context = self._prospect_context(seismic, log, fusion)
        images = [extra_image] if extra_image is not None else []
        return self.prospect_agent.run(images, context)

    def run_all(
        self,
        seismic_image=None,
        log_image=None,
        fusion_image=None,
        time_depth_pairs: list | None = None,
        well_info: dict | None = None,
        prospect_image=None,
    ) -> PipelineOutput:
        out = PipelineOutput()
        if seismic_image is not None:
            out.seismic = self.run_seismic(seismic_image)
        if log_image is not None:
            out.log = self.run_log(log_image)
        if fusion_image is not None:
            out.fusion = self.run_fusion(fusion_image, time_depth_pairs, well_info)
        if out.seismic or out.log or out.fusion:
            out.prospect = self.run_prospect(
                out.seismic,
                out.log,
                out.fusion,
                extra_image=prospect_image,
            )
        out.meta = {
            "seismic_ok": bool(out.seismic and out.seismic.ok),
            "log_ok": bool(out.log and out.log.ok),
            "fusion_ok": bool(out.fusion and out.fusion.ok),
            "prospect_ok": bool(out.prospect and out.prospect.ok),
        }
        return out

    @staticmethod
    def _prospect_context(
        seismic: AgentResult | None, log: AgentResult | None, fusion: AgentResult | None
    ) -> str:
        parts = ["请基于以下前序 Agent 输出评价勘探目标，仅输出JSON。\n"]
        if seismic and seismic.plan:
            parts.append(
                f"[Seismic Scene] {seismic.plan.get('scene_understanding', '')}"
            )
            if seismic.results:
                parts.append(
                    f"[Seismic Results] "
                    f"{json.dumps(seismic.results, ensure_ascii=False)[:1500]}"
                )
        if log and log.plan:
            parts.append(f"[Log Scene] {log.plan.get('scene_understanding', '')}")
            if log.results:
                parts.append(
                    f"[Log Results] "
                    f"{json.dumps(log.results, ensure_ascii=False)[:1500]}"
                )
        if fusion and fusion.output:
            parts.append(
                f"[Fusion] {json.dumps(fusion.output, ensure_ascii=False)[:1500]}"
            )
        return "\n".join(parts)
