"""Pipeline 编排：统一闭环——VLM自主选模型 → 执行 → 验证 → 迭代 → 输出。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from schemas.output_schemas import (
    FUSION_OUTPUT_SCHEMA, PROSPECT_OUTPUT_SCHEMA,
)

from .agents import AgentResult, LoopAgent, SingleShotAgent
from .prompts import (
    prospect_evaluation_prompt, seismic_interp_prompt,
    log_analysis_prompt, well_seismic_fusion_prompt,
    VERIFICATION_PROMPT, TASK_MODEL_MAP, workflow_planning_prompt,
)
from .vlm import VLMClient


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """两个 [x1,y1,x2,y2] bbox 的 IoU。"""
    xo = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    yo = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = xo * yo
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


@dataclass
class PipelineOutput:
    seismic: AgentResult | None = None
    log: AgentResult | None = None
    fusion: AgentResult | None = None
    prospect: AgentResult | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seismic":  self.seismic.to_dict() if self.seismic else None,
            "log":      self.log.to_dict() if self.log else None,
            "fusion":   self.fusion.to_dict() if self.fusion else None,
            "prospect": self.prospect.to_dict() if self.prospect else None,
            "meta":     self.meta,
        }


class Pipeline:
    """统一编排。赛题主流程 + Fallback SEG-Y + 4-Agent 语义解释。"""

    def __init__(self, vlm: VLMClient | None = None, verbose: bool = True):
        self.vlm = vlm or VLMClient()
        self.verbose = verbose
        self.loop_agent = LoopAgent(self.vlm, verbose=verbose)
        self.fusion_agent = SingleShotAgent(
            self.vlm, "well_seismic_fusion",
            well_seismic_fusion_prompt(), FUSION_OUTPUT_SCHEMA, verbose=verbose,
        )
        self.prospect_agent = SingleShotAgent(
            self.vlm, "prospect_evaluation",
            prospect_evaluation_prompt(), PROSPECT_OUTPUT_SCHEMA, verbose=verbose,
        )

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run_from_adapter(self, run_dir, out_dir=None,
                         verify: bool = True,
                         max_iterations: int = 3,
                         yolo_conf: float = 0.25) -> dict:
        """赛题主入口：吃 geo_adapter 的 runs/<sample_id>/ 目录。

        统一闭环:
          Phase 1: VLM 看到全部图像 + 任务描述 + 可用模型清单
                   → 自主规划 workflow_steps（选模型+参数+目标图像）
          Phase 2: 逐个执行 workflow_steps（调用对应的下游模型）
          Phase 3: VLM 验证——把下游结果+原图喂回 VLM，逐条判断真伪
          Phase 4: 如果 need_retry → 调整参数回到 Phase 2（最多 max_iterations 轮）
          Phase 5: 收敛后 → bbox 坐标反变换 → 3D 属性 SEG-Y + 标注 PNG
        """
        import numpy as np

        from . import downstream, tasks as tasks_mod
        from . import exporter as exporter_mod
        from .adapter import PackageImage, build_vlm_user_text, load_run
        from .exporter import (
            aggregate_adapter_detections, export_annotated_png,
            summary_report,
        )
        from .io.segy import (
            SegyVolume, write_attribute_segy, write_attribute_segy_like,
        )
        from schemas.output_schemas import (
            WORKFLOW_PLAN_SCHEMA, WORKFLOW_VERIFICATION_SCHEMA,
        )

        pkg = load_run(run_dir)
        if not pkg.images:
            raise RuntimeError(f"no model images found under {pkg.run_dir}/assets/*")

        out_dir = Path(out_dir) if out_dir else pkg.run_dir / "model_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        image_by_name = {im.name: im for im in pkg.images}
        self._log(f"[adapter] loaded {pkg.sample_id}: "
                  f"{len(pkg.images)} images, "
                  f"target_classes={pkg.target_classes}")

        # ── Phase 1: VLM 工作流规划 ──────────────────────────────────
        # 给 VLM 看图 + 任务描述 + 全部 8 个下游模型清单 → 让它自己选
        hint = tasks_mod.hint_for_target_classes(pkg.target_classes)
        plan_user_text = self._build_competition_plan_text(
            pkg, hint, image_by_name,
        )
        self._log(f"\n[adapter] Phase 1: VLM 工作流规划 "
                  f"(可用模型: {len(downstream.available_names())})")

        plan_resp = self.vlm.call_json(
            workflow_planning_prompt(),
            [im.pil for im in pkg.images],
            plan_user_text,
            schema=WORKFLOW_PLAN_SCHEMA,
            max_new_tokens=6144,
            temperature=0.1,
        )
        if plan_resp.data is None or not plan_resp.schema_valid:
            (out_dir / "vlm_plan_failed.txt").write_text(
                plan_resp.text, encoding="utf-8")
            report = {
                "sample_id": pkg.sample_id, "ok": False,
                "phase": "vlm_planning_failed",
                "errors": plan_resp.schema_errors,
            }
            summary_report(report, out_dir)
            return report

        plan = plan_resp.data
        steps = plan.get("workflow_steps", [])
        steps, plan_adjustments = self._sanitize_competition_steps(pkg, steps)
        if plan_adjustments:
            plan["workflow_steps"] = steps
            plan["execution_adjustments"] = plan_adjustments
        (out_dir / "vlm_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        self._log(f"  ✅ VLM 规划 ({plan_resp.elapsed_s:.0f}s, "
                  f"attempts={plan_resp.attempts})")
        self._log(f"  Scene: {plan.get('scene_understanding', '')[:150]}")
        if self.verbose:
            for s in steps:
                img = s.get('image_name', 'auto')
                self._log(f"    Step{s.get('step')}: {s.get('model')} "
                          f"on {img} → {s.get('reason','')[:80]}")

        # ── Phase 2+3+4: 执行 → 验证 → 迭代 ──────────────────────────
        max_iter = min(plan.get("max_iterations", max_iterations), max_iterations)
        all_detections: dict[str, list[dict]] = {}
        verifications: list[dict] = []
        total_elapsed = plan_resp.elapsed_s

        for iteration in range(max_iter):
            self._log(f"\n[adapter] Phase 2: 执行下游模型 "
                      f"(iter {iteration+1}/{max_iter})")

            round_dets = self._execute_competition_steps(
                steps, image_by_name, pkg,
            )
            round_dets, candidate_limit = self._limit_fault_candidates(round_dets)
            if candidate_limit["dropped"]:
                message = (
                    f"iteration {iteration + 1}: retained top "
                    f"{candidate_limit['limit_per_image']} seismic-domain fault "
                    f"candidates per image ({candidate_limit['raw']} raw, "
                    f"{candidate_limit['kept']} kept, "
                    f"{candidate_limit['dropped']} dropped)"
                )
                plan_adjustments.append(message)
                self._log(f"  candidate limit: {message}")
            # Each retry replaces the previous attempt. Keeping all attempts would
            # re-introduce detections that the verifier explicitly rejected.
            all_detections = round_dets
            n_dets = sum(len(v) for v in round_dets.values())
            if self.verbose:
                models_used = set(
                    d.get("model", "?") for v in round_dets.values() for d in v
                )
                print(f"  {n_dets} detections from models: {models_used}")

            if not verify or n_dets == 0:
                break

            # Phase 3: VLM 验证
            self._log(f"\n[adapter] Phase 3: VLM 验证 "
                      f"(iter {iteration+1}/{max_iter})")
            verification_payload = self._summarize_for_verification(round_dets)
            ver_text = (
                f"原始工作流计划:\n{json.dumps(steps, ensure_ascii=False)[:3000]}\n\n"
                f"下游模型实际检测结果:\n"
                f"{json.dumps(verification_payload, ensure_ascii=False)[:12000]}\n\n"
                "请逐条对照原图验证每条检测。仅输出JSON。"
            )
            ver_resp = self.vlm.call_json(
                VERIFICATION_PROMPT,
                [im.pil for im in pkg.images],
                ver_text,
                schema=WORKFLOW_VERIFICATION_SCHEMA,
                max_new_tokens=4096,
                temperature=0.0,
            )
            total_elapsed += ver_resp.elapsed_s
            if ver_resp.data is None:
                self._log(f"  ❌ 验证失败: {ver_resp.schema_errors}")
                break
            ver_data = ver_resp.data
            verified = ver_data.get("verified", [])
            real_n = sum(1 for v in verified if v.get("is_real"))
            fp_n = sum(1 for v in verified if not v.get("is_real"))
            all_detections = self._filter_competition_detections(
                round_dets, verified,
            )
            verifications.append({
                "iteration": iteration + 1,
                "real": real_n, "false_positive": fp_n,
                "verification": ver_data,
            })
            self._log(f"  ✅ 验证: {real_n} real, {fp_n} false positive, "
                      f"need_retry={ver_data.get('need_retry')} "
                      f"({ver_resp.elapsed_s:.0f}s)")

            if not ver_data.get("need_retry"):
                break
            # Phase 4: 应用重试指令
            retry_adjustments = self._apply_competition_retry(steps, ver_data)
            if retry_adjustments:
                plan_adjustments.extend(retry_adjustments)
                for message in retry_adjustments:
                    self._log(f"  retry guard: {message}")

        # ── Phase 5: 聚合输出 ────────────────────────────────────────
        self._log(f"\n[adapter] Phase 5: 聚合输出")

        # 5a: 归一化——各模型不同输出格式 → 统一 {class_name, bbox_norm, confidence}
        n_raw = sum(len(v) for v in all_detections.values())
        normalized = exporter_mod.normalize_detection_format(
            all_detections, pkg.manifest, image_by_name,
        )
        # 5b: 去重——归一化后 class_name 统一，去重准确
        n_before_dedup = sum(len(v) for v in normalized.values())
        normalized = self._dedup_detections(normalized)
        n_total_dets = sum(len(v) for v in normalized.values())
        if self.verbose and n_before_dedup != n_total_dets:
            print(f"  dedup: {n_before_dedup} → {n_total_dets} detections")
        self._log(f"  normalized+dedup: {n_raw} raw → {n_total_dets} detections")

        # 5c: 坐标反变换 → 3D 属性 SEG-Y
        seismic_meta = pkg.manifest.get("seismic") or {}
        shape = seismic_meta.get("shape")
        attr_sgy_paths: dict[str, str] = {}
        export_warnings: list[str] = []
        if shape and len(shape) == 3:
            per_class = aggregate_adapter_detections(
                normalized, pkg.manifest, tuple(shape),
            )
            source_path = seismic_meta.get("source_path")
            source_segy = None
            if source_path:
                source_candidate = Path(source_path)
                if not source_candidate.is_absolute():
                    source_candidate = pkg.run_dir / source_candidate
                if (source_candidate.is_file()
                        and source_candidate.suffix.lower() in (".sgy", ".segy")):
                    source_segy = source_candidate
            qc_meta = (seismic_meta.get("qc") or {}).get("metadata") or {}
            interval_ms = float(qc_meta.get("sample_interval_ms") or 1.0)
            for cls, cube in per_class.items():
                out_path = out_dir / f"{cls.replace(' ', '_')}_attribute.sgy"
                try:
                    if source_segy is not None:
                        write_attribute_segy_like(
                            str(source_segy), cube, str(out_path),
                        )
                    else:
                        fake_vol = SegyVolume(
                            cube=np.zeros(shape, dtype=np.float32),
                            inlines=np.arange(shape[0], dtype=np.int32),
                            xlines=np.arange(shape[1], dtype=np.int32),
                            sample_interval_ms=interval_ms,
                            n_samples=shape[2],
                        )
                        write_attribute_segy(fake_vol, cube, str(out_path))
                        export_warnings.append(
                            f"{cls}: reference SEG-Y unavailable; wrote synthetic geometry"
                        )
                    attr_sgy_paths[cls] = str(out_path)
                except Exception as e:
                    print(f"  [SEG-Y write failed for {cls}: {e}]")

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
            has_facies_output = any(
                d.get("model") == "facies_classifier"
                or str(d.get("class_name", "")).startswith("cluster_")
                for d in dets
            )
            default_task = tasks_mod.get("facies" if has_facies_output else "fault")
            first_task = next((t for t in class_to_task.values() if t), default_task)
            # normalized dets 没有 bbox_pixel，用 image 尺寸从 bbox_norm 反算
            w, h = im.pil.size
            results_for_png = []
            for d in dets:
                bn = d.get("bbox_norm")
                if bn and len(bn) == 4:
                    px = [bn[0]*w, bn[1]*h, bn[2]*w, bn[3]*h]
                else:
                    continue
                results_for_png.append({**d, "bbox_pixel": px})
            if not results_for_png:
                continue
            fake = AgentResult(agent=im.name, ok=True, results=results_for_png)
            p = export_annotated_png(fake, im.pil, None, first_task, out_dir)
            png_paths.append(str(p))

        # ── 汇总报告 ──────────────────────────────────────────────────
        models_used = sorted(set(
            d.get("model", "?") for v in all_detections.values() for d in v
        ))
        verification_coverage = self._fault_verification_coverage(all_detections)
        # Record retry guards and the exact parameters that were finally run.
        plan["workflow_steps"] = steps
        if plan_adjustments:
            plan["execution_adjustments"] = plan_adjustments
        (out_dir / "vlm_plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8",
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
                "verification_coverage": verification_coverage,
                "detections_by_image": exporter_mod.summarize_detections_for_json(
                    all_detections,
                ),
            },
            "outputs": {
                "annotated_pngs": png_paths,
                "attribute_sgy": attr_sgy_paths,
                "vlm_plan_json": str(out_dir / "vlm_plan.json"),
            },
            "warnings": export_warnings,
        }
        if plan_adjustments:
            report["warnings"].extend(plan_adjustments)
        summary_report(report, out_dir)
        self._log(f"[adapter] wrote {out_dir} "
                  f"(models={models_used}, dets={n_total_dets})")
        return report

    # ── 竞赛流程辅助方法 ────────────────────────────────────────────────

    def _build_competition_plan_text(self, pkg, task_hint: str,
                                     image_by_name: dict) -> str:
        """构建给 VLM 的竞赛规划 user text。
        包含: 任务描述 + 图像清单 + 推荐模型 + 约束说明。
        """
        from . import downstream
        img_lines = []
        for i, im in enumerate(pkg.images, 1):
            view = im.physical_view
            view_meta = pkg.view_meta(view) or {}
            array_shape = view_meta.get("array_shape")
            axis_labels = view_meta.get("axis_labels") or []
            bounds = ""
            if isinstance(array_shape, (list, tuple)) and len(array_shape) == 2:
                sample_axis = next(
                    (index for index, label in enumerate(axis_labels)
                     if "sample" in str(label).lower()),
                    None,
                )
                if sample_axis is not None:
                    trace_axis = 1 - sample_axis
                    bounds = (
                        f" raw_shape={list(array_shape)} axes={list(axis_labels)}"
                        f" valid_trace_idx=0..{int(array_shape[trace_axis]) - 1}"
                        f" valid_sample_idx=0..{int(array_shape[sample_axis]) - 1}"
                    )
            img_lines.append(
                f"  {i}. image_name=\"{im.name}\"  view={view}  "
                f"rendered_size={im.pil.size}{bounds}"
            )
        well_meta = pkg.manifest.get("well_logs") or {}
        curve_meta = well_meta.get("curves") or {}
        available_curves = sorted(
            name for name, meta in curve_meta.items() if meta.get("available")
        )
        missing_curves = sorted(
            name for name, meta in curve_meta.items() if not meta.get("available")
        )
        alignment = pkg.manifest.get("alignment") or {}
        time_depth = pkg.manifest.get("time_depth_relation") or {}
        evidence_context = {
            "seismic_shape": (pkg.manifest.get("seismic") or {}).get("shape"),
            "seismic_domain": (pkg.manifest.get("seismic") or {}).get("domain"),
            "full_3d_context_available": bool(
                (pkg.manifest.get("seismic") or {}).get("volume_array_path")
            ),
            "available_curves": available_curves,
            "missing_curves": missing_curves,
            "well_depth_range": well_meta.get("depth_range"),
            "well_depth_unit": (well_meta.get("depth_axis") or {}).get("unit"),
            "time_depth_available": bool(time_depth.get("available")),
            "time_depth_confidence": time_depth.get("confidence", "none"),
            "fusion_permission": alignment.get("fusion_permission"),
        }
        plan_text = (
            f"你是地球物理AI工作流规划器。以下是 geo_adapter 预处理好的 "
            f"赛题数据，共 {len(pkg.images)} 张图像。\n\n"
            f"=== 赛题任务 ===\n"
            f"sample_id: {pkg.sample_id}\n"
            f"target_classes: {pkg.target_classes}\n"
            f"{task_hint}\n\n"
            f"=== 可用证据约束 ===\n"
            f"{json.dumps(evidence_context, ensure_ascii=False, indent=2)}\n"
            "只能使用 available_curves 中存在的曲线。缺失曲线不得用常数替代，"
            "不得据此生成岩性或流体结论。只有 fusion_permission 允许且时深关系"
            "可用时，才能把测井深度约束转换到地震时间域。只有 "
            "full_3d_context_available=true 时才能选择 CIG 3D 模型。\n\n"
            f"=== 可用图像 (用 image_name 指定在哪个图上跑) ===\n"
            + "\n".join(img_lines) + "\n\n"
            f"=== 可用下游模型 ===\n"
            f"{downstream.available_models_desc()}\n\n"
            f"=== 任务→模型推荐 ===\n"
            f"{TASK_MODEL_MAP}\n\n"
            "请根据任务和图像，规划 workflow_steps。每步必须指定:"
            "step(序号)、model(模型名)、image_name(要处理的图像名)、"
            "reason(选择原因)、instruction(模型参数)。\n"
            "不同图像适合不同模型，例如:\n"
            "- inline/crossline 剖面 → seismic_domain_model(断层) 或 "
            "horizon_tracker(层位)\n"
            "- 所有地震图像 → attribute_extractor + facies_classifier(沉积相)\n"
            "- well_log_panel → well_log_analyzer 或 traditional_code(测井分析)\n"
            "禁止训练或微调模型，只能调用已注册的纯推理模块。\n\n"
            "仅输出JSON。"
        )
        # 注入 RAG 知识
        plan_text += (
            "\nIndex constraints are hard constraints: every horizon seed must "
            "stay inside the valid_trace_idx and valid_sample_idx ranges printed "
            "for its image. When both inline and crossline images exist and fault "
            "is a target, plan fault evidence on both orientations. Facies GMM "
            "outputs are attribute clusters, not named geological facies without "
            "independent geological evidence.\n"
        )
        from pipeline.rag import retrieve_for_task
        views = [im.physical_view for im in pkg.images]
        rag_knowledge = retrieve_for_task(pkg.target_classes, views)
        if rag_knowledge:
            plan_text += rag_knowledge
        return plan_text

    def _execute_competition_steps(self, steps: list[dict],
                                    image_by_name: dict,
                                    pkg) -> dict[str, list[dict]]:
        """执行 competition workflow_steps。返回 {image_name: [detections]}。

        与 LoopAgent._execute_steps 不同，这里每步可以指定不同的 image_name，
        且会把 context 里的 array 数据传给下游模型。
        """
        from . import downstream
        detections: dict[str, list[dict]] = {}

        for step in steps:
            model_name = step.get("model")
            instruction = step.get("instruction") or {}
            image_name = step.get("image_name", "")
            step_num = step.get("step", "?")

            # 解析目标图像
            im = image_by_name.get(image_name)
            if im is None:
                # fallback: 用第一张图
                im = next(iter(image_by_name.values()), None)
            if im is None:
                continue

            model = downstream.get(model_name)
            if model is None:
                self._log(f"  ⚠️ Step{step_num}: unknown '{model_name}', skip")
                continue

            self._log(f"  Step{step_num}: {model_name} on {im.name} ...")

            # 构建 context（传递原始数组数据给下游模型）
            ctx = self._build_step_context(im, pkg)

            try:
                out = model.detect(instruction, image=im.pil, context=ctx)
            except Exception as e:
                self._log(f"    ❌ {model_name} failed: {e}")
                continue

            self._log(f"    → {len(out)} results")
            if out:
                tagged = []
                for result_index, detection in enumerate(out):
                    if not isinstance(detection, dict):
                        continue
                    original_id = detection.get("id", result_index)
                    tagged.append({
                        **detection,
                        "id": f"{im.name}:step{step_num}:{original_id}",
                        "downstream_result_id": original_id,
                        "workflow_step": step_num,
                    })
                detections.setdefault(im.name, []).extend(tagged)

        return detections

    @staticmethod
    def _build_step_context(image, pkg) -> dict | None:
        """从 RunPackage 构建下游模型的 context dict。

        如果 manifest 中有 views 且能定位到 npy 数组，则传入 array 数据。
        这样 seismic_domain_model、attribute_extractor 等可以直接在 raw 数据上计算。
        """
        from .context import build_downstream_context
        return build_downstream_context(image, pkg)

    @staticmethod
    def _summarize_for_verification(
        detections_by_image: dict[str, list[dict]],
        max_per_image_model: int = 8,
    ) -> dict[str, list[dict]]:
        """Give every image/model a fair share of the verifier context window."""
        from . import exporter as exporter_mod
        summarized = exporter_mod.summarize_detections_for_json(detections_by_image)
        result: dict[str, list[dict]] = {}
        for image_name, detections in summarized.items():
            by_model: dict[str, list[dict]] = {}
            for detection in detections:
                by_model.setdefault(
                    str(detection.get("model", "unknown")), []
                ).append(detection)
            selected: list[dict] = []
            for model_name in sorted(by_model):
                ranked = sorted(
                    by_model[model_name],
                    key=lambda item: float(item.get("confidence", 0.0)),
                    reverse=True,
                )
                selected.extend(ranked[:max_per_image_model])
            result[image_name] = selected
        return result

    @staticmethod
    def _limit_fault_candidates(
        detections_by_image: dict[str, list[dict]],
        max_per_image: int = 8,
    ) -> tuple[dict[str, list[dict]], dict]:
        """Keep a bounded, high-value set of fault candidate objects per view."""
        result: dict[str, list[dict]] = {}
        raw = kept_total = 0
        for image_name, detections in detections_by_image.items():
            fault_candidates = []
            other = []
            for detection in detections:
                is_domain_fault = (
                    detection.get("model") == "seismic_domain_model"
                    and str(detection.get("class_name", "")).lower()
                    in {"fault", "fault_candidate"}
                )
                (fault_candidates if is_domain_fault else other).append(detection)
            raw += len(fault_candidates)
            ranked = sorted(
                fault_candidates,
                key=lambda item: (
                    float(item.get("confidence", 0.0)),
                    float(item.get("area_pixels", 0.0)),
                    float(item.get("aspect_ratio", 0.0)),
                ),
                reverse=True,
            )
            selected = ranked[:max(0, int(max_per_image))]
            kept_total += len(selected)
            result[image_name] = other + selected
        return result, {
            "limit_per_image": max(0, int(max_per_image)),
            "raw": raw,
            "kept": kept_total,
            "dropped": raw - kept_total,
        }

    @staticmethod
    def _fault_verification_coverage(
        detections_by_image: dict[str, list[dict]],
    ) -> dict:
        """Report reviewed fault evidence without treating unreviewed as verified."""
        counts = {
            "total_candidates": 0,
            "reviewed": 0,
            "verified": 0,
            "rejected": 0,
            "unreviewed": 0,
        }
        for detections in detections_by_image.values():
            for detection in detections:
                is_domain_fault = (
                    detection.get("model") == "seismic_domain_model"
                    and str(
                        detection.get("original_class_name")
                        or detection.get("class_name", "")
                    ).lower() in {"fault", "fault_candidate"}
                )
                if not is_domain_fault:
                    continue
                counts["total_candidates"] += 1
                status = detection.get("verification_status")
                if status == "verified":
                    counts["verified"] += 1
                    counts["reviewed"] += 1
                elif status == "rejected_by_vlm":
                    counts["rejected"] += 1
                    counts["reviewed"] += 1
                else:
                    counts["unreviewed"] += 1
        total = counts["total_candidates"]
        counts["review_fraction"] = (
            round(counts["reviewed"] / total, 6) if total else None
        )
        return counts

    @staticmethod
    def _filter_competition_detections(
        detections_by_image: dict[str, list[dict]],
        verified: list[dict],
    ) -> dict[str, list[dict]]:
        """Apply VLM rejection only where the verifier saw enough evidence.

        The verifier receives JSON summaries and rendered images, not the dense
        probability arrays themselves.  Its opinion remains useful for retry
        guidance, but it must not destructively remove a domain-model pixel map
        that it could not inspect.  Ordinary bbox detections remain filterable.
        """
        decisions = {
            str(item.get("result_id")): bool(item.get("is_real"))
            for item in verified
            if item.get("result_id") is not None
        }
        filtered: dict[str, list[dict]] = {}
        for image_name, detections in detections_by_image.items():
            kept: list[dict] = []
            for detection in detections:
                result_id = str(detection.get("id"))
                decision = decisions.get(result_id)
                dense_domain_map = (
                    detection.get("model") == "seismic_domain_model"
                    and detection.get("_probability_map") is not None
                )
                if decision is False and not dense_domain_map:
                    continue

                updated = dict(detection)
                if decision is True:
                    updated["verification_status"] = "verified"
                elif dense_domain_map:
                    # Numerical evidence survives, but rejected/unreviewed maps
                    # must not be presented as verified geological faults.
                    updated["verification_status"] = (
                        "rejected_by_vlm" if decision is False else "unreviewed"
                    )
                    if str(updated.get("class_name", "")).lower() == "fault":
                        updated["original_class_name"] = "fault"
                        updated["class_name"] = "fault_candidate"
                kept.append(updated)
            filtered[image_name] = kept
        return filtered

    @staticmethod
    def _sanitize_competition_steps(pkg, steps: list[dict]) -> tuple[list[dict], list[str]]:
        """Constrain VLM plans to the data actually present in the run package."""
        seismic = pkg.manifest.get("seismic") or {}
        has_full_3d = bool(seismic.get("volume_array_path"))
        well_logs = pkg.manifest.get("well_logs") or {}
        available_depth_range = well_logs.get("depth_range") or []

        image_by_name = {image.name: image for image in pkg.images}
        kept: list[dict] = []
        adjustments: list[str] = []
        for step in steps:
            model = str(step.get("model", ""))
            if not has_full_3d and model in {"cig_fault", "cig_channel"}:
                adjustments.append(
                    f"step {step.get('step', '?')} ({model}) disabled: "
                    "run package has no full 3D volume/tiles"
                )
                continue

            if model == "well_log_analyzer" and len(available_depth_range) >= 2:
                instruction = dict(step.get("instruction") or {})
                requested = instruction.get("depth_range")
                if isinstance(requested, dict):
                    try:
                        data_top, data_bottom = sorted(
                            (float(available_depth_range[0]), float(available_depth_range[1]))
                        )
                        req_top, req_bottom = sorted(
                            (float(requested["top_m"]), float(requested["bottom_m"]))
                        )
                        top = max(data_top, req_top)
                        bottom = min(data_bottom, req_bottom)
                        if bottom <= top:
                            top, bottom = data_top, data_bottom
                        if top != req_top or bottom != req_bottom:
                            instruction["depth_range"] = {
                                "top_m": top,
                                "bottom_m": bottom,
                            }
                            step = {**step, "instruction": instruction}
                            adjustments.append(
                                f"step {step.get('step', '?')} ({model}) depth_range "
                                f"adjusted from [{req_top}, {req_bottom}] to "
                                f"[{top}, {bottom}] using manifest bounds"
                            )
                    except (KeyError, TypeError, ValueError):
                        adjustments.append(
                            f"step {step.get('step', '?')} ({model}) ignored invalid "
                            "depth_range; downstream will use manifest bounds"
                        )
            if model == "horizon_tracker":
                instruction = dict(step.get("instruction") or {})
                seeds = instruction.get("seed_points") or []
                image = image_by_name.get(str(step.get("image_name", "")))
                view_meta = pkg.view_meta(image.physical_view) if image else None
                shape = (view_meta or {}).get("array_shape") or []
                labels = (view_meta or {}).get("axis_labels") or []
                sample_axis = next(
                    (index for index, label in enumerate(labels)
                     if "sample" in str(label).lower()),
                    None,
                )
                if len(shape) == 2 and sample_axis is not None:
                    trace_axis = 1 - sample_axis
                    max_trace = int(shape[trace_axis]) - 1
                    max_sample = int(shape[sample_axis]) - 1
                    sanitized_seeds = []
                    for seed_index, seed in enumerate(seeds):
                        if not isinstance(seed, dict):
                            continue
                        original_trace = int(seed.get("trace_idx", seed.get("cdp", 0)))
                        original_sample = int(seed.get("sample_idx", max_sample // 2))
                        trace = max(0, min(max_trace, original_trace))
                        sample = max(0, min(max_sample, original_sample))
                        sanitized_seeds.append({
                            **seed,
                            "trace_idx": trace,
                            "sample_idx": sample,
                        })
                        if trace != original_trace or sample != original_sample:
                            adjustments.append(
                                f"step {step.get('step', '?')} (horizon_tracker) "
                                f"seed {seed_index} adjusted from "
                                f"[{original_trace}, {original_sample}] to "
                                f"[{trace}, {sample}] for {image.name} bounds "
                                f"trace=0..{max_trace}, sample=0..{max_sample}"
                            )
                    instruction["seed_points"] = sanitized_seeds
                    step = {**step, "instruction": instruction}
            kept.append(step)

        # Always collect fault evidence from both available vertical directions.
        directional_images = {
            image.physical_view: image.name
            for image in pkg.images
            if image.physical_view in {"inline", "crossline"}
        }
        if set(directional_images) == {"inline", "crossline"}:
            fault_steps = [
                step for step in kept
                if step.get("model") == "seismic_domain_model"
                and (step.get("instruction") or {}).get("task") == "fault_detection"
            ]
            planned_views = {
                image_by_name[str(step.get("image_name", ""))].physical_view
                for step in fault_steps
                if str(step.get("image_name", "")) in image_by_name
            }
            missing_views = {"inline", "crossline"} - planned_views
            if fault_steps and missing_views:
                numeric_steps = [
                    int(step.get("step", 0)) for step in kept
                    if str(step.get("step", "")).isdigit()
                ]
                next_step = max(numeric_steps or [0]) + 1
                template = fault_steps[0]
                for view in sorted(missing_views):
                    instruction = dict(template.get("instruction") or {})
                    instruction.pop("regions_of_interest", None)
                    mirrored = {
                        **template,
                        "step": next_step,
                        "image_name": directional_images[view],
                        "instruction": instruction,
                        "reason": (
                            f"Directional consistency pass mirrored from step "
                            f"{template.get('step', '?')} onto {view}; global "
                            "evidence is used because ROI coordinates are view-specific."
                        ),
                    }
                    kept.append(mirrored)
                    adjustments.append(
                        f"step {next_step} (seismic_domain_model) added on "
                        f"{directional_images[view]} for inline/crossline fault consistency"
                    )
                    next_step += 1
        return kept, adjustments

    @staticmethod
    def _dedup_detections(
        detections_by_image: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        """对每张图的检测去重：同一 class_name + 相近 bbox → 取 max confidence。"""
        import numpy as np
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
            for cname, cdets in by_class.items():
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
                        iou = _bbox_iou(b1, b2)
                        if iou > 0.5:
                            is_dup = True
                            break
                    if not is_dup:
                        kept.append(d)
                deduped.extend(kept)
            result[img_name] = deduped
        return result

    @staticmethod
    def _apply_competition_retry(steps: list[dict], ver_data: dict) -> list[str]:
        """把 VLM 验证给出的 retry_instructions 应用到对应 step。"""
        retry = ver_data.get("retry_instructions") or {}
        target = retry.get("step")
        adjusted = retry.get("adjusted_params") or retry.get("adjusted_instruction")
        if target is None or not adjusted:
            return []
        if not isinstance(adjusted, dict):
            return []

        target_step = next((s for s in steps if s.get("step") == target), None)
        if target_step is None:
            return [f"retry target step {target} was not found; advice ignored"]

        verified = ver_data.get("verified") or []
        real_n = sum(1 for item in verified if item.get("is_real") is True)
        fp_n = sum(1 for item in verified if item.get("is_real") is False)
        target_instruction = target_step.get("instruction") or {}
        is_fault_retry = (
            target_step.get("model") == "seismic_domain_model"
            and target_instruction.get("task", "fault_detection") == "fault_detection"
        )

        if is_fault_retry and fp_n > real_n:
            messages = []
            for step in steps:
                instruction = step.get("instruction") or {}
                if not (
                    step.get("model") == "seismic_domain_model"
                    and instruction.get("task", "fault_detection") == "fault_detection"
                ):
                    continue
                old_threshold = float(instruction.get("confidence_threshold", 0.3))
                old_area = int(instruction.get("min_region_area_pixels", 100))
                try:
                    requested_threshold = float(
                        adjusted.get("confidence_threshold", old_threshold)
                    )
                except (TypeError, ValueError):
                    requested_threshold = old_threshold
                try:
                    requested_area = int(
                        adjusted.get("min_region_area_pixels", old_area)
                    )
                except (TypeError, ValueError):
                    requested_area = old_area
                # Keep view-specific ROI local to its original direction.
                step_adjusted = adjusted if step is target_step else {}
                guarded = {
                    **instruction,
                    **step_adjusted,
                    "confidence_threshold": max(
                        old_threshold, requested_threshold, 0.55,
                    ),
                    "min_region_area_pixels": max(
                        old_area, requested_area, 1000,
                    ),
                }
                step["instruction"] = guarded
                messages.append(
                    f"FP-dominant fault retry synchronized step {step.get('step')} "
                    f"({step.get('image_name')}): requested threshold/area "
                    f"{requested_threshold}/{requested_area}, applied "
                    f"{guarded['confidence_threshold']}/"
                    f"{guarded['min_region_area_pixels']}"
                )
            return messages

        target_step["instruction"] = {**target_instruction, **adjusted}
        return [f"retry advice applied to step {target}"]

    # ============================================================
    # Fallback: 不经 geo_adapter，直读 SEG-Y
    # ============================================================

    def run_slice_for_tasks(self, image, geom, tasks: list[str],
                            out_dir=None) -> dict:
        from . import tasks as tasks_mod
        from .exporter import build_slice_mask, export_annotated_png, export_json
        out: dict[str, dict] = {}
        for tname in tasks:
            spec = tasks_mod.get(tname)
            hint = tasks_mod.tasks_prompt_hint([tname])
            r = self.loop_agent.run(
                image, agent_name=f"{tname}_slice", task_hint=hint,
            )
            entry: dict = {"result": r}
            if out_dir is not None:
                entry["png"] = str(export_annotated_png(r, image, geom, spec, out_dir))
                entry["json"] = str(export_json(r, geom, tname, out_dir))
            out[tname] = entry
        return out

    def run_volume(self, volume, tasks: list[str],
                   slice_axis: str = "inline",
                   slice_stride: int = 5,
                   out_dir=None) -> dict:
        import numpy as np
        from . import tasks as tasks_mod
        from .exporter import (
            build_slice_mask, export_annotated_png, export_json,
            export_volume_attribute, summary_report,
        )
        from .io.render import render_slice
        from .io.segy import extract_inline_slice, extract_xline_slice

        if slice_axis == "inline":
            n_slices = volume.cube.shape[0]
            coord_arr = volume.inlines; other_arr = volume.xlines
            extract_fn = extract_inline_slice; axis_x_name = "crossline"
        elif slice_axis == "crossline":
            n_slices = volume.cube.shape[1]
            coord_arr = volume.xlines; other_arr = volume.inlines
            extract_fn = extract_xline_slice; axis_x_name = "inline"
        else:
            raise ValueError(f"slice_axis must be inline|crossline, got {slice_axis}")

        picked = list(range(0, n_slices, slice_stride))
        report: dict = {"volume_meta": volume.to_meta(),
                        "slice_axis": slice_axis,
                        "picked_indices": picked, "tasks": {}}

        for tname in tasks:
            spec = tasks_mod.get(tname)
            hint = tasks_mod.tasks_prompt_hint([tname])
            per_slice_masks: dict[int, np.ndarray] = {}
            per_slice_report: list = []

            for s_idx in picked:
                arr2d = extract_fn(volume, s_idx)
                img, geom = render_slice(
                    arr2d,
                    x_min=float(other_arr.min()), x_max=float(other_arr.max()),
                    y_top=0.0, y_bottom=float(volume.time_axis_ms.max()),
                    axis_x_name=axis_x_name, axis_y_name="time_ms",
                    slice_kind=slice_axis, slice_index=int(coord_arr[s_idx]),
                    title=f"{slice_axis} {int(coord_arr[s_idx])} — {tname}",
                )
                r = self.loop_agent.run(
                    img, agent_name=f"{tname}_{slice_axis}{int(coord_arr[s_idx])}",
                    task_hint=hint,
                )
                per_slice_masks[s_idx] = build_slice_mask(
                    r, geom, arr2d.shape, spec)
                slice_entry = {
                    "slice_index": int(coord_arr[s_idx]),
                    "geometry": geom.to_dict(),
                    "n_detections": len(r.results), "ok": r.ok,
                }
                if out_dir is not None:
                    slice_entry["png"] = str(
                        export_annotated_png(r, img, geom, spec, out_dir))
                    slice_entry["json"] = str(
                        export_json(r, geom, tname, out_dir))
                per_slice_report.append(slice_entry)

            attr_path = None
            if out_dir is not None and slice_axis == "inline":
                try:
                    attr_path = str(export_volume_attribute(
                        volume, per_slice_masks, spec, out_dir,
                        slice_axis=slice_axis))
                except Exception as e:
                    print(f"  [export_volume_attribute failed: {e}]")
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
        return self.loop_agent.run(image, agent_name="seismic",
                                   task_hint=task_hint)

    def run_log(self, image, task_hint: str | None = None) -> AgentResult:
        return self.loop_agent.run(image, agent_name="log", task_hint=task_hint)

    def run_fusion(self, image, time_depth_pairs: list | None = None,
                   well_info: dict | None = None) -> AgentResult:
        parts = []
        if well_info:
            parts.append(f"井信息: {json.dumps(well_info, ensure_ascii=False)}")
        if time_depth_pairs:
            parts.append(f"时深关系: {time_depth_pairs}")
        parts.append("分析井震对比图，仅输出JSON。")
        return self.fusion_agent.run([image], "\n".join(parts))

    def run_prospect(self, seismic: AgentResult | None,
                     log: AgentResult | None,
                     fusion: AgentResult | None = None,
                     extra_image=None) -> AgentResult:
        context = self._prospect_context(seismic, log, fusion)
        images = [extra_image] if extra_image is not None else []
        return self.prospect_agent.run(images, context)

    def run_all(self, seismic_image=None, log_image=None,
                fusion_image=None, time_depth_pairs: list | None = None,
                well_info: dict | None = None,
                prospect_image=None) -> PipelineOutput:
        out = PipelineOutput()
        if seismic_image is not None:
            out.seismic = self.run_seismic(seismic_image)
        if log_image is not None:
            out.log = self.run_log(log_image)
        if fusion_image is not None:
            out.fusion = self.run_fusion(fusion_image, time_depth_pairs, well_info)
        if out.seismic or out.log or out.fusion:
            out.prospect = self.run_prospect(
                out.seismic, out.log, out.fusion, extra_image=prospect_image,
            )
        out.meta = {
            "seismic_ok":  bool(out.seismic and out.seismic.ok),
            "log_ok":      bool(out.log and out.log.ok),
            "fusion_ok":   bool(out.fusion and out.fusion.ok),
            "prospect_ok": bool(out.prospect and out.prospect.ok),
        }
        return out

    @staticmethod
    def _prospect_context(seismic: AgentResult | None,
                          log: AgentResult | None,
                          fusion: AgentResult | None) -> str:
        parts = ["请基于以下前序 Agent 输出评价勘探目标，仅输出JSON。\n"]
        if seismic and seismic.plan:
            parts.append(f"[Seismic Scene] {seismic.plan.get('scene_understanding','')}")
            if seismic.results:
                parts.append(f"[Seismic Results] "
                             f"{json.dumps(seismic.results, ensure_ascii=False)[:1500]}")
        if log and log.plan:
            parts.append(f"[Log Scene] {log.plan.get('scene_understanding','')}")
            if log.results:
                parts.append(f"[Log Results] "
                             f"{json.dumps(log.results, ensure_ascii=False)[:1500]}")
        if fusion and fusion.output:
            parts.append(f"[Fusion] "
                         f"{json.dumps(fusion.output, ensure_ascii=False)[:1500]}")
        return "\n".join(parts)
