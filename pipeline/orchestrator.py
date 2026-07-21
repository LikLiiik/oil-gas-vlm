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
        from .io.segy import SegyVolume, write_attribute_segy
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
            # 合并到 all_detections（同一 image 同一 class 取 max confidence）
            for img_name, dets in round_dets.items():
                if img_name not in all_detections:
                    all_detections[img_name] = []
                all_detections[img_name].extend(dets)
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
            ver_text = (
                f"原始工作流计划:\n{json.dumps(steps, ensure_ascii=False)[:3000]}\n\n"
                f"下游模型实际检测结果:\n"
                f"{json.dumps(round_dets, ensure_ascii=False)[:4000]}\n\n"
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
            self._apply_competition_retry(steps, ver_data)

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
        if shape and len(shape) == 3:
            per_class = aggregate_adapter_detections(
                normalized, pkg.manifest, tuple(shape),
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
            default_task = tasks_mod.get("fault")
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
            img_lines.append(
                f"  {i}. image_name=\"{im.name}\"  view={view}  "
                f"size={im.pil.size}"
            )
        plan_text = (
            f"你是地球物理AI工作流规划器。以下是 geo_adapter 预处理好的 "
            f"赛题数据，共 {len(pkg.images)} 张图像。\n\n"
            f"=== 赛题任务 ===\n"
            f"sample_id: {pkg.sample_id}\n"
            f"target_classes: {pkg.target_classes}\n"
            f"{task_hint}\n\n"
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
            "- well_log_panel → well_log_ml 或 well_log_analyzer(测井分析)\n\n"
            "仅输出JSON。"
        )
        # 注入 RAG 知识
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
                detections.setdefault(im.name, []).extend(out)

        return detections

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
            # 尝试加载原始数组
            arrays_dir = pkg.run_dir / "arrays"
            arr_path = vm.get("array_path")
            if not arr_path:
                arr_path = vm.get("source_array")
            if arr_path and arrays_dir.exists():
                arr_file = arrays_dir / arr_path
                if arr_file.is_file():
                    import numpy as np
                    try:
                        ctx["array"] = np.load(arr_file)
                    except Exception:
                        pass
        return ctx if ctx else None

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
    def _apply_competition_retry(steps: list[dict], ver_data: dict):
        """把 VLM 验证给出的 retry_instructions 应用到对应 step。"""
        retry = ver_data.get("retry_instructions") or {}
        target = retry.get("step")
        adjusted = retry.get("adjusted_params") or retry.get("adjusted_instruction")
        if target is None or not adjusted:
            return
        for s in steps:
            if s.get("step") == target:
                if isinstance(adjusted, dict):
                    s["instruction"] = {**s.get("instruction", {}), **adjusted}
                break

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
