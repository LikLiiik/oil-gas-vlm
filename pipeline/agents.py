"""Agent 类：LoopAgent（自主workflow闭环） + SingleShotAgent（单次 VLM）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from schemas.output_schemas import (
    WORKFLOW_PLAN_SCHEMA,
    WORKFLOW_VERIFICATION_SCHEMA,
)

from . import downstream
from .loop_core import (
    apply_retry,
    ensure_bbox_norm,
    match_false_positives,
    tag_detection,
)
from .prompts import VERIFICATION_PROMPT, workflow_planning_prompt
from .vlm import VLMClient


def _dedup_by_det_id(dets: list[dict]) -> list[dict]:
    """按 det_id 去重，保留 confidence 最高者；无 det_id 的全部保留。"""
    best: dict[str, dict] = {}
    no_id: list[dict] = []
    for d in dets:
        rid = d.get("det_id")
        if rid is None:
            no_id.append(d)
            continue
        prev = best.get(rid)
        if prev is None or float(d.get("confidence", 0.0) or 0.0) \
                > float(prev.get("confidence", 0.0) or 0.0):
            best[rid] = d
    return list(best.values()) + no_id


@dataclass
class AgentResult:
    """一次 Agent 运行的完整产出，可 json.dump 直接落盘。"""
    agent: str
    ok: bool = False
    plan: dict | None = None
    results: list[dict] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)
    output: dict | None = None      # 单次 Agent 的最终 JSON
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent": self.agent, "ok": self.ok, "plan": self.plan,
            "results": self.results, "verifications": self.verifications,
            "output": self.output, "errors": self.errors,
            "elapsed_s": self.elapsed_s,
        }


class LoopAgent:
    """VLM 自主规划 → 下游执行 → VLM 验证 → 迭代 的闭环 Agent。
    对应 test_loop.py 的 run_agent_loop，从散逻辑抽出来。"""

    def __init__(self, vlm: VLMClient, max_iterations: int = 3, verbose: bool = True):
        self.vlm = vlm
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, image, agent_name: str = "agent",
            task_hint: str | None = None) -> AgentResult:
        """跑闭环。task_hint 是一段前缀文本，可用 tasks.tasks_prompt_hint() 生成，
        用来把 VLM 的规划偏向指定的地质任务（断层/层位/沉积相/裂缝）。"""
        result = AgentResult(agent=agent_name)
        t_total = 0.0

        # Phase 1: Planning
        self._log(f"\n{'='*60}\n[{agent_name}] Phase 1: VLM 规划下游模型\n{'='*60}")
        user_text = "分析图像，输出完整workflow计划。仅输出JSON。"
        if task_hint:
            user_text = f"{task_hint}\n\n{user_text}"
        plan_resp = self.vlm.call_json(
            workflow_planning_prompt(), [image], user_text,
            schema=WORKFLOW_PLAN_SCHEMA,
        )
        t_total += plan_resp.elapsed_s
        if not plan_resp.schema_valid or plan_resp.data is None:
            result.errors.append(
                f"planning failed after {plan_resp.attempts} attempts: "
                f"{plan_resp.schema_errors}"
            )
            self._log(f"  ❌ Planning failed: {plan_resp.schema_errors}")
            result.elapsed_s = t_total
            return result
        plan = plan_resp.data
        result.plan = plan
        steps = plan.get("workflow_steps", [])
        self._log(f"  ✅ Plan ({plan_resp.elapsed_s:.0f}s, attempts={plan_resp.attempts})")
        self._log(f"  Scene: {plan.get('scene_understanding', '')[:150]}")
        self._log(f"  Steps: {len(steps)}")
        for s in steps:
            self._log(f"    Step{s.get('step')}: {s.get('model')} "
                      f"→ {s.get('reason','')[:80]}")

        max_iter = min(plan.get("max_iterations", self.max_iterations),
                       self.max_iterations)

        # Phase 2+3: Execute + Verify + Retry
        ver_data: dict | None = None
        for iteration in range(max_iter):
            self._log(f"\n[{agent_name}] Phase 2: 执行下游模型 (iter {iteration+1}/{max_iter})")
            round_results = self._execute_steps(steps, image, agent_name)
            if not round_results:
                self._log("  No results, skip verification")
                break

            self._log(f"\n[{agent_name}] Phase 3: VLM 验证 (iter {iteration+1}/{max_iter})")
            user_text = (
                f"Workflow: {json.dumps(steps, ensure_ascii=False)}\n"
                f"Results (每条含 det_id): {json.dumps(round_results, ensure_ascii=False)}\n"
                "逐条验证检测结果，verified[].result_id 必须填该条检测的 det_id 原值。仅输出JSON。"
            )
            ver_resp = self.vlm.call_json(
                VERIFICATION_PROMPT, [image], user_text,
                schema=WORKFLOW_VERIFICATION_SCHEMA,
            )
            t_total += ver_resp.elapsed_s
            if not ver_resp.schema_valid or ver_resp.data is None:
                result.errors.append(
                    f"verification failed at iter {iteration+1}: "
                    f"{ver_resp.schema_errors}"
                )
                self._log(f"  ❌ Verification failed: {ver_resp.schema_errors}")
                break
            ver_data = ver_resp.data
            verified = ver_data.get("verified", [])
            real = sum(1 for v in verified if v.get("is_real"))
            fp = sum(1 for v in verified if not v.get("is_real"))

            # 过滤假阳性：det_id 精确匹配，回退 bbox-IoU；高置信才删，存疑进 review
            drop_ids, dropped, review = match_false_positives(ver_data, round_results)
            if drop_ids:
                round_results = [r for r in round_results
                                 if r.get("det_id") not in drop_ids]
                self._log(f"  🗑 过滤假阳性 {len(drop_ids)} 条: {sorted(drop_ids)}")
            if review:
                self._log(f"  ⚠️ 存疑 {len(review)} 条(低置信/未匹配, 保留): "
                          f"{[r['det_id'] for r in review]}")
            result.results.extend(round_results)

            self._log(f"  ✅ Verified ({ver_resp.elapsed_s:.0f}s, "
                      f"attempts={ver_resp.attempts})")
            self._log(f"  Real: {real} | FalsePositive: {fp} | "
                      f"NeedRetry: {ver_data.get('need_retry')}")
            result.verifications.append({
                "iteration": iteration + 1,
                "real": real, "false_positive": fp,
                "filtered": {"dropped": dropped, "review": review},
                "filtered_ids": sorted(drop_ids),   # 向后兼容旧字段
                "verification": ver_data,
            })
            if not ver_data.get("need_retry"):
                break
            apply_retry(steps, ver_data.get("retry_instructions") or {})

        # 跨迭代同一 det_id 可能重复累积（每轮重跑全部 step），按 det_id 去重保留高置信
        result.results = _dedup_by_det_id(result.results)

        result.ok = (
            bool(result.plan)
            and (len(result.results) > 0 or len(result.verifications) > 0)
            and len(result.errors) == 0
        )
        result.elapsed_s = t_total
        result.output = ver_data  # 最后一轮验证结论作为 Agent 的对外输出
        return result

    def _execute_steps(self, steps: list[dict], image,
                      agent_name: str = "agent") -> list[dict]:
        results = []
        for step in steps:
            model_name = step.get("model")
            step_num = step.get("step")
            model = downstream.get(model_name)
            if model is None:
                self._log(f"  ⚠️ Step{step_num}: unknown model '{model_name}', skip")
                continue
            self._log(f"  Step{step_num}: calling {model_name} ...")
            try:
                out = model.detect(step.get("instruction", {}), image=image)
            except Exception as e:
                self._log(f"    ❌ {model_name} failed: {e}")
                continue
            self._log(f"    → {len(out)} results")
            w, h = image.size
            tagged = []
            for i, d in enumerate(out):
                t = tag_detection(d, step=step_num or 0, image_name=agent_name,
                                  model_name=model_name, index=i)
                ensure_bbox_norm(t, w, h)
                tagged.append(t)
            for r in tagged:
                self._log(f"      {str(r)[:120]}")
            results.extend(tagged)
        return results


class SingleShotAgent:
    """单次 VLM 调用 + schema 校验。适合 fusion/prospect 这类文本导向 agent。"""

    def __init__(self, vlm: VLMClient, name: str, system_prompt: str,
                 schema: dict | None = None, verbose: bool = True):
        self.vlm = vlm
        self.name = name
        self.system_prompt = system_prompt
        self.schema = schema
        self.verbose = verbose

    def run(self, images: list, user_text: str,
            max_new_tokens: int = 4096) -> AgentResult:
        result = AgentResult(agent=self.name)
        resp = self.vlm.call_json(
            self.system_prompt, images, user_text,
            schema=self.schema, max_new_tokens=max_new_tokens, temperature=0.0,
        )
        result.elapsed_s = resp.elapsed_s
        if not resp.schema_valid or resp.data is None:
            result.errors = resp.schema_errors or ["JSON extraction failed"]
            if self.verbose:
                print(f"  ❌ [{self.name}] failed: {result.errors}")
            return result
        result.output = resp.data
        result.ok = True
        if self.verbose:
            print(f"  ✅ [{self.name}] ok ({resp.elapsed_s:.0f}s, "
                  f"attempts={resp.attempts})")
        return result
