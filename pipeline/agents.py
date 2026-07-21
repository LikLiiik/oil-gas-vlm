"""Agent 类：LoopAgent（自主workflow闭环） + SingleShotAgent（单次 VLM）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from schemas.output_schemas import (
    WORKFLOW_PLAN_SCHEMA, WORKFLOW_VERIFICATION_SCHEMA,
)

from . import downstream
from .prompts import VERIFICATION_PROMPT, workflow_planning_prompt
from .vlm import VLMClient


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
            round_results = self._execute_steps(steps, image)
            if not round_results:
                self._log("  No results, skip verification")
                break
            result.results.extend(round_results)

            self._log(f"\n[{agent_name}] Phase 3: VLM 验证 (iter {iteration+1}/{max_iter})")
            user_text = (
                f"Workflow: {json.dumps(steps, ensure_ascii=False)}\n"
                f"Results: {json.dumps(round_results, ensure_ascii=False)}\n"
                "逐条验证检测结果。仅输出JSON。"
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
            self._log(f"  ✅ Verified ({ver_resp.elapsed_s:.0f}s, "
                      f"attempts={ver_resp.attempts})")
            self._log(f"  Real: {real} | FalsePositive: {fp} | "
                      f"NeedRetry: {ver_data.get('need_retry')}")
            result.verifications.append(
                {"iteration": iteration + 1, "verification": ver_data}
            )
            if not ver_data.get("need_retry"):
                break
            self._apply_retry(steps, ver_data.get("retry_instructions") or {})

        result.ok = (
            bool(result.plan)
            and (len(result.results) > 0 or len(result.verifications) > 0)
            and len(result.errors) == 0
        )
        result.elapsed_s = t_total
        result.output = ver_data  # 最后一轮验证结论作为 Agent 的对外输出
        return result

    def _execute_steps(self, steps: list[dict], image) -> list[dict]:
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
            for r in out:
                self._log(f"      {str(r)[:120]}")
            results.extend(out)
        return results

    @staticmethod
    def _apply_retry(steps: list[dict], retry: dict):
        target_step = retry.get("step")
        adjusted = retry.get("adjusted_params") or retry.get("adjusted_instruction")
        if target_step is None or adjusted is None:
            return
        for s in steps:
            if s.get("step") == target_step:
                if isinstance(adjusted, dict):
                    s["instruction"] = {**s.get("instruction", {}), **adjusted}
                else:
                    s["instruction"] = adjusted


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
