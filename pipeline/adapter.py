"""对接前置模块 geo_adapter 的输出。

geo_adapter (多模态接口/geo_adapter) 把原始 .sgy/.las/.csv 转成标准 run 目录:
    runs/<sample_id>/
        assets/seismic/*_model.png
        assets/well_logs/well_log_panel.png
        prompts/system_prompt.txt
        prompts/user_prompt.txt
        manifest.json                        # 元数据 (shape/CRS/views/alignment)
        request.json                         # 完整消息结构
        schemas/expected_model_output.schema.json  # VLM 输出契约
        arrays/*.npy                          # 原始/归一化数组
        tables/*.csv                          # 曲线/时深表

这个模块把上面的目录载入成一个 RunPackage，供 Pipeline.run_from_adapter 使用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class PackageImage:
    """一张 geo_adapter 已渲染好的输入图像。"""
    name: str
    path: Path
    physical_view: str
    pil: Image.Image

    def to_dict(self) -> dict:
        return {"name": self.name, "path": str(self.path),
                "physical_view": self.physical_view}


@dataclass
class RunPackage:
    """geo_adapter 产出的运行包。所有字段都来自 runs/<sample_id>/。"""
    run_dir: Path
    sample_id: str
    manifest: dict
    request: dict
    system_prompt: str
    user_prompt: str
    images: list[PackageImage]
    expected_schema: dict
    target_classes: list[str] = field(default_factory=list)
    task_type: str | None = None

    def image_by_name(self, name: str) -> PackageImage | None:
        for im in self.images:
            if im.name == name:
                return im
        return None

    def view_meta(self, view_name: str) -> dict | None:
        """从 manifest 里取某个地震 view 的元数据（inline/crossline/slice/local_patch）。"""
        views = (self.manifest.get("seismic", {}) or {}).get("views", {})
        direct = views.get(view_name)
        if direct is not None:
            return direct
        return next(
            (
                meta
                for meta in views.values()
                if isinstance(meta, dict) and meta.get("physical_view") == view_name
            ),
            None,
        )

    def to_summary(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "sample_id": self.sample_id,
            "task_type": self.task_type,
            "target_classes": list(self.target_classes),
            "n_images": len(self.images),
            "image_views": [im.physical_view for im in self.images],
            "seismic_shape": (self.manifest.get("seismic", {}) or {}).get("shape"),
            "run_mode": self.manifest.get("run_mode"),
            "fusion_permission": (self.manifest.get("alignment", {}) or {})
                                     .get("fusion_permission"),
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: str | Path) -> RunPackage:
    """载入 geo_adapter 产出的一个 run 目录。文件缺失时抛清晰的错误。"""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    def _need(rel: str) -> Path:
        p = run_dir / rel
        if not p.is_file():
            raise FileNotFoundError(f"missing required file in run dir: {rel}")
        return p

    manifest = _read_json(_need("manifest.json"))
    request = _read_json(_need("request.json"))
    system_prompt = _read_text(_need("prompts/system_prompt.txt"))
    user_prompt = _read_text(_need("prompts/user_prompt.txt"))
    expected_schema = _read_json(_need("schemas/expected_model_output.schema.json"))

    # 从 request.messages[].content 里抓 image 项，路径相对于 run_dir
    images: list[PackageImage] = []
    for msg in request.get("messages", []):
        if msg.get("role") != "user":
            continue
        for c in msg.get("content", []):
            if c.get("type") != "image":
                continue
            rel = c.get("path")
            if not rel:
                continue
            img_path = (run_dir / rel).resolve()
            if not img_path.is_file():
                # geo_adapter 允许 QC 图缺失，模型图必须在
                continue
            images.append(PackageImage(
                name=c.get("name", img_path.stem),
                path=img_path,
                physical_view=c.get("physical_view", "unknown"),
                pil=Image.open(img_path).convert("RGB"),
            ))

    task = manifest.get("task") or {}
    return RunPackage(
        run_dir=run_dir,
        sample_id=str(manifest.get("sample_id") or run_dir.name),
        manifest=manifest,
        request=request,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        expected_schema=expected_schema,
        target_classes=list(task.get("target_classes") or []),
        task_type=task.get("type"),
    )


def build_vlm_user_text(pkg: RunPackage, task_hint: str | None = None) -> str:
    """把 geo_adapter 的 user_prompt + manifest + schema 要点拼成完整的 user text。"""
    parts = [pkg.user_prompt]
    if task_hint:
        parts.append("\n补充说明：\n" + task_hint)
    parts.append("\nmanifest.json 关键上下文（供你参考，不要复述）:\n"
                 + json.dumps(_slim_manifest(pkg.manifest),
                              ensure_ascii=False, indent=2))
    # 把 expected schema 的 required 结构明文注入，减少 schema 校验失败
    parts.append("\n" + _schema_cheat_sheet(pkg.expected_schema))
    return "\n".join(parts)


def _schema_cheat_sheet(schema: dict) -> str:
    """从 schema 里抽一段简洁的 JSON 骨架（only 关键 key，值用类型占位），
    帮助 VLM 一次输出就命中 required 字段，递归展开到 2 层。"""
    defs = schema.get("$defs") or {}
    lines = ["期望的 JSON 结构（required 字段必须存在）:"]
    _render(lines, schema, set(schema.get("required") or []), defs,
            indent="", max_depth=3)
    return "\n".join(lines)


def _render(lines: list[str], obj: dict, root_req: set, defs: dict,
            indent: str, max_depth: int):
    if max_depth <= 0:
        lines.append(f"{indent}  // ... nested fields required")
        return
    req = set(obj.get("required") or [])
    props = obj.get("properties") or {}
    for k, v in props.items():
        r = " *" if k in root_req or k in req else ""
        typ = v.get("type", "object")
        # 1) 纯 $ref
        ref_key = v.get("$ref")
        if not ref_key:
            for cand in v.get("anyOf") or []:
                if "$ref" in cand:
                    ref_key = cand["$ref"]
                    break
        if not ref_key:
            ref_key = (v.get("items") or {}).get("$ref")
        if ref_key and ref_key.startswith("#/"):
            dn = ref_key.split("/")[-1]
            resolved = defs.get(dn, {})
            if resolved:
                hint = ""
                if typ == "array":
                    hint = " (空时用[]不是{})"
                elif typ == "object" and "properties" not in v:
                    hint = ""
                lines.append(f"{indent}  {k}: {typ}[{dn}]{r}{hint}")
                _render(lines, resolved, req, defs, indent + "    ", max_depth - 1)
        elif typ == "object" and v.get("properties"):
            lines.append(f"{indent}  {k}: {typ}{r}")
            _render(lines, v, req, defs, indent + "    ", max_depth - 1)
        elif typ == "string":
            enum = v.get("enum")
            if enum:
                lines.append(f"{indent}  {k}: string{r} 可选={enum}")
            else:
                lines.append(f"{indent}  {k}: {typ}{r}")
        elif typ == "array":
            lines.append(f"{indent}  {k}: array{r} (空时用[])")
        else:
            lines.append(f"{indent}  {k}: {typ}{r}")


def _slim_manifest(m: dict) -> dict:
    """给 VLM 看的精简 manifest：去掉 raw path、qc 明细等，减少 token。"""
    if not m:
        return {}
    out: dict[str, Any] = {}
    for k in ("schema_version", "sample_id", "task", "run_mode",
              "availability", "alignment"):
        if k in m:
            out[k] = m[k]
    seismic = m.get("seismic") or {}
    out["seismic"] = {
        "shape": seismic.get("shape"),
        "domain": seismic.get("domain"),
        "crs": (seismic.get("crs") or {}).get("name"),
        "views": {vn: {
            "physical_view": v.get("physical_view"),
            "array_shape": v.get("array_shape"),
            "axis_labels": v.get("axis_labels"),
            "source_indices": v.get("source_indices"),
        } for vn, v in (seismic.get("views") or {}).items()},
    }
    wl = m.get("well_logs") or {}
    curves = wl.get("curves") or wl.get("curves_present") or {}
    if isinstance(curves, dict):
        curve_iter = curves.values()
    elif isinstance(curves, list):
        curve_iter = curves
    else:
        curve_iter = []
    present = []
    for curve in curve_iter:
        if isinstance(curve, str):
            present.append(curve)
            continue
        if not isinstance(curve, dict):
            continue
        name = curve.get("canonical_name")
        if name and curve.get("available", True):
            present.append(name)
    out["well_logs"] = {
        "available": wl.get("available"),
        "curves_present": present,
        "depth_range": wl.get("depth_range"),
    }
    td = m.get("time_depth_relation") or {}
    if td:
        out["time_depth_relation"] = {
            "source": td.get("source"),
            "calibration": td.get("calibration"),
        }
    return out
