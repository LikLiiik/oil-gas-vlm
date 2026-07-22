from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from geo_adapter.pipeline import inspect_geo_sample, prepare_geo_sample, validate_run
from geo_adapter.schemas.manifest import Manifest


app = typer.Typer(help="地球物理多模态输入适配接口", no_args_is_help=True)
console = Console()


def _emit_messages(warnings: list[str], errors: list[str]) -> None:
    for warning in warnings:
        console.print(f"[yellow]警告:[/yellow] {warning}")
    for error in errors:
        console.print(f"[red]错误:[/red] {error}")


@app.command()
def inspect(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
) -> None:
    """只读检查输入、语义映射、缺失和配准风险。"""
    result = inspect_geo_sample(config)
    console.print_json(json.dumps(result.inputs, ensure_ascii=False, default=str))
    _emit_messages(result.warnings, result.errors)
    if not result.success:
        raise typer.Exit(code=1)


@app.command()
def prepare(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False, readable=True),
) -> None:
    """执行完整流水线并生成标准运行包。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = prepare_geo_sample(config)
    _emit_messages(result.warnings, result.errors)
    if result.success:
        console.print(
            Panel.fit(
                f"输出目录: {result.output_directory}\n"
                f"运行模式: {result.run_mode}\n"
                f"水平/垂向: {result.horizontal_alignment} / {result.vertical_alignment}\n"
                f"融合权限: {result.fusion_permission}",
                title="准备完成",
                border_style="green",
            )
        )
    else:
        raise typer.Exit(code=1)


@app.command()
def validate(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, readable=True),
) -> None:
    """验证 Schema、路径引用、PNG、数组/Mask 和配准权限。"""
    result = validate_run(run_dir)
    _emit_messages(result.warnings, result.errors)
    if result.success:
        console.print(f"[green]验证通过[/green]，检查文件数: {result.checked_files}")
    else:
        raise typer.Exit(code=1)


@app.command("show-manifest")
def show_manifest(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, readable=True),
) -> None:
    """以易读方式显示 manifest 关键状态。"""
    path = run_dir / "manifest.json"
    if not path.is_file():
        console.print(f"[red]错误:[/red] manifest 不存在: {path}")
        raise typer.Exit(code=1)
    try:
        manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        console.print(f"[red]错误:[/red] manifest 无效: {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title=f"Manifest · {manifest.sample_id}")
    table.add_column("项目")
    table.add_column("值")
    table.add_row("运行模式", manifest.run_mode)
    table.add_row("可用模态", ", ".join(name for name, value in manifest.availability.model_dump().items() if value))
    table.add_row("水平配准", manifest.alignment.horizontal_level)
    table.add_row("垂向配准", manifest.alignment.vertical_level)
    table.add_row("融合权限", manifest.alignment.fusion_permission)
    table.add_row("时深来源", manifest.time_depth_relation.source)
    table.add_row("质量状态", manifest.quality.status)
    table.add_row("可用曲线", ", ".join(name for name, curve in manifest.well_logs.curves.items() if curve.available) or "无")
    console.print(table)


if __name__ == "__main__":
    app()

