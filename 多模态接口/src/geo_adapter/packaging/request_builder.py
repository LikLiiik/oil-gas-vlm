from __future__ import annotations

from pathlib import Path

from geo_adapter.schemas.request import ContentItem, Message, ModelRequest


def build_request(
    *,
    sample_id: str,
    run_dir: Path,
    seismic_images: dict[str, dict[str, str]],
    well_images: dict[str, str],
) -> ModelRequest:
    """Build a model-family-neutral request containing only existing assets."""
    user_content: list[ContentItem] = []
    for name, info in seismic_images.items():
        path = Path(info["model"])
        if path.is_file():
            user_content.append(
                ContentItem(
                    type="image",
                    name=f"seismic_{name}",
                    path=path.relative_to(run_dir).as_posix(),
                    analysis_path=Path(info["analysis"])
                    .relative_to(run_dir)
                    .as_posix(),
                    physical_view=info["physical_view"],
                    native_shape=info.get("native_shape"),
                    axis_labels=info.get("axis_labels"),
                    source_indices=info.get("source_indices"),
                )
            )
    panel = well_images.get("well_log_panel")
    if panel and Path(panel).is_file():
        panel_path = Path(panel).relative_to(run_dir).as_posix()
        user_content.append(
            ContentItem(
                type="image",
                name="well_log_panel",
                path=panel_path,
                analysis_path=panel_path,
                physical_view="well_log_panel",
            )
        )
    user_content.extend(
        [
            ContentItem(type="text", text_path="prompts/user_prompt.txt"),
            ContentItem(type="json", name="manifest", path="manifest.json"),
        ]
    )
    numeric_summary = run_dir / "tables/well_numeric_summary.json"
    if numeric_summary.is_file():
        user_content.append(
            ContentItem(
                type="json",
                name="well_numeric_summary",
                path=numeric_summary.relative_to(run_dir).as_posix(),
            )
        )
    return ModelRequest(
        sample_id=sample_id,
        messages=[
            Message(
                role="system",
                content=[
                    ContentItem(type="text", text_path="prompts/system_prompt.txt")
                ],
            ),
            Message(role="user", content=user_content),
        ],
        expected_output_schema="schemas/expected_model_output.schema.json",
    )
