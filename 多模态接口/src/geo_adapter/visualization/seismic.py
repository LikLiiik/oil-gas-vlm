from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_adapter.models import SeismicData


def save_seismic_images(data: SeismicData, directory: Path) -> dict[str, dict[str, str]]:
    """Save a clean model image and an annotated QC image for every view."""
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}
    for name, view in data.views.items():
        model_path = directory / f"{name}_model.png"
        qc_path = directory / f"{name}_qc.png"
        fig, axis = plt.subplots(figsize=(8, 6), dpi=160)
        axis.imshow(view.processed.T if name in {"inline", "crossline", "patch"} else view.processed, cmap="gray", aspect="auto", origin="upper")
        axis.set_axis_off()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(model_path, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(9, 6), dpi=150)
        image = axis.imshow(view.processed.T if name in {"inline", "crossline", "patch"} else view.processed, cmap="gray", aspect="auto", origin="upper")
        axis.set_title(f"{view.physical_view} QC | domain={data.domain}")
        axis.set_xlabel(view.axis_labels[0])
        axis.set_ylabel(view.axis_labels[1])
        fig.colorbar(image, ax=axis, label="normalized amplitude")
        info = ", ".join(f"{key}={value}" for key, value in view.source_indices.items()) or "provided 2D view"
        axis.text(0.01, 0.01, info, transform=axis.transAxes, fontsize=8, color="darkred", bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
        fig.tight_layout()
        fig.savefig(qc_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs[name] = {"model": str(model_path), "qc": str(qc_path), "physical_view": view.physical_view}
    return outputs

