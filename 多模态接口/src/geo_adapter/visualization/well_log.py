from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np

from geo_adapter.models import ProcessedCurve, WellLogData


COLORS = {
    "GR": "#2E8B57",
    "SP": "#1E66F5",
    "CAL": "#D97706",
    "RES_DEEP": "#B91C1C",
    "RES_MEDIUM_SHALLOW": "#7C3AED",
    "RES_MICRO": "#0891B2",
    "AC": "#334155",
    "DEN": "#EA580C",
    "CNL": "#0F766E",
}


def _legend_label(curve: ProcessedCurve) -> str:
    label = curve.selected_curve or curve.canonical_name
    if curve.measurement_family:
        label += f" [{curve.measurement_family}]"
    return label


def _plot_curve(axis: plt.Axes, depth: np.ndarray, curve: ProcessedCurve, label: str | None = None) -> None:
    if curve.values is None:
        return
    values = curve.values.copy()
    usable = (curve.valid_mask | curve.interpolated_mask) if curve.valid_mask is not None and curve.interpolated_mask is not None else np.isfinite(values)
    values[~usable] = np.nan
    axis.plot(values, depth, color=COLORS[curve.canonical_name], linewidth=1.0, label=label or _legend_label(curve))


def save_well_log_images(data: WellLogData, directory: Path) -> dict[str, str]:
    """Save an editable-source-independent four-track panel and available slot images."""
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    fig, axes = plt.subplots(1, 4, figsize=(15, 9), dpi=150, sharey=True)
    fig.suptitle(f"Well log panel | {data.well_id or 'unknown well'} | depth={data.depth_name} ({data.depth_unit})")

    track1 = axes[0]
    track1.set_title("GR / SP / CAL")
    offsets = 0
    for slot in ("GR", "SP", "CAL"):
        curve = data.curves[slot]
        if not curve.available:
            continue
        current = track1 if offsets == 0 else track1.twiny()
        if offsets:
            current.spines.top.set_position(("axes", 1.0 + 0.1 * offsets))
            current.patch.set_visible(False)
        _plot_curve(current, data.depth, curve)
        current.set_xlabel(f"{slot} · {curve.selected_curve} ({curve.canonical_unit})", color=COLORS[slot])
        current.tick_params(axis="x", colors=COLORS[slot], labelsize=7)
        offsets += 1

    track2 = axes[1]
    track2.set_title("Resistivity")
    plotted_res = False
    for slot in ("RES_DEEP", "RES_MEDIUM_SHALLOW", "RES_MICRO"):
        curve = data.curves[slot]
        if curve.available:
            _plot_curve(track2, data.depth, curve)
            plotted_res = True
    if plotted_res:
        track2.set_xscale("log")
        track2.xaxis.set_major_locator(LogLocator(base=10.0))
        track2.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
        track2.xaxis.set_minor_formatter(NullFormatter())
        track2.set_xlabel("ohm_m (log scale)")
        track2.legend(fontsize=7, loc="best")

    track3 = axes[2]
    track3.set_title("AC / DT")
    if data.curves["AC"].available:
        _plot_curve(track3, data.depth, data.curves["AC"])
        track3.set_xlabel(f"AC · {data.curves['AC'].selected_curve} (us/m)")

    track4 = axes[3]
    track4.set_title("DEN / CNL")
    available_track4 = [slot for slot in ("DEN", "CNL") if data.curves[slot].available]
    for index, slot in enumerate(available_track4):
        current = track4 if index == 0 else track4.twiny()
        if index:
            current.spines.top.set_position(("axes", 1.1))
            current.patch.set_visible(False)
        _plot_curve(current, data.depth, data.curves[slot])
        current.set_xlabel(
            f"{slot} · {data.curves[slot].selected_curve} ({data.curves[slot].canonical_unit})",
            color=COLORS[slot],
        )
        current.tick_params(axis="x", colors=COLORS[slot], labelsize=7)

    missing = [slot for slot, curve in data.curves.items() if not curve.available]
    for axis in axes:
        axis.grid(alpha=0.18)
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"{data.depth_name} ({data.depth_unit})")
    if missing:
        fig.text(0.5, 0.012, "Missing slots (not substituted): " + ", ".join(missing), ha="center", fontsize=9, color="#991B1B")
    fig.subplots_adjust(top=0.84, bottom=0.08, wspace=0.35)
    panel_path = directory / "well_log_panel.png"
    fig.savefig(panel_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs["well_log_panel"] = str(panel_path)

    for slot, curve in data.curves.items():
        if not curve.available or curve.values is None:
            continue
        fig, axis = plt.subplots(figsize=(4.5, 8), dpi=150)
        _plot_curve(axis, data.depth, curve)
        if slot.startswith("RES_"):
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(LogLocator(base=10.0))
            axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.invert_yaxis()
        axis.grid(alpha=0.2)
        axis.set_xlabel(f"{slot} ({curve.canonical_unit})")
        axis.set_ylabel(f"{data.depth_name} ({data.depth_unit})")
        axis.set_title(_legend_label(curve))
        path = directory / f"{slot.lower()}.png"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs[slot.lower()] = str(path)
    return outputs
