"""合成测试数据：地震剖面、测井曲线、井震对比图。仅用于开发/测试。"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _ricker(length: int, dt: float, freq: float) -> np.ndarray:
    t = np.arange(length) * dt - (length * dt) / 2
    return (1 - 2 * (np.pi * freq * t) ** 2) * np.exp(-(np.pi * freq * t) ** 2)


def _fig_to_pil(fig, dpi: int = 150) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def generate_seismic(seed: int = 42) -> Image.Image:
    np.random.seed(seed)
    n_traces, n_samples = 300, 500
    reflectivity = np.zeros((n_samples, n_traces))
    for layer_idx in range(8):
        base_depth = 50 + layer_idx * 55
        for i in range(n_traces):
            depth = base_depth + (15 + np.random.rand() * 20) * np.sin(
                i / (80 + np.random.rand() * 40) * 2 * np.pi)
            d_idx = int(np.clip(depth + np.random.randn() * 3, 0, n_samples - 1))
            if d_idx < n_samples:
                reflectivity[d_idx, i] = np.random.rand() * 0.6 + 0.2
    reflectivity[:, 120:] = np.roll(reflectivity[:, 120:], 15, axis=0)
    reflectivity[240:243, 180:210] = -1.5
    for i in range(50, 80):
        thickness = int(8 * (1 - ((i - 65) / 15) ** 2))
        if thickness > 0:
            reflectivity[300:300 + thickness, i] = (
                0.4 + np.random.randn(thickness) * 0.15)

    wavelet = _ricker(64, 0.004, 25)
    seismic = np.zeros_like(reflectivity)
    for i in range(n_traces):
        seismic[:, i] = np.convolve(reflectivity[:, i], wavelet, mode="same")
    seismic += np.random.randn(*seismic.shape) * 0.03

    fig, ax = plt.subplots(figsize=(14, 10))
    vmax = np.percentile(np.abs(seismic), 98)
    ax.imshow(seismic, cmap="seismic", aspect="auto",
              vmin=-vmax, vmax=vmax, extent=[1, n_traces, 2500, 0])
    ax.set_xlabel("CDP")
    ax.set_ylabel("Two-Way Time (ms)")
    ax.set_title("Seismic Inline Section")
    plt.tight_layout()
    return _fig_to_pil(fig)


def generate_log(seed: int = 42) -> Image.Image:
    np.random.seed(seed)
    depth = np.linspace(1000, 2000, 2000)
    gr = np.full(2000, 95.0)
    rt = np.full(2000, 3.0)
    den = np.full(2000, 2.50)
    cnl = np.full(2000, 0.32)
    for top, bot in [(1200, 1255), (1400, 1435), (1550, 1625), (1750, 1805)]:
        mask = (depth >= top) & (depth <= bot)
        n = mask.sum()
        gr[mask] = 35 + np.random.randn(n) * 6
        den[mask] = 2.28 + np.random.randn(n) * 0.06
        cnl[mask] = 0.16 + np.random.randn(n) * 0.04
    gas = (depth >= 1555) & (depth <= 1595)
    rt[gas] = 85 + np.random.randn(gas.sum()) * 10

    curves = [
        ("GR", gr, "green", "GR(API)", (0, 150)),
        ("RT", rt, "red", "RT(Ohm.m)", (0.1, 100)),
        ("DEN", den, "black", "DEN(g/cm3)", (1.8, 2.8)),
        ("CNL", cnl, "magenta", "CNL(v/v)", (0.45, -0.15)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 14), sharey=True)
    for ax, (_name, data, color, label, xlim) in zip(axes, curves):
        ax.plot(data, depth, color=color, linewidth=0.5)
        ax.set_xlabel(label, fontsize=8)
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        if xlim[0] > xlim[1]:
            ax.invert_xaxis()
    axes[0].set_ylabel("Depth(m)")
    fig.suptitle("Well Log", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _fig_to_pil(fig, dpi=120)


def generate_fusion(seed: int = 42) -> Image.Image:
    """简版井震并排对比图：左侧地震道，右侧 GR 曲线。"""
    np.random.seed(seed)
    time_ms = np.linspace(0, 2500, 500)
    depth_m = np.linspace(1000, 2000, 500)
    wavelet = _ricker(200, 0.004, 25)
    trace = np.convolve(np.random.randn(500) * 0.3, wavelet, mode="same")
    gr = 80 + 30 * np.sin(depth_m / 40) + np.random.randn(500) * 5
    for top, bot in [(1200, 1255), (1550, 1625)]:
        m = (depth_m >= top) & (depth_m <= bot)
        gr[m] = 40 + np.random.randn(m.sum()) * 5

    fig, axes = plt.subplots(1, 2, figsize=(10, 12), sharey=True)
    axes[0].plot(trace, time_ms, color="black", linewidth=0.6)
    axes[0].fill_betweenx(time_ms, 0, trace, where=trace > 0,
                          color="red", alpha=0.4)
    axes[0].set_xlabel("Amplitude")
    axes[0].set_ylabel("Two-Way Time (ms)")
    axes[0].invert_yaxis()
    axes[0].set_title("Well-side Seismic")
    axes[1].plot(gr, depth_m, color="green", linewidth=0.6)
    axes[1].set_xlabel("GR (API)")
    axes[1].set_title("Well Log GR")
    axes[1].invert_yaxis()
    fig.suptitle("Well-Seismic Fusion", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _fig_to_pil(fig, dpi=120)
