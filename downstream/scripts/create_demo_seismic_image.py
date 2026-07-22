"""Create a small synthetic seismic PNG for YOLO-World smoke testing.

The image is not geological training data. It only exists so the downstream
YOLO-World adapter can be tested end-to-end without waiting for real SEG-Y
slices from the upstream preprocessing step.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import zlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/demo_inline_120.png")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=202604)
    return parser.parse_args()


def _png_chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    raw = bytearray()
    row_bytes = width * 3
    for y in range(height):
        raw.append(0)  # PNG filter type 0
        start = y * row_bytes
        raw.extend(pixels[start : start + row_bytes])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def build_demo_image(width: int, height: int, seed: int) -> bytes:
    rng = random.Random(seed)
    pixels = bytearray(width * height * 3)

    for y in range(height):
        for x in range(width):
            xn = x / max(width - 1, 1)
            yn = y / max(height - 1, 1)

            wave = (
                0.55 * math.sin(2 * math.pi * (6.0 * yn + 0.8 * math.sin(2 * math.pi * xn)))
                + 0.30 * math.sin(2 * math.pi * (11.0 * yn + 0.35 * xn))
                + 0.18 * math.sin(2 * math.pi * (18.0 * yn - 0.25 * xn))
            )

            # Fault-like vertical break near the left-middle region.
            if 0.23 < xn < 0.34 and 0.18 < yn < 0.84:
                wave += 0.28 * math.sin(2 * math.pi * (22.0 * yn + 2.2 * xn))

            # Bright-spot-like anomaly on the right side.
            dx = (xn - 0.78) / 0.09
            dy = (yn - 0.48) / 0.05
            anomaly = math.exp(-(dx * dx + dy * dy))
            wave += 0.90 * anomaly

            # Channel-like lens near the lower-left.
            channel_center = 0.73 + 0.035 * math.sin(2 * math.pi * (xn * 1.6))
            channel = math.exp(-(((yn - channel_center) / 0.035) ** 2)) if 0.04 < xn < 0.33 else 0.0
            wave -= 0.55 * channel

            noise = (rng.random() - 0.5) * 0.08
            value = max(-1.0, min(1.0, wave + noise))

            if value >= 0:
                r = int(245 * value + 35 * (1 - value))
                g = int(78 * value + 35 * (1 - value))
                b = int(58 * value + 35 * (1 - value))
            else:
                a = -value
                r = int(35 * (1 - a) + 42 * a)
                g = int(35 * (1 - a) + 112 * a)
                b = int(35 * (1 - a) + 230 * a)

            idx = (y * width + x) * 3
            pixels[idx] = r
            pixels[idx + 1] = g
            pixels[idx + 2] = b

    return bytes(pixels)


def main() -> int:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    pixels = build_demo_image(args.width, args.height, args.seed)
    _write_png(output, args.width, args.height, pixels)
    print(f"Wrote {output} ({args.width}x{args.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
