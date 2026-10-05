"""Derive line masks from card art with a pure-Python Sobel filter.

Blender 5.x removed Scene.node_tree and CompositorNodeFilter.filter_type, so the
old compositor Sobel path failed (#172). Running under Blender's embedded
Python also lacked Pillow. This script uses the system interpreter and numpy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def sobel_edges(path: Path) -> Image.Image:
    image = Image.open(path).convert("L")
    # Pillow 10+ has Kernel; fall back to FIND_EDGES if needed.
    try:
        return image.filter(
            ImageFilter.Kernel((3, 3), [-1, 0, 1, -2, 0, 2, -1, 0, 1], scale=1)
        ).point(lambda p: 255 if p > 24 else 0)
    except Exception:
        return image.filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 24 else 0)


def apply_face_box(mask: Image.Image, box, size: tuple[int, int]) -> Image.Image:
    if not box:
        return mask
    x0, y0, x1, y1 = box
    w, h = size
    arr = np.asarray(mask, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    rx = max((x1 - x0) * 0.6, 1.0)
    ry = max((y1 - y0) * 0.625, 1.0)
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    arr[ellipse] = 0
    return Image.fromarray(arr.astype(np.uint8), mode="L")


def main(root: Path) -> None:
    config = json.loads((root / "card-config.json").read_text(encoding="utf-8-sig"))
    box = config.get("identity", {}).get("face_box")
    assets = root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (root / "sources").mkdir(parents=True, exist_ok=True)
    for state in ("a", "b"):
        src = assets / f"image_{state}.png"
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        edges = sobel_edges(src)
        edges = apply_face_box(edges, box, edges.size)
        # Match the old compositor ramp: keep mid/high edges as white lines on black.
        out = edges.convert("RGB")
        out.save(assets / f"line_{state}.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: prepare_fx.py <project_root>")
    main(Path(sys.argv[1]).resolve())
