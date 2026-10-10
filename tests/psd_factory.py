"""Programmatic PSD/PSB fixtures.

The importer is only exercisable against real files, and there are no PSD
samples in the repo. ``psd-tools`` can author PSDs (``PixelLayer.frompil``
/ ``Group.new``), so the fixtures are built here instead of being checked
in as binaries — they stay diffable and cost nothing in the working tree.

The layer stack produced by :func:`build_sample_psd`::

    [0] Group 1                (group, closed folder)
    [1]   Nested               (pixel layer, Multiply, at 40,30, 60x40)
    [2]   Masked               (pixel layer, 40x40 mask at 20,60, blocks of 8)
    [3] Top                    (pixel layer, hidden, at 0,0, full canvas)

    Background                 (pixel layer, full canvas, Normal)

The mask blocks read back as ``[255, 255, 0, 0, 255]`` per 8 pixels — that
is "hidden, hidden, visible, visible, hidden" in Compositor's polarity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

CANVAS_W = 200
CANVAS_H = 150


def _rgba(r: int, g: int, b: int, a: int = 255, w: int = 1, h: int = 1) -> "object":
    from PIL import Image

    return Image.fromarray(
        np.full((h, w, 4), (r, g, b, a), dtype=np.uint8), mode="RGBA"
    )


def build_sample_psd(path: Path, *, use_mask: bool = True) -> Path:
    """Write a small layered PSD and return its path."""
    from psd_tools.api.layers import Group, PixelLayer
    from psd_tools.api.psd_image import PSDImage
    from psd_tools.constants import ChannelID, Compression
    from psd_tools.psd.layer_and_mask import MaskData

    # ``color`` fills the merged preview: 255 = white.
    psd = PSDImage.new("RGB", (CANVAS_W, CANVAS_H), color=255)

    # Bottom of the stack.
    background = PixelLayer.frompil(
        _rgba(20, 40, 80, 255, CANVAS_W, CANVAS_H),
        psd,
        layer_name="Background",
        top=0,
        left=0,
    )

    top = PixelLayer.frompil(
        _rgba(200, 30, 30, 255, CANVAS_W, CANVAS_H),
        psd,
        layer_name="Top",
        top=0,
        left=0,
    )
    top.visible = False

    masked = PixelLayer.frompil(
        _rgba(30, 200, 60, 255, 60, 40),
        psd,
        layer_name="Masked",
        top=60,
        left=20,
    )
    if use_mask:
        # The mask rectangle is in document coordinates, like Photoshop writes it.
        _attach_pixel_mask(masked, MaskData, ChannelID, Compression.RAW, top=60, left=20)

    nested = PixelLayer.frompil(
        _rgba(240, 200, 40, 255, 60, 40),
        psd,
        layer_name="Nested",
        top=30,
        left=40,
    )
    nested.blend_mode = _blend_multiply()

    group = Group.new(name="Group 1", open_folder=False)
    psd.append(group)
    group.append(nested)
    group.append(masked)
    psd.append(background)
    psd.append(top)

    psd.save(str(path))
    return path


def _blend_multiply():
    from psd_tools.constants import BlendMode

    return BlendMode.MULTIPLY


def _attach_pixel_mask(layer, mask_data_cls, channel_id, compression, top=0, left=0) -> None:
    """Give ``layer`` a 40x40 half-white mask over its top-left corner."""
    width = height = 40
    # psd-tools reads the user-mask channel as one byte per pixel, so write it
    # unpacked. The pattern is constant over 8-pixel blocks because psd-tools
    # packs/unpacks each row's bits in its own order — a per-pixel ramp would
    # not come back in the positions we put it.
    blocks = [255, 255, 0, 0, 255]
    pattern = np.array([b for b in blocks for _ in range(8)], dtype=np.uint8)
    row = np.tile(pattern, (height, 1))
    data = row.tobytes()

    from psd_tools.psd.layer_and_mask import ChannelData, ChannelInfo

    layer._record.mask_data = mask_data_cls(
        top=top, left=left, bottom=top + height, right=left + width, background_color=0
    )
    layer._channels.append(ChannelData(compression=compression, data=data))
    layer._record.channel_info.append(
        ChannelInfo(id=channel_id.USER_LAYER_MASK, length=len(data) + 2)
    )


def write_flat_psd(path: Path, rgba: np.ndarray) -> Path:
    """Write a single-layer PSD from an HxWx4 array — used by the Photoshop
    round-trip tests, where the file comes back out of Photoshop itself."""
    from PIL import Image

    from psd_tools.api.layers import PixelLayer
    from psd_tools.api.psd_image import PSDImage

    height, width = rgba.shape[:2]
    psd = PSDImage.new("RGB", (width, height), color=255)
    layer = PixelLayer.frompil(
        Image.fromarray(rgba, mode="RGBA"), psd, layer_name="Layer 1"
    )
    psd.append(layer)
    psd.save(str(path))
    return path
