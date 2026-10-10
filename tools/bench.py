"""Quick performance benchmark for the rendering pipeline.

Run:
    python tools/bench.py
    python tools/bench.py --size 1920 --layers 8 --iterations 20

Reports per-operation median + p95 wall-clock time and pixels-per-second so
the numbers survive resolution changes.
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_REPO / "src"))


@dataclass
class BenchResult:
    name: str
    median_ms: float
    p95_ms: float
    pixels_per_second: float
    pixels: int


def _time(fn: Callable[[], np.ndarray], iterations: int) -> tuple[float, float, int]:
    durations: List[float] = []
    pixels = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)
        pixels = int(np.prod(out.shape))
    durations.sort()
    p50 = statistics.median(durations)
    p95 = durations[max(0, int(len(durations) * 0.95) - 1)]
    return p50, p95, pixels


def bench_gaussian_blur(size: int, iterations: int) -> BenchResult:
    from compositor.rendering.pixel_ops import gaussian_blur
    arr = (np.random.rand(size, size, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    p50, p95, px = _time(lambda: gaussian_blur(arr, sigma=5.0), iterations)
    return BenchResult("gaussian_blur", p50, p95, px / (p50 / 1000.0), px)


def bench_levels_image(size: int, iterations: int) -> BenchResult:
    from compositor.rendering.pixel_ops import ChannelRange, levels_image
    arr = (np.random.rand(size, size, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    ranges = [ChannelRange() for _ in range(4)]
    p50, p95, px = _time(lambda: levels_image(arr, ranges), iterations)
    return BenchResult("levels_image", p50, p95, px / (p50 / 1000.0), px)


def bench_curves(size: int, iterations: int) -> BenchResult:
    from compositor.rendering.pixel_ops import apply_curves_image
    arr = (np.random.rand(size, size, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    luts = [np.arange(256, dtype=np.uint8) for _ in range(4)]
    p50, p95, px = _time(lambda: apply_curves_image(arr, luts), iterations)
    return BenchResult("curves_image", p50, p95, px / (p50 / 1000.0), px)


def bench_hue_sat(size: int, iterations: int) -> BenchResult:
    from compositor.rendering.pixel_ops import hue_saturation_lightness
    arr = (np.random.rand(size, size, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    p50, p95, px = _time(lambda: hue_saturation_lightness(arr, hue_shift=30.0, sat_shift=10.0), iterations)
    return BenchResult("hue_sat", p50, p95, px / (p50 / 1000.0), px)


def bench_vignette(size: int, iterations: int) -> BenchResult:
    from compositor.rendering.pixel_ops import vignette
    arr = (np.random.rand(size, size, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    p50, p95, px = _time(lambda: vignette(arr, amount=0.5), iterations)
    return BenchResult("vignette", p50, p95, px / (p50 / 1000.0), px)


def bench_lens_correct(size: int, iterations: int) -> BenchResult:
    from compositor.rendering.pixel_ops import lens_correct
    arr = (np.random.rand(size, size, 4) * 255).astype(np.uint8)
    arr[..., 3] = 255
    p50, p95, px = _time(lambda: lens_correct(arr, strength=0.4), iterations)
    return BenchResult("lens_correct", p50, p95, px / (p50 / 1000.0), px)


def bench_composite(size: int, layers: int, iterations: int) -> BenchResult:
    """Composite ``layers`` 256×256 stacked RGBA images of size ``size``×``size``."""
    from compositor.rendering.layer_renderer import composite_document, CompositorOptions
    from compositor.document.editor_session import EditorSession
    from compositor.document.layers import ImageLayer, Transform, new_id
    rng = np.random.default_rng(0)
    cache = {}
    session = EditorSession(width=size, height=size)
    for i in range(layers):
        layer_id = new_id()
        cache[layer_id] = (rng.random((size, size, 4)) * 255).astype(np.uint8)
        cache[layer_id][..., 3] = 255
        session.add_layer(ImageLayer(
            id=layer_id,
            name=f"L{i}",
            transform=Transform(size=(size, size)),
            image_file=f"{layer_id}.png",
        ))
    opts = CompositorOptions(
        canvas_width=size,
        canvas_height=size,
        image_loader=cache.get,
    )

    def run() -> np.ndarray:
        return composite_document(session, opts)

    p50, p95, px = _time(run, iterations)
    return BenchResult(f"composite({layers}L)", p50, p95, px / (p50 / 1000.0), px)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=512, help="square canvas side, in pixels")
    p.add_argument("--layers", type=int, default=4, help="number of layers in composite benchmark")
    p.add_argument("--iterations", type=int, default=10)
    args = p.parse_args(argv)

    benches: List[BenchResult] = [
        bench_gaussian_blur(args.size, args.iterations),
        bench_levels_image(args.size, args.iterations),
        bench_curves(args.size, args.iterations),
        bench_hue_sat(args.size, args.iterations),
        bench_vignette(args.size, args.iterations),
        bench_lens_correct(args.size, args.iterations),
        bench_composite(args.size, args.layers, args.iterations),
    ]
    print(f"\n=== Bench ({args.size}×{args.size}, {args.iterations} iters) ===")
    print(f"{'op':<22} {'median ms':>10} {'p95 ms':>9} {'MP/s':>10}")
    for r in benches:
        mps = r.pixels_per_second / 1e6
        print(f"{r.name:<22} {r.median_ms:>10.2f} {r.p95_ms:>9.2f} {mps:>10.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
