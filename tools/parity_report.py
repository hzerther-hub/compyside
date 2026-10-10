"""Visual parity report — render a .comp through the PySide pipeline and
compare it against a reference PNG (typically produced by the Swift build).

Usage:
    # Single project
    python -m tools.parity_report path/to/project.comp \\
        --reference path/to/reference.png \\
        --output-dir dist/parity/

    # Batch mode — every .comp under a directory
    python -m tools.parity_report --reference-dir refs/renders/ projects/

Outputs:
    dist/parity/<name>/pyside.png     — the local render
    dist/parity/<name>/reference.png  — copy of the reference (if separate)
    dist/parity/<name>/diff.png       — heatmap of mismatched pixels
    dist/parity/<name>/report.json    — per-project metrics
    dist/parity/summary.md            — overall table across all projects
    dist/parity/summary.json          — same data, machine-readable

Metrics:
    rmse                Root-mean-square error per channel (0..255).
    mae                 Mean absolute error per channel (0..255).
    max_delta           Worst single-channel difference observed.
    pixels_above_1      Fraction of pixels with any channel > 1.
    pixels_above_5      Fraction with > 5.
    pixels_above_10     Fraction with > 10.
    p99                 99th-percentile absolute error.
    dimensions_match    True if local and reference share (H, W, C).

The script is intentionally standalone — it does not require Qt, so it runs
in headless CI on every commit. The renderer is pure NumPy + cv2.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running from the repo root without installing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from compositor.format.manifest import load_manifest  # noqa: E402
from compositor.io.image_importer import load_image  # noqa: E402
from compositor.io.project_store import ProjectStore  # noqa: E402
from compositor.rendering.layer_renderer import (  # noqa: E402
    CompositorOptions,
    composite_document,
)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_project(project_path: Path) -> Tuple[np.ndarray, dict]:
    """Load a .comp project and composite it through the PySide renderer.

    Returns the (HxWx4 uint8) composite plus a dict of useful metadata
    (dimensions, layer count, manifest version).
    """
    project_path = project_path.resolve()
    if project_path.is_dir():
        store = ProjectStore.open(project_path)
        manifest = store.load_manifest()
        cache: Dict[str, np.ndarray] = {}
        mask_cache: Dict[str, np.ndarray] = {}
        for layer in manifest.layers:
            if layer.get("imageFile"):
                img_path = store.paths.images_dir / layer["imageFile"]
                if img_path.is_file():
                    cache[layer["id"]] = load_image(img_path)
            if layer.get("maskFile"):
                mask_path = store.paths.images_dir / layer["maskFile"]
                if mask_path.is_file():
                    arr = load_image(mask_path)
                    mask_cache[layer["id"]] = arr[..., 0] if arr.ndim == 3 else arr
        opts = CompositorOptions(
            canvas_width=manifest.width,
            canvas_height=manifest.height,
            image_loader=cache.get,
            mask_loader=mask_cache.get,
        )
        # Build a transient EditorSession so the renderer can walk the layer
        # list. We don't need undo/redo here.
        from compositor.document.editor_session import EditorSession
        session = EditorSession(
            width=manifest.width,
            height=manifest.height,
            color_space=manifest.color_space,
            document_id=manifest.document_id,
            resolution=manifest.resolution,
        )
        session.load_manifest_layers(manifest.layers)
        composite = composite_document(session, opts)
        meta = {
            "project": str(project_path),
            "width": manifest.width,
            "height": manifest.height,
            "layer_count": len(manifest.layers),
            "manifest_version": manifest.version,
        }
        return composite, meta
    elif project_path.suffix.lower() == ".json" and project_path.name == "manifest.json":
        # Bare manifest without an enclosing .comp folder — bail out gracefully.
        manifest = load_manifest(project_path)
        return (
            np.zeros((manifest.height, manifest.width, 4), dtype=np.uint8),
            {"project": str(project_path), "note": "bare manifest, no images"},
        )
    else:
        raise ValueError(f"not a .comp folder or manifest.json: {project_path}")


# ---------------------------------------------------------------------------
# Diff metrics
# ---------------------------------------------------------------------------


@dataclass
class DiffMetrics:
    rmse_per_channel: List[float] = field(default_factory=list)
    mae_per_channel: List[float] = field(default_factory=list)
    max_delta: int = 0
    pixels_above_1: float = 0.0
    pixels_above_5: float = 0.0
    pixels_above_10: float = 0.0
    p99: float = 0.0
    dimensions_match: bool = False
    local_shape: Tuple[int, int, int] = (0, 0, 0)
    reference_shape: Tuple[int, int, int] = (0, 0, 0)


def _ensure_rgba(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img, np.full(img.shape, 255, dtype=np.uint8)], axis=-1)
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    return img


def diff(local: np.ndarray, reference: np.ndarray) -> DiffMetrics:
    """Compute per-channel diff metrics between two RGBA uint8 arrays."""
    local = _ensure_rgba(local)
    reference = _ensure_rgba(reference)
    metrics = DiffMetrics(
        local_shape=tuple(local.shape),
        reference_shape=tuple(reference.shape),
        dimensions_match=local.shape == reference.shape,
    )
    if not metrics.dimensions_match:
        return metrics
    diff_arr = np.abs(local.astype(np.int32) - reference.astype(np.int32))
    # Per-channel metrics.
    for c in range(diff_arr.shape[-1]):
        ch = diff_arr[..., c].astype(np.float64)
        metrics.rmse_per_channel.append(float(np.sqrt(np.mean(ch ** 2))))
        metrics.mae_per_channel.append(float(np.mean(ch)))
    metrics.max_delta = int(diff_arr.max())
    # Pixel-level (any-channel) thresholds.
    pixel_diff = diff_arr.max(axis=-1)
    metrics.pixels_above_1 = float(np.mean(pixel_diff > 1))
    metrics.pixels_above_5 = float(np.mean(pixel_diff > 5))
    metrics.pixels_above_10 = float(np.mean(pixel_diff > 10))
    metrics.p99 = float(np.percentile(pixel_diff, 99))
    return metrics


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def render_diff_heatmap(
    local: np.ndarray,
    reference: np.ndarray,
    *,
    out_path: Path,
    max_intensity: int = 64,
) -> None:
    """Save a heatmap PNG highlighting where the local render diverges.

    Pixels within ``max_intensity`` of the reference stay grayscale; larger
    deltas glow red proportionally.
    """
    local = _ensure_rgba(local)
    reference = _ensure_rgba(reference)
    if local.shape != reference.shape:
        # Render a placeholder if shapes mismatch.
        h = max(local.shape[0], reference.shape[0])
        w = max(local.shape[1], reference.shape[1])
        canvas = np.full((h, w, 3), 32, dtype=np.uint8)
        cv2_rgb = canvas
        from PIL import Image
        Image.fromarray(cv2_rgb).save(out_path)
        return
    diff_arr = np.abs(local.astype(np.int32) - reference.astype(np.int32)).max(axis=-1)
    # Base layer: a faded version of the reference so the heatmap sits on
    # something readable.
    base = (reference[..., :3].astype(np.float32) * 0.4).astype(np.uint8)
    intensity = np.clip(diff_arr.astype(np.float32) / max(1, max_intensity), 0.0, 1.0)
    red = np.zeros_like(base)
    red[..., 0] = 255
    blended = (base.astype(np.float32) * (1 - intensity[..., None])
               + red.astype(np.float32) * intensity[..., None]).astype(np.uint8)
    from PIL import Image
    Image.fromarray(blended).save(out_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _read_reference(path: Path) -> np.ndarray:
    """Load a reference PNG as RGBA uint8."""
    from PIL import Image
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    return arr


def run_single(
    project: Path,
    reference: Optional[Path],
    output_dir: Path,
) -> dict:
    """Render one project + optional reference, write artefacts, return metrics."""
    project = project.resolve()
    name = project.stem if project.is_dir() else project.stem
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    local, meta = render_project(project)
    from PIL import Image
    Image.fromarray(local).save(out_dir / "pyside.png")

    report: dict = {"project": str(project), "local": meta}

    if reference is None:
        report["status"] = "no_reference"
        report["metrics"] = asdict(DiffMetrics())
    else:
        ref_img = _read_reference(reference)
        Image.fromarray(ref_img).save(out_dir / "reference.png")
        m = diff(local, ref_img)
        render_diff_heatmap(local, ref_img, out_path=out_dir / "diff.png")
        report["reference"] = str(reference)
        report["metrics"] = asdict(m)
        report["status"] = "ok" if m.dimensions_match and m.max_delta < 2 else "differs"

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_batch(
    projects_dir: Path,
    reference_dir: Path,
    output_dir: Path,
) -> List[dict]:
    """Render every .comp under ``projects_dir``, compare against matching PNG
    in ``reference_dir`` (same filename + .png extension).
    """
    projects: List[Path] = []
    for entry in sorted(projects_dir.iterdir()):
        if entry.is_dir() and (entry / "manifest.json").is_file():
            projects.append(entry)
        elif entry.suffix.lower() == ".json" and entry.name == "manifest.json":
            projects.append(entry.parent)

    reports: List[dict] = []
    for proj in projects:
        ref = reference_dir / f"{proj.stem}.png"
        if not ref.is_file():
            ref = None
        try:
            report = run_single(proj, ref, output_dir)
            reports.append(report)
        except Exception as exc:  # noqa: BLE001
            reports.append({"project": str(proj), "status": "error", "error": str(exc)})
    return reports


def write_summary(reports: List[dict], output_dir: Path) -> None:
    """Emit a Markdown + JSON summary aggregating every project's metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    (output_dir / "summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown
    lines = ["# Visual Parity Summary", ""]
    lines.append(f"Projects: **{len(reports)}**")
    lines.append("")
    lines.append("| Project | Status | RMSE (avg) | Max Δ | Pixels > 5 |")
    lines.append("|---|---|---|---|---|")
    for r in reports:
        if r.get("status") == "error":
            lines.append(f"| `{Path(r['project']).name}` | ❌ error | — | — | — |")
            continue
        m = r.get("metrics", {})
        rmse = m.get("rmse_per_channel") or [0.0]
        avg_rmse = sum(rmse) / max(1, len(rmse))
        lines.append(
            f"| `{Path(r['project']).name}` | "
            f"{r.get('status', '?')} | "
            f"{avg_rmse:.2f} | "
            f"{m.get('max_delta', 0)} | "
            f"{(m.get('pixels_above_5', 0) * 100):.2f}% |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("project", nargs="?", help=".comp folder or manifest.json")
    parser.add_argument("--reference", help="Reference PNG produced by the Swift build")
    parser.add_argument("--reference-dir", help="Directory of reference PNGs (batch mode)")
    parser.add_argument("--projects-dir", help="Directory of .comp projects (batch mode)")
    parser.add_argument("--output-dir", default="dist/parity", help="Output directory")
    parser.add_argument(
        "--max-intensity",
        type=int,
        default=64,
        help="Pixel-delta value that maps to full red in the diff heatmap",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    output_dir = Path(args.output_dir).resolve()

    if args.projects_dir and args.reference_dir:
        projects_dir = Path(args.projects_dir).resolve()
        reference_dir = Path(args.reference_dir).resolve()
        reports = run_batch(projects_dir, reference_dir, output_dir)
        write_summary(reports, output_dir)
        print(f"Wrote summary to {output_dir / 'summary.md'}")
        return 0

    if args.project:
        project = Path(args.project).resolve()
        reference = Path(args.reference).resolve() if args.reference else None
        report = run_single(project, reference, output_dir)
        print(json.dumps(report, indent=2))
        return 0

    _build_parser().print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
