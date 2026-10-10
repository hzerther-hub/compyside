"""Selection mask tests."""
from __future__ import annotations

import numpy as np

from compositor.rendering import selections as sel


def test_rectangle_mask() -> None:
    mask = sel.rectangle_mask((20, 30), (5, 5, 15, 15))
    assert mask.shape == (20, 30)
    assert mask[10, 10] == 255
    assert mask[0, 0] == 0


def test_ellipse_mask() -> None:
    mask = sel.ellipse_mask((20, 20), (5, 5, 15, 15))
    # Centre should be inside; corners outside.
    assert mask[10, 10] == 255
    assert mask[0, 0] == 0


def test_lasso_triangle() -> None:
    pts = [(2, 2), (18, 2), (10, 18)]
    mask = sel.lasso_mask((20, 20), pts)
    # Centroid should be inside.
    assert mask[10, 10] == 255


def test_wand_uniform() -> None:
    img = np.full((20, 20, 3), 100, dtype=np.uint8)
    mask = sel.wand_mask(img, (10, 10), tolerance=10)
    assert mask.sum() == 20 * 20 * 255


def test_wand_two_color() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, :10, :] = 100
    mask = sel.wand_mask(img, (0, 0), tolerance=10)
    # Only the left half should be selected.
    assert mask[:, :10].sum() > mask[:, 10:].sum()


def test_feather() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    blurred = sel.feather(mask, radius=3.0)
    # Edge pixels should be partial.
    assert 0 < blurred[10, 20] < 255


def test_union_difference_intersection() -> None:
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10), dtype=np.uint8)
    a[2:5, :] = 255  # 30 cells
    b[4:7, :] = 255  # 30 cells
    u = sel.union(a, b)
    d = sel.difference(a, b)
    i = sel.intersection(a, b)
    # Union: rows 2..6 (5 rows of 10 cols).
    assert u[2:7, :].sum() == 255 * 10 * 5
    # Intersection: row 4 only.
    assert i.sum() == 255 * 10 * 1
    # Difference: |a-b| = 255 where exactly one is set (rows 2, 3 from a; rows 5, 6 from b).
    assert d.sum() == 255 * 10 * 4
