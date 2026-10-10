"""PSD/PSB import tests.

The fixtures are built by :mod:`tests.psd_factory` rather than checked in as
binaries, so every assertion is against a file whose structure is written
down in that module.
"""
from __future__ import annotations

import numpy as np
import pytest

psd_tools = pytest.importorskip("psd_tools")

from compositor.format.manifest import CURRENT_VERSION, Manifest  # noqa: E402
from compositor.io.image_importer import encode_png, encode_png_mask  # noqa: E402
from compositor.io.project_store import ProjectStore  # noqa: E402
from compositor.io.psd_importer import import_psd, is_psd  # noqa: E402

from .psd_factory import build_sample_psd  # noqa: E402


@pytest.fixture(scope="module")
def sample_psd(tmp_path_factory):
    path = build_sample_psd(tmp_path_factory.mktemp("psd") / "sample.psd")
    return str(path)


@pytest.fixture(scope="module")
def sample_result(sample_psd):
    return import_psd(sample_psd)


def _by_name(result, name):
    return next(layer for layer in result["layers"] if layer["name"] == name)


# ---------------------------------------------------------------------------
# File recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [("a.psd", True), ("a.PSB", True), ("a.psb", True), ("a.png", False), ("a", False)],
)
def test_is_psd(name, expected):
    assert is_psd(name) is expected


def test_missing_dependency_raises_runtime_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psd_tools":
            raise ImportError("no psd_tools")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="psd-tools not installed"):
        import_psd("whatever.psd")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


def test_canvas_size(sample_result):
    assert sample_result["width"] == 200
    assert sample_result["height"] == 150


def test_flattened_is_a_real_composite(sample_result):
    flattened = sample_result["flattened"]
    assert flattened.shape == (150, 200, 4)
    assert flattened.dtype == np.uint8
    # The background layer is opaque dark blue, so the composite cannot be
    # the all-transparent placeholder the old importer returned.
    assert flattened[..., 3].max() == 255
    assert tuple(flattened[0, 0, :3]) == (20, 40, 80)


# ---------------------------------------------------------------------------
# Layer structure
# ---------------------------------------------------------------------------


def test_layer_names_and_order(sample_result):
    # psd-tools iterates bottom-to-top, same as Compositor.
    assert [layer["name"] for layer in sample_result["layers"]] == [
        "Group 1",
        "Nested",
        "Masked",
        "Background",
        "Top",
    ]


def test_groups_are_groups(sample_result):
    group = _by_name(sample_result, "Group 1")
    assert group["isGroup"] is True
    assert "imageFile" not in group


def test_children_point_at_their_group(sample_result):
    group = _by_name(sample_result, "Group 1")
    for name in ("Nested", "Masked"):
        assert _by_name(sample_result, name)["parentID"] == group["id"]


def test_top_level_layers_have_no_parent(sample_result):
    for name in ("Group 1", "Background", "Top"):
        assert "parentID" not in _by_name(sample_result, name)


def test_visibility_survives(sample_result):
    assert _by_name(sample_result, "Top")["isVisible"] is False
    assert _by_name(sample_result, "Background")["isVisible"] is True


def test_blend_mode_survives(sample_result):
    assert _by_name(sample_result, "Nested")["blendMode"] == "Multiply"
    assert _by_name(sample_result, "Background")["blendMode"] == "Normal"


def test_layer_offset_becomes_transform_origin(sample_result):
    nested = _by_name(sample_result, "Nested")
    assert nested["transform"]["origin"] == [40.0, 30.0]
    assert nested["transform"]["size"] == [60.0, 40.0]


# ---------------------------------------------------------------------------
# Pixels
# ---------------------------------------------------------------------------


def test_pixels_are_rgba_uint8(sample_result):
    pixels = sample_result["pixels"][_by_name(sample_result, "Background")["id"]]
    assert pixels.shape == (150, 200, 4)
    assert pixels.dtype == np.uint8
    assert tuple(pixels[0, 0]) == (20, 40, 80, 255)


def test_pixels_map_is_keyed_by_layer_id(sample_result):
    image_layers = [
        layer for layer in sample_result["layers"] if "imageFile" in layer
    ]
    assert set(sample_result["pixels"]) == {layer["id"] for layer in image_layers}


def test_layer_dicts_carry_no_manifest_violations(sample_result):
    for layer in sample_result["layers"]:
        # Pixel arrays travel in ``pixels``/``masks``, never inside the dict,
        # because these dicts are written to the manifest verbatim.
        assert "pixels" not in layer
        assert "mask_pixels" not in layer
        if "imageFile" in layer:
            assert layer["imageFile"] == f"{layer['id']}.png"


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_mask_is_imported_and_enabled(sample_result):
    masked = _by_name(sample_result, "Masked")
    assert masked["maskEnabled"] is True
    assert masked["maskFile"] == f"{masked['id']}.mask.png"


def test_mask_matches_the_layer_extent(sample_result):
    masked = _by_name(sample_result, "Masked")
    mask = sample_result["masks"][masked["id"]]
    assert mask.shape == (40, 60)  # the layer's own bbox, not the canvas
    assert mask.dtype == np.uint8


def test_mask_polarity_and_placement(sample_result):
    """White means visible in Compositor; PSD stores the opposite."""
    masked = _by_name(sample_result, "Masked")
    row = sample_result["masks"][masked["id"]][0]
    # Fixture blocks are [hidden, hidden, visible, visible, hidden] in
    # Compositor terms; the mask sits at the layer's top-left.
    assert row[0] == 0
    assert row[8] == 0
    assert row[16] == 255
    assert row[24] == 255
    assert row[32] == 0
    # Past the mask rectangle the PSD's background colour (black) applies,
    # i.e. fully visible.
    assert row[45] == 255
    assert row[59] == 255


def test_layers_without_masks_have_no_mask_file(sample_result):
    for name in ("Nested", "Background", "Top"):
        assert "maskFile" not in _by_name(sample_result, name)


# ---------------------------------------------------------------------------
# Round trip into a project
# ---------------------------------------------------------------------------


def test_result_writes_into_a_valid_project(sample_result, tmp_path):
    manifest = Manifest(
        version=CURRENT_VERSION,
        color_space="sRGB",
        document_id="A" * 8,
        width=sample_result["width"],
        height=sample_result["height"],
        layers=sample_result["layers"],
    )
    store = ProjectStore.create(tmp_path / "sample.comp", manifest)
    for layer_id, pixels in sample_result["pixels"].items():
        store.write_image(layer_id, encode_png(pixels))
    for layer_id, mask in sample_result["masks"].items():
        store.write_image(layer_id, encode_png_mask(mask), mask=True)
    store.save_manifest(manifest)

    reloaded = ProjectStore.open(tmp_path / "sample.comp").load_manifest()
    assert len(reloaded.layers) == len(sample_result["layers"])
    names = [layer["name"] for layer in reloaded.layers]
    assert names == ["Group 1", "Nested", "Masked", "Background", "Top"]

    masked = next(layer for layer in reloaded.layers if layer["name"] == "Masked")
    assert (store.paths.images_dir / masked["maskFile"]).is_file()


def test_layers_from_manifest_round_trip(sample_result):
    from compositor.document.layers import layers_from_manifest

    layers = layers_from_manifest(sample_result["layers"])
    names = [layer.name for layer in layers]
    assert names == ["Group 1", "Nested", "Masked", "Background", "Top"]
    nested = next(layer for layer in layers if layer.name == "Nested")
    assert nested.blend_mode.value == "Multiply"
    assert nested.transform.origin == (40.0, 30.0)
    top = next(layer for layer in layers if layer.name == "Top")
    assert top.is_visible is False


def test_flat_psd_without_masks_imports(tmp_path):
    from .psd_factory import write_flat_psd

    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[..., :3] = 10
    rgba[..., 3] = 255
    result = import_psd(write_flat_psd(tmp_path / "flat.psd", rgba))
    assert result["masks"] == {}
    assert len(result["pixels"]) == 1
    Manifest(
        version=CURRENT_VERSION,
        color_space="sRGB",
        document_id="B" * 8,
        width=result["width"],
        height=result["height"],
        layers=result["layers"],
    ).to_dict()
