"""Axis-order detection from TIFF / OME-TIFF metadata.

Regression coverage for the case where tifffile cannot classify a stack axis
and labels it 'I' (sequence) or 'Q' (other) -- e.g. a generic multi-page TIFF,
or an OME/ImageJ file whose metadata tifffile could not fully resolve. Such a
file must be opened as a time series, not rejected with
``ValueError: Invalid axis letter(s): I``.
"""
import numpy as np
import tifffile
import pytest

from sarcasm import SarcAsM
from sarcasm.core import SarcAsMBase


def _detect(path):
    """Run the real detection + validation on a written file."""
    with tifffile.TiffFile(path) as tif:
        axes = SarcAsMBase._determine_axes(tif.series[0], tif)
    SarcAsMBase._validate_axes(axes)  # raises on illegal / duplicate letters
    return axes


# --- generic (metadata-less) stacks: tifffile tags the stack axis 'I' -------

def test_generic_multipage_stack_detected_as_time(tmp_path):
    """A plain multi-page TIFF (tifffile axes 'IYX') must resolve to 'TYX'."""
    p = tmp_path / "generic_stack.tif"
    tifffile.imwrite(p, np.zeros((8, 32, 32), np.uint16), metadata=None, ome=False)
    with tifffile.TiffFile(p) as tif:
        assert tif.series[0].axes == "IYX"  # precondition: this is the 'I' case
    assert _detect(p) == "TYX"


def test_generic_stack_opens_end_to_end(tmp_path):
    """Constructing on an 'I'-axis file must succeed and read as a movie."""
    p = tmp_path / "compiled_2D.ome.tif"
    # ome=False + no metadata reproduces a stack tifffile cannot tag as OME
    tifffile.imwrite(p, np.zeros((8, 32, 32), np.uint16), metadata=None, ome=False)
    sarc = SarcAsM(p, pixelsize=0.1, restart=True)
    assert sarc.metadata.axes == "TYX"
    assert sarc.read_imgs().shape == (8, 32, 32)


# --- well-formed OME-TIFFs (as the cv8000 compiler writes them) still work ---

@pytest.mark.parametrize(
    "shape, axes",
    [
        ((10, 32, 32), "TYX"),   # 2D time series
        ((32, 32), "YX"),        # single frame
        ((6, 32, 32), "ZYX"),    # z-stack
        ((2, 10, 32, 32), "CTYX"),  # multi-channel movie
    ],
)
def test_ome_axes_roundtrip(tmp_path, shape, axes):
    p = tmp_path / f"ome_{axes}.ome.tif"
    tifffile.imwrite(p, np.zeros(shape, np.uint16),
                     metadata={"axes": axes, "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.1})
    assert _detect(p) == axes


# --- direct unit coverage of the I/Q remap ----------------------------------

def test_validate_rejects_raw_generic_letters():
    with pytest.raises(ValueError, match="Invalid axis letter"):
        SarcAsMBase._validate_axes("IYX")
    with pytest.raises(ValueError, match="Invalid axis letter"):
        SarcAsMBase._validate_axes("QYX")
