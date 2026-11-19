import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import assign_crs

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import GeoMADPostProcessor  # noqa: E402


@pytest.fixture
def geomad():
    gb = GeoBox.from_bbox((0, 0, 100, 100), "epsg:32660", resolution=10)
    shape = (10, 10)
    red = np.full(shape, 1500, dtype="uint16")
    red[0, 0] = 0  # nodata
    ds = xr.Dataset(
        {
            "red": (("y", "x"), red),
            "green": (("y", "x"), np.full(shape, 3000, dtype="uint16")),
            "blue": (("y", "x"), np.full(shape, 6000, dtype="uint16")),
        }
    )
    ds = assign_crs(
        ds.assign_coords(y=gb.coords["y"].values, x=gb.coords["x"].values), gb.crs
    )
    for v in ds:
        ds[v].attrs["nodata"] = 0
    return ds


def test_visual_added(geomad):
    out = GeoMADPostProcessor(vars=["red", "green", "blue"], visual=True).process(
        geomad
    )
    v = out["visual"]
    assert v.dtype == "uint8" and v.dims == ("y", "x", "band")
    assert list(v.band.values) == ["r", "g", "b", "a"]
    # stretched 0-3000: 1500 -> mid, 3000 and above -> 255
    assert abs(int(v.values[5, 5, 0]) - 128) <= 1
    assert v.values[5, 5, 1] == 255 and v.values[5, 5, 2] == 255
    # nodata is transparent
    assert v.values[0, 0, 3] == 0 and v.values[5, 5, 3] == 255


def test_visual_off_by_default(geomad):
    out = GeoMADPostProcessor(vars=["red", "green", "blue"]).process(geomad)
    assert "visual" not in out


def test_visual_respects_scale_offset(geomad):
    scaled = geomad.copy()
    for v in ["red", "green", "blue"]:
        scaled[v] = (scaled[v] * 1e-4).astype("float32")
        scaled[v].attrs["nodata"] = float("nan")
    scaled["red"].values[0, 0] = np.nan
    out = GeoMADPostProcessor(
        vars=["red", "green", "blue"], scale=1e-4, offset=0, visual=True
    ).process(scaled)
    assert abs(int(out["visual"].values[5, 5, 0]) - 128) <= 1
    assert out["visual"].values[0, 0, 3] == 0
