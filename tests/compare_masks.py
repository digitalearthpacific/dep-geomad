"""Compare the production S-2 cloud mask against several candidate configurations.

For each region in test_regions.geojson, loads a small Sentinel-2 chip
(capped at ~1000x1000 px), computes a full geomedian under each masking
config, and writes an RGB PNG per config to tests/output/.

Old (production, "s2_old_filter" style): dep_tools.s2_utils.mask_clouds,
filters=[("dilation", 3), ("erosion", 2)], applied to the combined
{saturated/defective, cloud shadows, cloud medium/high probability, thin
cirrus} SCL mask.

New (candidate): saturated/defective pixels masked with no morphology,
unioned with the combined cloud-medium + cloud-high mask after
opening(3) -> closing(3) -> dilation(3).

Omni (candidate): cloud detection via OmniCloudMask (a Red/Green/NIR
deep-learning segmenter) instead of the SCL band. Run per-timestep since
predict_from_array takes a single (3, H, W) scene; only thick cloud (class 1)
is masked, cleaned up with the same 3,3,3 morphology as "new".

(s2cloudless was also tried and dropped -- it needs a real B10/cirrus band,
which L2A doesn't have, and zero-filling it badly under-detects cloud.)

Tmask (candidate): a simplified version of Zhu & Woodcock's (2014) Tmask --
multi-temporal robust-outlier detection instead of per-scene classification.
Flags an observation as cloud when it's anomalously bright (blue and green)
relative to that pixel's own median/MAD across the whole time series. Tried
and found to plateau well short of the SCL-based configs on a heavily
cloudy test region -- see conversation, kept for reference.

And (candidate): intersection ensemble of the raw SCL medium+high classes
and OmniCloudMask's raw thick-cloud detection -- only masked where BOTH
agree, morphology applied to the intersection afterwards. Never masks more
than the more generous of the two, trading missed-cloud risk for keeping
more observations per pixel.

Usage:
    python tests/compare_masks.py
    python tests/compare_masks.py --region fiji_gao --region fiji_suva
    python tests/compare_masks.py --overwrite
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import typer
from datacube_compute import geomedian_with_mads
from dep_tools.loaders import OdcLoader
from dep_tools.s2_utils import mask_clouds as mask_clouds_old
from dep_tools.searchers import PystacSearcher
from odc.algo import erase_bad, mask_cleanup
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.stac import configure_rio
import odc.geo.xr  # noqa: F401  (registers the .odc xarray accessor)
from omnicloudmask import predict_from_array
from PIL import Image
from typing_extensions import Annotated

CATALOG = "https://earth-search.aws.element84.com/v1/"
COLLECTION = "sentinel-2-l2a"
PACIFIC_EPSG = "EPSG:3832"
TARGET_PX = 1000

OLD_FILTERS = [("dilation", 3), ("erosion", 2)]
NEW_FILTERS = [("opening", 3), ("closing", 3), ("dilation", 3)]

SATURATED_OR_DEFECTIVE = 1
CLOUD_MEDIUM_PROBABILITY = 8
CLOUD_HIGH_PROBABILITY = 9

# Extra bands only needed to feed a masker, dropped again before the geomedian.
AUX_BANDS = ("scl", "nir08")


def _scl_medium_high(xr):
    """Raw, pre-morphology SCL medium+high cloud probability classes."""
    return xr.scl.isin([CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY])


def mask_clouds_new(xr):
    cloud = mask_cleanup(_scl_medium_high(xr), NEW_FILTERS)
    cloud = cloud | (xr.scl == SATURATED_OR_DEFECTIVE)
    return erase_bad(xr, cloud)


def _omni_thick_cloud(loaded):
    """Raw, pre-morphology OmniCloudMask thick-cloud (class 1) detection.
    Takes an already-`.compute()`d Dataset (predict_from_array needs numpy)."""
    red = loaded["red"].values.astype("float32")
    green = loaded["green"].values.astype("float32")
    nir = loaded["nir08"].values.astype("float32")
    mask = np.zeros(red.shape, dtype=bool)
    for t in range(red.shape[0]):
        pred = predict_from_array(np.stack([red[t], green[t], nir[t]]), no_data_value=0)
        mask[t] = pred[0] == 1
    return loaded["red"].copy(data=mask)


def mask_clouds_omni(xr):
    """OmniCloudMask instead of SCL, thick cloud only, cleaned up with the
    same opening(3)/closing(3)/dilation(3) morphology as "new"."""
    loaded = xr.compute()
    cloud = mask_cleanup(_omni_thick_cloud(loaded), NEW_FILTERS)
    return erase_bad(loaded, cloud)


def mask_clouds_ensemble_and(xr):
    """Cloud only where the raw SCL medium+high classes AND OmniCloudMask's
    raw thick-cloud detection agree -- an intersection of the two RAW masks,
    cleaned up with the 3,3,3 morphology afterwards (not applied to each
    mask separately first). A cloud call from either method alone isn't
    enough on its own, so this never masks more than the more generous of
    the two; the risk is missing cloud that only one method catches."""
    loaded = xr.compute()
    cloud = _scl_medium_high(loaded) & _omni_thick_cloud(loaded)
    cloud = mask_cleanup(cloud, NEW_FILTERS)
    return erase_bad(loaded, cloud)


def mask_clouds_tmask(
    xr,
    pass1_thresh: float = 1.0,
    pass2_thresh: float = 2.0,
    nir_thresh: tuple[float, float] = (0.8, 1.5),
    clear_percentile: float = 20.0,
):
    """Multi-temporal robust-outlier cloud mask, inspired by Zhu & Woodcock's
    (2014) Tmask. Rather than classifying each scene independently, flags an
    observation as cloud when it's anomalously bright relative to that SAME
    pixel's own robust distribution across the whole time series --
    self-referencing, so it isn't fooled by permanently bright surfaces
    (sand, coral, roofs) the way single-scene brightness thresholds are.

    Uses a low percentile (not the median) as the "clear" reference: cloud
    only ever brightens a pixel relative to its true clear-sky value, never
    darkens it, so the lower tail of the distribution stays a reasonable
    clear-sky proxy even when most observations at a pixel are cloudy.

    Two passes per band: the first pass's own outliers are excluded before
    re-estimating the reference/spread, since with a high cloud fraction
    (e.g. wet-season Fiji, where well under a quarter of scenes are clear)
    a single pass' "robust" stats are themselves cloud-contaminated. Flags
    a pixel as cloud if it's a two-pass outlier in any 2 of {blue, green,
    nir08} -- ceiling of this approach: still detects far less cloud than
    the SCL-based configs (see conversation), needs a real harmonic-
    regression Tmask to close that gap.
    """
    loaded = xr.compute()

    def is_outlier(da, thresh1, thresh2):
        v = da.values  # uint16, no copy
        vf = np.where(v == 0, np.nan, v).astype("float32")  # transient
        ref = np.nanpercentile(vf, clear_percentile, axis=0, keepdims=True)
        mad = np.nanmedian(np.abs(vf - ref), axis=0, keepdims=True) * 1.4826
        mad = np.where(mad == 0, 1.0, mad)
        pass1 = (vf - ref) / mad > thresh1
        vf2 = np.where(pass1, np.nan, vf)
        ref2 = np.nanpercentile(vf2, clear_percentile, axis=0, keepdims=True)
        mad2 = np.nanmedian(np.abs(vf2 - ref2), axis=0, keepdims=True) * 1.4826
        del vf, vf2
        mad2 = np.where(mad2 == 0, 1.0, mad2)
        z2 = (v.astype("float32") - ref2) / mad2
        return z2 > thresh2

    blue = is_outlier(loaded["blue"], pass1_thresh, pass2_thresh)
    green = is_outlier(loaded["green"], pass1_thresh, pass2_thresh)
    nir = is_outlier(loaded["nir08"], *nir_thresh)
    cloud = (blue & green) | (blue & nir) | (green & nir)
    mask_da = mask_cleanup(loaded["red"].copy(data=cloud), NEW_FILTERS)
    return erase_bad(loaded, mask_da)


def geobox_for(geom_json: dict, target_px: int = TARGET_PX) -> GeoBox:
    geom = Geometry(geom_json, crs="EPSG:4326").to_crs(PACIFIC_EPSG)
    bbox = geom.boundingbox
    resolution = max(bbox.span_x, bbox.span_y) / target_px
    return GeoBox.from_bbox(
        (bbox.left, bbox.bottom, bbox.right, bbox.top),
        crs=PACIFIC_EPSG,
        resolution=resolution,
    )


NODATA_COLOR = np.array([255, 165, 0, 255], dtype="uint8")  # orange, opaque


def write_rgb_png(ds, path: Path, vmin: float = 0.0, vmax: float = 3000.0) -> None:
    path.unlink(missing_ok=True)  # avoid any doubt about a stale file being seen
    rgba = ds.odc.to_rgba(vmin=vmin, vmax=vmax).values.copy()
    rgba[rgba[..., 3] == 0] = NODATA_COLOR
    Image.fromarray(rgba, "RGBA").save(path)


def main(
    datetime: Annotated[str, typer.Option(help="STAC datetime range.")] = "2024",
    regions_path: Annotated[Path, typer.Option()] = Path("tests/test_regions.geojson"),
    output_dir: Annotated[Path, typer.Option()] = Path("tests/output"),
    region: Annotated[
        list[str] | None, typer.Option(help="Filter to region name(s). May repeat.")
    ] = None,
    min_timesteps: Annotated[int, typer.Option()] = 5,
    overwrite: Annotated[
        bool, typer.Option("--overwrite/--no-overwrite", help="Redo files that already exist.")
    ] = False,
) -> None:
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(regions_path)
    if region:
        gdf = gdf[gdf["name"].isin(region)]

    maskers = {
        "old": lambda d: mask_clouds_old(d, filters=OLD_FILTERS, keep_ints=True),
        "new": mask_clouds_new,
        "omni": mask_clouds_omni,
        "tmask": mask_clouds_tmask,
        "and": mask_clouds_ensemble_and,
    }

    for _, row in gdf.iterrows():
        name = row["name"]

        pending = {tag: output_dir / f"{name}_{tag}.png" for tag in maskers}
        if not overwrite:
            pending = {tag: path for tag, path in pending.items() if not path.exists()}
        if not pending:
            print(f"[{name}] all outputs exist, skipping (use --overwrite to redo)")
            continue

        geobox = geobox_for(row.geometry.__geo_interface__)
        print(f"[{name}] geobox shape={geobox.shape}")

        items = PystacSearcher(
            catalog=CATALOG, collections=[COLLECTION], datetime=datetime
        ).search(area=geobox)
        if len(items) < min_timesteps:
            print(f"[{name}] only {len(items)} scenes (<{min_timesteps}), skipping")
            continue

        data = OdcLoader(
            bands=["scl", "red", "green", "blue", *AUX_BANDS[1:]],
            chunks=dict(time=1, x=1024, y=1024),
            groupby="solar_day",
            fail_on_error=False,
            nodata=0,
        ).load(items, areas=geobox)

        for tag, out_path in pending.items():
            masked = maskers[tag](data)
            masked = masked.drop_vars([v for v in AUX_BANDS if v in masked.data_vars])
            geomad = geomedian_with_mads(
                masked, work_chunks=(1000, 1000), num_threads=4, maxiters=100
            ).compute()
            write_rgb_png(geomad, out_path)
            print(f"[{name}] wrote {out_path}")


if __name__ == "__main__":
    typer.run(main)
