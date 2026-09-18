"""Compare the production S-2 cloud mask against a candidate configuration.

For each region in test_regions.geojson, loads a small Sentinel-2 chip
(capped at ~1000x1000 px), computes a full geomedian under both masking
configs, and writes an RGB PNG per config to tests/output/.

Old (production, "s2_old_filter" style): dep_tools.s2_utils.mask_clouds,
filters=[("dilation", 3), ("erosion", 2)], applied to the combined
{saturated/defective, cloud shadows, cloud medium/high probability, thin
cirrus} SCL mask.

New (candidate): saturated/defective pixels masked with no morphology,
unioned with the combined cloud-medium + cloud-high mask after
opening(3) -> closing(3) -> dilation(3).

Usage:
    python tests/compare_masks.py
    python tests/compare_masks.py --region fiji_gao --region fiji_suva
"""

from pathlib import Path

import geopandas as gpd
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


def mask_clouds_new(xr):
    cloud = xr.scl.isin([CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY])
    cloud = mask_cleanup(cloud, NEW_FILTERS)
    saturated = xr.scl == SATURATED_OR_DEFECTIVE
    return erase_bad(xr, cloud | saturated)


def geobox_for(geom_json: dict, target_px: int = TARGET_PX) -> GeoBox:
    geom = Geometry(geom_json, crs="EPSG:4326").to_crs(PACIFIC_EPSG)
    bbox = geom.boundingbox
    resolution = max(bbox.span_x, bbox.span_y) / target_px
    return GeoBox.from_bbox(
        (bbox.left, bbox.bottom, bbox.right, bbox.top),
        crs=PACIFIC_EPSG,
        resolution=resolution,
    )


def write_rgb_png(ds, path: Path, vmin: float = 0.0, vmax: float = 3000.0) -> None:
    rgba = ds.odc.to_rgba(vmin=vmin, vmax=vmax)
    Image.fromarray(rgba.values, "RGBA").save(path)


def main(
    datetime: Annotated[str, typer.Option(help="STAC datetime range.")] = "2024",
    regions_path: Annotated[Path, typer.Option()] = Path("tests/test_regions.geojson"),
    output_dir: Annotated[Path, typer.Option()] = Path("tests/output"),
    region: Annotated[
        list[str] | None, typer.Option(help="Filter to region name(s). May repeat.")
    ] = None,
    min_timesteps: Annotated[int, typer.Option()] = 5,
) -> None:
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(regions_path)
    if region:
        gdf = gdf[gdf["name"].isin(region)]

    for _, row in gdf.iterrows():
        name = row["name"]
        geobox = geobox_for(row.geometry.__geo_interface__)
        print(f"[{name}] geobox shape={geobox.shape}")

        items = PystacSearcher(
            catalog=CATALOG, collections=[COLLECTION], datetime=datetime
        ).search(area=geobox)
        if len(items) < min_timesteps:
            print(f"[{name}] only {len(items)} scenes (<{min_timesteps}), skipping")
            continue

        data = OdcLoader(
            bands=["scl", "red", "green", "blue"],
            chunks=dict(time=1, x=1024, y=1024),
            groupby="solar_day",
            fail_on_error=False,
            nodata=0,
        ).load(items, areas=geobox)

        maskers = {
            "old": lambda d: mask_clouds_old(d, filters=OLD_FILTERS, keep_ints=True),
            "new": mask_clouds_new,
        }
        for tag, masker in maskers.items():
            masked = masker(data).drop_vars("scl")
            geomad = geomedian_with_mads(
                masked, work_chunks=(1000, 1000), num_threads=4, maxiters=100
            ).compute()
            out_path = output_dir / f"{name}_{tag}.png"
            write_rgb_png(geomad, out_path)
            print(f"[{name}] wrote {out_path}")


if __name__ == "__main__":
    typer.run(main)
