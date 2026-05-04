import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Annotated, Optional

import boto3
import geopandas as gpd
import pandas as pd
import typer
from dep_tools.grids import (
    PACIFIC_EPSG,
    PACIFIC_GRID_10,
    get_tiles,
    grid,
)
from dep_tools.namers import S3ItemPath
from dep_tools.utils import bbox_across_180
from odc.geo import Geometry
from pystac_client import Client as StacClient


def get_tiles_for_countries(country_codes, buffer_distance=None, resolution=30):
    """Fetch tiles for specific countries, downloading only their GADM data."""
    geometries = pd.concat(
        [
            gpd.read_file(
                f"https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_{code}.gpkg",
                layer="ADM_ADM_0",
            )
            for code in country_codes
        ]
    )

    gridspec = grid(resolution=resolution)
    geo_dict = geometries.to_crs(PACIFIC_EPSG).simplify(0.1).to_frame().to_geo_dict()
    geometry = Geometry(geo_dict, crs=PACIFIC_EPSG)
    geometry = geometry.buffer(buffer_distance if buffer_distance is not None else 0.0)
    return gridspec.tiles_from_geopolygon(geopolygon=geometry)


def _has_s2_data(tile_index, year):
    """Fast check for S2 data existence using max_items=1 instead of fetching all."""
    geobox = PACIFIC_GRID_10.tile_geobox(tile_index)
    bbox = bbox_across_180(geobox)
    client = StacClient.open("https://earth-search.aws.element84.com/v1")

    search_kwargs = dict(
        collections=["sentinel-2-l2a"],
        datetime=str(year),
        max_items=1,
    )

    if isinstance(bbox, tuple):
        # Antimeridian crossing: check both sides
        for b in bbox:
            results = list(client.search(bbox=b, **search_kwargs).items())
            if len(results) > 0:
                return True
        return False
    else:
        results = list(client.search(bbox=bbox, **search_kwargs).items())
        return len(results) > 0


def _get_existing_stac_paths(
    output_bucket, base_product, version, output_prefix, years, full_path_prefix, tasks
):
    """Check which tasks already have outputs using concurrent HEAD requests."""
    import threading
    from dep_tools.aws import object_exists

    thread_local = threading.local()

    def _get_client():
        if not hasattr(thread_local, "client"):
            thread_local.client = boto3.client("s3")
        return thread_local.client

    def _check_exists(task):
        itempath = S3ItemPath(
            bucket=output_bucket,
            sensor=base_product,
            dataset_id="geomad",
            version=version,
            time=task["year"],
            full_path_prefix=full_path_prefix,
        )
        stac_path = itempath.stac_path(task["tile-id"].split(","))
        if output_prefix is not None:
            stac_path = f"{output_prefix}/{stac_path}"
        return (task, object_exists(output_bucket, stac_path, client=_get_client()))

    existing = set()
    with ThreadPoolExecutor(max_workers=50) as executor:
        for task, exists in executor.map(_check_exists, tasks):
            if exists:
                existing.add((task["tile-id"], str(task["year"])))
    return existing


def _filter_existing_tasks(
    tasks, output_bucket, base_product, version, output_prefix, limit, check_stac
):
    """Filter tasks, keeping only those that need processing."""
    # Resolve bucket region once, to avoid a head_bucket call per task
    from dep_tools.aws import get_s3_bucket_region

    aws_region = get_s3_bucket_region(output_bucket)
    full_path_prefix = f"https://{output_bucket}.s3.{aws_region}.amazonaws.com/"

    # Check S3 existence concurrently (50 workers for fast HEAD requests)
    years = set(str(t["year"]) for t in tasks)
    existing = _get_existing_stac_paths(
        output_bucket,
        base_product,
        version,
        output_prefix,
        years,
        full_path_prefix,
        tasks,
    )

    # Filter out tasks that already exist
    remaining_tasks = [
        t for t in tasks if (t["tile-id"], str(t["year"])) not in existing
    ]

    if not check_stac:
        if limit is not None:
            remaining_tasks = remaining_tasks[:limit]
        return remaining_tasks

    # With STAC checks, run concurrently
    valid_tasks = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(
                _has_s2_data,
                tuple(int(i) for i in task["tile-id"].split(",")),
                task["year"],
            ): task
            for task in remaining_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            if future.result():
                valid_tasks.append(task)
                if limit is not None and len(valid_tasks) >= limit:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    return valid_tasks


def main(
    years: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    regions: Optional[str] = "ALL",
    tile_buffer_kms: Optional[int] = 0.0,
    limit: Optional[str] = None,
    base_product: str = "ls",
    output_bucket: Annotated[
        Optional[str], typer.Option("--output-bucket", "--bucket")
    ] = None,
    output_prefix: Optional[str] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
    check_stac: Annotated[
        bool,
        typer.Option(
            help="Check STAC API for data existence before adding a tile to the task list."
        ),
    ] = False,
) -> None:
    country_codes = None if regions.upper() == "ALL" else regions.split(",")

    if country_codes is not None:
        tiles = get_tiles_for_countries(
            country_codes, buffer_distance=tile_buffer_kms * 1000
        )
    else:
        tiles = get_tiles(country_codes=None, buffer_distance=tile_buffer_kms * 1000)

    if limit is not None:
        limit = int(limit)

    # Makes a list no matter what
    years = years.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{years} is not a valid value for --years")

    tasks = [
        {
            "tile-id": ",".join([str(i) for i in tile[0]]),
            "year": year,
            "version": version,
        }
        for tile, year in product(list(tiles), years)
    ]

    # If we don't want to overwrite, then we should only run tasks that don't already exist
    # i.e., they failed in the past or they're missing for some other reason
    if not overwrite:
        tasks = _filter_existing_tasks(
            tasks,
            output_bucket,
            base_product,
            version,
            output_prefix,
            limit,
            check_stac,
        )
    else:
        # If we are overwriting, we just keep going
        pass

    if limit is not None:
        tasks = tasks[0:limit]

    json.dump(tasks, sys.stdout)


if __name__ == "__main__":
    typer.run(main)
