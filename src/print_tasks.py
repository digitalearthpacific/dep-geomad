import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Annotated, Optional

import boto3
import geopandas as gpd
import pandas as pd
import typer
from dep_tools.aws import get_s3_bucket_region, object_exists
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

# Catalog and collection settings per base product, matching run_task.py
STAC_SETTINGS = {
    "s2": {
        "catalog": "https://earth-search.aws.element84.com/v1",
        "collections": ["sentinel-2-l2a"],
    },
    "s1": {
        "catalog": "https://planetarycomputer.microsoft.com/api/stac/v1/",
        "collections": ["sentinel-1-rtc"],
    },
}


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


def _has_stac_data(tile_index, year, base_product):
    """Fast check for data existence using max_items=1."""
    if base_product not in STAC_SETTINGS:
        return True

    settings = STAC_SETTINGS[base_product]
    geobox = PACIFIC_GRID_10.tile_geobox(tile_index)
    bbox = bbox_across_180(geobox)
    client = StacClient.open(settings["catalog"])

    search_kwargs = dict(
        collections=settings["collections"],
        datetime=str(year),
        max_items=1,
    )

    if isinstance(bbox, tuple):
        for b in bbox:
            results = list(client.search(bbox=b, **search_kwargs).items())
            if len(results) > 0:
                return True
        return False
    else:
        results = list(client.search(bbox=bbox, **search_kwargs).items())
        return len(results) > 0


def _find_existing_tasks(
    tasks, output_bucket, base_product, version, output_prefix, full_path_prefix, limit
):
    """Check which tasks already have outputs using concurrent HEAD requests.
    Stops early once enough non-existing tasks are found to satisfy limit."""
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
        return object_exists(output_bucket, stac_path, client=_get_client())

    existing = set()
    non_existing_count = 0
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_task = {executor.submit(_check_exists, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            if future.result():
                existing.add((task["tile-id"], str(task["year"])))
            else:
                non_existing_count += 1
                if limit is not None and non_existing_count >= limit:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
    return existing


def _filter_stac(tasks, base_product, limit):
    """Filter tasks by STAC data existence, concurrently with early termination."""
    valid_tasks = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_task = {
            executor.submit(
                _has_stac_data,
                tuple(int(i) for i in task["tile-id"].split(",")),
                task["year"],
                base_product,
            ): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
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

    # Filter out tasks whose output already exists in S3
    if not overwrite and output_bucket is not None:
        aws_region = get_s3_bucket_region(output_bucket)
        full_path_prefix = f"https://{output_bucket}.s3.{aws_region}.amazonaws.com/"

        existing = _find_existing_tasks(
            tasks,
            output_bucket,
            base_product,
            version,
            output_prefix,
            full_path_prefix,
            limit,
        )
        tasks = [t for t in tasks if (t["tile-id"], str(t["year"])) not in existing]

    # Filter out tiles with no source data in the STAC catalog
    if check_stac:
        tasks = _filter_stac(tasks, base_product, limit)

    if limit is not None:
        tasks = tasks[0:limit]

    json.dump(tasks, sys.stdout)


if __name__ == "__main__":
    typer.run(main)
