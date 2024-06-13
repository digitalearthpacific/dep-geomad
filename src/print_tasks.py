import json
import sys
from itertools import product
from typing import Annotated, Optional

import typer
from dep_tools.aws import object_exists
from dep_tools.azure import blob_exists
from dep_tools.namers import DepItemPath

from utils import get_grid


def main(
    years: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    regions: Optional[str] = "ALL",
    limit: Optional[str] = None,
    base_product: str = "ls",
    output_bucket: Optional[str] = None,
    output_prefix: Optional[str] = None,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    tiles = get_grid()

    region_codes = None if regions.upper() == "ALL" else regions.split(",")

    if limit is not None:
        limit = int(limit)

    # Makes a list no matter what
    years = years.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{years} is not a valid value for --years")

    # Filter by country codes if we have them
    if region_codes is not None:
        tiles = tiles.loc[tiles.country_code.isin(region_codes)]

    tasks = [
        {"tile-id": tile, "year": year, "version": version}
        for tile, year in product(list(tiles.tile_id), years)
    ]

    # If we don't want to overwrite, then we should only run tasks that don't already exist
    # i.e., they failed in the past or they're missing for some other reason
    if not overwrite:
        valid_tasks = []
        for task in tasks:
            itempath = DepItemPath(
                base_product, "geomad", version, task["year"], zero_pad_numbers=True
            )
            stac_path = itempath.stac_path(task["tile-id"])

            if output_prefix is not None:
                stac_path = f"{output_prefix}/{stac_path}"

            exists = False
            if output_bucket is not None:
                exists = object_exists(output_bucket, stac_path)
            else:
                exists = blob_exists(stac_path)
            if not exists:
                valid_tasks.append(task)
            if len(valid_tasks) == limit:
                break
        # Switch to this list of tasks, which has been filtered
        tasks = valid_tasks
    else:
        # If we are overwriting, we just keep going
        pass

    if limit is not None:
        tasks = tasks[0:limit]

    json.dump(tasks, sys.stdout)


if __name__ == "__main__":
    typer.run(main)
