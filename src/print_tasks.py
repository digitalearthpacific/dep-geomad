import json
import sys
from itertools import product
from typing import Annotated, Optional

import typer
from dep_tools.azure import blob_exists
from dep_tools.namers import DepItemPath

from run_task import get_grid


def main(
    regions: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    limit: Optional[str] = None,
    base_product: str = "ls",
    dataset_id: str = "geomad",
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    grid = get_grid()
    region_codes = None if regions.upper() == "ALL" else regions.split(",")

    if limit is not None:
        limit = int(limit)

    # Makes a list no matter what
    years = datetime.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{datetime} is not a valid value for --datetime")

    # Filter by country codes if we have them
    if region_codes is not None:
        grid = grid.loc[grid.country_code.isin(region_codes)]

    tasks = [
        {
            "base-product": base_product,
            "region-code": region[0][0],
            "datetime": region[1],
        }
        for region in product(grid.index, years)
    ]

    # If we don't want to overwrite, then we should only run tasks that don't already exist
    # i.e., they failed in the past or they're missing for some other reason
    itempath = DepItemPath(base_product, dataset_id, version, datetime)
    if not overwrite:
        valid_tasks = []
        for task in tasks:
            if not blob_exists(itempath.stac_path(task["region-code"])):
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
