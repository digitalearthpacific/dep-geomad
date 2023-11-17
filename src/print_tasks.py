import json
import sys
from itertools import product
from typing import Annotated, Optional

import typer
from azure_logger import CsvLogger, filter_by_log
from dep_tools.azure import get_container_client
from dep_tools.namers import DepItemPath

from run_task import get_grid


def main(
    regions: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    limit: Optional[str] = None,
    base_product: str = "ls",
    dataset_id: str = "geomad",
    no_retry_errors: Optional[bool] = False,
) -> None:
    grid = get_grid()
    region_codes = None if regions.upper() == "ALL" else regions.split(",")

    # Makes a list no matter what
    years = datetime.split("-")
    if len(years) == 2:
        years = range(int(years[0]), int(years[1]) + 1)
    elif len(years) > 2:
        ValueError(f"{datetime} is not a valid value for --datetime")

    grid_subset = (
        grid.loc[grid.country_code.isin(region_codes)]
        if region_codes is not None
        else grid
    )

    itempath = DepItemPath(base_product, dataset_id, version, datetime)
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=itempath.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    grid_subset = filter_by_log(grid_subset, logger.parse_log(), not no_retry_errors)
    params = [
        {
            "base-product": base_product,
            "region-code": region[0][0],
            "datetime": region[1],
        }
        for region in product(grid_subset.index, years)
    ]

    if limit is not None:
        params = params[0 : int(limit)]

    json.dump(params, sys.stdout)


if __name__ == "__main__":
    typer.run(main)
