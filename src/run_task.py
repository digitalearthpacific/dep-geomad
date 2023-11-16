from typing import Tuple

import geopandas as gpd
import typer
from azure_logger import CsvLogger
from dask.distributed import Client
from dep_tools.azure import get_container_client
from dep_tools.loaders import LandsatOdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import LandsatProcessor
from dep_tools.s2_utils import S2Processor
from dep_tools.stac_utils import set_stac_properties
from dep_tools.task import ErrorCategoryAreaTask
from dep_tools.writers import AzureDsWriter
from odc.algo import geomedian_with_mads
from typing_extensions import Annotated
from xarray import DataArray, Dataset


def get_grid() -> gpd.GeoDataFrame:
    return (
        gpd.read_file(
            "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
        )
        .astype({"tile_id": str, "country_code": str})
        .set_index(["tile_id", "country_code"], drop=False)
    )


class GeoMADLandsatProcessor(LandsatProcessor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = False,
        mask_clouds: bool = True,
        dilate_mask: Tuple[int, int] | None = [2, 3],
        num_threads: int = 4,
        work_chunks: Tuple[int, int] = (1000, 1000),
        keep_ints: bool = True,
    ) -> None:
        super().__init__(
            send_area_to_processor,
            scale_and_offset,
            mask_clouds,
            dilate_mask,
            keep_ints,
        )
        self.num_threads = num_threads
        self.work_chunks = work_chunks

    def process(self, xr: DataArray) -> Dataset:
        xr = super().process(xr)
        data = xr.drop_vars(["qa_pixel"])
        geomad = geomedian_with_mads(
            data, num_threads=self.num_threads, work_chunks=self.work_chunks
        )
        output = set_stac_properties(data, geomad)
        return output


class GeoMADSentinelProcessor(S2Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = False,
        mask_clouds: bool = True,
        dilate_mask: bool = True,
        num_threads: int = 4,
        work_chunks: Tuple[int, int] = (1000, 1000),
        keep_ints: bool = True,
    ) -> None:
        super().__init__(
            send_area_to_processor,
            scale_and_offset,
            mask_clouds,
            dilate_mask,
            keep_ints,
        )
        self.num_threads = num_threads
        self.work_chunks = work_chunks

    def process(self, xr: DataArray) -> Dataset:
        xr = super().process(xr)
        data = xr.drop_vars(["SCL"])
        geomad = geomedian_with_mads(
            data, num_threads=self.num_threads, work_chunks=self.work_chunks
        )
        output = set_stac_properties(data, geomad)
        return output


def main(
    region_code: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    dataset_id: str = "geomad",
    base_product: str = "ls",
    memory_limit: str = "24GB",
    n_workers: int = 1,
    threads_per_worker: int = 16,
) -> None:
    grid = get_grid()

    loader = LandsatOdcLoader(
        epsg=3832,
        datetime=datetime,
        dask_chunksize=dict(time=1, x=4096, y=4096),
        odc_load_kwargs=dict(
            fail_on_error=False,
            resolution=30,
            bands=["qa_pixel", "red", "green", "blue", "nir08", "swir16", "swir22"],
        ),
        exclude_platforms=["landsat-7"],
        nodata_value=0,
        keep_ints=True,
        flat_array=True,
    )

    processor = GeoMADLandsatProcessor(
        scale_and_offset=False,
        dilate_mask=[2, 3],
        work_chunks=(601, 601),
        num_threads=10,
        keep_ints=True,
    )

    itempath = DepItemPath(base_product, dataset_id, version, datetime)

    writer = AzureDsWriter(
        itempath=itempath,
        overwrite=True,
        extra_attrs=dict(dep_version=version),
    )

    # TODO: consider refactoring to use a normal logger
    # and if we do, then changing the logic around checking
    # if a tile is already done.
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=itempath.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    runner = ErrorCategoryAreaTask(
        id=region_code,
        area=grid.loc[[region_code]],
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
    )

    with Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    ):
        paths = runner.run_one(region_code)

    if paths is not None:
        print(f"Completed writing to {paths[-1]}")
    else:
        print("ERROR: Failed to process...")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
