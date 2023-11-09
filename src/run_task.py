from typing import Tuple

import typer
from azure_logger import CsvLogger
from dep_tools.azure import get_container_client
from dep_tools.loaders import FlatLandsatOdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import LandsatProcessor
from dep_tools.runner import run_by_area
from dep_tools.stac_utils import set_stac_properties
from dep_tools.writers import AzureDsWriter
from odc.algo import geomedian_with_mads
from typing_extensions import Annotated
from xarray import DataArray, Dataset

from src.grid import grid


class GeoMADLandsatProcessor(LandsatProcessor):

    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = False,
        dilate_mask: bool = True,
        num_threads: int = 1,
        work_chunks: Tuple[int, int] = (100, 100),
        keep_ints: bool = True,
    ) -> None:
        super().__init__(send_area_to_processor, scale_and_offset, dilate_mask, keep_ints)
        self.num_threads = num_threads
        self.work_chunks = work_chunks

    def process(self, xr: DataArray) -> Dataset:
        xr = super().process(xr)
        data = xr.drop_vars(["qa_pixel"])
        geomad = geomedian_with_mads(data, num_threads=self.num_threads, work_chunks=self.work_chunks)
        output = set_stac_properties(data, geomad)
        return output


def main(
    region_code: Annotated[str, typer.Option()],
    region_index: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    dataset_id: str = "geomad",
    base_product: str = "landsat",
) -> None:
    cell = grid.loc[[(region_code, region_index)]]

    if base_product == "landsat":
        base = "ls"

    loader = FlatLandsatOdcLoader(
        epsg=3832,
        datetime=datetime,
        dask_chunksize=dict(band=1, time=1, x=4096, y=4096),
        odc_load_kwargs=dict(
            fail_on_error=False,
            resolution=30,
            bands=["qa_pixel", "red", "green", "blue", "nir08", "swir16", "swir22"],
        ),
        exclude_platforms=["landsat-7"],
        nodata_value=0
    )

    processor = GeoMADLandsatProcessor(
        scale_and_offset=False,
        dilate_mask=True,
        work_chunks=(1801, 1801),
        num_threads=4,
        keep_ints=True
    )

    itempath = DepItemPath(base, dataset_id, version, datetime)

    writer = AzureDsWriter(
        itempath=itempath,
        convert_to_int16=True,
        overwrite=True,
        output_value_multiplier=100,
        extra_attrs=dict(dep_version=version),
    )
    logger = CsvLogger(
        name=dataset_id,
        container_client=get_container_client(),
        path=itempath.log_path(),
        overwrite=False,
        header="time|index|status|paths|comment\n",
    )

    run_by_area(
        areas=cell,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=logger,
        continue_on_error=False,
    )


if __name__ == "__main__":
    typer.run(main)
