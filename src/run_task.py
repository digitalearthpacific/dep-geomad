from logging import INFO, Formatter, Logger, StreamHandler, getLogger

import geopandas as gpd
import typer
from dask.distributed import Client
from dep_tools.azure import blob_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.loaders import LandsatOdcLoader, Sentinel2OdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.processors import LandsatProcessor, Processor, S2Processor
from dep_tools.stac_utils import set_stac_properties

# from dep_tools.task import SimpleLoggingAreaTask
from dep_tools.task import AreaTask
from dep_tools.writers import AzureDsWriter
from odc.algo import geomedian_with_mads
from typing_extensions import Annotated
from xarray import DataArray, Dataset


def get_logger(region_code: str) -> Logger:
    """Set up a simple logger"""
    console = StreamHandler()
    time_format = "%Y-%m-%d %H:%M:%S"
    console.setFormatter(
        Formatter(
            fmt=f"%(asctime)s %(levelname)s ({region_code}):  %(message)s",
            datefmt=time_format,
        )
    )

    log = getLogger("GEOMAD")
    log.addHandler(console)
    log.setLevel(INFO)
    return log


def get_grid() -> gpd.GeoDataFrame:
    return (
        gpd.read_file(
            "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
        )
        .astype({"tile_id": str, "country_code": str})
        .set_index(["tile_id", "country_code"], drop=False)
    )


class SimpleLoggingAreaTask(AreaTask):
    def run(self, load_data_before_writing=False):
        self.logger.info("Preparing to load data")
        input_data = self.loader.load(self.area)
        self.logger.info(f"Found {len(input_data.time)} timesteps to load")

        self.logger.info("Preparing to process data")
        processor_kwargs = {}
        if self.processor.send_area_to_processor:
            processor_kwargs["area"] = self.area
        output_data = self.processor.process(input_data, **processor_kwargs)
        self.logger.info(
            f"Processed data will have a result of shape: {[output_data.dims[d] for d in ['x', 'y']]}"
        )

        if load_data_before_writing:
            self.logger.info("Loading data into memory and processing")
            output_data = output_data.compute()
            self.logger.info("Data processed and loaded")

            self.logger.info("Writing data")
        else:
            self.logger.info("Processing and writing data...")

        paths = self.writer.write(output_data, self.id)
        self.logger.info(f"Succesfully wrote data to {len(paths)} paths")

        return paths


class GeoMADProcessor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = False,
        mask_clouds: bool = True,
        geomad_options: dict = {
            "num_threads": 4,
            "work_chunks": (1000, 1000),
            "maxiters": 1000,
        },
        filters: list | None = [("closing", 5), ("opening", 5)],
        keep_ints: bool = True,
        drop_vars: list[str] = [],
    ) -> None:
        super().__init__(
            send_area_to_processor,
            scale_and_offset,
            mask_clouds,
            mask_clouds_kwargs={"filters": filters, "keep_ints": keep_ints},
        )
        self.geomad_options = geomad_options
        self.drop_vars = drop_vars

    def process(self, xr: DataArray) -> Dataset:
        xr = super().process(xr)
        data = xr.drop_vars(self.drop_vars)
        geomad = geomedian_with_mads(data, **self.geomad_options)
        output = set_stac_properties(data, geomad)
        return output


class GeoMADSentinel2Processor(GeoMADProcessor, S2Processor):
    def __init__(self, drop_vars=["SCL"], **kwargs) -> None:
        super(GeoMADSentinel2Processor, self).__init__(drop_vars=drop_vars, **kwargs)


class GeoMADLandsatProcessor(GeoMADProcessor, LandsatProcessor):
    def __init__(self, drop_vars=["qa_pixel"], **kwargs) -> None:
        super(GeoMADLandsatProcessor, self).__init__(drop_vars=drop_vars, **kwargs)


def main(
    region_code: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    dataset_id: str = "geomad",
    base_product: str = "ls",
    memory_limit_per_worker: str = "50GB",
    n_workers: int = 2,
    threads_per_worker: int = 32,
    xy_chunk_size: int = 4096,
    all_bands: Annotated[bool, typer.Option()] = True,
    overwrite: Annotated[bool, typer.Option()] = False,
    only_tier_one: Annotated[bool, typer.Option()] = True,
    fall_back_to_tier_two: Annotated[bool, typer.Option()] = True,
) -> None:
    grid = get_grid()
    area = grid.loc[[region_code]]

    log = get_logger(region_code)
    log.info(f"Starting processing for {region_code}")

    itempath = DepItemPath(base_product, dataset_id, version, datetime)
    stac_document = itempath.stac_path(region_code)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and blob_exists(stac_document):
        log.info(f"Item already exists at {stac_document}")
        # This is an exit with success
        raise typer.Exit()

    resolution = 10
    if base_product == "ls":
        resolution = 30

    common_load_args = dict(
        epsg=3832,
        datetime=datetime,
        dask_chunksize=dict(time=1, x=xy_chunk_size, y=xy_chunk_size),
        nodata_value=0,
        keep_ints=True,
        load_as_dataset=True,
    )

    if base_product == "ls":
        log.info("Configuring Landsat process")

        resolution = 30
        bands = ["qa_pixel", "red", "green", "blue", "nir08", "swir16", "swir22"]
        if not all_bands:
            bands = ["qa_pixel", "red", "green", "blue"]

        loader = LandsatOdcLoader(
            **common_load_args,
            odc_load_kwargs=dict(
                fail_on_error=False,
                resolution=resolution,
                groupby="solar_day",
                bands=bands,
            ),
            exclude_platforms=["landsat-7"],
            only_tier_one=only_tier_one,
            fall_back_to_tier_two=fall_back_to_tier_two,
        )
        ProcessorClass = GeoMADLandsatProcessor
    elif base_product == "s2":
        log.info("Configuring Sentinel-2 process")

        resolution = 10
        bands = [
            "SCL",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]
        if not all_bands:
            bands = ["SCL", "B04", "B03", "B02"]

        loader = Sentinel2OdcLoader(
            **common_load_args,
            odc_load_kwargs=dict(
                fail_on_error=False,
                resolution=resolution,
                groupby="solar_day",
                bands=bands,
            ),
        )
        ProcessorClass = GeoMADSentinel2Processor
    else:
        raise Exception("Only LS is supported at the moment")

    log.info("Configuring processor")
    processor = ProcessorClass(
        scale_and_offset=False,
        filters=[("closing", 5), ("opening", 5)],
        keep_ints=True,
        geomad_options=dict(
            num_threads=2,
            work_chunks=(601, 601),
            maxiters=100,
        ),
    )

    log.info("Configuring writer")
    writer = AzureDsWriter(
        itempath=itempath,
        overwrite=overwrite,
        convert_to_int16=False,
        extra_attrs=dict(dep_version=version),
        write_multithreaded=True,
    )

    runner = SimpleLoggingAreaTask(
        id=region_code,
        area=area,
        loader=loader,
        processor=processor,
        writer=writer,
        logger=log,
    )

    with Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_per_worker,
    ):
        try:
            paths = runner.run(load_data_before_writing=True)
            log.info(f"Completed writing to {paths[-1]}")
        except EmptyCollectionError:
            log.warning("No data found for this tile.")
        except Exception as e:
            log.exception(f"Failed to process {region_code} with error: {e}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
    typer.run(main)
