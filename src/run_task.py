from logging import INFO, Formatter, Logger, StreamHandler, getLogger

import boto3
import dask
import typer
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.grids import PACIFIC_GRID_10, PACIFIC_GRID_30
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath
from dep_tools.searchers import PystacSearcher
from dep_tools.task import AwsStacTask as Task
from odc.stac import configure_s3_access
from typing_extensions import Annotated
from utils import GeoMADSentinel2Processor

S2_BANDS = [
    "scl",
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "nir09",
    "swir16",
    "swir22",
]

LANDSAT_BANDS = ["qa_pixel", "red", "green", "blue", "nir08", "swir16", "swir22"]


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


def main(
    tile_id: Annotated[str, typer.Option()],
    datetime: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    output_bucket: str = None,
    base_product: str = "ls",
    memory_limit: str = "50GB",
    n_workers: int = 2,
    threads_per_worker: int = 32,
    xy_chunk_size: int = 3201,
    geomad_threads: int = 10,
    decimated: bool = False,
    all_bands: Annotated[bool, typer.Option()] = True,
    overwrite: Annotated[bool, typer.Option()] = False,
    only_tier_one: Annotated[bool, typer.Option()] = True,
    fall_back_to_tier_two: Annotated[bool, typer.Option()] = True,
) -> None:
    log = get_logger(tile_id)
    log.info("Starting processing.")

    if base_product == "s2":
        grid = PACIFIC_GRID_10
    else:
        grid = PACIFIC_GRID_30

    tile_index = tuple(int(i) for i in tile_id.split(","))
    geobox = grid.tile_geobox(tile_index)

    if decimated:
        log.warning("Running at 1/10th resolution")
        geobox = geobox.zoom_out(10)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)

    client = boto3.client("s3")

    itempath = S3ItemPath(
        bucket=output_bucket,
        sensor=base_product,
        dataset_id="geomad",
        version=version,
        time=datetime,
    )
    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_document, client=client):
        log.info(f"Item already exists at {stac_document}")
        # This is an exit with success
        raise typer.Exit()

    if base_product == "ls":
        raise Exception("Only S2 is supported at the moment")

        # bands = LANDSAT_BANDS
        # if not all_bands:
        #     bands = ["qa_pixel", "red", "green", "blue"]

        # loader = LandsatOdcLoader(
        #     **common_load_args,
        #     odc_load_kwargs=dict(
        #         fail_on_error=False,
        #         resolution=resolution,
        #         groupby="solar_day",
        #         bands=bands,
        #     ),
        #     exclude_platforms=["landsat-7"],
        #     only_tier_one=only_tier_one,
        #     fall_back_to_tier_two=fall_back_to_tier_two,
        # )
        # ProcessorClass = GeoMADLandsatProcessor
    elif base_product == "s2":
        log.info("Configuring Sentinel-2 process")

        if not all_bands:
            bands = ["scl", "red", "green", "blue"]
        else:
            bands = S2_BANDS

        catalog = "https://earth-search.aws.element84.com/v1/"
        collection = "sentinel-2-c1-l2a"
        ProcessorClass = GeoMADSentinel2Processor
        chunks = dict(time=1, x=xy_chunk_size, y=xy_chunk_size)
        drop_vars = ["scl"]
        filters = [("dilation", 3), ("erosion", 2)]
    else:
        raise Exception("Only LS is supported at the moment")

    searcher = PystacSearcher(
        catalog=catalog,
        collections=collection,
        datetime=datetime,
    )

    loader = OdcLoader(
        bands=bands, chunks=chunks, groupby="solar_day", fail_on_error=False
    )

    processor = ProcessorClass(
        geomad_options=dict(
            work_chunks=(xy_chunk_size, xy_chunk_size),
            num_threads=geomad_threads,
            maxiters=100,
        ),
        filters=filters,
        min_timesteps=5,
        drop_vars=drop_vars,
    )

    try:
        with dask.config.set(
            {
                "dataframe.shuffle.method": "p2p",
                "distributed.worker.memory.target": False,
                "distributed.worker.memory.spill": False,
                "distributed.worker.memory.pause": 0.8,
                "distributed.worker.memory.terminate": 0.95,
            }
        ):
            with Client(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
            ):
                log.info(
                    (
                        f"Started dask client with {n_workers} workers "
                        f"and {threads_per_worker} threads with "
                        f"{memory_limit} memory"
                    )
                )
                paths = Task(
                    itempath=itempath,
                    id=tile_index,
                    area=geobox,
                    searcher=searcher,
                    loader=loader,
                    processor=processor,
                    logger=log,
                ).run()
    except Exception as e:
        log.exception(f"Failed to process with error: {e}")
        raise typer.Exit(code=1)

    log.info(
        f"Completed processing. Wrote {len(paths)} items to https://{output_bucket}.s3.us-west-2.amazonaws.com/{ stac_document}"
    )


if __name__ == "__main__":
    typer.run(main)
