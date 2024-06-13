from logging import INFO, Formatter, Logger, StreamHandler, getLogger

import boto3
import typer
from dask.distributed import Client
from dep_tools.aws import object_exists
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.grids import PACIFIC_GRID_10, PACIFIC_GRID_30
from dep_tools.loaders import OdcLoader
from dep_tools.namers import DepItemPath
from dep_tools.searchers import PystacSearcher
from dep_tools.writers import AwsDsCogWriter
from odc.stac import configure_s3_access
from typing_extensions import Annotated

from utils import GeoMADAWSSentinel2Processor

S2_BANDS = [
    "cloud",
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
    dataset_id: str = "geomad",
    base_product: str = "ls",
    memory_limit_per_worker: str = "50GB",
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
    log.info(f"Starting processing for {tile_id}")

    grid = PACIFIC_GRID_30
    if base_product == "s2":
        grid = PACIFIC_GRID_10

    tile_index = tuple(int(i) for i in tile_id.split(","))
    geobox = grid.tile_geobox(tile_index)

    if decimated:
        log.warning("Running at 1/10th resolution")
        geobox = geobox.zoom_out(10)

    # Make sure we can access S3
    log.info("Configuring S3 access")
    configure_s3_access(cloud_defaults=True)

    itempath = DepItemPath(
        base_product, dataset_id, version, datetime, zero_pad_numbers=True
    )
    stac_document = itempath.stac_path(tile_id)

    # If we don't want to overwrite, and the destination file already exists, skip it
    if not overwrite and object_exists(output_bucket, stac_document):
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
            bands = ["cloud", "red", "green", "blue"]
        else:
            bands = S2_BANDS

        catalog = "https://earth-search.aws.element84.com/v1/"
        collection = "sentinel-2-c1-l2a"
        ProcessorClass = GeoMADAWSSentinel2Processor
        chunks = dict(time=1, x=xy_chunk_size, y=xy_chunk_size)
        drop_vars = ["cloud"]
    else:
        raise Exception("Only LS is supported at the moment")

    searcher = PystacSearcher(
        catalog=catalog,
        collections=collection,
        datetime=datetime,
    )

    loader = OdcLoader(bands=bands, chunks=chunks, groupby="solar_day")

    processor = ProcessorClass(
        geomad_options=dict(
            work_chunks=(600, 600),
            num_threads=geomad_threads,
            maxiters=100,
        ),
        filters=[("closing", 5), ("opening", 5)],
        mask_cloud_percentage=5,  # only used for S-2
        min_timesteps=5,
        drop_vars=drop_vars,
    )

    client = boto3.client("s3")
    writer = AwsDsCogWriter(
        itempath=itempath,
        overwrite=overwrite,
        convert_to_int16=False,
        extra_attrs=dict(dep_version=version),
        write_multithreaded=True,
        bucket=output_bucket,
        client=client,
    )

    paths = []
    with Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_per_worker,
    ):
        try:
            # Find items
            items = searcher.search(area=geobox)
            log.info(f"Found {len(items)} items")

            # Run the load process, which is lazy-loaded
            data = loader.load(items, areas=geobox)
            log.info(f"Found {len(data.time)} timesteps to load")

            output_data = processor.process(data)
            output_sizes = [output_data.sizes[d] for d in ["x", "y"]]
            log.info(f"Processed data to shape {output_sizes}")

            paths = writer.write(output_data, tile_id)

            if paths is not None:
                log.info(f"Completed writing to {paths[-1]}")
            else:
                log.warning("No paths returned from writer")

        except EmptyCollectionError:
            log.warning("No data found for this tile.")
        except Exception as e:
            log.exception(f"Failed to process with error: {e}")
            raise typer.Exit(code=1)

        log.info("Completed processing.")
        if len(paths) > 0:
            log.info(f"Item written to {stac_document}")
        else:
            log.warning("Nothing written")


if __name__ == "__main__":
    typer.run(main)
