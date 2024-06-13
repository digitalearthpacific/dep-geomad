from datacube_compute import geomedian_with_mads
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.processors import LandsatProcessor, Processor, S2Processor
from dep_tools.s2_utils import mask_clouds
from dep_tools.stac_utils import set_stac_properties
from dep_tools.utils import scale_and_offset
from geopandas import GeoDataFrame, read_file
from odc.algo import erase_bad, mask_cleanup
from xarray import DataArray, Dataset


def get_grid() -> GeoDataFrame:
    return (
        read_file(
            "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
        )
        .astype({"tile_id": str, "country_code": str})
        .set_index(["tile_id", "country_code"], drop=False)
    )


def mask_clouds_s2_aws(
    xr: DataArray,
    filters: list | None = [("closing", 5), ("opening", 5)],
    keep_ints: bool = True,
    use_scl: bool = False,
    mask_cloud_percentage: int = 20,
) -> DataArray:
    if use_scl:
        return mask_clouds(xr, filters=filters, keep_ints=keep_ints)

    try:
        cloud_mask = xr.sel(band="cloud").astype("uint16") > mask_cloud_percentage
    except KeyError:
        cloud_mask = xr["cloud"].astype("uint16") > mask_cloud_percentage

    if filters is not None:
        cloud_mask = mask_cleanup(cloud_mask, filters)

    if keep_ints:
        return erase_bad(xr, cloud_mask)
    else:
        return xr.where(~cloud_mask)


class AWSS2Processor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = True,
        mask_clouds: bool = True,
        mask_clouds_kwargs: dict = dict(),
    ) -> None:
        super().__init__(send_area_to_processor)
        self.scale_and_offset = scale_and_offset
        self.mask_clouds = mask_clouds
        self.mask_kwargs = mask_clouds_kwargs

    def process(self, xr: DataArray) -> DataArray:
        if self.mask_clouds:
            xr = mask_clouds_s2_aws(xr, **self.mask_kwargs)

        if self.scale_and_offset:
            scale = 1 / 10000
            offset = 0.1  # Should this be 1000?
            xr = scale_and_offset(xr, scale=[scale], offset=offset)

        return xr


class GeoMADProcessor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        scale_and_offset: bool = False,
        mask_clouds: bool = True,
        load_data_before_writing: bool = True,
        min_timesteps: int = 0,
        geomad_options: dict = {
            "num_threads": 4,
            "work_chunks": (1000, 1000),
            "maxiters": 1000,
        },
        filters: list | None = [("closing", 5), ("opening", 5)],
        keep_ints: bool = True,
        use_scl: bool = False,
        mask_cloud_percentage: int = 20,
        drop_vars: list[str] = [],
    ) -> None:
        super().__init__(
            send_area_to_processor,
            scale_and_offset,
            mask_clouds,
            mask_clouds_kwargs={
                "filters": filters,
                "keep_ints": keep_ints,
                "mask_cloud_percentage": mask_cloud_percentage,
                "use_scl": use_scl,
            },
        )
        self.scale_and_offset = scale_and_offset
        self.load_data_before_writing = load_data_before_writing
        self.min_timesteps = min_timesteps
        self.geomad_options = geomad_options
        self.drop_vars = drop_vars

    def process(self, xr: DataArray) -> Dataset:
        # Raise an exception if there's not enough data
        if xr.time.size < self.min_timesteps:
            raise EmptyCollectionError(
                f"{xr.time.size} is less than {self.min_timesteps} timesteps"
            )

        xr = super().process(xr)
        data = xr.drop_vars(self.drop_vars)
        geomad = geomedian_with_mads(data, **self.geomad_options)

        if self.load_data_before_writing:
            geomad = geomad.compute()

        output = set_stac_properties(data, geomad)
        return output


class GeoMADSentinel2Processor(GeoMADProcessor, S2Processor):
    def __init__(self, drop_vars=["scl"], **kwargs) -> None:
        super(GeoMADSentinel2Processor, self).__init__(drop_vars=drop_vars, **kwargs)


class GeoMADLandsatProcessor(GeoMADProcessor, LandsatProcessor):
    def __init__(self, drop_vars=["qa_pixel"], **kwargs) -> None:
        super(GeoMADLandsatProcessor, self).__init__(drop_vars=drop_vars, **kwargs)


class GeoMADAWSSentinel2Processor(GeoMADProcessor, AWSS2Processor):
    def __init__(self, drop_vars=["cloud"], **kwargs) -> None:
        super(GeoMADAWSSentinel2Processor, self).__init__(drop_vars=drop_vars, **kwargs)
