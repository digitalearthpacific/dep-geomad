from datacube_compute import geomedian_with_mads
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.processors import LandsatProcessor, Processor, S2Processor
from dep_tools.stac_utils import set_stac_properties
from xarray import DataArray, Dataset


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
        # For Sentinel-2, the default filters are dilation of 3 and erosion of 2.
        # This is very conservative, so will let clouds through.
        # Dilation of 8 and erosion of 6 removes most clouds
        filters: list | None = [("dilation", 3), ("erosion", 2)],
        keep_ints: bool = True,
        drop_vars: list[str] = [],
    ) -> None:
        super().__init__(
            send_area_to_processor,
            scale_and_offset,
            mask_clouds,
            mask_clouds_kwargs={"filters": filters, "keep_ints": keep_ints},
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
