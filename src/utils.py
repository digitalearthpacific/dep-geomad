from datacube_compute import geomedian_with_mads
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.processors import LandsatProcessor, Processor, S2Processor
from dep_tools.stac_utils import set_stac_properties
from xarray import DataArray, Dataset


class GeoMADProcessor(Processor):
    def __init__(
        self,
        send_area_to_processor: bool = False,
        load_data_before_writing: bool = True,
        min_timesteps: int = 0,
        geomad_options: dict = {
            "num_threads": 4,
            "work_chunks": (1000, 1000),
            "maxiters": 1000,
        },
        drop_vars: list[str] = [],
        preprocessor: Processor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(send_area_to_processor, **kwargs)
        self.load_data_before_writing = load_data_before_writing
        self.min_timesteps = min_timesteps
        self.geomad_options = geomad_options
        self.drop_vars = drop_vars
        self.preprocessor = preprocessor

    def process(self, xr: DataArray) -> Dataset:
        # Raise an exception if there's not enough data
        if xr.time.size < self.min_timesteps:
            raise EmptyCollectionError(
                f"{xr.time.size} is less than {self.min_timesteps} timesteps"
            )

        if self.preprocessor is not None:
            xr = self.preprocessor.process(xr)

        data = xr

        if len(self.drop_vars) > 0:
            data = data.drop_vars(self.drop_vars)

        geomad = geomedian_with_mads(data, **self.geomad_options)

        if self.load_data_before_writing:
            geomad = geomad.compute()

        # Add nodata as 0 to the count variable
        geomad["count"].odc.nodata = 0

        output = set_stac_properties(data, geomad)

        return output


class GeoMADSentinel1Processor(GeoMADProcessor):
    def __init__(self, **kwargs) -> None:
        super(GeoMADSentinel1Processor, self).__init__(**kwargs)

    def process(self, xr: DataArray) -> Dataset:
        # Raise an exception if there's not enough data
        if xr.time.size < self.min_timesteps:
            raise EmptyCollectionError(
                f"{xr.time.size} is less than {self.min_timesteps} timesteps"
            )

        if self.preprocessor is not None:
            xr = self.preprocessor.process(xr)

        data = xr

        if len(self.drop_vars) > 0:
            data = data.drop_vars(self.drop_vars)

        # First compute the mean and standard deviation
        data = data.compute()  # Load into memory, so we only do it once
        stats = {}
        for var in ["vv", "vh"]:
            stats[f"mean_{var}"] = data[var].mean(dim="time", skipna=True)
            stats[f"stdev_{var}"] = data[var].std(dim="time", skipna=True)
        geomad = geomedian_with_mads(data, **self.geomad_options)

        # Append the computed statistics to the geomad output
        geomad = geomad.assign(stats)

        if self.load_data_before_writing:
            geomad = geomad.compute()

        # Add nodata as 0 to the count variable
        geomad["count"].odc.nodata = 0

        output = set_stac_properties(data, geomad)

        return output


class GeoMADSentinel2Processor(GeoMADProcessor):
    def __init__(
        self,
        preprocessor_args: dict = {
            "mask_clouds": True,
            "mask_clouds_kwargs": {
                "filters": [("dilation", 3), ("erosion", 2)],
                "keep_ints": True,
            },
        },
        drop_vars=["scl"],
        **kwargs,
    ) -> None:
        super().__init__(
            preprocessor=S2Processor(**preprocessor_args), drop_vars=drop_vars, **kwargs
        )


class GeoMADLandsatProcessor(GeoMADProcessor, LandsatProcessor):
    def __init__(self, drop_vars=["qa_pixel"], **kwargs) -> None:
        super(GeoMADLandsatProcessor, self).__init__(drop_vars=drop_vars, **kwargs)


class GeoMADPostProcessor(Processor):
    def __init__(
        self,
        vars: list[str] = [],
        drop_vars: list[str] = [],
        scale: float | None = None,
        offset: float | None = None,
    ):
        self._vars = [v for v in vars if v not in drop_vars]
        self._scale = scale
        self._offset = offset

    def process(self, xr: Dataset):
        if len(self._vars) != 0:
            for var in self._vars:
                if self._scale is not None:
                    xr[var].attrs["scales"] = self._scale
                if self._offset is not None:
                    xr[var].attrs["offsets"] = self._offset
        return xr
