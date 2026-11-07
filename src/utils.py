from datacube_compute import geomedian_with_mads
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.processors import Processor
from dep_tools.stac_utils import set_stac_properties
from xarray import DataArray, Dataset
from odc.algo import mask_cleanup, erase_bad


def mask_clouds(
    xr: DataArray,
    mask_filters: dict,
    mask_high_values: bool = False,
    return_mask: bool = False,
) -> DataArray:
    """Takes a dictionary of value to a list of functions to mask that value in the scl
    band of the input xarray. Do this in the Datacubey way...

    Example mask_filters: {"cloud shadows":[["dilation", 5]], "cloud medium probability":[["opening", 5], ["dilation", 5]], "cloud high probability":[["opening", 5], ["dilation", 5]], "thin cirrus":[["dilation", 5]]}
    """
    SCL_VALUE_LOOKUP = {
        "saturated or defective": 1,
        "cloud shadows": 3,
        "cloud medium probability": 8,
        "cloud high probability": 9,
        "thin cirrus": 10,
    }

    scl = xr["scl"]

    # Maintain nodata (0)
    mask = scl == 0
    assert mask.dtype == bool, f"mask dtype is {mask.dtype}, expected bool"

    # For each value, apply the list of functions to mask that value in the scl band
    for value, operation in mask_filters.items():
        if isinstance(value, tuple):
            to_mask = scl.isin([SCL_VALUE_LOOKUP[v] for v in value])
        else:
            to_mask = scl == SCL_VALUE_LOOKUP[value]
        # make sure we're still using ints
        assert to_mask.dtype == bool, f"to_mask dtype is {to_mask.dtype}, expected bool"
        next_mask = mask_cleanup(to_mask, operation) if operation else to_mask
        mask = mask | next_mask

    # Exclude pixels where R, G and B are > THRESHOLD (these are likely to be clouds)
    THRESHOLD = 4000
    if mask_high_values:
        high_values = (
            (xr["green"] > THRESHOLD)
            & (xr["red"] > THRESHOLD)
            & (xr["blue"] > THRESHOLD)
        )
        mask = mask | high_values

    if return_mask:
        return erase_bad(xr, mask), mask
    else:
        return erase_bad(xr, mask)


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
        # Raise an exception because this does nothing
        raise NotImplementedError(
            "GeoMADProcessor is an abstract class. Use GeoMADSentinel1Processor or GeoMADSentinel2Processor instead."
        )


class GeoMADSentinel2Processor(GeoMADProcessor):
    def __init__(
        self,
        drop_vars: list[str] = ["scl"],
        mask_filters: dict = {
            "saturated or defective": [],
            ("cloud medium probability", "cloud high probability"): [
                ["opening", 3],
                ["closing", 3],
                ["dilation", 3],
            ],
        },
        mask_high_values: bool = False,
        scale_and_offset: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(drop_vars=drop_vars, **kwargs)
        self.mask_filters = mask_filters
        self.mask_high_values = mask_high_values
        self.scale_and_offset = scale_and_offset

    def _mask_clouds(self, xr: DataArray) -> DataArray:
        return mask_clouds(
            xr, mask_filters=self.mask_filters, mask_high_values=self.mask_high_values
        )

    def process(self, xr: DataArray) -> Dataset:
        # Raise an exception if there's not enough data
        if xr.time.size < self.min_timesteps:
            raise EmptyCollectionError(
                f"{xr.time.size} is less than {self.min_timesteps} timesteps"
            )

        xr = self._mask_clouds(xr)

        if self.scale_and_offset:
            self.geomad_options["scale"] = 1 / 10_000
            self.geomad_options["offset"] = 0

        if len(self.drop_vars) > 0:
            xr = xr.drop_vars(self.drop_vars)

        geomad = geomedian_with_mads(xr, **self.geomad_options)

        if self.load_data_before_writing:
            geomad = geomad.compute()

        # Add nodata as 0 to the count variable
        geomad["count"].odc.nodata = 0

        output = set_stac_properties(xr, geomad)

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
