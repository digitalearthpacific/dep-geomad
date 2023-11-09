import geopandas as gpd

grid = (
    gpd.read_file(
        "https://raw.githubusercontent.com/digitalearthpacific/dep-grid/master/grid_pacific.geojson"
    )
    .astype({"tile_id": str, "country_code": str})
    .set_index(["tile_id", "country_code"], drop=False)
)
