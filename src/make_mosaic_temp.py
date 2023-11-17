#!/usr/bin/env python3

import odc.geo  # noqa: F401
from dep_tools.azure import get_container_client, list_blob_container
from odc.stac import load
from pystac import Item

client = get_container_client()
lazy_docs = list_blob_container(client, prefix="dep_ls_geomad/0-0-1/", suffix=".json")

docs = [d for d in lazy_docs]
print(f"Found {len(docs)} documents")

base_url = "https://deppcpublicstorage.blob.core.windows.net/output/{key}"
items = [Item.from_file(base_url.format(key=d)) for d in docs]

data = load(items, chunks={}, resolution=30, crs="epsg:3832")
print("Prepared data")

rgba = data.odc.to_rgba(vmin=7000, vmax=12000)
print("Converted to RGBA")

# Writing to Azure
print("Writing to Azure...")
data = rgba.squeeze().odc.to_cog()
output_file = "dep_ls_geomad/0-0-1/fiji_rgb_30m.tif"
blob_thing = client.get_blob_client(output_file)
blob_thing.upload_blob(data, overwrite=True)

print(f"{base_url}/{output_file}")
print("I think it worked!!")
