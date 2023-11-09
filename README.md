# Digital Earth Pacific GeoMAD

## Annual GeoMAD

This is a work in progress.

Note that currently it needs the version of `odc-algo` from
[this branch](https://github.com/opendatacube/odc-algo/tree/add-rust-geomedian-impl).

And it needs the version of `dep_tools` from
[this PR](https://github.com/digitalearthpacific/dep-tools/pull/20)

It can run in the Notebook.

Things to do:

* Test performance on large datasets (full year)
* Evaluate different configurations of chunks/threads
* Ensure that MADs are coming out valid
* After Landsat is working and stable, implement for Sentinel-2.
