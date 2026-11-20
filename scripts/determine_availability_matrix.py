# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Determine the share of each cutout grid cell that is available for one renewable technology in each network region.

Land eligibility is computed with [atlite](https://github.com/pypsa/atlite)
by excluding rasters and geometries: selected classes of the
[CORINE Land Cover](https://land.copernicus.eu/pan-european/corine-land-cover)
inventory (44 land use classes at 100 m resolution) and of the similar 50 m
[LUISA Base Map](https://publications.jrc.ec.europa.eu/repository/handle/JRC124621),
optionally with a buffer distance around some classes; Natura 2000 nature
reserves; and for offshore wind, water depths from the
[GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/)
bathymetry, shipping lane density and distance to the shore. Onshore regions
are used for onshore technologies and offshore regions for offshore wind. For
Moldova and Ukraine, areas not covered by CORINE are overwritten with
externally determined availabilities.

![](../img/corine.png)

![](../img/gebco_2019_grid_image.jpg)
"""

import functools
import logging
import time

import atlite
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from atlite.gis import shape_availability
from rasterio.io import MemoryFile
from rasterio.plot import show

from scripts._helpers import configure_logging, load_cutout, set_scenario_config

logger = logging.getLogger(__name__)


def _drop_subnormals(array):
    values = np.array(array, copy=True)
    if np.issubdtype(values.dtype, np.floating):
        values[np.abs(values) < 1e-6] = 0
    return values


def use_geotiff_for_memory_warps() -> None:
    """Warp in-memory masks through a GeoTIFF.

    GDAL 3.13 writes a numpy mask into a float destination as stray
    bytes, so ``astype(bool)`` then excludes every offshore cell.
    """
    import atlite.gis as gis

    original = gis.rio.warp.reproject

    def reproject(source, destination, *args, **kwargs):
        if isinstance(source, np.ndarray):
            src = np.ascontiguousarray(source)
            if src.ndim == 3 and src.shape[0] == 1:
                src = src[0]
            if src.dtype == np.bool_:
                src = src.astype(np.uint8)
            profile = {
                "driver": "GTiff",
                "height": src.shape[-2],
                "width": src.shape[-1],
                "count": 1,
                "dtype": src.dtype.name,
                "crs": kwargs.pop("src_crs", None),
                "transform": kwargs.pop("src_transform", None),
            }
            # float64 destinations come back with stray subnormal bytes on
            # GDAL 3.13, and those bytes are truthy in the exclusion mask.
            dest = np.zeros(np.shape(destination), dtype=np.float32)
            with MemoryFile() as mem:
                with mem.open(**profile) as dataset:
                    dataset.write(src, 1)
                with mem.open() as dataset:
                    out, transform = original(
                        rasterio.band(dataset, 1), dest, *args, **kwargs
                    )
            return _drop_subnormals(out), transform
        dest = np.zeros(np.shape(destination), dtype=np.float32)
        out, transform = original(source, dest, *args, **kwargs)
        return _drop_subnormals(out), transform

    gis.rio.warp.reproject = reproject


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "determine_availability_matrix", technology="offwind-dc"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    noprogress = snakemake.params.disable_progressbar
    noprogress = noprogress or not snakemake.params.show_progress
    technology = snakemake.wildcards.technology
    params = snakemake.params.technology

    cutout = load_cutout(snakemake.input.cutout)
    regions = gpd.read_file(snakemake.input.regions)
    assert not regions.empty, (
        f"List of regions in {snakemake.input.regions} is empty, please "
        "disable the corresponding renewable technology"
    )
    # do not pull up, set_index does not work if geo dataframe is empty
    regions = regions.set_index("name").rename_axis("bus")

    res = params.get("excluder_resolution", 100)
    excluder = atlite.ExclusionContainer(crs=3035, res=res)

    if params["natura"]:
        excluder.add_raster(snakemake.input.natura, nodata=0, allow_no_overlap=True)

    for dataset in ["corine", "luisa"]:
        kwargs = {"nodata": 0} if dataset == "luisa" else {}
        settings = params.get(dataset, {})
        if not settings:
            continue
        if dataset == "luisa" and res > 50:
            logger.info(
                "LUISA data is available at 50m resolution, "
                f"but coarser {res}m resolution is used."
            )
        if isinstance(settings, list):
            settings = {"grid_codes": settings}
        if "grid_codes" in settings:
            codes = settings["grid_codes"]
            excluder.add_raster(
                snakemake.input[dataset], codes=codes, invert=True, crs=3035, **kwargs
            )
        if settings.get("distance", 0.0) > 0.0:
            codes = settings["distance_grid_codes"]
            buffer = settings["distance"]
            excluder.add_raster(
                snakemake.input[dataset], codes=codes, buffer=buffer, crs=3035, **kwargs
            )

    if params.get("ship_threshold"):
        shipping_threshold = (
            params["ship_threshold"] * 8760 * 6
        )  # approximation because 6 years of data which is hourly collected
        func = functools.partial(np.less, shipping_threshold)
        excluder.add_raster(
            snakemake.input.ship_density, codes=func, crs=4326, allow_no_overlap=True
        )

    if params.get("max_depth"):
        # lambda not supported for atlite + multiprocessing
        # use named function np.greater with partially frozen argument instead
        # and exclude areas where: -max_depth > grid cell depth
        func = functools.partial(np.greater, -params["max_depth"])
        excluder.add_raster(snakemake.input.gebco, codes=func, crs=4326, nodata=-1000)

    if params.get("min_depth"):
        func = functools.partial(np.greater, -params["min_depth"])
        excluder.add_raster(
            snakemake.input.gebco, codes=func, crs=4326, nodata=-1000, invert=True
        )

    if params.get("min_shore_distance") is not None:
        buffer = params["min_shore_distance"]
        excluder.add_geometry(snakemake.input.country_shapes, buffer=buffer)

    if params.get("max_shore_distance") is not None:
        buffer = params["max_shore_distance"]
        excluder.add_geometry(
            snakemake.input.country_shapes, buffer=buffer, invert=True
        )

    logger.info(f"Calculate landuse availability for {technology}...")
    start = time.time()

    # The warp fix lives in this process. A spawned worker would import a
    # fresh atlite and hit the broken in-memory reprojection again.
    use_geotiff_for_memory_warps()
    kwargs = {"nprocesses": None, "disable_progressbar": noprogress}
    availability = cutout.availabilitymatrix(regions, excluder, **kwargs)

    duration = time.time() - start
    logger.info(
        f"Completed landuse availability calculation for {technology} ({duration:2.2f}s)"
    )

    if snakemake.params.plot_availability_matrix:
        logger.info(f"Plotting landuse availability matrix for {technology}.")
        band, transform = shape_availability(
            regions.geometry.to_crs(excluder.crs), excluder
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        regions.to_crs(excluder.crs).plot(ax=ax, color="none")
        show(band, transform=transform, cmap="Greens", ax=ax)
        plt.savefig(snakemake.output["plot"], dpi=300)

    # For Moldova and Ukraine: Overwrite parts not covered by Corine with
    # externally determined available areas
    if "availability_matrix_MD_UA" in snakemake.input.keys():  # noqa: SIM118
        availability_MDUA = xr.open_dataarray(
            snakemake.input["availability_matrix_MD_UA"]
        )
        availability.loc[availability_MDUA.coords] = availability_MDUA

    availability.to_netcdf(snakemake.output["nc"])
