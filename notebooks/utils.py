# Utils for loading inputs, manipulating data, and writing out results
import pandas as pd
import s3fs
import xarray as xr
import numpy as np
import geopandas as gpd
import sparse
from shapely.geometry import Polygon
import fsspec
import dask

gcm_list = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "CNRM-CM6-1",
    "CNRM-ESM2-1",
    "EC-Earth3-Veg-LR",
    "EC-Earth3",
    "FGOALS-g3",
    "GFDL-CM4",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "INM-CM4-8",
    "INM-CM5-0",
    "KACE-1-0-G",
    "KIOST-ESM",
    "MIROC-ES2L",
    "MPI-ESM1-2-HR",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "NorESM2-LM",
    "NorESM2-MM",
    "UKESM1-0-LL",
]

gcms_with_nonstandard_calendars_list = [
    "BCC-CSM2-MR",
    "CanESM5",
    "CMCC-CM2-SR5",
    "CMCC-ESM2",
    "FGOALS-g3",
    "GFDL-CM4",
    "GFDL-ESM4",
    "GISS-E2-1-G",
    "HadGEM3-GC31-LL",
    "INM-CM4-8",
    "INM-CM5-0",
    "KACE-1-0-G",
    "KIOST-ESM",
    "NorESM2-LM",
    "NorESM2-MM",
    "UKESM1-0-LL",
]

## loading
df = pd.read_csv(
    "s3://carbonplan-climate-impacts/extreme-heat/v1.0/inputs/nex-gddp-cmip6-files.csv"
)
nasa_nex_runs_df = pd.DataFrame([run.split("/") for run in df[" fileURL"].values]).drop(
    [0, 1, 2, 3], axis=1
)
nasa_nex_runs_df.columns = [
    "GCM",
    "scenario",
    "ensemble_member",
    "variable",
    "file_name",
]


def find_nasanex_filename(gcm, scenario):
    """
    Load list of NASA-NEX files downloaded from their docs. We will use it to create
    the catalog of available datasets. Largely this is used to filter out the GCMs
    that don't have tasmax available.
    """
    template_filename = nasa_nex_runs_df[
        (nasa_nex_runs_df["GCM"] == gcm)
        & (nasa_nex_runs_df["scenario"] == scenario)
        & (nasa_nex_runs_df["variable"] == "tasmax")
    ]["file_name"].iloc[0]
    (
        _variable,
        _timestep,
        _gcm,
        _scenario,
        ensemble_member,
        grid_code,
        _yearnc,
    ) = template_filename.split("_")
    return ensemble_member, grid_code


##
def load_nasanex(scenario, gcm, variables, years, chunk_dict=None):
    """
    Read in NEX-GDDP-CMIP6 data from S3.
    """
    fs = s3fs.S3FileSystem(anon=True, default_fill_cache=False)

    file_objs = {}
    ds = xr.Dataset()
    ensemble_member, grid_code = find_nasanex_filename(gcm, scenario)
    for i, var in enumerate(variables):
        file_objs[var] = [
            fs.open(
                f"nex-gddp-cmip6/NEX-GDDP-CMIP6/{gcm}/{scenario}/"
                f"{ensemble_member}/{var}/{var}_day_{gcm}_{scenario}"
                f"_{ensemble_member}_{grid_code}_{year}.nc"
            )
            for year in years
        ]
        if i == 0:
            ds[var] = xr.open_mfdataset(file_objs[var], engine="h5netcdf")[var]
        else:
            new_var = xr.open_mfdataset(file_objs[var], engine="h5netcdf")
            new_var["time"] = ds[variables[0]]["time"].values
            ds[var] = new_var[var]
    if chunk_dict is not None:
        ds = ds.chunk(chunk_dict)
    return ds


## calc wbgt
def wbgt(wbt, bgt, tas):
    """
    Calculate wet bulb globe temperature as linear combination of component
    temperatures. All should be in Celcius.
    """
    wbgt = 0.7 * wbt + 0.2 * bgt + 0.1 * tas
    return wbgt


# aggregations and weighting
def apply_weights_matmul_sparse(weights, data):
    """
    Create sparse matrix of weights that collapses gridded data over a
    region into a weighted average point estimate.
    """
    assert isinstance(weights, sparse.SparseArray)
    assert isinstance(data, np.ndarray)
    data = sparse.COO.from_numpy(data)
    data_shape = data.shape
    n, k = data_shape[0], data_shape[1] * data_shape[2]
    data = data.reshape((n, k))
    weights_shape = weights.shape
    k_, m = weights_shape[0] * weights_shape[1], weights_shape[2]
    assert k == k_
    weights_data = weights.reshape((k, m))

    regridded = sparse.matmul(data, weights_data)
    assert regridded.shape == (n, m)
    return regridded.todense()


def bounds_to_poly(lon_bounds, lat_bounds):
    """
    Create polygon encompassing the bounds of the analysis domain.
    """
    if lon_bounds[0] >= 180:
        # geopandas needs this
        lon_bounds = lon_bounds - 360
    return Polygon(
        [
            (lon_bounds[0], lat_bounds[0]),
            (lon_bounds[0], lat_bounds[1]),
            (lon_bounds[1], lat_bounds[1]),
            (lon_bounds[1], lat_bounds[0]),
        ]
    )


def ds_to_grid(ds, variables_to_drop):
    """
    Create a grid of lat and lon bounds that matches the 0.25 degree grid of
    the NEX-GDDP-CMIP6 dataset.
    """
    grid = ds.drop(["time"] + variables_to_drop)
    grid_spacing = 0.25
    grid = grid.assign_coords(
        {
            "lat_bounds": xr.DataArray(
                np.stack(
                    (
                        grid.lat.values - grid_spacing / 2,
                        grid.lat.values + grid_spacing / 2,
                    ),
                    axis=1,
                ),
                dims=["lat", "nv"],
                coords={"lat": grid.lat},
            )
        }
    )
    grid = grid.assign_coords(
        {
            "lon_bounds": xr.DataArray(
                np.stack(
                    (
                        grid.lon.values - grid_spacing / 2,
                        grid.lon.values + grid_spacing / 2,
                    ),
                    axis=1,
                ),
                dims=["lon", "nv"],
                coords={"lon": grid.lon},
            )
        }
    )
    return grid.reset_coords().load()


def calc_sparse_weights(
    ds,
    regions_df,
    variables_to_drop,
    mask_nulls=None,
    population_weight=None,
    crs_orig="EPSG:4326",
):
    """
    Calculate weights as sparse matrix.
    """
    print("Generating weights...")
    grid = ds_to_grid(ds, variables_to_drop)
    points = grid.stack(point=("lat", "lon"))
    boxes = xr.apply_ufunc(
        bounds_to_poly,
        points.lon_bounds,
        points.lat_bounds,
        input_core_dims=[("nv",), ("nv",)],
        output_dtypes=[np.dtype("O")],
        vectorize=True,
    )
    grid_df = gpd.GeoDataFrame(
        data={"geometry": boxes.values, "lat": boxes["lat"], "lon": boxes["lon"]},
        index=boxes.indexes["point"],
        crs=crs_orig,
    )
    crs_area = "ESRI:53034"

    if mask_nulls is not None:
        mask = mask_nulls.isnull().compute().values.flatten()
        grid_df["isnull"] = mask
    if population_weight is not None:
        assert (population_weight["lon"].values == ds["lon"].values).all()
        assert (population_weight["lat"].values == ds["lat"].values).all()
        assert (
            population_weight["population"].values.shape
            == ds[variables_to_drop[0]].isel(time=0).values.shape
        )
        grid_df["population"] = population_weight["population"].values.flatten()
    print("population_weights calculated")
    grid_df = grid_df.to_crs(crs_area)
    overlay = grid_df.overlay(regions_df, keep_geom_type=True)
    if mask_nulls is not None:
        overlay = overlay[~overlay["isnull"]]
    grid_cell_fraction = overlay.geometry.area.groupby(
        overlay["processing_id"]
    ).transform(lambda x: x / x.sum())
    overlay["weights"] = grid_cell_fraction
    if population_weight is not None:
        overlay["population_informed_weights"] = (
            overlay["weights"] * overlay["population"]
        )
        population_weights = (
            overlay["population_informed_weights"]
            .groupby(overlay["processing_id"])
            .transform(lambda x: x / x.sum())
        )
        population_weights = population_weights.where(
            ~population_weights.isnull(), overlay["weights"]
        )
        overlay["weights"] = population_weights
        del overlay["population_informed_weights"]
    multi_index = overlay.set_index(["lat", "lon", "processing_id"]).index
    df_weights = pd.DataFrame({"weights": overlay["weights"].values}, index=multi_index)
    ds_weights = xr.Dataset(df_weights)
    weights_sparse = ds_weights.unstack(sparse=True, fill_value=0.0).weights
    return weights_sparse


def remove_360_longitudes(ds):
    """
    Rename coordinates so that indices span -180 to 180 instead of 0 to 360.
    Only change the coordinates; don't reindex it because that's expensive and
    we don't need to.
    """
    new_lons = ds["lon"].where(ds["lon"] < 180, ds["lon"] - 360)
    ds = ds.assign_coords(lon=new_lons)
    return ds


def prep_sparse(
    sample_ds,
    population,
    return_population=False,
    variables_to_drop=["rsds", "sfcWind"],
):
    """
    Prepare the inputs for the sparse matrix derivation.
    """
    path = (
        "s3://carbonplan-climate-impacts/extreme-heat/v1.0/inputs/"
        "all_regions_and_cities.json"
    )
    with fsspec.open(path) as file:
        regions_df = gpd.read_file(file)

    regions_df.crs = "epsg:4326"

    tuple(regions_df.total_bounds)
    crs_orig = f"EPSG:{regions_df.crs.to_epsg()}"
    # use an area preserving projections
    crs_area = "ESRI:53034"
    regions_df = regions_df.to_crs(crs_area)
    population = population.rename({"x": "lon", "y": "lat"}).drop("spatial_ref")

    sample_ds = remove_360_longitudes(sample_ds)
    population = population.reindex({"lon": sample_ds.lon.values}).load()
    sample_time_slice = sample_ds.isel(time=0)[variables_to_drop[0]].load()
    sparse_weights = calc_sparse_weights(
        sample_ds,
        regions_df,
        variables_to_drop,
        crs_orig=crs_orig,
        mask_nulls=sample_time_slice,
        population_weight=population,
    )
    if return_population:
        return sparse_weights, population
    else:
        return sparse_weights


def spatial_aggregation(ds, weights_sparse, load=True):
    """
    Use pre-calculated weights to aggregate gridded fields in `ds` into region-averaged
    point-estimates.
    """
    lon = "lon"
    lat = "lat"
    print("Applying weights...")
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        regridded = xr.apply_ufunc(
            apply_weights_matmul_sparse,
            weights_sparse,
            ds,
            join="left",
            input_core_dims=[[lat, lon, "processing_id"], [lat, lon]],
            output_core_dims=[["processing_id"]],
            dask="parallelized",
            dask_gufunc_kwargs=dict(meta=[np.ndarray((0,))]),
        )
        if load:
            regridded.load()
    return regridded


def clean_up_times(ds):
    """
    Make calendars conform, dropping five years for which we don't have
    modeled results and allowing for calendars that use different timestamps.
    """
    midnight_indexed_suffix = " 00:00:00"
    noon_indexed_suffix = " 12:00:00"
    try:
        ds = ds.drop_sel(
            {
                "time": pd.date_range(
                    "2015-01-01" + noon_indexed_suffix,
                    "2019-12-31" + noon_indexed_suffix,
                )
            }
        ).drop_sel(
            {
                "time": pd.date_range(
                    "2060-01-01" + noon_indexed_suffix,
                    "2060-12-31" + noon_indexed_suffix,
                )
            }
        )
    except Exception:
        ds = ds.drop_sel(
            {
                "time": pd.date_range(
                    "2015-01-01" + midnight_indexed_suffix,
                    "2019-12-31" + midnight_indexed_suffix,
                )
            }
        ).drop_sel(
            {
                "time": pd.date_range(
                    "2060-01-01" + midnight_indexed_suffix,
                    "2060-12-31" + midnight_indexed_suffix,
                )
            }
        )
    return ds


## processing
def summarize(da, metric):
    """
    Calculate two key kinds of aggregated metrics: annual maximum and number
    of days over a set of thresholds. Default thresholds selected from
    Liljegren (2008).
    """
    celcius_thresholds = [29, 30.5, 32, 35]
    annual_max = da.groupby("time.year").max()
    results = xr.Dataset({"annual_maximum": annual_max})
    for threshold in celcius_thresholds:
        results[f"days_exceeding_{threshold}degC"] = calc_days_over_threshold(
            da, threshold
        )
        results[f"days_exceeding_{threshold}degC"].attrs["units"] = "ndays"

    return results


def calc_days_over_threshold(da, threshold):
    """
    Given a threshold calculate the days per year over that threshold
    """
    return (da > threshold).groupby("time.year").sum()


def load_modelled_results(metric, gcm):
    if metric == "wbgt-sun":
        store_string = (
            f"s3://carbonplan-scratch/extreme-heat/wbgt-sun-regions/"
            f"wbgt-sun-{gcm}.zarr"
        )
        return xr.open_zarr(store_string)
    elif metric == "wbgt-shade":
        store_string_list = [
            f"s3://carbonplan-scratch/extreme-heat/wbgt-shade-regions/"
            f"{gcm}-{scenario}-bc.zarr"
            for scenario in ["historical", "ssp245-2030", "ssp245-2050"]
        ]
        return xr.concat(
            [xr.open_zarr(store) for store in store_string_list], dim="time"
        )


def load_multimodel_results(gcms, metric):
    """
    Read in the annualized results from different GCMs into a single dataset.
    """
    ds_gcm_list = []
    first = True
    for gcm in gcms:
        ds = load_modelled_results(metric, gcm)
        if not first:
            ds["time"] = ds_gcm_list[0].time.values
        ds_gcm_list.append(ds)
        first = False

    full_ds = xr.concat(ds_gcm_list, dim="gcm")
    full_ds = full_ds.assign_coords({"gcm": gcms})
    full_ds.time.attrs["standard_name"] = "time"
    return full_ds
