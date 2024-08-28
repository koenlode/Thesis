import dataclasses
import datetime
import functools
import math
import re
from typing import Optional, Tuple, Dict
import sys
import argparse
import time

import cartopy.crs as ccrs
from google.cloud import storage
import gcsfs
import zarr

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint 
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization 
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree

import haiku as hk
import jax

import numpy as np
import xarray
import dask.array as da

# supress warnings
import warnings
import os
import logging
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress TensorFlow and XLA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs: 0 = all logs, 1 = INFO, 2 = WARNINGS, 3 = ERRORS
os.environ['XLA_FLAGS'] = '--xla_dump_to=/dev/null'  # Redirect XLA dump output

# Suppress warnings from specific libraries
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress specific warnings in Python
warnings.filterwarnings("ignore", message="Skipping gradient checkpointing for sequence length of 1")

# Redirect stderr to /dev/null to suppress all error output
sys.stderr = open(os.devnull, 'w')



def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

def datetime_steps_slicer(ds, target_start_date, num_timesteps):
    num_timesteps = num_timesteps - 1 # Ensure that the total number of timesteps is equal to num_timesteps provided
    start_date = np.datetime64('2016-01-01T00:00:00.000000000')
    target_start_date = np.datetime64(target_start_date)
    timedelta_start = target_start_date - start_date
    timestep_duration = np.timedelta64(6, 'h')
    timedelta_end = timedelta_start + num_timesteps * timestep_duration
    ds_slice = ds.sel(time=slice(timedelta_start, timedelta_end))
    return ds_slice

def main(start_date: str, timesteps: int, netherlands: bool=True):
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")

    source = "Checkpoint"
    params_file_value = "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"

    assert source == "Checkpoint"
    with gcs_bucket.blob(f"params/{params_file_value}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    params = ckpt.params
    state = {}

    model_config = ckpt.model_config
    task_config = ckpt.task_config
    print("Model description:\n", ckpt.description, "\n")

    fs = gcsfs.GCSFileSystem(anon=True)
    store = gcsfs.GCSMap(root='gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr', gcs=fs, check=False)
    hres_6h = xarray.open_zarr(store, consolidated=True)
    print("Dataset opened successfully")

    hres_6h = hres_6h.rename({'latitude': 'lat', 'longitude': 'lon'})
    hres_6h_var_sliced = hres_6h.assign_coords(datetime=hres_6h['time'])
    batch_dim = da.zeros(hres_6h_var_sliced.sizes['time'], dtype=int)
    hres_6h_var_sliced = hres_6h_var_sliced.assign_coords(batch=('time', batch_dim))
    reshaped_datetime = hres_6h_var_sliced['datetime'].values.reshape((1, hres_6h_var_sliced.sizes['time']))
    hres_6h_var_sliced = hres_6h_var_sliced.assign_coords(datetime=(['batch', 'time'], reshaped_datetime))

    start_time = hres_6h_var_sliced['time'].values[0]
    time_deltas = hres_6h_var_sliced['time'].values - start_time
    hres_6h_var_sliced = hres_6h_var_sliced.assign_coords(time=('time', time_deltas))

    reshaped_vars = {}
    for var in hres_6h_var_sliced.data_vars:
        reshaped_vars[var] = (['batch', 'time'] + [dim for dim in hres_6h_var_sliced[var].dims if dim != 'time'],
                              hres_6h_var_sliced[var].data.reshape((1, len(hres_6h_var_sliced['time'])) + hres_6h_var_sliced[var].shape[1:]))

    ds_new = xarray.Dataset(reshaped_vars,
                            coords={'lon': hres_6h_var_sliced['lon'], 'lat': hres_6h_var_sliced['lat'], 'level': hres_6h_var_sliced['level'], 'time': hres_6h_var_sliced['time'],
                                    'datetime': (['batch', 'time'], hres_6h_var_sliced['datetime'].values.reshape((1, len(hres_6h_var_sliced['time']))))})

    for coord in hres_6h_var_sliced.coords:
        if coord not in ds_new.coords and coord != 'datetime':
            ds_new = ds_new.assign_coords({coord: hres_6h_var_sliced[coord]})

    ds_land_sea_mask = xarray.open_dataset('ds_land_sea_mask.nc')
    ds_geopotential_at_surface = xarray.open_dataset('ds_geopotential_at_surface.nc')

    ds_new['land_sea_mask'] = ds_land_sea_mask['land_sea_mask']
    ds_new['geopotential_at_surface'] = ds_geopotential_at_surface['geopotential_at_surface']
    ds_new = ds_new.drop_vars('batch')

    ds_new = ds_new[['geopotential_at_surface', 'land_sea_mask', '2m_temperature', 'mean_sea_level_pressure', '10m_v_component_of_wind',
                     '10m_u_component_of_wind', 'total_precipitation_6hr', 'temperature', 'geopotential', 'u_component_of_wind',
                     'v_component_of_wind', 'vertical_velocity', 'specific_humidity']]

    hres_gc_shaped = ds_new

    hres_gc_shaped_sliced = datetime_steps_slicer(hres_gc_shaped, start_date, timesteps + 2)
    print('Dataset transformed successfully')

    example_batch = hres_gc_shaped_sliced.compute()
    print('Dataset computed succesfully')

    # train_steps = 1
    eval_steps = timesteps

    # train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    #     example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    #     **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
        **dataclasses.asdict(task_config))

    with gcs_bucket.blob("stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob("stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob("stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

    def construct_wrapped_graphcast(model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level)
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    # @hk.transform_with_state
    # def loss_fn(model_config, task_config, inputs, targets, forcings):
    #     predictor = construct_wrapped_graphcast(model_config, task_config)
    #     loss, diagnostics = predictor.loss(inputs, targets, forcings)
    #     return xarray_tree.map_structure(
    #         lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
    #         (loss, diagnostics))

    # def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    #     def _aux(params, state, i, t, f):
    #         (loss, diagnostics), next_state = loss_fn.apply(
    #             params, state, jax.random.PRNGKey(0), model_config, task_config,
    #             i, t, f)
    #         return loss, (diagnostics, next_state)
    #     (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
    #         _aux, has_aux=True)(params, state, inputs, targets, forcings)
    #     return loss, diagnostics, next_state, grads

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    # init_jitted = jax.jit(with_configs(run_forward.init))

    # if params is None:
    #     params, state = init_jitted(
    #         rng=jax.random.PRNGKey(0),
    #         inputs=train_inputs,
    #         targets_template=train_targets,
    #         forcings=train_forcings)

    # loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
    # grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings)
    
    predictions = predictions.rename({'time': 'prediction_timedelta'})

    predictions = predictions.expand_dims(time=[np.datetime64(start_date,'ns')])

    # Reorder the coordinates in the desired order
    predictions = predictions.assign_coords({'lat': predictions.lat, 
                       'level': predictions.level, 
                       'lon': predictions.lon, 
                       'prediction_timedelta': predictions.prediction_timedelta, 
                       'time': predictions.time})

    if netherlands:
        # Limiting extent to the Netherlands
        predictions = predictions.sel(lat=slice(50.5, 55.75), lon=slice(2.25, 7.5))

        # Naming convention for the NetCDF file
        filename = f'gc_nl_{start_date}_{timesteps}.nc'

    else:
        filename = f'gc_wrld_{start_date}_{timesteps}.nc'

    # Write the predictions to a NetCDF file
    predictions.to_netcdf(filename)
    print(f"Predictions written to {filename}")

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run weather prediction model")
    parser.add_argument("start_date", type=str, help="Start date for the prediction (format: YYYY-MM-DD)")
    parser.add_argument("timesteps", type=int, help="Number of timesteps for the prediction")

    args = parser.parse_args()

    # Measure the elapsed time of the main function
    start_time = time.time()
    main(args.start_date, args.timesteps)
    elapsed_time = time.time() - start_time

print(f"Elapsed time: {elapsed_time} seconds")
