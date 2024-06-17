# conda activate glmval #

import netCDF4
import xarray as xr
import sys
import numpy as np
import configparser
from datetime import datetime, timedelta
import glob 

input_file = sys.argv[1]
output_path = sys.argv[2]

ds = xr.open_dataset(input_file)
#x1_val = ds['x'][0].values
#x2_val = ds['x'][2099].values
#y1_val = ds['y'][0].values
#y2_val = ds['y'][2099].values
x1, x2 = 0, 2100
y1, y2 = 0, 2100
#crop_ds = ds.sel(x=slice(x1_val, x2_val), y=slice(y1_val, y2_val))
crop_ds = ds.isel(x=slice(x1, x2), y=slice(y1, y2))
crop_ds.to_netcdf(path='GLM_BR_finaldomain.nc', mode='w')

