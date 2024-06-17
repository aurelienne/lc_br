# conda activate glmval #

import netCDF4
import xarray as xr
import sys
import numpy as np
import configparser
from datetime import datetime, timedelta
import glob 

input_file = sys.argv[1]

ds = xr.open_dataset(input_file)
#ds_f = ds.number_of_flashes
#ds_br = ds.where((ds.flash_lat==slice(-33.80, 5.35))&(ds.flash_lon==slice(-74.20, -34)), drop=True)
#ds_br.to_netcdf(path='test.nc', mode='w')

vals = ds.flash_extent_density.data
vals = vals[~np.isnan(vals)]
print(vals)
print("Sum:")
print(np.sum(vals))

