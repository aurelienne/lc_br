# conda activate glmval #

import netCDF4
import xarray as xr
import sys
import numpy as np
import configparser
from datetime import datetime, timedelta
import glob 

"""
input_file = sys.argv[1]

ds = xr.open_dataset(input_file)
ds_f = ds.number_of_flashes
ds_br = ds.where((ds.flash_lat==slice(-33.80, 5.35))&(ds.flash_lon==slice(-74.20, -34)), drop=True)
ds_br.to_netcdf(path='test.nc', mode='w')
num_flashes = np.sum(ds_br.values)
print(num_flashes)
"""
input_path = '/ships22/grain/ajorge/data/glm_grids_1min/'
input_path2 = '/ships22/grain/ajorge/data/glm_grids_60min/'

startdt = dt = datetime(2020,1,10,19,1)
enddt = datetime(2020,1,10,20,0)

ncfile2 = '/ships22/grain/ajorge/data/glm_grids_60min/agg/20200110-190100.netcdf'
ds = xr.open_dataset(ncfile2)
vals = ds.flash_extent_density.data
print(len(vals[vals>0]))
sys.exit()
print(vals.shape)
vals_60min = vals[~np.isnan(vals)]
print(vals_60min.shape)
print(np.sum(vals_60min))

while dt < enddt:
    filepattern = input_path + dt.strftime('%Y/%b/%d/OR_GLM-L2-*s%Y%j%H%M*.nc*')
    print(filepattern)
    ncfile1 = np.sort(glob.glob(filepattern))
    print(ncfile1[0])
    ds = xr.open_dataset(ncfile1[0])
    vals = ds.flash_extent_density.values
    vals_1min = vals[~np.isnan(vals)]
    print(np.sum(vals_1min))
    print(len(vals_1min))
    print(vals_1min[0:10])

    dt += timedelta(minutes=1)

