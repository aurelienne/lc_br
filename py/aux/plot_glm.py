import sys
import netCDF4
from matplotlib import pyplot as plt
from datetime import datetime
import os

filename = sys.argv[1]

dt = datetime.strptime(os.path.basename(filename),'%Y%m%d-%H%M00.netcdf')
nc = netCDF4.Dataset(filename)
data = nc.variables['flash_extent_density'][:]
nc.close()
plt.imshow(data,vmin=0,vmax=100)
plt.title(f'Aggregated GLM flash-extent density for one hour, ending {dt.strftime("%d %b %Y %H:%M")} UTC')
cbar = plt.colorbar(orientation='horizontal',label='Flash-extent density [flashes/hour]')
plt.show()
