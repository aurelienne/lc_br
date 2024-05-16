import aux
import sys
import numpy as np
import os

#lon_range = [-75, -34]
#lat_range = [-34, 5]
lon_ini_glm = -76.1876
lat_ini_glm = 6.30
NX, NY = 2100, 2100

input_file = sys.argv[1]
lons, lats = aux.rad2latlon(input_file)
#lats, lons = aux.calculate_degrees(input_file)

print(lons[0, NX-10:])
print(lats[NY-10:, NX-10:])

#print(lons[:,2100])
#print(lats[2100, :])

#idx_lon1 = aux.find_nearest(lons, lon_range[0])
#idx_lon2 = aux.find_nearest(lons, lon_range[1])

#idx_lat2 = aux.find_nearest(lats, lat_range[0])
#idx_lat1 = aux.find_nearest(lats, lat_range[1])

idx_lon1 = aux.find_nearest(lons, lon_ini_glm)
idx_lat1 = aux.find_nearest(lats, lat_ini_glm)
print(lons[idx_lat1[0],idx_lon1[1]])
print(lats[idx_lat1[0],idx_lon1[1]])

# Para GLM: Usar 0,2100 como boundaries, tanto em x quanto em y 
#print(lons[0,0], lons[0,2100])
#print(lats[0,0], lats[2100,0])

if 'RadF-M6C02' in os.path.basename(input_file):
    CH_dim = (NY*4, NX*4)
elif 'RadF-M6C05' in os.path.basename(input_file):
    CH_dim = (NY*2, NX*2)
else:
    CH_dim = (NY, NX)

print("")
print(lons[idx_lat1[0], idx_lon1[1]+CH_dim[1]])
print(lats[idx_lat1[0]+CH_dim[0], idx_lon1[1]])

#print(lons[idx_lat1[0]+NY*2, idx_lon1[1]+NX*2])
#print(lats[idx_lat1[0]+NY*2, idx_lon1[1]+NX*2])
