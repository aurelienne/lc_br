import aux
import sys
import numpy as np
import os
from sklearn.neighbors import BallTree, KDTree

input_file = sys.argv[1]
lon_ini_glm = -76.187677
lat_ini_glm = 6.300088
NX, NY = 2100, 2100

# Defining multiplicator according to channel resolution
if 'RadF-M6C02' in os.path.basename(input_file):
    mult = 4
elif 'RadF-M6C05' in os.path.basename(input_file):
    mult = 2
else:
    mult = 1

lons2D, lats2D = aux.rad2latlon(input_file)

# Getting likely subgrid (considering resolution of Channels 13 or 15)
Y_st, Y_end = 2100*mult, 2500*mult
X_st, X_end = 2500*mult, 2800*mult
nlins_subg = Y_end - Y_st
ncols_subg = X_end - X_st
lats_ = lats2D[Y_st:Y_end, X_st:X_end].flatten()
lons_ = lons2D[Y_st:Y_end, X_st:X_end].flatten()
#lats_ = lats2D.flatten()
#lons_ = lons2D.flatten()
lats_[lats_ == np.inf] = 0
lons_[lons_ == np.inf] = 0

coords = np.zeros((len(lats_), 2))
i = 0
for lat, lon in zip(lats_, lons_):
    coords[i] = (lat, lon)
    i += 1

bt = BallTree(coords, metric='haversine')
# Finding index of nearest neightbor considering the subgrid
dist, idx = bt.query([[lat_ini_glm, lon_ini_glm]], k=1)
idx_subg = np.unravel_index(idx, (nlins_subg, ncols_subg))

# Converting index to global grid
idx_Y = Y_st + idx_subg[0]
idx_X = X_st + idx_subg[1]
print("Y, X:")
print(idx_Y, idx_X)
print("Lat, Lon:")
print(lats2D[idx_Y, idx_X], lons2D[idx_Y, idx_X])

