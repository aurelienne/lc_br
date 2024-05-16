import numpy as np
import netCDF4
from pyproj import Proj
import sys

input_file = sys.argv[1]

def rad2latlon(input_file):
    nc = netCDF4.Dataset(input_file, "r")
    gip = nc.variables["goes_imager_projection"]
    x = nc.variables["x"][:]
    y = nc.variables["y"][:]
    projection = Proj(
        f"+proj=geos +lon_0={gip.longitude_of_projection_origin}"
        + f" +h={gip.perspective_point_height} +x_0=0.0 +y_0=0.0 +a="
        + f"{gip.semi_major_axis} +b={gip.semi_minor_axis}"
    )

    x *= gip.perspective_point_height
    y *= gip.perspective_point_height

    xx, yy = np.meshgrid(x, y)
    lons2D, lats2D = projection(xx, yy, inverse=True)
    return lons2D, lats2D

def find_nearest(array, value):
    #array = np.asarray(array)
    print("Value to search: "+str(value))
    idx = np.unravel_index((np.abs(array - value)).argmin(), array.shape)
    print(idx)
    return idx

# Do exactly the same as the function 'rad2latlon'
def calculate_degrees(input_file):
    file_id = netCDF4.Dataset(input_file, "r")

    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables['y'][:]  # N/S elevation angle in radians
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')

    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)

    return abi_lat, abi_lon
