import logging
from logging.config import dictConfig

# from lightningcast.logging_def import *
from subprocess import Popen, PIPE, call
from datetime import datetime, timedelta
import time
import os, sys, re
import smtplib
import getpass
from datetime import datetime
from netCDF4 import Dataset
import errno
import numpy as np

# import pygrib as pg
import pickle

# from pyhdf.SD import SD, SDC
# import pyhdf
import json
import glob
from _ctypes import PyObj_FromPtr

from lightningcast.lightningcast_yaml_config import LightningcastYamlCascade
from collections import OrderedDict
import inspect
import subprocess

#####################################################################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


#####################################################################################################
def get_abi_data(
    dt,
    ch,
    datadir_patt,
    startY=None,
    endY=None,
    startX=None,
    endX=None,
    metadata={},
    return_proj_info=False,
):

    # returns the 2D reflectance or BT array
    try:
        datafile = glob.glob(dt.strftime(f"{datadir_patt}/*C{ch}_*_s%Y%j%H%M%S*.nc"))[0]
    except IndexError:
        try:
            datafile = glob.glob(dt.strftime(f"{datadir_patt}/*C{ch}_*_s%Y%j%H%M*.nc"))[
                0
            ]
        except IndexError:
            logging.error(
                f"Could not get "
                + dt.strftime(f"{datadir_patt}/*C{ch}_*_s%Y%j%H%M%S*.nc")
            )
            raise

    # copy file to $PWD
    #  bf = os.path.basename(datafile)
    #  #make tmp dir with UTCnow microseconds
    #  localfile = os.path.join(os.environ['PLTG_DATA'],f"{sector}_{datetime.utcnow().strftime('%f')}",bf)
    #  tmpdir = os.path.dirname(localfile)
    #  utils.mkdir_p(tmpdir)
    #  shutil.copy(datafile,localfile)
    localfile = datafile
    nc = Dataset(localfile, "r")
    metadata["goes_position"] = nc.orbital_slot
    metadata["scene_id"] = nc.scene_id
    metadata["platform_ID"] = nc.platform_ID

    if ch == "02":
        # convert vis radiance to reflectance; crop if necessary
        if startY:
            radiance = nc.variables["Rad"][startY * 4 : endY * 4, startX * 4 : endX * 4]
        else:
            radiance = nc.variables["Rad"][:]
        result = radiance * nc.variables["kappa0"][:]
    elif ch in ["01", "03", "05"]:
        # convert vis radiance to reflectance; crop if necessary
        if startY:
            radiance = nc.variables["Rad"][startY * 2 : endY * 2, startX * 2 : endX * 2]
        else:
            radiance = nc.variables["Rad"][:]
        result = radiance * nc.variables["kappa0"][:]
    elif ch == "04" or ch == "06":
        # convert vis radiance to reflectance; crop if necessary
        if startY:
            radiance = nc.variables["Rad"][startY:endY, startX:endX]
        else:
            radiance = nc.variables["Rad"][:]
        result = radiance * nc.variables["kappa0"][:]
    else:  # IR channels
        # convert IR radiance to BT; crop if necessary
        if startY:
            radiance = nc.variables["Rad"][startY:endY, startX:endX]
        else:
            radiance = nc.variables["Rad"][:]
        result = (
            nc.variables["planck_fk2"][:]
            / (np.log((nc.variables["planck_fk1"][:] / radiance) + 1))
            - nc.variables["planck_bc1"][:]
        ) / nc.variables["planck_bc2"][:]

    if return_proj_info:
        satheight = nc.variables["goes_imager_projection"].getncattr(
            "perspective_point_height"
        )
        x = nc.variables["x"][:] * satheight
        y = nc.variables["y"][:] * satheight

    try:
        nc.close()
    except RuntimeError:
        pass

    # shutil.rmtree(tmpdir, ignore_errors=False) #remove tmpdir

    if return_proj_info:
        return x, y, result
    else:
        return result


####################################################################################################
def plax(
    lats,
    lons,
    satlon=-75.2,
    elevation=9000.0,
    gip=None,
    return_dict=False,
    write_plax_file=False,
):
    # By default, it will only return the parallax-corrected lats and lons.

    # elevation is height of object, in meters, above earth surface (assuming constant height)
    # ----->Local constants
    if np.ma.is_masked(lats) or np.ma.is_masked(lons):
        mask = lats.mask  # to be used later
        masked = True
    else:
        masked = False

    logging.info(
        f"Performing parallax correction with satlon = {satlon} deg. and elevation = {elevation} m."
    )
    to_radians = np.pi / 180.0  # Conversion to radians from degrees
    to_degrees = 180.0 / np.pi  # Conversion to degrees from radians
    right_angle = np.pi / 2.0  # Right angle in radians for reference
    satlat = 0.0  # Satellite latitude; 0.0 for GEOs
    satlat_rad = satlat * to_radians  # Satellite latitude in radians
    if gip:
        earth_radius = (gip["semi_major_axis"] + gip["semi_minor_axis"]) / 2.0
        altitude = earth_radius + gip["perspective_point_height"]
    else:
        earth_radius = 6371009.0  # Mean earth radius, in meters, based on WGS-84 data
        altitude = 42166111.9  # Distance of satellite, in meters, from center of earth

    earth_radius_elevation = (
        earth_radius + elevation
    )  # Mean radius of the earth plus height of object above surface, in meters
    satlon_rad = satlon * to_radians

    # ----->General variables defined
    datalat_rad = lats * to_radians

    # Check to see if data longitude is past -180 W
    minlon = np.min(lons)
    if minlon < -180.0:
        logging.warning("Longitudes less than -180; returning.")
        raise ValueError
        # return {'lat_pc':0,'lon_pc':0}

    datalon_rad = lons * to_radians

    datalat_pc_rad = np.zeros(lats.shape) - 999.0
    datalon_pc_rad = np.zeros(lons.shape) - 999.0

    # ----->Calculate the distance between satellite nadir and base of object (using haversine formula)
    londiff = satlon_rad - datalon_rad
    latdiff = satlat_rad - datalat_rad

    centralangle = 2.0 * np.arcsin(
        np.sqrt(
            (np.sin(latdiff / 2.0)) ** 2
            + np.cos(satlat_rad) * np.cos(datalat_rad) * (np.sin(londiff / 2.0)) ** 2
        )
    )
    gcdistance = earth_radius * centralangle

    # ----->Calculate the central angle that subtends the great circle distance between nadir and true position
    # ...First find distance between satellite and non-true position using the Law of Cosines
    satviewdist = np.sqrt(
        altitude**2
        + earth_radius**2
        - 2.0 * altitude * earth_radius * np.cos(centralangle)
    )

    # ...Then find the satellite view angle from nadir on great circle plane also using the Law of Cosines
    satviewangle = np.arccos(
        (earth_radius**2 - altitude**2 - satviewdist**2)
        / (-2.0 * altitude * satviewdist)
    )

    # ...Then find the adjusted central angle for parallax
    # First find angle between satellite and center of the earth where the true position is the vertex using the Law of Sines
    angleSNO = np.arcsin((altitude * np.sin(satviewangle)) / earth_radius_elevation)
    ind = np.where(angleSNO < right_angle)
    if len(ind) > 0:
        angleSNO[ind] = np.pi - angleSNO[ind]

    # Then calculate adjusted central angle for parallax using identity that all angles of a triangle add up to pi radians
    sat_true_centralangle = np.pi - angleSNO - satviewangle
    sat_true_gcdist = earth_radius * sat_true_centralangle

    # ----->Calculate latitudinal component of parallax correction (assumes great circle is equivalent to satellite meridian)
    # ...First find distance between satellite and non-true latitude position using the Law of Cosines
    satviewdist_lat = np.sqrt(
        altitude**2
        + earth_radius**2
        - 2.0 * altitude * earth_radius * np.cos(datalat_rad)
    )

    # ...Then find the satellite view zenith angle from nadir also using the Law of Cosines
    satviewangle_lat = np.arccos(
        (earth_radius**2 - altitude**2 - satviewdist_lat**2)
        / (-2.0 * altitude * satviewdist_lat)
    )

    # ...Then find the adjusted zenith angle for parallax
    # First find angle between satellite and center of the earth where the true position is the vertex using the Law of Sines
    angleSNO_lat = np.arcsin(
        (altitude * np.sin(satviewangle_lat)) / earth_radius_elevation
    )
    ind = np.where(angleSNO_lat < right_angle)
    if len(ind[0]) > 0:
        angleSNO_lat[ind] = np.pi - angleSNO_lat[ind]

    # Then calculate adjusted zenith angle for parallax using identity that all angles of a triangle add up to pi radians
    ind = np.where(lats > satlat)
    other_ind = np.where(lats <= satlat)
    if len(ind[0]) > 0:
        datalat_pc_rad[ind] = np.pi - angleSNO_lat[ind] - satviewangle_lat[ind]

    if len(other_ind) > 0:
        datalat_pc_rad[other_ind] = (
            angleSNO_lat[other_ind] + satviewangle_lat[other_ind] - np.pi
        )

    # ----->Calculate longitudinal component of parallax correction (using haversine formula)
    latdiff_pc = satlat_rad - datalat_pc_rad
    ind = np.where(lons < satlon)
    other_ind = np.where(lons >= satlon)
    if len(ind[0]) > 0:
        datalon_pc_rad[ind] = satlon_rad - 2.0 * np.arcsin(
            np.sqrt(
                np.abs(
                    (
                        (np.sin(sat_true_centralangle[ind] / 2.0)) ** 2
                        - (np.sin(latdiff_pc[ind] / 2.0)) ** 2
                    )
                    / (np.cos(satlat_rad) * np.cos(datalat_pc_rad[ind]))
                )
            )
        )

    if len(other_ind[0]) > 0:
        datalon_pc_rad[other_ind] = satlon_rad + 2.0 * np.arcsin(
            np.sqrt(
                np.abs(
                    (
                        (np.sin(sat_true_centralangle[other_ind] / 2.0)) ** 2
                        - (np.sin(latdiff_pc[other_ind] / 2.0)) ** 2
                    )
                    / (np.cos(satlat_rad) * np.cos(datalat_pc_rad[other_ind]))
                )
            )
        )

    # ----->Place parallax-corrected latitudes and longitudes into respective arrays
    lat_pc = datalat_pc_rad * to_degrees
    lon_pc = datalon_pc_rad * to_degrees

    if masked:
        lat_pc = np.ma.MaskedArray(lat_pc, mask=mask)
        lon_pc = np.ma.MaskedArray(lon_pc, mask=mask)
    # logging.info('Some error checking...')

    # ind = np.where(lon_pc < -180.0)
    # if(len(ind[0]) > 0): lon_pc[ind] = 180.0 - (-180.0 - lon_pc[ind])
    # ind = np.where(lats == -999)
    # if(len(ind[0]) > 0):
    #  lat_pc[ind] = -999.
    #  lon_pc[ind] = -999.

    if not (return_dict) and not (write_plax_file):
        logging.info("Process ended.")
        return lat_pc, lon_pc

    else:
        # NB: NW corner should be the 0,0 point for your lats and lons
        logging.info("Getting new xinds (for lons) and yinds (for lats)")

        latdiff = lats - lat_pc
        londiff = lons - lon_pc
        nlats, nlons = lons.shape
        xind = np.zeros((nlats, nlons), dtype=int)
        yind = np.zeros((nlats, nlons), dtype=int)

        t1 = time.time()
        mask = (
            np.ma.is_masked(lats[1:-1, 1:-1])
            | np.ma.is_masked(lats[:-2, 1:-1])
            | np.ma.is_masked(lats[2:, 1:-1])
            | np.ma.is_masked(lons[1:-1, 1:-1])
            | np.ma.is_masked(lons[1:-1, :-2])
            | np.ma.is_masked(lons[1:-1, 2:])
        )

        dy = (lats[:-2, 1:-1] - lats[2:, 1:-1]) / 2.0
        dx = (lons[1:-1, :-2] - lons[1:-1, 2:]) / 2.0

        latshift = np.round(latdiff[1:-1, 1:-1] / dy)  # number of pixels to shift
        lonshift = np.round(londiff[1:-1, 1:-1] / dx)  # number of pixels to shift

        new_elem = np.arange(1, nlons - 1) + lonshift  # add vector to each column
        new_line = (
            np.arange(1, nlats - 1).reshape(-1, 1) + latshift
        )  # reshape so that we add to each row

        xind[1:-1, 1:-1] = new_elem
        yind[1:-1, 1:-1] = new_line

        logging.info("Accounting for pixels off the grid")

        # Account for edges
        yind[0, :] = yind[1, :]
        yind[-1, :] = yind[-2, :]
        yind[:, 0] = yind[:, 1]
        yind[:, -1] = yind[:, -2]
        xind[0, :] = xind[1, :]
        xind[-1, :] = xind[-2, :]
        xind[:, 0] = xind[:, 1]
        xind[:, -1] = xind[:, -2]

        # Account for pixels off the grid
        yind[yind >= nlats] = nlats - 1
        yind[yind < 0] = 0
        xind[xind >= nlons] = nlons - 1
        xind[xind < 0] = 0

        outdata = {
            "xind": xind,
            "yind": yind,
            "lat_orig": lats,
            "lon_orig": lons,
            "lat_pc": lat_pc,
            "lon_pc": lon_pc,
            "elevation_in_m": elevation,
            "satlon": round(satlon, 1),
            "dx": f"{np.round(lons[0,1] - lons[0,0],2) * 100}km",
            "dy": f"{np.round(lats[0,0] - lats[1,0],2) * 100}km",
        }

        # import matplotlib.pyplot as plt
        # fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,9))
        # im = ax[0].imshow(latshift); ax[0].set_title('latshift'); plt.colorbar(im,ax=ax[0],shrink=0.5)
        # im = ax[1].imshow(lonshift); ax[1].set_title('lonshift'); plt.colorbar(im,ax=ax[1],shrink=0.5)
        # plt.savefig('tmp.png',bbox_inches='tight')
        # sys.exit()

        if return_dict:
            return outdata

        if write_plax_file:
            outfile = "PLAX_" + "{:.1f}".format(satlon) + f"_{int(elevation)}m.pkl"
            pickle.dump(outdata, open(outfile, "wb"))
            logging.info(f"Wrote out {outfile}")


#####################################################################################################
# custom json output classes
# see https://stackoverflow.com/questions/13249415/how-to-implement-custom-indentation-when-pretty-printing-with-the-json-module
class NoIndent(object):
    """Value wrapper."""

    def __init__(self, value):
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = "@@{}@@"
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get("sort_keys", None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, NoIndent)
            else super(MyEncoder, self).default(obj)
        )

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), json_obj_repr
            )

        return json_repr


#####################################################################################################
def mkdir_p(path):
    try:
        os.makedirs(path)
    except (OSError, IOError) as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


#####################################################################################################
def read_data(file, var_name, atts=None, global_atts=None):
    logging.info("Process started. Received " + file)

    if file[-3:] == ".gz":
        gzip = 1
        unzipped_file = file[0 : len(file) - 3]
        the_file = unzipped_file
        s = call(["gunzip", "-f", file])
    else:
        gzip = 0
        the_file = file

    try:
        openFile = Dataset(the_file, "r")
    except (OSError, IOError) as err:
        logging.error(str(err))
        return -1

    data = openFile.variables[var_name][:]
    # get variable attributes
    if isinstance(atts, dict):
        for att in atts:
            if att in openFile.variables[var_name].ncattrs():
                atts[att] = getattr(
                    openFile.variables[var_name], att
                )  # atts is modified and returned
    # get global attributes
    if isinstance(global_atts, dict):
        for gatt in global_atts:
            if gatt in openFile.ncattrs():
                global_atts[gatt] = getattr(
                    openFile, gatt
                )  # global_atts is modified and returned
    openFile.close()

    # --gzip file after closing?
    if gzip:
        # s = Popen(['gzip','-f',unzipped_file]) #==will run in background since it doesn't need output from command.
        s = call(["gzip", "-f", unzipped_file])

    logging.info("Got new " + var_name + ". Process ended.")
    return data


#####################################################################################################
def read_netcdf(file, datasets, global_atts=None):
    logging.info("Process started. Received " + file)
    # datasets is a dict, formatted thusly:
    # datasets = {'varname1':{'atts':['attname1','attname2']}, 'varname2':{'atts':[]}}
    # atts are variable attributes. If 'atts':['ALL'], then it will grab all variable attributes.
    # global_atts are file global attributes.
    # Each key/var in datasets is returned with a 'data' key...which contains a numpy array

    if file[-3:] == ".gz":
        gzip = 1
        unzipped_file = file[0 : len(file) - 3]
        the_file = unzipped_file
        s = call(["gunzip", "-f", file])
    else:
        gzip = 0
        the_file = file

    try:
        openFile = Dataset(the_file, "r")
    except (OSError, IOError) as err:
        raise

    for key in datasets:
        datasets[key]["data"] = openFile.variables[key][:]
        # get each variable attribute
        if datasets[key]["atts"] == ["ALL"]:
            datasets[key]["atts"] = {}
            for att in openFile.variables[key].ncattrs():
                datasets[key]["atts"][att] = getattr(openFile.variables[key], att)
        # get select variable attributes
        else:
            tmpdict = {}
            for att in datasets[key]["atts"]:
                if att in openFile.variables[key].ncattrs():
                    tmpdict[att] = getattr(openFile.variables[key], att)
            datasets[key]["atts"] = tmpdict
    # get global attributes
    if isinstance(global_atts, dict):
        for gatt in global_atts:
            if gatt in openFile.ncattrs():
                global_atts[gatt] = getattr(
                    openFile, gatt
                )  # global_atts is modified and returned
    openFile.close()

    # --gzip file after closing?
    if gzip:
        # s = Popen(['gzip','-f',unzipped_file]) #==will run in background since it doesn't need output from command.
        s = call(["gzip", "-f", unzipped_file])

    logging.info("Process ended.")


#####################################################################################################
def read_grib2(infile, gpd_info, constants):
    logging.info("Process started. Received " + infile)

    # now read grib2
    grb = pg.open(infile)
    msg = grb[1]  # should only be 1 'message' in grib2 file
    data = msg.values
    # make file lats/lons. Could pull from grib, e.g.: lats = grb[1]['latitudes'] ; but making them is faster than pulling from grib.
    grb.close()
    orig_nlat, orig_nlon = np.shape(data)
    origNWlat = constants["MRMS"]["origNWlat"]  # usu. 55.0
    origNWlon = constants["MRMS"]["origNWlon"]  # usu. -130.0

    # gpd_info = read_gpd_file(gpd)
    NW_lat = gpd_info["NW_lat"]
    NW_lon = gpd_info["NW_lon"]
    nlon = gpd_info["nlon"]
    nlat = gpd_info["nlat"]
    dy = gpd_info["dy"]
    dx = gpd_info["dx"]

    # just cut out the part we want
    offset_from_origNWlat = int((origNWlat - NW_lat) / dy)
    offset_from_origNWlon = int((NW_lon - origNWlon) / dx)
    new_data = (
        data[
            offset_from_origNWlat : (offset_from_origNWlat + nlat),
            offset_from_origNWlon : (offset_from_origNWlon + nlon),
        ]
    ).astype(np.float32)

    logging.info("Got new data. Process ended.")
    return new_data


#####################################################################################################
def write_netcdf(
    output_file, datasets, dims, atts={}, gzip=False, wait=False, nc4=False, **kwargs
):
    logging.info("Process started")

    try:
        mkdir_p(os.path.dirname(output_file))
    except (OSError, IOError) as err:
        logging.error(str(err))
        return False
    else:
        if nc4:
            ncfile = Dataset(output_file, "w")
        else:
            ncfile = Dataset(output_file, "w")  # ,format='NETCDF3_CLASSIC')
    # dimensions
    for dim in dims:
        ncfile.createDimension(dim, dims[dim])
    # variables
    for varname in datasets:
        try:
            dtype = datasets[varname]["data"].dtype.str
        except AttributeError:
            if isinstance(datasets[varname]["data"], int):
                dtype = "int"
            elif isinstance(datasets[varname]["data"], float):
                dtype = "float"

        if "_FillValue" in datasets[varname]["atts"]:
            dat = ncfile.createVariable(
                varname,
                dtype,
                datasets[varname]["dims"],
                fill_value=datasets[varname]["atts"]["_FillValue"],
            )
        else:
            dat = ncfile.createVariable(varname, dtype, datasets[varname]["dims"])
        dat[:] = datasets[varname]["data"]
        # variable attributes
        if "atts" in datasets[varname]:
            for att in datasets[varname]["atts"]:
                if att != "_FillValue":
                    dat.__setattr__(
                        att, datasets[varname]["atts"][att]
                    )  # _FillValue is made in 'createVariable'
    # global attributes
    for key in atts:
        ncfile.__setattr__(key, atts[key])
    ncfile.close()
    logging.info("Wrote out " + output_file)

    if gzip:
        if wait:
            s = call(["gzip", "-f", output_file])  # ==wait to finish
        else:
            s = Popen(["gzip", "-f", output_file])  # ==will run in background

    logging.info("Process ended")
    return True


#####################################################################################################
def read_hdf_sd(file, varnames, att_names=[], FillValue=-999, unscale=True):
    logging.info("Process started")

    data = {}
    atts = {}
    if not (isinstance(varnames, list)):
        varnames = [varnames]
    if not (os.path.isfile(file)):
        logging.error(file + " does not exist")
        return data, atts

    # try to read the data 3 times
    for i in range(3):
        try:
            hdf = SD(file, SDC.READ)
        except pyhdf.error.HDF4Error as err:
            logging.warning(str(err))
            logging.info("Retrying read of " + file)
            time.sleep(2)
            if i == 2:
                logging.error("Cannot read " + file)
                return data, atts
        else:
            break  # read in just fine

    for var in varnames:
        if var in hdf.datasets().keys():
            sd = hdf.select(var)
            data[var] = {}
            data[var]["data"] = sd[:]
            data[var]["atts"] = sd.attributes()
            data[var]["isValid"] = True
            sd.endaccess()  # close dataset
        else:
            logging.warning(var + " is not present in " + file)
            data[var] = {}
            data[var]["data"] = -1
            data[var]["atts"] = {}
            data[var]["isValid"] = False

        if "_FillValue" in data[var]["atts"]:
            bad_ind = data[var]["data"] == data[var]["atts"]["_FillValue"]
        if (
            "scale_factor" in data[var]["atts"]
            and "add_offset" in data[var]["atts"]
            and unscale
        ):
            data[var]["data"] = (
                data[var]["data"] * data[var]["atts"]["scale_factor"]
                + data[var]["atts"]["add_offset"]
            )
        if "_FillValue" in data[var]["atts"]:
            data[var]["data"][bad_ind] = FillValue

    for att in att_names:
        if att in hdf.attributes():
            atts[att] = hdf.attributes()[att]

    hdf.end()  # close hdf file

    logging.info("Process ended")

    return data, atts


#####################################################################################################
def restart_w2alg_proc(group, proc_name, email=False):
    logging.info("Process started")

    machine = os.uname()[1]

    logging.info("executing `w2alg stop " + group + " " + proc_name + "` on " + machine)

    FNULL = open(os.devnull, "w")  # write output to /dev/null

    s = Popen(["w2alg", "stop", group, proc_name], stdout=FNULL, stderr=FNULL)

    time.sleep(2)  # make sure proc_name completely shuts down

    logging.info(
        "executing `w2alg start " + group + " " + proc_name + "` on " + machine
    )

    s = Popen(["w2alg", "start", group, proc_name], stdout=FNULL, stderr=FNULL)

    if email:
        sub = "ProbSevere processing notification on " + machine
        msg = (
            "`w2alg stop/start "
            + group
            + " "
            + proc_name
            + "` has occurred on "
            + machine
            + ". This may be a routine restart because objectID# got too large, "
            + "or because there was too large a time gap between scans."
        )
        to = os.environ["ADDRESS"]
        send_email(sub, msg, to)

    logging.info("Process ended")


#####################################################################################################
def send_email(subject, message, to):

    logging.info("Process started")

    user = getpass.getuser()
    hostname = os.uname()[1]
    FROM = "john.cintineo@ssec.wisc.edu"
    TO = to.split(",")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    theEmail = "Subject: %s\n\n%s\n%s" % (subject, message, now)

    s = smtplib.SMTP("localhost")
    s.sendmail(FROM, TO, theEmail)
    s.quit()

    logging.info("Process ended")


#####################################################################################################
def read_gpd_file(gpd_file):
    logging.info("Process started")
    gpd_info = {}

    try:
        f = open(gpd_file, "r")
    except (OSError, IOError) as err:
        logging.error(str(err))
        raise
    for line in f:
        parts = re.split(r"\s{2,}", line)
        if parts[0] == "Map Projection:":
            gpd_info["projection"] = parts[1].rstrip("\n")
        elif parts[0] == "Map Reference Latitude:":
            gpd_info["reflat"] = float(parts[1])
        elif parts[0] == "Map Reference Longitude:":
            gpd_info["reflon"] = float(parts[1])
        elif parts[0] == "Map Scale:":
            gpd_info["map_scale"] = float(parts[1])
        elif parts[0] == "Map Origin Latitude:":
            gpd_info["NW_lat"] = float(parts[1])
        elif parts[0] == "Map Origin Longitude:":
            gpd_info["NW_lon"] = float(parts[1])
        elif parts[0] == "Map Equatorial Radius:":
            gpd_info["equatorial_radius"] = float(parts[1])
        elif parts[0] == "Grid Width:":
            gpd_info["nlon"] = int(parts[1])
        elif parts[0] == "Grid Height:":
            gpd_info["nlat"] = int(parts[1])
        elif parts[0] == "Grid Map Units per Cell:":
            gpd_info["dy"] = gpd_info["dx"] = int(parts[1]) / 3600.0

    logging.info("Read " + gpd_file + ". Process ended.")

    return gpd_info


########################################################################################################
def get_area_definition(
    projection_file, outny=-1, outnx=-1, return_latlons=False, return_xy=False
):
    from pyproj import Proj
    import pyresample as pr
    import numpy.ma as ma

    logging.info(f"Getting AreaDefinition from {projection_file}")

    nc = Dataset(projection_file)
    gip = nc.variables["goes_imager_projection"]
    x = nc.variables["x"][:]
    y = nc.variables["y"][:]

    # p = Proj('+proj=geos +lon_0='+str(gip.longitude_of_projection_origin)+\
    #         ' +h='+str(gip.perspective_point_height)+' +x_0=0.0 +y_0=0.0 +a=' + \
    #         str(gip.semi_major_axis) + ' +b=' + str(gip.semi_minor_axis) + ' +sweep=x')

    x *= gip.perspective_point_height
    y *= gip.perspective_point_height
    # print(x.min(),x.max())
    # xx,yy = np.meshgrid(x,y)
    ny, nx = len(y), len(x)  # np.shape(xx)
    # newlons, newlats = p(xx, yy, inverse=True)
    # mlats = ma.masked_where((newlats < -90) | (newlats > 90),newlats)
    # mlons = ma.masked_where((newlons < -180) | (newlons > 180),newlons)
    # print(mlons.max(),mlons.min())
    # newx,newy = p(mlons,mlats)
    # print(newx.min(),newx.max(),newy.min(),newy.max())
    # max_x = newx.max(); min_x = newx.min(); max_y = newy.max(); min_y = newy.min()
    # half_x = (max_x - min_x) / newlons.shape[1] / 2.
    # half_y = (max_y - min_y) / newlats.shape[0] / 2.
    # extents = (min_x - half_x, min_y - half_y, max_x + half_x, max_y + half_y)
    # print(min_x,max_x)
    extents = (x.min(), y.min(), x.max(), y.max())
    if outny > 0:
        # Increase or decrease spatial resolution. Note that outny and outnx should be
        # a whole-number multiple of ny and nx.
        # assert(ny % outny == 0 and nx % outnx == 0)
        ny = outny
        nx = outnx

    newgrid = pr.geometry.AreaDefinition(
        "geos",
        "goes_conus",
        "goes",
        {
            "proj": "geos",
            "h": gip.perspective_point_height,
            "lon_0": gip.longitude_of_projection_origin,
            "a": gip.semi_major_axis,
            "b": gip.semi_minor_axis,
            "sweep": gip.sweep_angle_axis,
        },
        nx,
        ny,
        extents,
    )
    nc.close()

    if return_latlons:
        return newgrid, ny, nx, newlons, newlats
    if return_xy:
        return newgrid, ny, nx, x, y
    else:
        return newgrid, ny, nx


########################################################################################################
def find_int_rem(elements, maxncpu=10):
    # elements is a list or 1D array
    # maxncpu is the maximum number of cpus you want to spin up

    # Returns the interval or length of elements to add to each CPU.
    # Also returns the remainder, which you need to add/distribute among the CPUs.
    # Also returns a possibly modified maxncpu, if your element list is too small
    # and your maxncpu too large

    nelem = len(elements)
    assert nelem >= 1

    while nelem // maxncpu == 0:
        maxncpu = (
            maxncpu // 2
        )  # reduce maxncpu by factor of 2 until we get 1 or greater CPUs

    if maxncpu == 1:
        logging.warning(
            f"max number cpu found to be {maxncpu} for list with length {nelem}"
        )
        logging.warning("Consider not using multiprocessing")

    interval = int(nelem / maxncpu)
    remainder = nelem % maxncpu
    return interval, remainder, maxncpu


# ------------------------------------------------------------------------------------------------------
def bytescale(data_arr, vmin, vmax):
    assert vmin < vmax
    DataImage = np.round((data_arr - vmin) / (vmax - vmin) * 255.9999)
    DataImage[DataImage < 0] = 0
    DataImage[DataImage > 255] = 255
    return DataImage.astype(np.uint8)


# ------------------------------------------------------------------------------------------------------
def unbytescale(scaled_arr, vmin, vmax):
    assert vmin < vmax
    scaled_arr = scaled_arr.astype(np.float32)
    unscaled_arr = scaled_arr / 255.9999 * (vmax - vmin) + vmin
    return unscaled_arr


# ------------------------------------------------------------------------------------------------------
# This function is taken from SkyCover's use of YamlCascade
def extract_keypath_values(seq):
    for s in seq:
        try:
            k, v = s.split("=")
        except ValueError as not_an_expression:
            raise ValueError(
                f"expression {repr(s)} is not of the form key-path=yaml-value"
            )
        yield k.strip(), v.strip()


# ------------------------------------------------------------------------------------------------------
def init_config_env(yaml_files, yaml_overrides, format_context, init_env=True):

    config = LightningcastYamlCascade(format_context, *yaml_files, **yaml_overrides)

    if init_env:
        if not os.environ.get("PWD"):
            os.environ["PWD"] = os.getcwd()
        os.environ["ONDEMAND_CSV"] = str(config["enviorment.ondemand_csv"])
        os.environ["DSS_DURATION_THRESH"] = str(
            config["enviorment.dss_duration_thresh"]
        )
        os.environ["PLTG"] = str(config["enviorment.pltg"])
        os.environ["PLTG_DATA"] = str(config["enviorment.pltg_data"])
        os.environ["SATPY_CONFIG_PATH"] = str(config["enviorment.satpy_config_path"])
        os.environ["PS_PYTHONPATH"] = str(sys.executable)
        sys.path.insert(0, str(config["enviorment.path_addition"]))

    #  print(os.environ["ONDEMAND_CSV"])
    #  print(os.environ["DSS_DURATION_THRESH"])
    #  print(os.environ["PLTG"])
    #  print(os.environ["PLTG_DATA"])
    #  print(os.environ["SATPY_CONFIG_PATH"])
    #  print(os.environ["PS_PYTHONPATH"])
    #  print(sys.path)

    return config
