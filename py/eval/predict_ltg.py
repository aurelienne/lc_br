from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import imageio.v2
import traceback
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os, sys
import copy
import mysql.connector
import uuid

pltg_data = f"{os.environ['PLTG_DATA']}/"
pltg = f"{os.environ['PLTG']}/"

import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(current_dir))

import pygrib as pg
from scipy.ndimage import zoom

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from netCDF4 import Dataset
import numpy as np
import argparse
import pandas as pd
import pyresample as pr

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    from lightningcast import keras_metrics
except ModuleNotFoundError:
    print("TF modules not found...make sure you are using pytorch.")
    pass

# Set the number of CPUs to use
# num_threads = 40
# os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
# os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
# os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"
# tf.config.threading.set_inter_op_parallelism_threads(num_threads)
# tf.config.threading.set_intra_op_parallelism_threads(num_threads)
# tf.config.set_soft_device_placement(enabled=True)

import satpy
from satpy.writers import get_enhanced_image
import time
import subprocess
from pyproj import Proj
from lightningcast.static.ABIcmaps import irw_ctc
from lightningcast import utils
import numpy.ma as ma
from datetime import datetime, timedelta, timezone
import re
import csv
from lightningcast.solar_angles import solar_angles
from lightningcast.sat_zen_angle import sat_zen_angle
import pickle
import glob
import shutil
import collections
import logging
from logging.config import dictConfig
from lightningcast.logging_def import *
from lightningcast.earth_to_fgf import earth_to_fgf
from lightningcast import ltg_utils
from lightningcast.__version__ import __version__
import rasterio
import rasterio.transform

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import ftplib

from lightningcast.pltg_gr_placefile import timeRangeGRFile
from lightningcast.__version__ import __version__
# Per NCCF requirements, 1.3.2 becomes 1r3
version = ('r').join(__version__.split('.')[0:2]) 
#################################################################################################################################
import shapely
import warnings
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message="The .xlabels_bottom attribute is deprecated. Please use .bottom_labels to toggle visibility instead.",
)
warnings.filterwarnings(
    "ignore",
    message="The .ylabels_right attribute is deprecated. Please use .right_labels to toggle visibility instead.",
)
warnings.filterwarnings(
    "ignore",
    message="You will likely lose important projection information when converting to a PROJ string from another format. See: https://proj.org/faq.html#what-is-the-best-format-for-describing-coordinate-reference-systems",
)
np.seterr(divide='ignore', invalid='ignore')
from shapely import geometry
from geojson import Polygon, Feature, FeatureCollection
import geojson
from geojson_rewind import rewind
import json

#################################################################################################################################
def append_to_fc_list(fc_list, poly, prob):
    # fc_list gets appended and returned
    poly = Feature(geometry=poly)
    poly["properties"]["PROBABILITY"] = str(prob)

    if poly.is_valid:
        fc_list.append(poly)
    else:
        logging.error("poly not valid:")
        print(poly)

#################################################################################################################################
def get_all_paths(contour_img):
    '''
    Making the process of getting the Paths backward-compatible with matplotlib < 3.8
    '''

    if hasattr(contour_img, 'get_paths'):
        # For 3.8+
        # all_paths is a list of matplotlib.path.Paths, one for each contour level
        # Each Path is a continuous set of vertices and codes (2 arrays) for all lines in the contour level
        all_paths = contour_img.get_paths()
    else:
        # For < 3.8
        all_paths = []
        for path_col in contour_img.collections:
            # One path_col per contour level
            all_verts = []
            all_codes = []
            for path in path_col.get_paths():
                # Combine all the vertices and codes for the given path_col
                # into a new Path (below), so that it is in the expected format
                # for the for loop below
                all_verts.append(path.vertices)
                all_codes.append(path.codes)

            if len(all_verts):
                newpath = matplotlib.path.Path(vertices=np.concatenate(all_verts),
                                               codes=np.concatenate(all_codes))
                all_paths.append(newpath)
           
    return all_paths

#################################################################################################################################
def write_json(
    new_preds,
    idict,
    meta,
    outdir,
    prefix="LtgCast",
    values=[0.1, 0.25, 0.5, 0.75],
):


    tj = time.time()
    stride = idict['stride']

    preds_shape = new_preds.shape
    if len(preds_shape) > 2:
      n_outputs = preds_shape[-1]
    else:
      n_outputs = 1
      new_preds = np.expand_dims(new_preds, axis=-1)

    for ii in range(n_outputs):

        fc_list = []

        if ii == 0:
            vals = values.copy()
            new_prefix = prefix
            new_outdir = outdir + '1fl-60min'
        else:
            vals = [0.1, 0.25]     # hard-coded info for input beyond 0th
            #rgb_colors = get_rgb_colors(["#FF0000", "#FF9999"])
            new_prefix = prefix + '-10fl' 
            new_outdir = outdir + '10fl-60min'
         
        preds = new_preds[...,ii]
 
        if np.max(preds) >= vals[0]:
        
            fig = plt.figure(figsize=(15, 15))
            ax = plt.axes(projection = idict['ccrs_proj'])
    
            # irlons = irlons % 360 --> all positive
            # irlons = (irlons - 360) % (-360) --> all negative        
            if '-pc' in prefix:
                contour_img = ax.contour(
                    #plax_lons are already converted to all negative!
                    idict['plax_lons'][0::stride, 0::stride],
                    idict['plax_lats'][0::stride, 0::stride],
                    preds[0::stride, 0::stride],
                    vals,
                    transform=ccrs.PlateCarree(),
                )
                
            else:
                contour_img = ax.contour(
                    (idict['irlons'][0::stride, 0::stride] - 360) % (-360),
                    idict['irlats'][0::stride, 0::stride],
                    preds[0::stride, 0::stride],
                    vals,
                    transform=ccrs.PlateCarree(),
                )


            all_paths = get_all_paths(contour_img)

            for cter, path in enumerate(all_paths):
                # Each path is a set of vertices for the same probability level. 
                # We use the codes to start, stop, and connect polygons (or not connect linestrings)
                # See more info here: https://matplotlib.org/stable/api/path_api.html
                verts = []
                for ii in range(len(path.vertices)):
               
                    # Each path has vertices and codes attributes of the same length
                    if path.codes[ii] == 1: 
                        # Pick up the pen and move to the given vertex
                        if len(verts) > 1:
                            # Save off previous Feature
                            new_shape = geometry.LineString(verts)
                            append_to_fc_list(
                                fc_list,
                                new_shape,
                                int(vals[cter] * 100),
                            )
                        verts = [] # Reset
                        # Begin new feature
                        verts.append(np.round(path.vertices[ii],3))
                    elif path.codes[ii] == 79:
                        #Draw a line segment to the start point of the current polyline
                        # Close a polygon
                        verts.append(np.round(path.vertices[ii],3))
                        # Save off this Feature
                        new_shape = geometry.Polygon(verts)
                        append_to_fc_list(
                            fc_list,
                            new_shape,
                            int(vals[cter] * 100),
                        )
                        verts = [] # Reset
                    elif path.codes[ii] == 2:
                        # Draw a line from the current position to the given vertex
                        verts.append(np.round(path.vertices[ii],3))
                    else:
                        logging.error(f"Code {path.codes[ii]} is not recognized! Returning.")
                        return
                # Save off final LineString for the contour level, potentially.
                # If ended with 79 (a Polygon), len(verts) should = 0 and feature should already be saved off
                if len(verts) > 1:
                    new_shape = geometry.LineString(verts)
                    append_to_fc_list(
                        fc_list,
                        new_shape,
                        int(vals[cter] * 100),
                    )


        outdict = collections.OrderedDict()

        # "Collection-Level" Metadata
        outdict["naming_authority"] = "gov.noaa.nesdis.ncei"
        # outdict['Conventions'] = "CF-1.8"
        # outdict['standard_name_vocabulary'] = "CF Standard Name Table (v83, 17 October 2023)"
        outdict["project"] = "NESDIS Common Cloud Framework"
        outdict["institution"] = idict["institution"]
        outdict["platform"] = meta["platform_name"]
        outdict["platform_type"] = "Geostationary"
        outdict["instrument"] = meta["sensor"].upper()
        outdict["sector"] = sector_outname = idict["sector_outname"]
        outdict["title"] = idict["title"]
        outdict[
            "summary"
        ] = "Short-term probabilistic lightning predictions using remotely sensed data and machine-learning models"
        outdict["product"] = "Probability of lightning in next 60 minutes"
        outdict["history"] = f"LightningCast v{version}"
        outdict[
            "processing_level"
        ] = "National Oceanic and Atmospheric Administration (NOAA) Level 2"
        outdict["source"] = meta["reader"]  # abi_l1b or ahi_l1b
        outdict["references"] = "https://doi.org/10.1175/WAF-D-22-0019.1"
        outdict[
            "keywords"
        ] = "LIGHTNING, INFRARED RADIANCE, VISIBLE RADIANCE, MACHINE-LEARNING"
        outdict["keywords_vocabulary"] = f"NASA Global Change Master Directory (GCMD) Science Keywords"
        # "Granule-Level" Metadata
        outdict["id"] = str(uuid.uuid4())
        outdict["production_site"] = idict["production_site"]
        outdict["production_environment"] = idict["production_environment"]
        outdict["metadata_link"] = "https://cimss.ssec.wisc.edu/severe_conv/pltg.html"
        outdict["time_coverage_start"] = meta["start_time"].strftime(idict["time_patt"])
        outdict["time_coverage_end"] = meta["end_time"].strftime(idict["time_patt"])
        now = datetime.utcnow()
        outdict["date_created"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        if meta["sensor"].upper() == 'ABI':
           sensor_band_identifier = f"02,05,13,15"
        else:
           sensor_band_identifier = f"03,05,13,15"
        outdict["sensor_band_identifier"] = sensor_band_identifier 
        outdict["sensor_band_central_radiation_wavelength"] = f"0.64um,1.6um,10.3um,12.3um"

        platform_abbr = (
            meta["platform_name"][0].lower() + meta["platform_name"].split("-")[1]
        )
        outjson = f"{new_prefix}-{sector_outname.replace('_','-')}_v{version}_{platform_abbr}"
        # Note: %f is microseconds. We take the first digit to get tenths of a second, which NESDIS requires.
        outjson += f"_s{meta['start_time'].strftime('%Y%m%d%H%M%S%f')[0:15]}"
        outjson += f"_e{meta['end_time'].strftime('%Y%m%d%H%M%S%f')[0:15]}"
        outjson += f"_c{now.strftime('%Y%m%d%H%M%S%f')[0:15]}.json"


        outfile = f"{new_outdir}/{outjson}"

        if outdict["platform"].startswith("GOES"):
            outdict["orbital_slot"] = meta["orbital_slot"]  # GOES specific
            slot_init = meta["orbital_slot"][5]  # GOES specific
            if "USSAMOA" in sector_outname:
                outdict["sector_id"] = sector_id = "USSAMOA"
            elif "AKCAN" in sector_outname:
                outdict["sector_id"] = sector_id = "AKCAN"
            else:
                if sector_outname.startswith("RadC"):
                    sector_id = "CONUS"
                elif sector_outname.startswith("RadM1"):
                    sector_id = "MESO1"
                elif sector_outname.startswith("RadM2"):
                    sector_id = "MESO2"
                elif sector_outname.startswith("RadF"):
                    sector_id = "FULL"
                outdict["sector_id"] = f"{slot_init}{sector_id}"
        elif outdict["platform"].startswith("Himawari"):
            sector_id = "GUAM"
            outdict["sector_id"] = sector_id

        outdict["model"] = meta["model"]

        # Geographic metadata
        outdict["cdm_data_type"] = "vector"
        outdict["geospatial_lat_min"] = np.min(
            idict["irlats"][np.isfinite(idict["irlats"])]
        )
        outdict["geospatial_lat_max"] = np.max(
            idict["irlats"][np.isfinite(idict["irlats"])]
        )
        outdict["geospatial_lon_min"] = np.min(
            idict["irlons"][np.isfinite(idict["irlons"])]
        )
        outdict["geospatial_lon_max"] = np.max(
            idict["irlons"][np.isfinite(idict["irlons"])]
        )
        outdict["geospatial_lat_units"] = "degrees_north"
        outdict["geospatial_lon_units"] = "degrees_east"

        # Product development team and publisher metadata
        outdict[
            "creator_name"
        ] = "DOC/NOAA/NESDIS/STAR > ProbSevere Team, Center for Satellite Applications and Research, NESDIS, NOAA, U.S. Department of Commerce."
        outdict["creator_email"] = "probsevere@gmail.com"
        outdict["creator_url"] = "https://cimss.ssec.wisc.edu/severe_conv/pltg.html"
        outdict[
            "publisher_name"
        ] = "DOC/NOAA/NESDIS/OSPO > Office of Satellite and Product Operations, NESDIS, NOAA, U.S. Department of Commerce."
        outdict["publisher_email"] = "espcoperations@noaa.gov"
        outdict["publisher_url"] = "http://www.ospo.noaa.gov"

        # Add the features to the dictionary
        outdict["type"] = "FeatureCollection"
        outdict["features"] = fc_list

        # Enforce polygon ring winding
        outdict = rewind(outdict)

        # outjson_obj = json.dumps(outdict, indent=2, sort_keys=True) #pretty printing
        outjson_obj = json.dumps(outdict, indent=None)
        try:
            os.makedirs(new_outdir, exist_ok=True)
            of = open(outfile, "w")
            of.write(outjson_obj)
            of.close()
        except (PermissionError, OSError, IOError):
            logging.error(traceback.format_exc())
            return None

        logging.info(f"wrote out {outfile}")



    return outfile

    # -- DEPRECATED -- JC 11/2023
    # if(ftp is not None): #remove LineStrings, save new filename.
    #  #remove LineStrings, since AWIPS currently can't handle them.
    #  fc_list[:] = [x for x in fc_list if not x['geometry']['type'].lower() == 'linestring']
    #  outdict['features'] = fc_list
    #  outdict = rewind(outdict) #just in case
    #  outjson_obj = json.dumps(outdict, indent=None)
    #  #Save new filename for AWIPS
    #  ldm_dirname = os.path.dirname(outfile)
    #  ldm_filename = f'SSEC_AWIPS_PROBSEVERE-LTGCast_{fileparts[1]}_{fileparts[2]}.json'
    #  ldm_outfile = f'{ldm_dirname}/{ldm_filename}'
    #  of = open(ldm_outfile,'w')
    #  of.write(outjson_obj)
    #  of.close()

    #  if(ftp is not None):
    #    fName = os.path.basename(ldm_outfile)
    #    session = ftplib.FTP('ftp.ssec.wisc.edu','anonymous','jcintineo') #needs to be a valid/credentialed user
    #    of = open(ldm_outfile,'rb')         # file to send
    #    session.cwd(ftp) #e.g., /pub/probsevere/ltgcast/
    #    session.storbinary(f"STOR {fName}", of)   # send the file
    #    of.close()                  # close file and FTP
    #    session.quit()
    #  #if(ldm): #not sending JSONs to LDM -- 02/10/2021 (JLC)
    #  #  #send to ldm
    #  #  status = subprocess.call(['/home/ldm/bin/pqinsert','-f','EXP','-q','/home/ldm/var/queues/ldm.pq',
    #  #               '-p',ldm_filename,ldm_outfile])


#################################################################################################################################
def find_latest(listened_file, sector):
    if not (os.path.isfile(listened_file)):
        logging.critical(listened_file + " is not a file or does not exist")
        sys.exit(1)
    f = open(listened_file, "r")
    lines = f.read().split("\n")
    logging.info("Checking lines for " + sector)
    latest = None
    nlines = len(lines)
    for i in range(nlines - 1, 0, -1):
        if sector in lines[i]:
            latest = lines[i]
            break

    if latest:
        parts = latest.split("#")
        machine = parts[1]
        partition = "/" + machine.split(".")[0] + "_data"
        filestr = parts[-1]

        fixed_full_file = filestr.replace(
            "/data", partition
        )  # partition should be /satbuf2_data or /satbuf3_data

        # Get dt info from filestring instead of {datetime} info from amqpfind logfile,
        # in case there is some incongruity
        if sector == "FLDK":  # H8
            fileparts = filestr.split("_")
            dt = datetime.strptime(fileparts[-6] + fileparts[-5], "%Y%m%d%H%M")
        else:  # GOES
            fileparts = filestr.split("_")
            timestamp = fileparts[-1][1:14]
            dt = datetime.strptime(timestamp, "%Y%j%H%M%S")

    else:
        fixed_full_file = dt = "None"

    try:
        if sector == "FLDK":
            fixed_full_file = glob.glob(fixed_full_file.replace("B??", "B13"))[0]
        elif sector.startswith("Rad"):
            fixed_full_file = glob.glob(fixed_full_file.replace("C??", "C13"))[0]
    except IndexError:
        logging.warning(f"Can't find {fixed_full_file}.")
        #sys.exit(1)

    return fixed_full_file, dt


#################################################################################################################################
def get_ahi_scn(dt, datadir, sector):

    if sector == "JP":
        mss = dt.strftime("%M%S")[1:]
        if mss == "000":
            mm = "01"
        elif mss == "230":
            mm = "02"
        elif mss == "500":
            mm = "03"
        else:
            mm = "04"

        datafiles = glob.glob(dt.strftime(f"{datadir}/*%Y%m%d_%H00_B*_{sector}{mm}*"))

    else:  # FLDK
        datafiles = glob.glob(dt.strftime(f"{datadir}/*%Y%m%d_%H%M_B*_{sector}*"))

    if len(datafiles) == 0:
        logging.critical(
            "Can't find AHI data for "
            + dt.strftime(f"{datadir}/*%Y%m%d_%H%M_B*_{sector}*")
        )
        sys.exit(1)
    else:
        scene = satpy.Scene(filenames=datafiles, reader="ahi_hsd")

    return scene


#################################################################################################################################
def get_meteosat_scn(dt, datadir_patt):
    # check decompressor
    try:
        foo = os.environ["XRIT_DECOMPRESS_PATH"]
    except KeyError as err:
        logging.error("XRIT_DECOMPRESS_PATH is not set in environmental vars.")
        sys.exit(1)
    else:
        if os.path.isfile(os.environ["XRIT_DECOMPRESS_PATH"]):
            # getting the needed channels, as well as the prolog and epilog files
            filenames = (
                glob.glob(
                    dt.strftime(f"{datadir_patt}/%Y/%Y_%m_%d_%j/*VIS006*%Y%m%d%H%M*")
                )
                + glob.glob(
                    dt.strftime(f"{datadir_patt}/%Y/%Y_%m_%d_%j/*IR_1[0,2]*%Y%m%d%H%M*")
                )
                + glob.glob(
                    dt.strftime(f"{datadir_patt}/%Y/%Y_%m_%d_%j/*IR_016*%Y%m%d%H%M*")
                )
                + glob.glob(
                    dt.strftime(f"{datadir_patt}/%Y/%Y_%m_%d_%j/*EPI*%Y%m%d%H%M*")
                )
                + glob.glob(
                    dt.strftime(f"{datadir_patt}/%Y/%Y_%m_%d_%j/*PRO*%Y%m%d%H%M*")
                )
            )
        else:
            logging.error(
                f"XRIT_DECOMPRESS_PATH {os.environ['XRIT_DECOMPRESS_PATH']} does not exist."
            )
            sys.exit(1)

    scn = satpy.Scene(reader="seviri_l1b_hrit", filenames=filenames)
    return scn


#################################################################################################################################
def get_mtg_scn(dt, datadir_patt):
    # 40 files for MTG date-time
    # FD is every 10 min. FD files' start times are every 15 seconds.
    ###datadir_patt is temporary###
    tens_min = dt.strftime("%M")[0]
    filenames = glob.glob(
        dt.strftime(f"{datadir_patt}/*DEV_%Y%m%d%H{tens_min}*00??.nc")
    )
    scn = satpy.Scene(reader="fci_l1c_nc", filenames=filenames)
    return scn


#################################################################################################################################
def start_end_inds(xmin, xmax, ymin, ymax, irx, iry):
    xmin_ind, xmin = utils.find_nearest(irx, xmin)
    xmax_ind, xmax = utils.find_nearest(irx, xmax)
    ymin_ind, ymin = utils.find_nearest(iry, ymin)
    ymax_ind, ymax = utils.find_nearest(iry, ymax)

    # We need to round to the nearest 4 for U-net to work with CH02.
    base = 4
    xmin_ind = int(base * round(xmin_ind / base))
    xmax_ind = int(base * round(xmax_ind / base))
    xmin = irx[xmin_ind]
    try:
        xmax = irx[xmax_ind]
    except IndexError:
        xmax = irx[xmax_ind - 1]  # at right edge of grid
    ymin_ind = int(base * round(ymin_ind / base))
    ymax_ind = int(base * round(ymax_ind / base))
    ymin = iry[ymin_ind]
    try:
        ymax = iry[ymax_ind]
    except IndexError:
        ymax = iry[ymax_ind - 1]  # at bottom edge of grid

    startX = xmin_ind
    endX = xmax_ind
    startY = ymax_ind
    endY = ymin_ind
    xy_bbox = [xmin, ymin, xmax, ymax]

    return startY, endY, startX, endX, xy_bbox


#################################################################################################################################
def save_netcdf(
    scn_ch,
    preds,
    dt,
    meta,
    outdir,
    idict,
    stride=4,
    savecode=1,
    scale_data=False,
    awips=False,
):

    atts = collections.OrderedDict()  # global attributes

    # "Collection-Level" Metadata
    atts["naming_authority"] = "gov.noaa.nesdis.ncei"
    atts["Conventions"] = "CF-1.8"
    atts["standard_name_vocabulary"] = "CF Standard Name Table (v83, 17 October 2023)"
    atts["project"] = "NESDIS Common Cloud Framework"
    atts["institution"] = idict["institution"]
    atts["platform"] = meta["platform_name"]
    atts["platform_type"] = "Geostationary"
    atts["instrument"] = meta["sensor"].upper()
    atts["sector"] = sector_outname = idict["sector_outname"]
    atts["title"] = idict["title"]
    atts[
        "summary"
    ] = "Short-term probabilistic lightning predictions using remotely sensed data and machine-learning models"
    atts["history"] = f"LightningCast v{version}"
    atts[
        "processing_level"
    ] = "National Oceanic and Atmospheric Administration (NOAA) Level 2"
    atts["source"] = meta["reader"]  # abi_l1b or ahi_l1b
    atts["references"] = "https://doi.org/10.1175/WAF-D-22-0019.1"
    atts[
        "keywords"
    ] = "LIGHTNING, INFRARED RADIANCE, VISIBLE RADIANCE, MACHINE-LEARNING"
    atts["keywords_vocabulary"] = f"NASA Global Change Master Directory (GCMD) Science Keywords"
    # "Granule-Level" Metadata
    atts["id"] = ""  # will be filled in with outfile name
    atts["production_site"] = idict["production_site"]
    atts["production_environment"] = idict["production_environment"]
    atts["metadata_link"] = "https://cimss.ssec.wisc.edu/severe_conv/pltg.html"
    atts["time_coverage_start"] = meta["start_time"].strftime(idict["time_patt"])
    atts["time_coverage_end"] = meta["end_time"].strftime(idict["time_patt"])
    now = datetime.utcnow()
    atts["date_created"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
   
    atts["id"] = str(uuid.uuid4())
    if meta["sensor"].upper() == 'ABI':
       sensor_band_identifier = f"02,05,13,15"
    else:
       sensor_band_identifier = f"03,05,13,15"

    atts["sensor_band_identifier"] = sensor_band_identifier 
    atts["sensor_band_central_radiation_wavelength"] = f"0.64um,1.6um,10.3um,12.3um"

    if idict["production_site"] == "NCCF":
        platform_abbr = (
            meta["platform_name"][0].lower() + meta["platform_name"].split("-")[1]
        )
        outfile = f"LtgCast-{sector_outname.replace('_','-')}_v{version}_{platform_abbr}"
        # Note: %f is microseconds. We take the first digit to get tenths of a second, which NESDIS requires.
        outfile += f"_s{meta['start_time'].strftime('%Y%m%d%H%M%S%f')[0:15]}"
        outfile += f"_e{meta['end_time'].strftime('%Y%m%d%H%M%S%f')[0:15]}"
        outfile += f"_c{now.strftime('%Y%m%d%H%M%S%f')[0:15]}.nc"
    else:
        outfile = f"LightningCast_{atts['platform']}_{sector_outname}_%Y%m%d-%H%M%S.nc"


    if atts["platform"].startswith("GOES"):
        atts["orbital_slot"] = meta["orbital_slot"]             # GOES specific
        orbslot = meta["orbital_slot"].lower().replace('-','_') # GOES specific 
        slot_init = meta["orbital_slot"][5]                     # GOES specific
        if "USSAMOA" in sector_outname:
            atts["sector_id"] = sector_id = "USSAMOA"
        elif "AKCAN" in sector_outname:
            atts["sector_id"] = sector_id = "AKCAN"
        else:
            if sector_outname.startswith("RadC"):
                sector_id = "CONUS"
            elif sector_outname.startswith("RadM1"):
                sector_id = "MESO1"
            elif sector_outname.startswith("RadM2"):
                sector_id = "MESO2"
            elif sector_outname.startswith("RadF"):
                sector_id = "FULL"
            atts["sector_id"] = f"{slot_init}{sector_id}"
    elif atts["platform"].startswith("Himawari"):
        sector_id = "GUAM"
        atts["sector_id"] = sector_id
        orbslot = atts["orbital_slot"] = "ahi_japan"

    atts["model"] = meta["model"]

    if len(preds.shape) == 2:
        preds = np.expand_dims(preds, axis=-1)
    ny, nx, nout_feature_maps = preds.shape

    lons, lats = idict["irlons"], idict["irlats"] #scn_ch.attrs["area"].get_lonlats()
    lons = lons[0:ny, 0:nx]  # force the dims to be the same.
    lats = lats[0:ny, 0:nx]
    lons = lons[::stride, ::stride]
    lats = lats[::stride, ::stride]
   
    orig_preds = np.copy(preds)
    preds = preds[::stride, ::stride]
    # fix bad lats and lons
    lons = np.ma.masked_where(((lons < -180) | (lons > 180)), lons)
    lats = np.ma.masked_where(((lats < -90) | (lats > 90)), lats)

    lon_0 = scn_ch.attrs["orbital_parameters"]["projection_longitude"]

    # Use pyresample to remap from geostationary projection to EQC, which is recognized by AWIPS plug-in.
    geosGrid = pr.geometry.GridDefinition(lons=lons, lats=lats)
    projection_dict = {
        "proj": "eqc",
        "units": "m",
    }  # {'proj':'lcc', 'lon_0':lon_0, 'lat_1':33, 'lat_2':45, 'units':'m'}
    target_proj = Proj(projection_dict)

    if awips:
        SWlat = idict['awips_grid']['SWlat']
        NElat = idict['awips_grid']['NElat']
        SWlon = idict['awips_grid']['SWlon']
        NElon = idict['awips_grid']['NElon']
        dy = idict['awips_grid']['dy']
        dx = idict['awips_grid']['dx']

        # Geographic Metadata
        atts["cdm_data_type"] = "Grid"
        atts["geospatial_lat_min"] = float(SWlat)
        atts["geospatial_lat_max"] = float(NElat)
        atts["geospatial_lon_min"] = float(SWlon)
        atts["geospatial_lon_max"] = float(NElon)
        atts["geospatial_lat_units"] = "degrees_north"
        atts["geospatial_lon_units"] = "degrees_east"
        atts["geospatial_lat_resolution"] = dy
        atts["geospatial_lon_resolution"] = dx

    # Product development team and publisher metadata
    atts[
        "creator_name"
    ] = "DOC/NOAA/NESDIS/STAR > ProbSevere Team, Center for Satellite Applications and Research, NESDIS, NOAA, U.S. Department of Commerce."
    atts["creator_email"] = "probsevere@gmail.com"
    atts["creator_url"] = "https://cimss.ssec.wisc.edu/severe_conv/pltg.html"
    atts[
        "publisher_name"
    ] = "DOC/NOAA/NESDIS/OSPO > Office of Satellite and Product Operations, NESDIS, NOAA, U.S. Department of Commerce."
    atts["publisher_email"] = "espcoperations@noaa.gov"
    atts["publisher_url"] = "http://www.ospo.noaa.gov"

    
    dims = collections.OrderedDict()
    dataset = {}

    if awips:
        # Check that our dimensions are whole numbers.
        # We do this to ensure that these hard-coded grids match up with the static parallax-correction grids.
        width = (NElon - SWlon) / dx
        height = (NElat - SWlat) / dy
        assert(width % 1 == 0)
        assert(height % 1 == 0)
        width = int(width)
        height = int(height)
    
        # Remap data to this equal-lat equal-lon projection
    
        if width < 0: # indicates that NElon is negative. I.e., the domains crosses the datetline
            width = int((180 - SWlon + 180 + NElon) / dx)

        dims["y"], dims["x"] = height, width        
    
        x, y = target_proj(np.array([SWlon, NElon]), np.array([SWlat, NElat]))
        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()
        area_extent = (x_min, y_min, x_max, y_max)
   
        targetArea = pr.geometry.AreaDefinition(
            "targetProj",
            "LC_targetProj",
            "targetProj",
            projection_dict,
            width=width,
            height=height,
            area_extent=area_extent,
        )
        new_lons, new_lats = targetArea.get_lonlats()
  
        new_preds = pr.kd_tree.resample_nearest(
            geosGrid, preds, targetArea, radius_of_influence=16000, fill_value=-1
        )
        assert new_preds.shape[0:2] == new_lons.shape == new_lats.shape
    else:
        new_preds = np.copy(preds)
        dims["y"], dims["x"], _ = new_preds.shape
        # additional projection data
        if scn_ch.attrs["area"].proj_id.lower().startswith('abi'):
            long_name = "GOES-R ABI fixed grid projection"
        elif scn_ch.attrs["area"].proj_id.lower().startswith('geosh'):
            long_name = "Himawari geostationary projection" 
        else:
            logging.error(f"Projection: {scn_ch.attrs['area'].proj_id.lower()} is not supported.")
            return -1, False

        # Create x and y, like they are in ABI/AHI files.
        # force the dims to be the same (e.g., [0:nx])
        projx = scn_ch.x.compute().data[0:nx] / scn_ch.attrs["area"].proj_dict['h']
        projy = scn_ch.y.compute().data[0:ny] / scn_ch.attrs["area"].proj_dict['h']
        projx = projx[::stride]
        projy = projy[::stride]
        x_scale_factor = 5.6e-05
        x_add_offset = -0.101332
        xpacked = ((projx - x_add_offset) / x_scale_factor).astype(np.int16)
        y_scale_factor = -5.6e-05
        y_add_offset = 0.128212
        ypacked = ((projy - y_add_offset) / y_scale_factor).astype(np.int16)
        
        dataset["y"] = {'data': ypacked,
                        'dims': ("y"),
                        'atts': {'scale_factor':y_scale_factor,
                                 'add_offset':y_add_offset,
                                 'units':'rad',
                                 'axis':'Y',
                                 'standard_name':'projection_y_coordinate',
                        }
        }
        dataset["x"] = {'data': xpacked,
                        'dims': ("x"),
                        'atts': {'scale_factor':x_scale_factor,
                                 'add_offset':x_add_offset,
                                 'units':'rad',
                                 'axis':'X',
                                 'standard_name':'projection_x_coordinate',
                        }
        }
 
        # Add imager_projection info

        dataset["imager_projection"] = {'data':np.ubyte(255),
                                        'dims':(),
                                        'atts': {'long_name':long_name,
                                                 'projection':'geostationary',
                                                 'perspective_point_height':scn_ch.attrs["area"].proj_dict['h'],
                                                 'longitude_of_projection_origin':lon_0,
                                                 'latitude_of_projection_origin':0.
                                        }
        }

 
    # For parallax-correction fields    
    if savecode > 1:
        if awips:
            plons, plats = idict['plax_lons'], idict['plax_lats']
            plons = plons[0:ny, 0:nx]  # force the dims to be the same.
            plats = plats[0:ny, 0:nx]
            plons = plons[::stride, ::stride]
            plats = plats[::stride, ::stride]
    
            # fix bad lats and lons
            plons = np.ma.masked_where(((plons < -180) | (plons > 180)), plons)
            plats = np.ma.masked_where(((plats < -90) | (plats > 90)), plats)
    
            plax_geosGrid = pr.geometry.GridDefinition(lons=plons, lats=plats)
    
            plax_preds = pr.kd_tree.resample_nearest(
                plax_geosGrid, preds, targetArea, radius_of_influence=16000, fill_value=-1
            )
            assert plax_preds.shape[0:2] == new_lons.shape == new_lats.shape

            # t1=time.time()
            # new_preds = pr.kd_tree.resample_gauss(geosGrid,tmp_preds,targetArea,radius_of_influence=16000,sigmas=20000,fill_value=-1)
            # print(time.time()-t1)
            logging.info("Remapped probability data.")
  
        else:
            yind, xind = idict["PLAX_inds_nonstatic"]
            plax_preds = np.zeros(orig_preds.shape, dtype=np.float32)
            for nn in range(nout_feature_maps):
                plax_preds[yind, xind, nn] = orig_preds[..., nn].ravel()
            plax_preds = plax_preds[::stride, ::stride]


    # Create the DQF mask
    if "05km" in idict["masks"]:
        half_km_mask = idict["masks"]["05km"][::4, ::4]
    else:
        half_km_mask = np.zeros(idict["masks"]["2km"].shape, dtype=np.uint8)
    if "1km" in idict["masks"]:
        one_km_mask = idict["masks"]["1km"][::2, ::2]
    else:
        one_km_mask = np.zeros(idict["masks"]["2km"].shape, dtype=np.uint8)
    two_km_mask = idict["masks"]["2km"]  # should always be present
    mask = np.max([half_km_mask, one_km_mask, two_km_mask], axis=0)
    # If all channels/resolutions have large % of bad data, set to 3 (i.e., "unusable_pixel_qf")
    mask[((half_km_mask == 1) & (one_km_mask == 1) & (two_km_mask == 1))] = 3
    # We "stride" the image to reduce resolution and noise
    mask = mask[::stride, ::stride]

    if awips:
        # Remap mask to AWIPS targetArea
        new_mask = pr.kd_tree.resample_nearest(
            geosGrid, mask, targetArea, radius_of_influence=16000, fill_value=-1
        ).astype(np.int8)
    else:
        new_mask = mask.astype(np.int8)

    total_number_attempted_retrievals = np.count_nonzero(new_mask >= 0)
    total_number_unattempted_retrievals = np.count_nonzero(new_mask == -1)
    total_number_good_retrievals = np.count_nonzero(new_mask == 0)
    total_number_sub_optimal_retrievals = np.count_nonzero(
        (new_mask == 1) | (new_mask == 2)
    )
    total_number_unusable_retrievals = np.count_nonzero((new_mask == 3))
    dataset["DQF"] = {
        "data": new_mask,
        "dims": ("y", "x"),
        "atts": {
            "long_name": "LightningCast retrieval data quality flags",
            "flag_values": np.array([-1, 0, 1, 2, 3], dtype=np.int8),
            "flag_meanings": "retrieval_unattempted_qf good_pixel_qf "
            + "conditionally_unusable_pixel_qf conditionally_usable_pixel_qf "
            + "unusable_pixel_qf",
            "total_number_attempted_retrievals": total_number_attempted_retrievals,
            "total_number_unattempted_retrievals": total_number_unattempted_retrievals,
            "total_number_good_retrievals": total_number_good_retrievals,
            "total_number_sub_optimal_retrievals": total_number_sub_optimal_retrievals,
            "total_number_unusable_retrievals": total_number_unusable_retrievals,
        },
    }

    #----------------------------------------------------------
    # addtional required quality information
    #----------------------------------------------------------
    try:
        percentage_optimal_retrievals = 100.0*total_number_good_retrievals/float(total_number_attempted_retrievals)
        percentage_sub_optimal_retrievals = 100.0*total_number_sub_optimal_retrievals/float(total_number_attempted_retrievals)
        percentage_bad_retrievals = 100.0*total_number_unusable_retrievals/float(total_number_attempted_retrievals)
    except ZeroDivisionError:
        # This could mean the sector is outside of the region we remap to.
        percentage_optimal_retrievals = percentage_sub_optimal_retrievals = percentage_bad_retrievals = -1

    quality_information = np.ubyte(255)
    dataset["quality_information"] = {
       "data": quality_information,
       "dims": (),
       "atts" : {
           "long_name": "total number of retrievals, percentage of optimal retrievals, " + \
                        "percentage of sub_optimal retrievals, percentage of bad retrievals",
           "total_number_retrievals": total_number_attempted_retrievals,
           "percentage_optimal_retrievals": percentage_optimal_retrievals,
           "percentage_sub_optimal_retrievals": percentage_sub_optimal_retrievals,
           "percentage_bad_retrievals": percentage_bad_retrievals,
       },
    }

    for nn in range(nout_feature_maps):

        if nn == 0:
          varname = f"PLTG_60min"
        elif nn == 1:
          flashrate = idict['binarize2']
          timedur = idict['target2'].split('_')[-2]
          varname = f"PLTG_{flashrate}fl-{timedur}"
        else:
          logging.critical(f"More than {nn+1} outputs not supported")
          sys.exit(1)


        if scale_data or sector_id == "FULL":
            # save as byte to save space
            scaled_preds = np.round(new_preds[...,nn] * 100).astype(np.byte)
            missing_val = np.byte(-100)  # fill_value from resample_nearest is -1
            if savecode > 1: # plax-corr data
                scaled_plax_preds = np.round(plax_preds[...,nn] * 100).astype(np.byte)
        else:
            scaled_preds = new_preds[...,nn] * 100
            missing_val = -100
            if savecode > 1: # plax-corr data
                scaled_plax_preds = plax_preds[...,nn] * 100

       # if(scale_data):
       #  #scale the lats and lons
       #  n = 16 #number of bits of the packed data type (16 bits in a np.uint16)
       #  lat_scale_factor = (new_lats.max() - new_lats.min()) / (2**n - 1)
       #  lat_add_offset = new_lats.min()
       #  lon_scale_factor = (new_lons.max() - new_lons.min()) / (2**n - 1)
       #  lon_add_offset = new_lons.min()
       #  scaled_lats = np.round((new_lats - lat_add_offset)/lat_scale_factor).astype(np.uint16)
       #  scaled_lons = np.round((new_lons - lon_add_offset)/lon_scale_factor).astype(np.uint16)
       # else:
       #  scaled_lats = (new_lats).astype(np.float32)
       #  scaled_lons = (new_lons).astype(np.float32)

        # savecode = 1: non-parallax-corrected grid only
        # savecode = 2: parallax-corrected grid only
        # savecode = 3: both plax and non-plax grids
        if savecode == 1 or savecode == 3:
            dataset[varname] = {
                "data": scaled_preds,
                "dims": ("y", "x"),
                "atts": {
                    "units": "%",
                    "long_name": "Probability of lightning in the next 60 min",
                    "missing_value": missing_val,
                },
            }
    
        if savecode == 2 or savecode == 3: 
            dataset[f"{varname}_plax"] = {
                "data": scaled_plax_preds,
                "dims": ("y", "x"),
                "atts": {
                    "units": "%",
                    "long_name": "Parallax-corrected probability of lightning in the next 60 min",
                    "missing_value": missing_val,
                    "parallax_cloud_height_in_m": idict['plax_hgt'],
                },
            }
    
        # if(scale_data):
        #  #scale the lats and lons
        #  n = 16 #number of bits of the packed data type (16 bits in a np.uint16)
        #  lat_scale_factor = (new_lats.max() - new_lats.min()) / (2**n - 1)
        #  lat_add_offset = new_lats.min()
        #  lon_scale_factor = (new_lons.max() - new_lons.min()) / (2**n - 1)
        #  lon_add_offset = new_lons.min()
        #  scaled_lats = np.round((new_lats - lat_add_offset)/lat_scale_factor).astype(np.uint16)
        #  scaled_lons = np.round((new_lons - lon_add_offset)/lon_scale_factor).astype(np.uint16)
        # else:
        #  scaled_lats = (new_lats).astype(np.float32)
        #  scaled_lons = (new_lons).astype(np.float32)
    
        # dataset['lat'] = {'data':scaled_lats,'dims':('y','x'),
        #                  'atts':{'units':'degrees north','long_name':'latitude'}}
        # if(scale_data):
        #  dataset['lat']['atts']['scale_factor'] = lat_scale_factor
        #  dataset['lat']['atts']['add_offset'] = lat_add_offset
        #
        # dataset['lon'] = {'data':scaled_lons,'dims':('y','x'),
        #                  'atts':{'units':'degrees east','long_name':'longitude'}}
        # if(scale_data):
        #  dataset['lon']['atts']['scale_factor'] = lon_scale_factor
        #  dataset['lon']['atts']['add_offset'] = lon_add_offset

    outnetcdf = dt.strftime(f"{outdir}/%Y%m%d/{orbslot}/{sector_outname}/netcdf/{outfile}")
    status = utils.write_netcdf(outnetcdf, dataset, dims, atts=atts, wait=True, gzip=False)

    return outnetcdf, status


#################################################################################################################################
def make_geotiff(
    scn_ch, preds, dt, projection, stride, slot, sector, outdir, re_upload=None
):

    logging.info("Making geotiff...")
    projx = scn_ch.x.compute().data
    projy = scn_ch.y.compute().data
    xmin = projx[0]
    ymax = projy[0]
    xsize = projx[stride] - projx[0]
    ysize = projy[0] - projy[stride]
    transform = rasterio.transform.from_origin(xmin, ymax, xsize, ysize)
    preds = preds[0::stride, 0::stride] * 100

    parts = slot.split("_")  # slot = 'goes_east'
    new_slot = parts[0].upper() + parts[1].title()  #'goes', 'east' --> GOESEast
    outtif = dt.strftime(
        f"{outdir}/PLTG{new_slot}{sector}Gridded_%Y%m%d_%H%M00.force.tif"
    )
    new_dataset = rasterio.open(
        outtif,
        "w",
        driver="GTiff",
        height=preds.shape[0],
        width=preds.shape[1],
        count=1,
        dtype=str(preds.dtype),
        crs=projection.definition_string(),  #'+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
        transform=transform,
    )
    try:
        new_dataset.write(preds, 1)
        new_dataset.close()
    except ValueError: # too many channels (last dim)
        # Do the 0th channel only
        new_dataset.write(preds[...,0], 1)
        new_dataset.close()
    if (
        re_upload
    ):  # re_upload is either None or a string to full path of the re_upload script
        s = subprocess.call([re_upload, outtif])
        os.remove(outtif)  # clean up
    logging.info(f"Completed {outtif}")

    # from rasterio.plot import show
    # img = rasterio.open(outtif)
    # show(img)


#################################################################################################################################
def get_FPT(abi_filenames):
    # get C13 file only
    index = [idx for idx, s in enumerate(abi_filenames) if "C13_" in s][0]
    c13file = abi_filenames[index]

    nc = Dataset(c13file, "r")
    max_focal_plane_temp = nc.variables["maximum_focal_plane_temperature"][:]
    nc.close()
    return max_focal_plane_temp


#################################################################################################################################
def get_eni_data(dt, irlats, irlons, eni_path="/ships19/grain/lightning/"):
    logging.info("Process started.")
    # /ships19/grain/lightning/2021/2021_08_11_223/19/entln_2021223_193500.txt
    # The timestamp of the eni files are the starttime (rounded to nearest minute) of flashes in the file,
    # and contain flashes for about the next 5 min
    # latmin = irlats.min(); latmax = irlats.max()
    # lonmin = irlons.min(); lonmax = irlons.max()
    # dt is the timestamp of the scan
    startdt = dt - timedelta(minutes=10)
    enddt = dt + timedelta(minutes=5)

    # Get all files for date of startdt and date of enddt
    files = glob.glob(
        startdt.strftime(f"{eni_path}/%Y/%Y_%m_%d_%j/%H/*txt")
    ) + glob.glob(enddt.strftime(f"{eni_path}/%Y/%Y_%m_%d_%j/%H/*txt"))
    files = np.sort(files)

    # select the files to open
    good_eni_files = []
    for f in files:
        filedt = datetime.strptime(os.path.basename(f), "entln_%Y%j_%H%M%S.txt")
        if startdt <= filedt <= enddt:
            good_eni_files.append(f)
    if len(good_eni_files) == 0:
        logging.error("Can't find any good ENI files")
        raise RuntimeError

    # select the flashes (within 5 min before scan dt)
    good_flashes = []
    for enif in good_eni_files:
        # iterate over csv files and keep appending flash data
        with open(enif, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                tmpdt = datetime.strptime(
                    row[0].split(".")[0], 'time:"%Y-%m-%dT%H:%M:%S'
                )  # NB: hard-coded time index
                lat = float(row[3].split(":")[1])  # NB: hard-coded lat index
                lon = float(row[2].split(":")[1])  # NB: hard-coded lon index
                if startdt <= tmpdt <= dt:
                    good_flashes.append(row)
    if len(good_flashes) == 0:
        logging.error("Can't find any good flashes in files")
        raise RuntimeError
    lats = [
        float(fl[3].split(":")[1]) for fl in good_flashes
    ]  # NB: hard-coded lat index
    lons = [
        float(fl[2].split(":")[1]) for fl in good_flashes
    ]  # NB: hard-coded lon index

    logging.info("Got ENI data.")
    return lats, lons  # these are ENI flashes within 5 min before scan dt


#################################################################################################################################
def get_channames(imgr):
    channames = {}
    if imgr == "ABI":
        for ii in np.arange(1, 17, 1, dtype=int):
            jj = str(ii).zfill(2)
            channames[f"CH{jj}"] = f"C{jj}"
    elif imgr == "AHI":
        for ii in np.arange(1, 17, 1, dtype=int):
            jj = str(ii).zfill(2)
            channames[f"CH{jj}"] = f"B{jj}"
        channames["CH02"] = "B03"  # band 3 is red on AHI
    elif imgr == "SEVIRI":
        channames["CH02"] = "VIS006"
        channames["CH03"] = "VIS008"
        channames["CH05"] = "IR_016"
        channames["CH07"] = "IR_039"
        channames["CH08"] = "WV_062"
        channames["CH10"] = "WV_073"
        channames["CH11"] = "IR_087"
        channames["CH12"] = "IR_097"
        channames["CH13"] = "IR_108"
        channames["CH15"] = "IR_120"
        channames["CH16"] = "IR_134"
    else:  # MTG/FCI
        channames["CH02"] = "vis_06"
        channames["CH05"] = "nir_16"
        channames["CH07"] = "nir_38"
        channames["CH08"] = "ir_063"
        channames["CH10"] = "ir_073"
        channames["CH13"] = "ir_105"
        channames["CH15"] = "ir_123"
        channames["CH16"] = "ir_133"

    return channames


#################################################################################################################################
def fix_coords_for_seviri(irlons, irlats, projx, projy):
    # converting from 3km to 2km

    base = 4  # (for 2-km data)

    irlons = irlons.repeat(3, axis=0).repeat(3, axis=1)  # from 3-km to 1-km
    irlons = irlons[::2, ::2]  # from 1-km to 2-km
    y_dim, x_dim = irlons.shape
    y_dim = int(base * np.floor(y_dim / base))
    x_dim = int(base * np.floor(x_dim / base))
    irlons = irlons[0:y_dim, 0:x_dim]

    irlats = irlats.repeat(3, axis=0).repeat(3, axis=1)  # from 3-km to 1-km
    irlats = irlats[::2, ::2]  # from 1-km to 2-km
    irlats = irlats[0:y_dim, 0:x_dim]

    projx = projx.repeat(3, axis=0)  # from 3-km to 1-km
    projx = projx[::2]  # from 1-km to 2-km
    projx = projx[0:x_dim]

    projy = projy.repeat(3, axis=0)  # from 3-km to 1-km
    projy = projy[::2]  # from 1-km to 2-km
    projy = projy[0:y_dim]

    return irlons, irlats, projx, projy


#################################################################################################################################
def spatial_align(sat_data, mc, imgr, scn, ch, idict):
    base = 4

    # Dims for 2-km input. Make divisible by 4
    y_dim, x_dim = idict["irlats"].shape
    y_dim = int(base * np.floor(y_dim / base))
    x_dim = int(base * np.floor(x_dim / base))

    if imgr == "AHI":
        # We need to make sure that the 2-km input is divisible by 4.
        # We need to ensure that the 0.5-km input is *4 of the 2-km input
        # and ensure that the 1-km input is *2 of the 2-km input

        if ch == "CH02":
            sat_data = sat_data[0 : y_dim * 4, 0 : x_dim * 4]
        elif ch == "CH05":
            sat_data = zoom(sat_data, 2, order=0)
            sat_data = sat_data[0 : y_dim * 2, 0 : x_dim * 2]
        else:  # 2km data
            sat_data = sat_data[0:y_dim, 0:x_dim]

    return sat_data


#################################################################################################################################
def spatial_align_old(all_preds, config, imgr, scn):
    base = 4

    if imgr == "AHI":
        # We need to make sure that the 2-km input is divisible by 4.
        # We need to ensure that the 0.5-km input is *4 of the 2-km input
        # and ensure that the 1-km input is *2 of the 2-km input

        # Find the dims for 2-km input such that they are divisible by 4
        for ii, inp in enumerate(config["inputs"]):
            if "CH13" in inp:  # assumes we'll always use CH13
                dims = all_preds[ii].shape
                y_dim = dims[-3]
                x_dim = dims[-2]  # -1 should be channels dim
                y_dim = int(base * np.floor(y_dim / base))
                x_dim = int(base * np.floor(x_dim / base))
                break

        # Zoom to new resolutions and truncate with proper divisible dims, based off the 2-km dims.
        for ii, inp in enumerate(config["inputs"]):
            dims = all_preds[ii].shape
            if "CH02" in inp:  # 0.5-km inputs (natively)
                all_preds[ii] = all_preds[ii][..., 0 : y_dim * 4, 0 : x_dim * 4, :]
            elif (
                "CH05" in inp
            ):  # 1.6-µm natively 2-km on AHI. But our model was trained with GOES and expects 1-km res,
                if len(dims) == 4:
                    zoomfac = (1, 2, 2, 1)  # convert from 2km to 1km
                elif len(dims) == 5:  # with time dim
                    zoomfac = (1, 1, 2, 2, 1)
                all_preds[ii] = zoom(all_preds[ii], zoomfac, order=0)
                all_preds[ii] = all_preds[ii][..., 0 : y_dim * 2, 0 : x_dim * 2, :]
            else:
                all_preds[ii] = all_preds[ii][..., 0:y_dim, 0:x_dim, :]

    elif imgr == "FCI":
        # We need to make sure that the 2-km input is divisible by 4.
        # We need to ensure that the 0.5-km input is *4 of the 2-km input
        # and ensure that the 1-km input is *2 of the 2-km input

        for ii, inp in enumerate(config["inputs"]):
            if (
                "CH13" in inp
            ):  # assumes we'll always use CH13. Also assuming CH13 is 2-km
                dims = all_preds[ii].shape
                # check resolution of CH13
                if scn["ir_105"].resolution == 2000:  # 2km
                    y_dim = dims[-3]
                    x_dim = dims[-2]  # -1 should be channels dim
                elif scn["ir_105"].resolution == 1000:  # 1km
                    y_dim = dims[-3] // 2
                    x_dim = dims[-2] // 2  # -1 should be channels dim
                else:
                    logging.error(
                        'Resolution of CH13 not 2km or 1km: {scn["ir_105"].resolution}'
                    )
                    sys.exit(1)
                y_dim = int(base * np.floor(y_dim / base))
                x_dim = int(base * np.floor(x_dim / base))
                break

        # Zoom to new resolutions and truncate with proper divisible dims, based off the 2-km dims.
        for ii, inp in enumerate(config["inputs"]):
            dims = all_preds[ii].shape
            if "CH02" in inp:  # Can be 0.5-km or 1-km
                if scn["vis_06"].resolution == 1000:  # 1km...zoom to 0.5km first
                    if len(dims) == 4:
                        zoomfac = (1, 2, 2, 1)
                    elif len(dims) == 5:  # with time dim
                        zoomfac = (1, 1, 2, 2, 1)
                    all_preds[ii] = zoom(all_preds[ii], zoomfac, order=0)
                # Make sure it's divisible by 4
                all_preds[ii] = all_preds[ii][..., 0 : y_dim * 4, 0 : x_dim * 4, :]

            elif (
                "CH05" in inp
            ):  # 1.6-µm can be 0.5km or 1km FCI. But our model was trained with GOES and expects 1-km res,
                if scn["nir_16"].resolution == 500:  # 0.5km
                    all_preds[ii] = all_preds[ii][
                        ..., 0::2, 0::2, :
                    ]  # reduce from 0.5-km to 1-km
                all_preds[ii] = all_preds[ii][
                    ..., 0 : y_dim * 2, 0 : x_dim * 2, :
                ]  # reduce from 0.5-km to 1-km
            else:  # 2-km input
                all_preds[ii] = all_preds[ii][..., 0:y_dim, 0:x_dim, :]

    elif imgr == "SEVIRI":
        # Each channel is 3-km. We need to interpolate to the GOES resolution.
        # We need to make sure that the 2-km input is divisible by 4.
        # We need to ensure that the 0.5-km input is *4 of the 2-km input
        # and ensure that the 1-km input is *2 of the 2-km input

        # Find the index for the 2-km input,
        # 1. Calc y_dim and x_dim to be divisible by 4 (due to model architecture)
        # 2. Truncate data at y_dim and x_dim
        # 3. Zoom to new resolution
        # 4. Calc new y_dim and x_dim to be divisible by 4
        # 5. In 2nd for loop, 2-km input will be truncated
        for ii, inp in enumerate(config["inputs"]):
            if "CH13" in inp:  # assumes we'll always use CH13
                # Step 1
                dims = all_preds[ii].shape
                y_dim = dims[-3]
                x_dim = dims[-2]  # -1 should be channels dim
                y_dim = int(base * np.floor(y_dim / base))
                x_dim = int(base * np.floor(x_dim / base))
                # Step 2
                all_preds[ii] = all_preds[ii][..., 0:y_dim, 0:x_dim, :]
                # Step 3
                if len(dims) == 4:
                    zoomfac = (1, 1.5, 1.5, 1)  # convert from 3km to 2km
                elif len(dims) == 5:  # with time dim
                    zoomfac = (1, 1, 1.5, 1.5, 1)  # convert from 3km to 2km
                all_preds[ii] = zoom(all_preds[ii], zoomfac, order=0)
                # Step 4
                dims = all_preds[ii].shape
                y_dim = dims[-3]
                x_dim = dims[-2]
                y_dim = int(base * np.floor(y_dim / base))
                x_dim = int(base * np.floor(x_dim / base))
                break

        # Zoom to new resolutions and truncate with proper divisible dims, based off the 2-km dims.
        for ii, inp in enumerate(config["inputs"]):
            dims = all_preds[ii].shape
            if "CH02" in inp:  # 0.5-km inputs
                if len(dims) == 4:
                    zoomfac = (1, 6, 6, 1)  # convert from 3km to 0.5km
                elif len(dims) == 5:  # with time dim
                    zoomfac = (1, 1, 6, 6, 1)
                all_preds[ii] = zoom(all_preds[ii], zoomfac, order=0)
                all_preds[ii] = all_preds[ii][..., 0 : y_dim * 4, 0 : x_dim * 4, :]
            elif "CH05" in inp or "CH01" in inp or "CH03" in inp:  # 1-km inputs
                if len(dims) == 4:
                    zoomfac = (1, 3, 3, 1)  # convert from 3km to 1km
                elif len(dims) == 5:  # with time dim
                    zoomfac = (1, 1, 3, 3, 1)
                all_preds[ii] = zoom(all_preds[ii], zoomfac, order=0)
                all_preds[ii] = all_preds[ii][..., 0 : y_dim * 2, 0 : x_dim * 2, :]
            else:  # 2-km inputs
                all_preds[ii] = all_preds[ii][..., 0:y_dim, 0:x_dim, :]


#################################################################################################################################
def remap_info(idict, meta):
    """Get the static remap info (corner lat/lons, dx, dy, and parallax-correction info, if necessary.
    These are static grids for netCDF creation. For each satellite, we have one output projection whereby each sector
    for that satellite gets remapped to that projection."""

    # Here we're making a static grid for GOES-East CONUS/Mesoscale sectors
    # Or a static grid for GOES-West PACUS/Mesoscale sectors.
    # Or a static grid for GUAM.
    # Or a static grid for USSAMOA.
    if meta["platform_name"].startswith("GOES"):
        if "USSAMOA" in idict['sector_outname']:
            # American Samoa
            SWlat = -20
            NElat = -4
            dy = 0.08
            SWlon = -180
            NElon = -160
            dx = 0.08
        elif meta["orbital_slot"].lower() == "goes-east":
            SWlat = -6
            NElat = 55
            dy = 0.1
            SWlon = -135
            NElon = -30
            dx = 0.1
        elif meta["orbital_slot"].lower() == "goes-west":
            SWlat = 0
            NElat = 65
            dy = 0.1
            SWlon = -175
            NElon = -90
            dx = 0.1
        else:
            logging.critical(f"{meta['orbital_slot']} is not supported.")
            sys.exit(1)
    elif meta["platform_name"].startswith("Himawari"):
        # GUAM
        SWlat = -1
        NElat = 25
        dy = 0.08
        SWlon = 130
        NElon = 180
        dx = 0.08

    idict['awips_grid'] = {}
    idict['awips_grid']['SWlat'] = SWlat
    idict['awips_grid']['NElat'] = NElat
    idict['awips_grid']['SWlon'] = SWlon
    idict['awips_grid']['NElon'] = NElon
    idict['awips_grid']['dy'] = dy
    idict['awips_grid']['dx'] = dx


#################################################################################################################################
def get_remap_cache(input_def, target_def, radius_of_influence=4000):
    logging.info("Making remap cache...")
    (
        input_index,
        output_index,
        index_array,
        distance_array,
    ) = pr.kd_tree.get_neighbour_info(
        input_def, target_def, radius_of_influence, neighbours=1
    )
    remap_cache = {
        "input_index": input_index,
        "output_index": output_index,
        "index_array": index_array,
        "target_def": target_def,
    }
    return remap_cache


#################################################################################################################################
def upload_re_json(dt, json_outdir, idict, re_upload):

    """
    Will make RE-compatible filename and upload to RE VM.
    Assumes that each directory in json_outdir is a product to be uploaded to RE VM
    args:
      dt: datetime.datetime - the satellite scan time
      json_outdir: str - the directory containing the jsons
      idict: dict - metadata
      re_upload: str - the RE upload executable
    """

    goes_slot_str = idict['slot'].replace("_", "")
    # need this b/c RealEarth doesn't do underscores
    sector_str = idict["sector_outname"].replace("_", "")

    directories = [entry.name for entry in os.scandir(json_outdir) if entry.is_dir()] 

    for product in directories:
        if product == '1fl-60min': 
            base_re_outjson = dt.strftime(
                f"PLTG{goes_slot_str}{sector_str}_%Y%m%d_%H%M%S.force.json"
            )
        else:
            base_re_outjson = dt.strftime(
                f"PLTG-{product}-{goes_slot_str}{sector_str}_%Y%m%d_%H%M%S.force.json"
            )
        new_re_json = f"{json_outdir}/{product}/{base_re_outjson}"

        try:
            orig_json = glob.glob(dt.strftime(f'{json_outdir}/{product}/*_s%Y%m%d%H%M%S*.json'))[-1]
        except IndexError as err:
            logging.warning(dt.strftime(f'{json_outdir}/{product}/*_s%Y%m%d%H%M%S*.json') + ' does not exist')
        else:

            shutil.copy(orig_json, new_re_json)
            
            try:
                s = subprocess.run(
                    [re_upload, new_re_json], timeout=10
                )  # 10-second timeout
            except subprocess.TimeoutExpired as err:
                logging.error(str(err))
            try:
                os.remove(new_re_json)
            except:
                pass
#################################################################################################################################
def get_model(model, mc):
    # setting up tensorflow or pytorch
    if model.endswith("h5"):
        is_tensorflow = True
        is_pytorch = False
        conv_model = load_model(model, compile=False)
    elif model.endswith("pt"):
        import torch
        import torch.nn as nn

        sys.path.insert(
            0, os.path.dirname(model)
        )  # Use torch_models.py in model directory, not the one in src,
        # as model def might have changed. This command just puts the
        # model directory at front of the path
        from lightningcast import torch_models

        is_tensorflow = False
        is_pytorch = True
        torch_model = getattr(torch_models, mc["cnn"])
        conv_model = torch_model(mc)
        if torch.cuda.is_available():
            ngpus = 1  # hardcoded for now
            device = torch.device("cpu")  # torch.device("cuda")
        else:
            ngpus = 0
            device = torch.device("cpu")

        logging.info("opening " + model)
        mdict = torch.load(model, map_location=device)
        conv_model.load_state_dict(mdict["model_state_dict"])
        # ngpu = torch.cuda.device_count()
        conv_model.to(device)
        conv_model.eval()  # Needed to set dropout and batch normalization layers to evaluation mode before running inference.
        # Failing to do this will yield inconsistent inference results.
        logging.info("Using device = " + str(device))

    return conv_model, is_tensorflow, is_pytorch


#################################################################################################################################
def get_s3_data(dt, chan):
    import s3fs

    s3 = s3fs.S3FileSystem(anon=True)  # connect to s3 system

    bucketpatt = f"s3://noaa-mrms-pds/CONUS/{chan}_00.50/%Y%m%d/MRMS_{chan}_00.50_%Y%m%d-%H%M*.grib2.gz"

    try:
        grib2_file = s3.glob(dt.strftime(bucketpatt))[0]
        logging.info(f"Got {chan} data from S3 for {dt}")
    except IndexError:
        for mmm in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
            tmpdt = dt + timedelta(minutes=mmm)
            try:
                grib2_file = s3.glob(tmpdt.strftime(bucketpatt))[0]
                logging.info(f"Got {chan} data from S3 for {tmpdt}")
            except IndexError:
                if mmm == -5:
                    logging.critical(f"Can't find remote MRMS grib2 data near {dt}.")
                    sys.exit(1)
            else:
                break

    tmpdir = f"{os.environ['PWD']}/tmp_s3/"  # wget will create it
    parts = grib2_file.split("/")
    url = ("/").join([f"https://{parts[0]}.s3.amazonaws.com"] + parts[1:])
    status = subprocess.run(["wget", "-q", "--directory-prefix", tmpdir, url])
    # gunzip!
    status = subprocess.run(["gunzip", f"{tmpdir}/{parts[-1]}"])

    return f"{tmpdir}/{parts[-1][:-3]}", tmpdir

    # https://noaa-mrms-pds.s3.amazonaws.com/CONUS/Reflectivity_-10C_00.50/20211123/MRMS_Reflectivity_-10C_00.50_20211123-012645.grib2.gz
    # ['noaa-mrms-pds/CONUS/Reflectivity_-10C_00.50/20211123/MRMS_Reflectivity_-10C_00.50_20211123-145648.grib2.gz']


#################################################################################################################################
def get_grib2_data(dt, ch, mrms_grib2_patt, target_area_def=None):
    tmpdir = None

    if ch == "REF10":
        chan = "Reflectivity_-10C"
    else:
        chan = ch
    mrms_grib2_patt = mrms_grib2_patt.replace("{ch}", chan)

    try:
        grib2_file = glob.glob(dt.strftime(mrms_grib2_patt))[0]
    except IndexError:
        for mmm in [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
            tmpdt = dt + timedelta(minutes=mmm)
            try:
                grib2_file = glob.glob(tmpdt.strftime(mrms_grib2_patt))[0]
            except IndexError:
                if mmm == -5:
                    logging.warning(
                        f"Can't find local MRMS grib2 data near {dt}. Trying S3 bucket."
                    )
                    grib2_file, tmpdir = get_s3_data(dt, chan)
            else:
                break

    grb = pg.open(grib2_file)
    msg = grb[1]  # should only be 1 'message' in grib2 file
    data = msg.values
    grb.close()

    if tmpdir:
        shutil.rmtree(tmpdir)  # Remove temporary directory for S3 data download

    # Remap to geos
    if target_area_def:
        nlats = 3500
        nlons = 7000  # hard-coded for MRMS
        radlats = np.arange(55, 20, -0.01)
        radlons = np.arange(-130, -60, 0.01)  # hard-coded for MRMS
        radlons, radlats = np.meshgrid(radlons, radlats)
        radgrid = pr.geometry.GridDefinition(lons=radlons, lats=radlats)

        data = pr.kd_tree.resample_nearest(
            radgrid, data, target_area_def, radius_of_influence=4000, fill_value=-999
        )

    return data


#################################################################################################################################
def setup_shapefiles():

    """
    Setup for shapefiles. Comment/uncomment as necessary. Perhaps use yaml cascade to configure in the future.
    """

    country_borders = cfeature.BORDERS.with_scale("50m")
    # state_borders = cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines',
    #                 scale='50m', facecolor='none',edgecolor='k') #includes Canadian provinces
    state_borders = cfeature.STATES.with_scale("10m")

    import cartopy.io.shapereader as shpreader

    ##offshore and high seas
    OFFSHORE_ZONES = HSMZ_ZONES = None
    # offshore_reader = shpreader.Reader(f"{pltg}/lightningcast/static/shapefiles/oz16ap19.shp")
    # recs = list(offshore_reader.records())
    ##remove non-CONUS offshore regions
    # delete=[]
    # for ii,rec in enumerate(recs):
    #  if("Hawaii" in rec.attributes['Name'] or
    #     "Alaska" in rec.attributes['Name'] or
    #     "Bering" in rec.attributes['Name'] or
    #     "Arctic" in rec.attributes['Name']):
    #    delete.append(ii)
    # offshore_zones = list(offshore_reader.geometries())
    # for index in sorted(delete, reverse=True): #reversed inds so we don't screw up the list
    #  del offshore_zones[index]
    # OFFSHORE_ZONES = cfeature.ShapelyFeature(offshore_zones, ccrs.PlateCarree())
    ##High-Seas Marine Zones
    # hsmz_reader = shpreader.Reader(f"{pltg}/lightningcast/static/shapefiles/hz30jn17.shp")
    # hsmz_zones = list(hsmz_reader.geometries())
    # HSMZ_ZONES = cfeature.ShapelyFeature(hsmz_zones, ccrs.PlateCarree())

    # BOU Burn areas
    burn_areas = None
    # burn_reader = shpreader.Reader(f"{pltg}/lightningcast/BOU_burns/BOU_BurnAreas.shp")
    # geoms = list(burn_reader.geometries())
    # for ii,geom in enumerate(geoms):
    #  geoms[ii] = geom.simplify(0.001)
    # burn_areas = cfeature.ShapelyFeature(geoms, ccrs.PlateCarree())

    return country_borders, state_borders, OFFSHORE_ZONES, HSMZ_ZONES, burn_areas


#################################################################################################################################
def get_imgr_chans(imgr, inputs, make_img):

    channames = get_channames(imgr)
    if make_img == 0:
        # Unpacking the list of lists in channels to a single list, getting the corresponding
        # channame for Satpy Scene if ABI, AHI, FCI, SEVIRI. Only naming the channels in the model.
        # Not exactly sure why this works, but see https://stackoverflow.com/questions/1198777/double-iteration-in-list-comprehension
        # Not adding non-imager channels (i.e., those that don't start with 'CH').
        satpy_chans = [
            channames[ch] for inp in inputs for ch in inp if ch.startswith("CH")
        ]
    else:
        # Just hard-code the channels for offline image-making mode. These are for Satpy Scenes.
        # Some RGBs need channels in addition to the ones LightningCast uses.
        if imgr == "ABI":
            satpy_chans = [
                "C13",
                "C15",
                "C11",
                "C14",
                "C02",
                "C03",
                "C05",
                "C07",
                "C01",
            ]
        elif imgr == "AHI":
            satpy_chans = ["B13", "B15", "B11", "B14", "B03", "B05", "B04"]
        elif imgr == "SEVIRI":
            satpy_chans = ["VIS006", "IR_016", "IR_108", "IR_120"]
        elif imgr == "FCI":
            satpy_chans = ["vis_06", "nir_16", "ir_105", "ir_123"]
        else:
            logging.critical(f"imgr {imgr} not supported.")
            sys.exit(1)

    return satpy_chans, channames


#################################################################################################################################
def other_model_config(mc):
    """
    Some checking and set up for other model config items.
    """
    try:
        svals = mc["scaling_values"]  # for  mean / std scaling
    except KeyError:
        svals = mc["scaling_vals"]
    mm_scaling = byte_scaling = False
    mm_svals = byte_scaling_vals = None
    if "minmax_scaling_vals" in mc:
        if len(mc["minmax_scaling_vals"]) > 0:
            # Indicates that we should scale to [0-1] (using minmax_scaling_vals)
            mm_scaling = True
            mm_svals = mc["minmax_scaling_vals"]
    if "byte_scaling_vals" in mc:
        if len(mc["byte_scaling_vals"]) > 0:
            # Use the byte_scaling_vals to normalize. This is almost always used with mm_scaling=True.
            # First you scale to 0-1, then you scale with these values, which are translated mean/std values.
            # This indicates that the model was trained with the bitmap methodology.
            byte_scaling = True
            byte_scaling_vals = mc["byte_scaling_vals"]
            logging.info("**Scaling data to bytes and then normalizing**")

    try:
        binarize1 = binarize = mc["binarize1"]
        binarize2 = mc["binarize2"]
    except KeyError:
        binarize = mc["binarize"]
        binarize2 = binarize1 = None

    try:
        target1 = mc["target1"]
        target2 = mc["target2"]
    except KeyError:
        target1 = mc["target"]
        target2 = None

    return (
        svals,
        mm_scaling,
        mm_svals,
        byte_scaling,
        byte_scaling_vals,
        binarize,
        binarize1,
        binarize2,
        target1,
        target2
    )



#################################################################################################################################
def run_overhead(
    model_config_file,
    sector,
    contour_cmap,
    values,
    colors,
    ref10,
    make_img,
    plot_engln,
    sector_suffix,
    infinite,
    filelist,
    datadir_patt,
    outdir,
    ll_bbox,
    ind_bbox,
    county_map,
    glmpatt,
    pickle_preds_labs,
    timeseries,
    ltg_stats,
    sat_info=None,
):
    """
    Run numerous one-time overhead processes and retrieve pertinent data for future use.
    """

    if infinite and sector is None:
        logging.critical("You must supply --sector if --infinite is True")
        sys.exit(1)
    elif sector and not (infinite):
        logging.critical(
            "Do not supply --sector except if --infinite is True. Code will get --sector dynamically."
        )
        sys.exit(1)

    idict = (
        {}
    )  # info dictionary that caries info that will be used later or iteratively.

    # Create the output directory
    os.makedirs(outdir, exist_ok=True)

    # Config info for the model we're running
    mc = pickle.load(open(model_config_file, "rb"))
    (
        svals,
        mm_scaling,
        mm_svals,
        byte_scaling,
        byte_scaling_vals,
        binarize,
        binarize1,
        binarize2,
        target1,
        target2,
    ) = other_model_config(mc)
    idict["svals"] = svals
    idict["mm_scaling"] = mm_scaling
    idict["mm_svals"] = mm_svals
    idict["byte_scaling"] = byte_scaling
    idict["byte_scaling_vals"] = byte_scaling_vals
    idict["binarize"] = binarize
    idict["binarize1"] = binarize1
    idict["binarize2"] = binarize2
    idict["target1"] = target1
    idict["target2"] = target2

    # If not using a contour colormap, make sure values and colors have same length. And make sure we don't have too many values.
    if not (contour_cmap):
        assert len(values) == len(colors)
    assert len(values) <= 8
    idict["values"] = values

    if contour_cmap:
        contour_cmap = cm.get_cmap(contour_cmap)
        colors = [
            contour_cmap(v / np.max(values)) for v in values
        ]  # normalize by max of values b/c vmax=np.max(values) in ax.contour
    idict["colors"] = colors
    idict["contour_cmap"] = contour_cmap

    # Set linewidths for images
    if len(values) == 4:  # defaults
        if ll_bbox:
            if ll_bbox[1] - ll_bbox[0] > 7 or ll_bbox[3] - ll_bbox[2] > 7:
                linewidths = [1, 1, 1, 1.5]
            else:
                linewidths = [1.5, 2, 2.5, 3]
        else:
            linewidths = [1] * len(values)
    else:
        linewidths = [2] * len(values)
    idict["linewidths"] = linewidths

    # Constants for MRMS Ref_-10C for images
    if ref10:
        from lightningcast import NWSColorMaps

        idict["ref_cmap"] = NWSColorMaps.NWSColorMaps()
        idict["gpd_info"] = {
            "NW_lat": 51,
            "NW_lon": -125,
            "nlon": 5900,
            "nlat": 2700,
            "dy": 0.01,
            "dx": 0.01,
        }
        idict["constants"] = {"MRMS": {"origNWlat": 55, "origNWlon": -130}}

    # Set up NWsatlat, SEsatlat to check if sector has changed positions
    idict["NWsatlat"] = -999
    idict[
        "SEsatlat"
    ] = -999  # for parallax and satzen-compute check, to see if the grid changed

    # nongoes sectors
    idict["nongoes"] = ["AHI", "SEVIRI", "FCI"]

    try:  # test if this is a valid himawari or goes l1b file
        satpy.Scene(filenames=[filelist], reader=["abi_l1b", "ahi_hsd"])
        files = [filelist]
    except FileNotFoundError:
        filelist = datadir_patt + "/" + filelist
        satpy.Scene(filenames=[filelist], reader=["abi_l1b", "ahi_hsd"])
        files = [filelist]
    except ValueError:
        # Read the filelist to get metadata (sector, imgr, etc.)
        with open(filelist, "r") as of:
            files = of.readlines()
            files = [
                f.strip() for f in files if f.strip()
            ]  # This trims whitespace before / after and excludes empty lines


    # If infinite, sector is defined. Get latest file that contains sector
    if infinite:
        file_path, dt = find_latest(filelist, sector)
        files = [file_path]

    out_files = []
    timestamp_set = set()
    for file in files:
        if os.path.isfile(file):
            full_path_filelist = True
            file_full_path = file
        else:
            if not datadir_patt:
                logging.critical(
                    f"Invalid path provided: {file}. Looks like you need to add datadir_patt to config or use full paths in your filelist"
                )
                sys.exit(1)
            file_full_path = datadir_patt + "/" + file
            if os.path.isfile(file_full_path):
                full_path_filelist = False
            else:
                logging.critical(f"Invalid path provided: {file_full_path}")
                sys.exit(1)

        if "full_path_filelist" in idict:
            if full_path_filelist != idict["full_path_filelist"]:
                logging.critical(
                    "Filelist contains invalid filenames. You cannot mix and match full paths and just basenames in your filelist."
                )
                sys.exit(1)
        else:
            idict["full_path_filelist"] = full_path_filelist

        imgr,  satname, sector, timestamp = utils.get_metadata_from_filename(file)

        if "satname" in idict:
            if satname != idict["satname"]:
                logging.critical("Inconsistent satname in filelist!")
                sys.exit(1)
        else:
            idict["satname"] = satname

        if "sector" in idict:
            if sector != idict["sector"]:
                logging.critical("Inconsistent Sectors in filelist!")
                sys.exit(1)
        else:
            idict["sector"] = sector

        if timestamp not in timestamp_set:
            timestamp_set.add(timestamp)
            out_files.append(file)
        else:
            logging.info(f"skipping duplicate")

    files = out_files

    idict["imgr"] = imgr
    if imgr == "ABI":
        idict["irCH"] = "C13"
    elif imgr == "AHI":
        idict["irCH"] = "B13"

    # tmp_scn.load([irCH])
    # meta = tmp_scn[irCH].attrs
    # print(meta)
    # sys.exit()

    # Lightning plotting config
    idict["plotGLM"] = (
        True if (glmpatt is not None and make_img and sector in ["RadC", "RadM1", "RadM2", "RadF"]) else False
    )
    if plot_engln is True:
        idict["plotGLM"] = False
    idict["plot_engln"] = plot_engln

    # Get the satellite name, slot, imager name, and channels
    from lightningcast.sat_config import sat_config

    if sat_info is None:
        sat_info = sat_config()

    satpy_chans, channames = get_imgr_chans(idict["imgr"], mc["inputs"], make_img)
    idict["channames"] = channames
    idict["satpy_chans"] = satpy_chans
    # The factor by which we subsample the predictions (in pixel-space).
    idict["stride"] = sat_info["stride"]
    # Threshold for skipping scenes with too hot of max focal plane temperature
    idict["mfpt_thresh"] = sat_info["max_focal_plane_temp_threshold"]
    # Threshold for block masking or median filtering bad data. Used in ltg_utils.clean_data
    idict["bad_data_fraction_threshold"] = sat_info["bad_data_fraction_threshold"]
    # Half-size at 2km res. for QC purposes. Used in ltg_utils.clean_data
    idict["qc_hs"] = sat_info["qc_hs"]

    # Used when we have a static subset of a sector
    idict["sector_outname"] = f"{sector}_{sector_suffix}" if (sector_suffix) else sector

    # Other metadata for output files
    idict["production_site"] = sat_info["production_site"]
    idict["production_environment"] = sat_info["production_environment"]
    idict["time_patt"] = sat_info["time_patt"]
    idict["institution"] = sat_info["institution"]
    idict["title"] = sat_info["title"]

    # Copy the filelist and ll_bbox info to outdir if not infinite. For future re-running of cases.
    if not (infinite):
        # copy filelist
        try:
            shutil.copy(filelist, outdir)
        except:
            pass
        # save the ll_bbox info in outdir in case we need to reference it again
        if ll_bbox:
            f = open(f"{outdir}/ll_bbox.txt", "w")
            f.write(str(ll_bbox) + "\n")
            f.close()

    # ==Shapefiles==#
    (
        country_borders,
        state_borders,
        OFFSHORE_ZONES,
        HSMZ_ZONES,
        burn_areas,
    ) = setup_shapefiles()
    idict["country_borders"] = country_borders
    idict["state_borders"] = state_borders
    idict["OFFSHORE_ZONES"] = OFFSHORE_ZONES
    idict["HSMZ_ZONES"] = HSMZ_ZONES
    idict["burn_areas"] = burn_areas
    idict["county_map"] = county_map
    if county_map:
        from metpy.plots import USCOUNTIES
        idict["uscounties"] = USCOUNTIES

    crop = False
    # A note about cropping...the crop should be completely contained
    # within the sector, or imaging (ABI and/or GLM data) and json results may be wrong.

    # ==Set default indexing for certain satellite/sectors==#
    if not (ll_bbox) and not (ind_bbox):
        if sector == "RadC":
            startY = 0
            endY = 1500
            startX = 0
            endX = 2500
        elif sector.startswith("RadM"):
            startY = 0
            endY = 500
            startX = 0
            endX = 500
        elif sector == "RadF":

            logging.warning(
                "Running RadF sector with no cut out. If it fails you may not have enough ram or you can try turning off onednn (see docs)."
            )
            #sys.exit(1)

            # For 2-km pixels. Dims should be divisible by 4.
            # ind_bbox = [700,4600,338,4000] #(yind_min is south of y_ind_max)
            # ind_bbox = [200,4800,400,4000]

            # ind_bbox = [4,5420,100,2000] #top sector
            # ind_bbox = [4,5420,1900,4000] #second sector
            # if('goes18' in datadir_patt):
            #  ind_bbox = [1100,4100,400,2900]
            # else:
            #  ind_bbox = [700,3900,500,3000] #covers OPC and TAFB offshore regions
            # ind_bbox = [2700, 4800, 2300, 4500] #Brazil

            #crop = True
            startY = 0
            endY = 5424
            startX = 0
            endX = 5424
        elif sector == "FLDK":

            logging.warning(
                "Running Himawark Fulldisk sector with no cut out. If it fails you may not have enough ram or you can try turning off onednn (see docs)."
            )

            startY = 0
            endY = 5500
            startX = 0
            endX = 5500

        elif sector == "JP":
            # chopping off missing data; dims also need to be div. by 4
            crop = True
            (startY, endY, startX, endX) = ind_bbox = [160, 1396, 95, 1115]

        idict["startY"] = startY
        idict["endY"] = endY
        idict["startX"] = startX
        idict["endX"] = endX

    else:
        crop = True
    idict["crop"] = crop

    # save the ind_bbox in outdir in case we need to reference it again. These are the data inds.
    if ind_bbox:
        idict["ind_bbox"] = ind_bbox
        assert ind_bbox[0] < ind_bbox[1] and ind_bbox[2] < ind_bbox[3]
        if not (infinite):
            f = open(f"{outdir}/ind_bbox.txt", "w")
            f.write(str(ind_bbox) + "\n")
            f.close()

    # Currently unused
    # Check if we should read static data
    # If daily ltg climo is predictor (HRAC_COM_FR), read static data
    # daily_ltg_climo_file = '/ships19/grain/probsevere/LC/ltg_climo_daily.nc'
    # for inp in inputs:
    #  if('HRAC_COM_FR' in inp):
    #    ltg_climo_data = utils.read_data(daily_ltg_climo_file,'HRAC_COM_FR')
    #    climo_fac = 4 #FIXME this should probably be a config item
    #    break

    return idict, mc, files


#################################################################################################################################
def get_glm_area_def(idict, glmpatt):

    """Make GE and GW GLM area definitions, to remap to ABI area
    This is only needed if we're using `make_img` (and `glmpatt` is not None), `timeseries`, `pickle_preds_labs`, or `ltg_stats`.
    The other time we would need glm_area_def is if GLM fields are used as predictors, which they are currently not."""

    try:
        assert (1 > 0)
    #    assert (("_g16_" in glmpatt.lower() or "_g19_" in glmpatt.lower() or "goes_east" in glmpatt.lower()) and idict["slot"].lower() == "goes_east") or \
    #           (("_g17_" in glmpatt.lower() or "_g18_" in glmpatt.lower() or "goes_west" in glmpatt.lower()) and idict["slot"].lower() == "goes_west")
    except AssertionError:
        logging.critical(
            f"Unable to get glm_area_def with this glmpatt: {glmpatt}"
        )
        logging.critical(
            f"Check that slot ({idict['slot']}) and satellite slot in glmpatt are compatible."
        )
        sys.exit(1)

    if glmpatt.startswith("https") or "GLMF" in glmpatt:  # indicates we're using FD GLM data
        glm_area_def, _, _ = utils.get_area_definition(
            f"{pltg}/lightningcast/static/{idict['slot']}_FD.nc",
            return_latlons=False,
        )
    else:  # Assumes CONUS/PACUS fixed-grid-format GLM data
        if (idict["slot"] == "GOES_East"):  # GOES_East GLM and ABI are exactly the same!
            glm_area_def, _, _ = utils.get_area_definition(
                #f"{pltg}/lightningcast/static/GOES_East.nc",
                "/ships22/grain/ajorge/data/glm_grids_60min_sum/agg_ln/20211110-180000_20211110-190000.netcdf", 
                return_latlons=False,
            )
        elif (idict["slot"] == "GOES_West"):  # GOES_West GLM and ABI are slightly different, for some reason.
            glm_area_def, _, _ = utils.get_area_definition(
                f"{pltg}/lightningcast/static/GOES_West_GLM.nc",
                return_latlons=False,
            )
    try:
        idict["glm_area_def"] = glm_area_def
    except (UnboundLocalError, NameError) as err:
        logging.warning("No glm_area_def available.")


#################################################################################################################################
def sector_shift(idict, plax_hgt, lon_0):

    """
    Checking if the sector has changed (usu. for mesos or first-time CONUS/PACUS).
    If new, will re-compute satzen and set flag, make_remap_cache=True, for GLM remapping.
    Also, creates new parallax-corrected info.
    """

    # Checking NW and SE corners in case one corner moves from one space-look pixel to another space-look pixel
    if idict["NWsatlat"] == np.round(idict["irlats"][0, 0], 3) and idict[
        "SEsatlat"
    ] == np.round(idict["irlats"][-1, -1], 3):
        idict["make_remap_cache"] = False
    else:
        idict["make_remap_cache"] = True
        # reset
        idict["NWsatlat"] = np.round(idict["irlats"][0, 0], 3)
        idict["SEsatlat"] = np.round(idict["irlats"][-1, -1], 3)

        # Get parallax correction info
        if plax_hgt > 0:
            idict["plax_hgt"] = plax_hgt
            logging.info("Performing parallax correction for nonstatic gridded data.")
            # irlons = irlons % 360 --> all positive
            # irlons = (irlons - 360) % (-360) --> all negative
            try:
                plax_outdict = utils.plax(
                    lons=(idict["irlons"] - 360) % (-360), # Convert all lons to negative 
                    lats=idict["irlats"],
                    satlon=lon_0,
                    elevation=plax_hgt,
                    return_dict=True,
                )
            except ValueError as err:
                logging.error("Parallax correction unsuccessful")
                idict["plaxcorr_success"] = False
            else:
                idict["plaxcorr_success"] = True
                # PLAX_inds_nonstatic is for non-static GRIDDED data.
                idict["PLAX_inds_nonstatic"] = (
                    np.ravel(plax_outdict["yind"]),
                    np.ravel(plax_outdict["xind"]),
                )
                idict["plax_lats"] = plax_outdict["lat_pc"]
                idict["plax_lons"] = plax_outdict["lon_pc"]

        # Make satellite-zenith angle for QC of preds
        # We only need to do this once, or when Mesos move
        # This takes 3-5 seconds, so we only want to do it when necessary
        idict["satzen"] = sat_zen_angle(
            xlat=idict["irlats"], xlon=idict["irlons"], satlat=0, satlon=lon_0
        )
        np.nan_to_num(idict["satzen"], nan=999, copy=False)

        # 0.5 km and 1 km satzens used for QCing in ltg_utils.clean_data
        idict["satzen_05km"] = idict["satzen"].repeat(4, axis=0).repeat(4, axis=1)
        idict["satzen_1km"] = idict["satzen"].repeat(2, axis=0).repeat(2, axis=1)

        logging.info("Computed satellite-zenith angle")

    # idict gets returned modified


#################################################################################################################################
def write_image(scn, dt, new_preds, projx, projy, idict, meta, files_for_ir_only_cloud_phase, glm_aggregate_time=5):


    image_type_str_dict={1:"DLC", 2:"VIS", 3:"DCC", 4:"DCPD", 5:"SW"}
    image_type_str=image_type_str_dict[idict["make_img"]]

    # Some shapefiles we may need
    country_borders = idict["country_borders"]
    state_borders = idict["state_borders"]
    OFFSHORE_ZONES = idict["OFFSHORE_ZONES"]
    HSMZ_ZONES = idict["HSMZ_ZONES"]
    burn_areas = idict["burn_areas"]

    # find solar zenith angle for midpoint, for making images
    # wlon, slat, elon, nlat = meta['area'].area_extent_ll
    llshape = idict["irlats"].shape
    ind0 = llshape[0] // 2
    ind1 = llshape[1] // 2  # works over the dateline
    mean_solzen, mean_solaz = solar_angles(
        dt, lat=idict["irlats"][ind0, ind1], lon=idict["irlons"][ind0, ind1]
    )
    mean_solzen = mean_solzen[0]

    if idict["ref10"]:
        gpd_info = idict["gpd_info"]
        constants = idict["gpd_info"]
        vmin = 0
        vmax = 62
        cmin = 10
        cmax = 61
        cinc = 10
        cmap = idict["ref_cmap"]
        IR = False
        cbar_label = "MRMS Reflectivity -10$^{o}$C"
        filled_mrms_patt = idict["mrms_patt"].replace("{ch}", "Reflectivity_-10C")
        # find and read Ref -10C data
        for mmm in [0, 1, 2, 3, 4]:
            try:  # hard-coded dir for now
                tmpdt = dt - timedelta(minutes=mmm)
                reffile = glob.glob(tmpdt.strftime(filled_mrms_patt))[0]
            except IndexError:
                if mmm == 4:
                    logging.critical(
                        f"Couldn't find Ref -10C data near {dt.strftime('%Y%m%d-%H%M%S')}"
                    )
                    sys.exit(1)
            else:
                imgdata = utils.read_grib2(reffile, gpd_info, constants)
                imgdata = np.nan_to_num(imgdata, nan=0)
                break
        # do remapping
        radlats = np.arange(
            gpd_info["NW_lat"],
            gpd_info["NW_lat"] - (gpd_info["nlat"]) * gpd_info["dy"],
            -gpd_info["dy"],
        )
        radlons = np.arange(
            gpd_info["NW_lon"],
            gpd_info["NW_lon"] + (gpd_info["nlon"]) * gpd_info["dx"],
            gpd_info["dx"],
        )
        oldlons, oldlats = np.meshgrid(radlons, radlats)
        mrmsgrid = pr.geometry.GridDefinition(lons=oldlons, lats=oldlats)
        crs = scn[idict["irCH"]].attrs["area"]  # .to_cartopy_crs()
        imgdata = pr.kd_tree.resample_nearest(
            mrmsgrid, imgdata, crs, radius_of_influence=4000, fill_value=-1
        )

    elif idict["plotGLM"]:
        try:
            imgdata = fed_data + 0.0
        except (UnboundLocalError, NameError):
            try:
                if idict["make_remap_cache"]:
                    idict["remap_cache"] = get_remap_cache(
                        idict["glm_area_def"], idict["abi_area_def"]
                    )  # If new corner point, cache info for remapping
                imgdata = ltg_utils.get_fed(
                    dt,
                    idict["glmpatt"],
                    glmvar=idict["glmvar"],
                    remap_cache=idict["remap_cache"],
                )
                imgdata = imgdata[
                    0 : (idict["endY"] - idict["startY"]),
                    0 : (idict["endX"] - idict["startX"]),
                ]  # force it to have the same dims as ABI data
            except (UnboundLocalError, ValueError) as err:  # couldn't find fed data
                logging.warning(str(err))
                logging.warning("Couldn't find FED for " + dt.strftime("%Y%m%d-%H%M%S"))
                return  # move on to next line / datetime

        vmin=0.1; cbticks=[16,32,48,64]; cmap = plt.get_cmap('jet'); norm=None
        vmax = np.max(cbticks)
        #vmin = 0.1
        #vmax = 64
        #cbticks = [16, 32, 48, 64]
        #cmap = plt.get_cmap("jet")
        #norm = None
        # vmin=idict['binarize']; vmax=128; cbticks=[8,16,32,64,128]; cmap = plt.get_cmap('jet'); norm=None #mcolors.PowerNorm(gamma=0.5)
        # assert(cbticks[0] >= vmin)
        cbar_label = f"GLM flash-extent density [fl ({glm_aggregate_time} min)$^{-1}$]"
        IR = False

    else:
        IR = True  # assume an IR image
        imgdata = scn[idict["irCH"]].compute().data
        vmin = 180
        vmax = 310
        inc = 10
        cmin = 180
        cmax = 301
        cinc = 40
        cmap = irw_ctc()
        cbar_label = idict["imgr"] + " 10.3-$\mu$m BT [K]"

    # fig = plt.figure(figsize=((fig_width,fig_height)))
    # ax=fig.add_axes([0,0,1,1],projection=geoproj)
    ##set extent of image
    # ax.set_extent([xmin,xmax,ymin,ymax], crs=geoproj)
    # if(idict['imgr']=='ABI'): ax.add_feature(state_borders, edgecolor='tan', linewidth=1.5)

    fs = 14

    figsize = 10
    if len(projx) / len(projy) >= 1:
        fig_width = figsize
        fig_height = figsize * len(projy) / len(projx)
    else:
        fig_width = figsize * len(projx) / len(projy)
        fig_height = figsize

    logging.info(
        f"figure width/height: {np.round(fig_width,2)},{np.round(fig_height,2)}"
    )
    fig = plt.figure(figsize=(fig_width, fig_height))
    crs = scn[idict["irCH"]].attrs["area"].to_cartopy_crs()
    ax = fig.add_axes([0, 0, 1, 1], projection=crs)
    ax.set_extent([projx.min(), projx.max(), projy.min(), projy.max()], crs=crs)

    # bottom left cbar axes
    cbar_axes = [0.02, -0.04, 0.45, 0.03]  # left bottom width height

    # imgdata *= 0 #set if you want to zero-out GLM data

    if mean_solzen < 85:  # and idict['sector'] != 'RadF'):
        chan = idict["channames"]["CH02"]
        if idict["make_img"] == 2 or idict["ref10"]:
            visdata = scn[chan].compute().data
            ax.imshow(
                np.sqrt(visdata / 100),
                transform=crs,
                extent=crs.bounds,
                vmin=0,
                vmax=1,
                cmap=plt.get_cmap("Greys_r"),
            )
        else:
            if idict["make_img"] == 1:
                composite = "DayLandCloud"
                alpha = 0.8
            elif idict["make_img"] == 3:
                composite = "COD_DCC"
                alpha = 0.8
            elif idict["make_img"] == 4:
                composite = "COD_DCP"
                alpha = 0.8
            elif idict["make_img"] == 5:
                composite = "ir_sandwich"
                alpha = 1
            scn.load([composite])
            resamp = scn.resample(scn.finest_area(), resampler="native")
            rgb = np.transpose(
                get_enhanced_image(resamp[composite]).data.values, axes=(1, 2, 0)
            )
            # rgb = scn.resample(resampler='native')
            rgb_shape = rgb.shape
            alpha_layer = np.full((rgb_shape[0], rgb_shape[1], 1), alpha)
            rgb = np.concatenate((rgb, alpha_layer), axis=-1)
            ax.imshow(rgb, transform=crs, extent=crs.bounds)

        if idict["imgr"] == "ABI":
            imgd = ma.masked_less(imgdata, 0.1)  # idict['binarize'])
            # if(IR):
            #  pcimg = ax.imshow(imgd, transform=crs, extent=crs.bounds, vmin=vmin, vmax=vmax,
            #               cmap=cmap)
            # else: #just make GLM opaque

            if idict["plotGLM"]:
                pcimg = ax.imshow(
                    imgd,
                    transform=crs,
                    extent=crs.bounds,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    norm=norm,
                )
                cbaxes2 = fig.add_axes(cbar_axes)
                cbar2 = plt.colorbar(
                    pcimg,
                    orientation="horizontal",
                    pad=0.00,
                    extend="max",
                    drawedges=0,
                    ax=ax,
                    ticks=cbticks,
                    cax=cbaxes2,
                )
                cbar2.set_label(cbar_label, fontsize=fs)
                cbar2.ax.tick_params(labelsize=fs)

    else:  # if "night time"
        # IR and glm imgs
        # buff=5; stride=buff*8
        # ny,nx = irdata.shape
        # for ii in range(buff,ny-buff,stride):
        #  for jj in range(buff,nx-buff,stride):
        #    irdata[-buff+ii:ii+buff+1,-buff+jj:jj+buff+1] = 310

        # Just IR

   #     irdata = scn[idict['irCH']].compute().data
   #     ax.imshow(irdata, transform=crs, extent=crs.bounds, vmin=180, vmax=310, cmap=plt.get_cmap('Greys')) #irw_ctc()

        logging.info("Input timestamp has solar zenith > 85 so defaulting to making image from IR only.")


        if files_for_ir_only_cloud_phase:
            image_type_str="IRCP"
            # IRCloudPhaseFC
            alpha = 0.8
            composite = "IRCloudPhaseFC"
            scn.load([composite])
            resamp = scn.resample(scn.finest_area(), resampler="native")
            rgb = np.transpose(
                get_enhanced_image(resamp[composite]).data.values, axes=(1, 2, 0)
            )
            rgb_shape = rgb.shape
            alpha_layer = np.full((rgb_shape[0], rgb_shape[1], 1), alpha)
            rgb = np.concatenate((rgb, alpha_layer), axis=-1)
            ax.imshow(rgb, transform=crs, extent=crs.bounds)
        else:
            logging.info("Channel / Band 11 and 14 missing so creating IR image from Channel / Band 13 only.")
            image_type_str = "C13"
            irdata = scn[idict['irCH']].compute().data
            ax.imshow(irdata, transform=crs, extent=crs.bounds, vmin=180, vmax=310, cmap=plt.get_cmap('Greys')) #irw_ctc()

        if idict["plotGLM"]:
            glmimg = ax.imshow(
                ma.masked_less(imgdata, 0.1),
                transform=crs,
                extent=crs.bounds,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            cbaxes = fig.add_axes(cbar_axes)
            cbar = plt.colorbar(
                glmimg,
                orientation="horizontal",
                pad=0.01,
                extend="max",
                drawedges=0,
                ticks=cbticks,
                cax=cbaxes,
            )
            cbar.set_label(cbar_label, fontsize=fs)
            cbar.ax.tick_params(labelsize=fs)

    # ENI lightning for non-GOES imagers
    eni_pts = None # default val
    if idict["imgr"] in idict["nongoes"] or idict["plot_engln"]:
        eni_ms = 6
        if idict["eni_path"]:
            try:
                eni_lats, eni_lons = get_eni_data(
                    dt, idict["irlats"], idict["irlons"], eni_path=idict["eni_path"]
                )
            except RuntimeError:
                pass
            else:
                eni_pts = ax.plot(
                    eni_lats,
                    eni_lons,
                    "o",
                    color="black",
                    markersize=eni_ms,
                    transform=ccrs.PlateCarree(),
                )


    # Maps and shapefiles
    if idict["imgr"] == "ABI":
        # ax.add_feature(burn_areas,facecolor='none', edgecolor='red',linewidth=1)
        if idict["sector"] == "RadF":
            ax.coastlines(color="black", linewidth=2.0)
            ax.add_feature(state_borders, edgecolor="black", linewidth=1.0)
            ax.add_feature(
                country_borders, facecolor="none", edgecolor="black", linewidth=2.0
            )
            # ax.add_feature(OFFSHORE_ZONES, facecolor='none', edgecolor='red',linewidth=0.5)
            # ax.add_feature(HSMZ_ZONES, facecolor='none', edgecolor='red', linewidth=0.75)
        else:
            if idict["county_map"]:
                ax.add_feature(
                    idict["uscounties"].with_scale("5m"),
                    edgecolor="black",
                    facecolor="none",
                    linewidth=0.5,
                )
            ax.add_feature(state_borders, edgecolor="black", linewidth=1.5)
            # ax.coastlines(color='tan',linewidth=1.5)
    if idict["imgr"] in idict["nongoes"]:
        ax.coastlines(color="black")

    # Plot symbols for points
    if idict["plot_points"] is not None:
        pp_lons = idict["plot_points"][::2]
        pp_lats = idict["plot_points"][
            1::2
        ]  # Assuming alternating lon/lat starting with lon
        ax.plot(
            pp_lons,
            pp_lats,
            color="red",
            linewidth=None,
            marker="o",
            markersize=12,
            transform=ccrs.PlateCarree(),
        )

    # ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='black', facecolor='#a15d10'))
    # ax.add_feature(cfeature.OCEAN.with_scale('50m'),color='aqua')

    # Add latitude and longitude gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=['top','left'],
        linewidth=0.5,
        color="gray",
        linestyle="--",
    )
    # Customize the gridline labels
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # contours
    labels = [str(int(v * 100)) + "%" for v in idict["values"]]

    # new_preds *= 0 #set if you want to zero-out the predictions

    if (
        len(new_preds.shape) == 3
    ):  # indicates that we have more than one predicted field
        vals1 = [0.1, 0.5]
        colors1 = ["#0000FF", "#00FFFF"]
        linewidths1 = [1, 1.5]
        contour_img1 = ax.contour(
            projx[0 :: idict["stride"]],
            projy[0 :: idict["stride"]],
            new_preds[0 :: idict["stride"], 0 :: idict["stride"], 0],
            vals1,
            transform=crs,
            extent=crs.bounds,
            colors=colors1,
            linewidths=linewidths1,
        )
        vals2 = [0.25, 0.5]
        colors2 = ["#FF0000", "#FF9999"]
        linewidths2 = [1, 1.5]
        contour_img2 = ax.contour(
            projx[0 :: idict["stride"]],
            projy[0 :: idict["stride"]],
            new_preds[0 :: idict["stride"], 0 :: idict["stride"], 1],
            vals2,
            transform=crs,
            extent=crs.bounds,
            colors=colors2,
            linewidths=linewidths2,
        )
        custom_lines = [
            Line2D([0], [0], color=c, lw=lw * 2)
            for c, lw in zip(colors1 + colors2, linewidths1 + linewidths2)
        ]
        labels = [str(int(v * 100)) + "%" for v in vals1 + vals2]

        ncol = len(labels)
        vert_offset_lab = -0.11
        vert_offset_leg = -0.08
        horiz_offset_leg = 1.0
        leg2 = ax.legend(
            custom_lines,
            labels,
            fontsize=fs,
            loc="lower right",
            handlelength=1.5,
            ncol=ncol,
            bbox_to_anchor=(horiz_offset_leg, vert_offset_leg),
        )
        prob_label1 = f"Prob. ≥ {idict['binarize1']} fl in 60 min"
        prob_label2 = f"Prob. ≥ {idict['binarize2']} fl in 60 min"
        ctext1 = plt.text(
            0.515, vert_offset_lab, prob_label1, fontsize=fs, transform=ax.transAxes
        )
        ctext2 = plt.text(
            0.765, vert_offset_lab, prob_label2, fontsize=fs, transform=ax.transAxes
        )
    else:
        if idict["contour_cmap"]:
            contour_img = ax.contour(
                projx[0 :: idict["stride"]],
                projy[0 :: idict["stride"]],
                new_preds[0 :: idict["stride"], 0 :: idict["stride"]],
                idict["values"],
                transform=crs,
                extent=crs.bounds,
                cmap=idict["contour_cmap"],
                vmin=0,
                vmax=np.max(idict["values"]),
                linewidths=idict["linewidths"],
            )
            # contour_img = ax.contour(irlons[0::stride,0::stride],irlats[0::stride,0::stride],new_preds[0::stride,0::stride], idict['values'],
            #                        transform=ccrs.PlateCarree(), colors=colors, linewidths=idict['linewidths'])
        else:
            contour_img = ax.contour(
                projx[0 :: idict["stride"]],
                projy[0 :: idict["stride"]],
                new_preds[0 :: idict["stride"], 0 :: idict["stride"]],
                idict["values"],
                transform=crs,
                extent=crs.bounds,
                colors=idict["colors"],
                linewidths=idict["linewidths"],
            )

        custom_lines = [
            Line2D([0], [0], color=c, lw=lw * 2)
            for c, lw in zip(idict["colors"], idict["linewidths"])
        ]
        if len(labels) <= 4:
            ncol = len(labels)
            vert_offset_lab = -0.11
            vert_offset_leg = -0.08
            horiz_offset_leg = 1.0
            # ncol=2
            # vert_offset_lab = -0.14
            # vert_offset_leg = -0.12
            # horiz_offset_leg = 0.97
        else:
            ncol = 4
            vert_offset_lab = -0.14
            vert_offset_leg = -0.11
            horiz_offset_leg = 1.0
        if not (idict["ref10"]):
            leg2 = ax.legend(
                custom_lines,
                labels,
                fontsize=fs,
                loc="lower right",
                handlelength=1.5,
                ncol=ncol,
                bbox_to_anchor=(horiz_offset_leg, vert_offset_leg),
            )
            prob_label = (
                "Probability of lightning in 60 min"
                if (idict["binarize"] <= 1)
                else f"Probability of ≥ {idict['binarize']} fl in 60 min"
            )
            ctext = plt.text(
                0.6, vert_offset_lab, prob_label, fontsize=fs, transform=ax.transAxes
            )

            # Legend for ENI
            if eni_pts and idict["imgr"] in idict["nongoes"] or idict["plot_engln"]:
                leg3 = ax.legend(
                    [
                        Line2D(
                            [],
                            [],
                            color="black",
                            linestyle="none",
                            marker="o",
                            markersize=eni_ms,
                        )
                    ],
                    ["ENI flashes in prev. 10 min"],
                    ncol=1,
                    fontsize=fs + 2,
                    loc="lower left",
                    bbox_to_anchor=(0.05, vert_offset_lab + 0.02),
                )
                ax.add_artist(leg2)  # add probs legend back

    # Add logo
    CIMSS_logo = (
        f"{pltg}/lightningcast/static/CIMSS_logo_web_multicolor_PNG_1100x800.png"
    )
    logo = imageio.v2.imread(CIMSS_logo)
    newax = fig.add_axes([0.88, 0.005, 0.12, 0.12], anchor="SE")
    newax.imshow(logo)
    newax.axis("off")

    bird = meta["platform_name"]
    if idict["imgr"] == "ABI":
        if idict["sector"] == "RadC":
            scene_id = "CONUS"
        elif idict["sector"] == "RadF":
            scene_id = "FullDisk"
        elif idict["sector"] == "RadM1":
            scene_id = "Mesoscale-1"
        elif idict["sector"] == "RadM2":
            scene_id = "Mesoscale-2"
    elif idict["imgr"] == "AHI":
        scene_id = "Japan" if (idict["sector"].startswith("JP")) else "FullDisk"
    elif idict["imgr"] == "SEVIRI" or idict["imgr"] == "FCI":
        scene_id = "FullDisk"
    timestamp = dt.strftime("%Y%m%d-%H%M")
    title_timestamp = dt.strftime("%Y-%m-%d %H:%M")
    ax.set_title(f"{bird} {scene_id} {title_timestamp} UTC", fontsize=fs + 4)

    image_name_out=f"{idict['outdir']}/LC_{bird}_{scene_id}_{timestamp}_{image_type_str}.png"

    fig.savefig(
        image_name_out, bbox_inches="tight", dpi=200
    )
    plt.close()

    logging.info(f"Wrote out {image_name_out}")


#################################################################################################################################
def get_dt(line):

    basefile = os.path.basename(line)
    if basefile.startswith("OR_ABI"):  # ABI
        dt = datetime.strptime(basefile.split("_s")[1][0:13], "%Y%j%H%M%S")
    elif basefile.startswith("HS") and basefile.endswith(".DAT"):  # Himawari
        if "FLDK" in basefile:
            dt = datetime.strptime(("").join(basefile.split("_")[2:4]), "%Y%m%d%H%M")
        elif "JP0" in basefile:
            parts = basefile.split("_")
            if parts[5] == "JP01":
                td = timedelta(minutes=0)
            elif parts[5] == "JP02":
                td = timedelta(minutes=2.5)
            elif parts[5] == "JP03":
                td = timedelta(minutes=5)
            elif parts[5] == "JP04":
                td = timedelta(minutes=7.5)
            else:
                logging.critical("Not a subfile time for JP")
                sys.exit(1)
            dt = datetime.strptime(("").join(parts[2] + parts[3]), "%Y%m%d%H%M") + td
        else:
            logging.critical(f'{basefile} must contain "JP0" or "FLDK"')
            sys.exit(1)
    else:
        logging.critical(f"{basefile} filetype not supported.")
        sys.exit(1)

    return dt


#################################################################################################################################
def predict_ltg(
    filelist,
    model,
    model_config_file,
    sector=None,
    outdir=os.environ["PWD"] + "/OUTPUT/",
    glmpatt=None,
    glmvar="flash_extent_density",
    eni_path="/ships19/grain/lightning/",
    mrms_patt=None,
    ll_bbox=None,
    ind_bbox=None,
    datadir_patt=None,
    values=[0.1, 0.25, 0.5, 0.75],
    vis=False,
    ltg_stats=False,
    make_img=0,
    colors=["#0000FF", "#00FFFF", "#00FF00", "#CC00FF"],
    contour_cmap=None,
    county_map=False,
    plot_points=None,
    sector_suffix=None,
    lightning_meteograms=False,
    geotiff=False,
    awips_nc=0,
    netcdf=0,
    fire_event_ts=False,
    infinite=False,
    re_upload=None,
    make_json=False,
    pickle_preds_labs=False,
    grplacefile=False,
    pixel_buff_and_stride=[5, 40],
    timeseries=[],
    plax_hgt=0,
    extra_script=None,
    ftp=None,
    ref10=False,
    plot_engln=False,
    sat_info=None,
    glm_aggregate_time=5,
):

    logging.info(f"LightningCast version {__version__} Process started.")
    t00 = time.time()

    # Getting/Setting some variables ("overhead")
    idict, mc, files = run_overhead(
        model_config_file,
        sector,
        contour_cmap,
        values,
        colors,
        ref10,
        make_img,
        plot_engln,
        sector_suffix,
        infinite,
        filelist,
        datadir_patt,
        outdir,
        ll_bbox,
        ind_bbox,
        county_map,
        glmpatt,
        pickle_preds_labs,
        timeseries,
        ltg_stats,
        sat_info,
    )
    sector = idict["sector"]
    imgr = idict["imgr"]
    stride = idict["stride"]
    if "ind_bbox" in idict:
        ind_bbox = idict["ind_bbox"]

    # If we're writing parallax-corrected netcdfs, ensure that plax_hgt > 0
    if awips_nc >= 2:
        try:
            assert(plax_hgt > 0)
        except AssertionError:
            logging.critical("Arg plax_hgt must be > 0 if awips_nc is >= 2")
            logging.critical(f"plax = {plax_hgt}")
            sys.exit(1)

    # Initialize file-check strings for inf processing
    if infinite:
        latest_filepath = file_path = "foo"

    # Start/end X/Y for *non-cropped* sectors
    if "startY" in idict:
        startY = idict["startY"]
        endY = idict["endY"]
        startX = idict["startX"]
        endX = idict["endX"]

    # Add some info to idict
    idict["mrms_patt"] = mrms_patt
    idict["outdir"] = outdir
    idict["glmvar"] = glmvar
    idict["glmpatt"] = glmpatt
    idict["make_img"] = make_img
    idict["ref10"] = ref10
    idict["eni_path"] = eni_path
    idict["plot_points"] = plot_points

    # place holder
    plax_dict = None

    # Load the model
    conv_model, is_tensorflow, is_pytorch = get_model(model, mc)

    # Get the file times or listing from listened file
    if infinite:
        with open(filelist) as fp:  # with automatically closes file when done with it.
            lines = [line.rstrip("\n") for line in fp.readlines()]
    else:  # since duplicate lines are removed in overhead we will get our processed filelist from there
        lines = files

    nlines = len(lines)

    # Set up for timeseries capability
    # Should work RadC or RadM (if RadM is within RadC, since that is where our GLM data are).
    if len(timeseries):
        (
            crop,
            ts_every,
            ts_dts,
            ts_pltg,
            ts_pltg_radius,
            ltg_obs,
            ltg_obs_radius,
        ) = ts_setup(lines, timeseries, ll_bbox)

    logging.info(f"Overhead: {np.round(time.time()-t00,3)} seconds")
    # === Set up is complete ===#

    # === Begin loop to cycle through date-times ===#

    line_cter = 0
    frame_cter = 0
    # if 'infinite' is set, this is an infinite loop
    # otherwise, go through each line or time in filelist
    while line_cter < nlines or infinite:
        try:
            del fed_data
        except NameError:
            pass

        if infinite:
            while latest_filepath == file_path or file_path == "None":
                file_path, dt = find_latest(filelist, sector)
                if latest_filepath == file_path or file_path == "None":
                    logging.info("Already processed: " + latest_filepath)
                    logging.info("Sleeping 10 sec...")
                    time.sleep(10)
            # reset
            logging.info("Beginning to process: " + file_path)
            latest_filepath = file_path

        else:
            file_path = lines[line_cter]
            dt = get_dt(file_path)
            logging.info(f"processing {dt.strftime('%Y%m%d-%H%M%S')} UTC...")

        tstart = time.time()



        # --------------------------------------------------
        #  Create, load, and crop scene
        # --------------------------------------------------
        # Check datadir
        if not os.path.isfile(file_path):
            file_path = datadir_patt + file_path
            if not os.path.isfile(file_path):
                logging.critical(
                    f"Invalid file: {file_path}. Check your input and your datadir_patt (if you're using that option)"
                )
                sys.exit(1)

        datadir = os.path.dirname(file_path)
        if not datadir:
            datadir = "./"


        valid_req_chan_bands, problem_chan_band, optional_bands_check=utils.check_chan_bands(imgr, datadir, dt, sector,  required_abi=["02", "05", "13", "15" ], required_ahi=["03", "05", "13", "15" ], optional_abi=["03", "11", "14"], optional_ahi=["04", "11", "14"])

        if not valid_req_chan_bands:
            logging.critical("Missing 1 or more Files for: "+problem_chan_band)
            sys.exit(1)

        if make_img==1 and not optional_bands_check[0]:
            logging.critical(
                f"For making DayLandCloud images we require an extra Channel / Band (02 for ABI and 03 for AHI). A needed file appears to be missing."
            )
            sys.exit(1)

        files_for_ir_only_cloud_phase=False
        if optional_bands_check[1] and optional_bands_check[2]:
            files_for_ir_only_cloud_phase=True



        if imgr == "ABI":
            abi_filenames = glob.glob(
                dt.strftime(f"{datadir}/OR*_*-{sector}-*_s%Y%j%H%M%S*nc")
            )

            if (pickle_preds_labs or ltg_stats) and len(abi_filenames) < 16:
                # For functions that require some time continuity, we don't want the process to
                # error out if there is a time with some ABI files missing. This won't affect real-time.
                logging.warning(f'Only {len(abi_filenames)} abi files. Skipping.')
                continue

            try:
                orig_scn = satpy.Scene(reader="abi_l1b", filenames=abi_filenames)
            except ValueError:
                logging.critical(
                    "abi_filenames is empty. Check your --sector, or --datadir_patt, or filelist."
                )
                sys.exit(1)
            else:
                # Check max_focal_plane_temperature (This is for GOES-17 LHP problem).
                # If max_focal_plan_temperature >= X kelvin, then the scene is too degraded and we just skip it.
                if infinite:
                    max_focal_plane_temp = get_FPT(abi_filenames)
                    if max_focal_plane_temp >= idict["mfpt_thresh"]:
                        logging.warning(
                            f"Max focal plane temp. is {max_focal_plane_temp}K. Skipping this datetime."
                        )
                        line_cter += 1
                        continue  # next file/iteration

                orig_scn.load(idict["satpy_chans"])
                irCH = "C13"
        elif imgr == "AHI":
            orig_scn = get_ahi_scn(dt, datadir, sector)
            orig_scn.load(idict["satpy_chans"])
            irCH = "B13"
        elif imgr == "SEVIRI":
            orig_scn = get_meteosat_scn(dt, datadir)
            orig_scn.load(
                idict["satpy_chans"], upper_right_corner="NE"
            )  # 2nd arg flips it correctly
            irCH = "IR_108"  # all channels are 3km, except HRV (high-res vis).
        elif imgr == "FCI":  # on Meteosat Third Generation (MTG)
            orig_scn = get_mtg_scn(dt, datadir)
            orig_scn.load(
                idict["satpy_chans"], upper_right_corner="NE"
            )  # 2nd arg flips it correct
            irCH = "ir_105"
        idict["irCH"] = irCH

        try:
            meta = orig_scn[irCH].attrs
        except KeyError as err:
            logging.critical(str(err))
            sys.exit(1)

        # Get the "slot" and satname
        if imgr == "ABI":
            slot = idict["slot"] = meta["orbital_slot"].replace("-", "_")
            satname = meta["platform_name"].lower().replace("-", "")
        elif imgr == "AHI":
            slot = idict["slot"] = "AHI_JAPAN"
            satname = meta["platform_name"].lower()
        else:
            logging.critical(f"Imager {imgr} is not supported.")
            sys.exit(1)

        lon_0 = meta["area"].proj_dict["lon_0"] 
        projection = Proj(orig_scn[irCH].attrs["area"].crs)
        ccrs_proj = idict['ccrs_proj'] = ccrs.Geostationary(
            central_longitude=lon_0, satellite_height=meta["area"].proj_dict["h"]
        )

        # Get the grid remap data for the AWIPS remapped grid. It is statically saved in the remap_info.
        # Also get the parallax-correction info, if appropriate. idict gets returned with added fields.
        if line_cter == 0 and  awips_nc > 0:
            remap_info(idict, meta) 

        # Get GLM area definition. glm_area_def gets added to idict
        if line_cter == 0 and idict["imgr"] == "ABI" and ((
                make_img > 0 and glmpatt is not None) or len(timeseries) > 0 or pickle_preds_labs or ltg_stats
        ):
            get_glm_area_def(idict, glmpatt)
            

        # get proj coords (in case no crop)
        projx = orig_scn[irCH].x.compute().data
        projy = orig_scn[irCH].y.compute().data
        if ll_bbox:
            xmin, ymin = projection(ll_bbox[0], ll_bbox[2])
            xmax, ymax = projection(ll_bbox[1], ll_bbox[3])
            try:
                assert xmin < xmax and ymin < ymax
            except AssertionError:
                logging.critical(
                    "Your ll_bbox is incorrect (should be wlon, elon, slat, nlat)."
                )
                sys.exit(1)
            xy_bbox = [xmin, ymin, xmax, ymax]

        elif ind_bbox:
            # Use proj coords to crop grids. You want to do this if your cropped image will have space looks.
            # This is generally if we want to crop the FD or CONUS (use ll_bbox for CONUS, if no space looks).
            # It's best to just use ll_bbox if cropping the RadM sectors.
            print(projx.shape)
            xmin = projx[ind_bbox[0]]
            ymin = projy[ind_bbox[3]]
            xmax = projx[ind_bbox[1]]
            ymax = projy[ind_bbox[2]]
            xy_bbox = [xmin, ymin, xmax, ymax]

        if idict["crop"]:
            startY, endY, startX, endX, xy_bbox = start_end_inds(
                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, irx=projx, iry=projy
            )
            idict["startY"] = startY
            idict["endY"] = endY
            idict["startX"] = startX
            idict["endX"] = endX
            scn = orig_scn.crop(xy_bbox=xy_bbox)
            projx = scn[irCH].x.compute().data  # get cropped proj coords
            projy = scn[irCH].y.compute().data
            #scn.save_datasets(writer='simple_image', datasets=['B13'], filename='ldp-cropped-valid.png') #use to visualize cropped scn
            meta = scn[irCH].attrs  # reset with cropped info
        else:
            scn = orig_scn
        abi_area_def = idict["abi_area_def"] = scn[irCH].attrs["area"]

        try:
            # currently, only used when REF10 is a predictor
            abi_1km_def = scn["C05"].attrs["area"]
        except:
            pass

        # additional metadata
        meta["model"] = model

        # get the lats/lons and make sure everything is the right len/size
        projx = projx[0 : endX - startX]
        projy = projy[0 : endY - startY]
        irlons, irlats = meta["area"].get_lonlats()
        
        if ll_bbox and (True in np.isinf(irlons) or True in np.isinf(irlons)):
            logging.critical("Invalid bounding box! Most likely your ll_bbox parameter is to blame! ")
            sys.exit(1)
        
        irlons = irlons[0 : endY - startY, 0 : endX - startX]
        irlats = irlats[0 : endY - startY, 0 : endX - startX]

        idict["irlons"] = irlons
        idict["irlats"] = irlats
        # foo = utils.plax(lons=np.ma.masked_invalid(irlons),lats=np.ma.masked_invalid(irlats),satlon=lon_0,write_plax_file=True)

        if imgr == "SEVIRI":  # need to convert from 3km to 2km
            idict["irlons"], idict["irlats"], projx, projy = fix_coords_for_seviri(
                irlons, irlats, projx, projy
            )

        # Check if domain has changed/shifted from previous iteration
        # N.B. elements of idict get modified, incl. satzen, plax info, and make_remap_cache
        sector_shift(idict, plax_hgt, lon_0)

        if idict["crop"]:
            logging.info("got crop...")

        # Reset QC masks for each datetime
        idict["masks"] = {}  # for QCing data, if necessary
 
        # --------------------------------------------------
        #  Now begin organizing data to feed into model
        # --------------------------------------------------
        all_preds = []  # inputs of "tensors", but dtype=np.ndarray
        for inp in mc[
            "inputs"
        ]:  # this is for each input into the model (inputs is a list of lists)
            X = None  # this becomes the numpy tensor for input inp
            for ch in inp:  # this is for each channel in the input

                # ABI, AHI, SEVIRI, & MTG predictors
                if ch.startswith("CH"):
                    # makes the 2D numpy arrays
                    try:
                        chan = idict["channames"][ch]
                        if imgr == "ABI":
                            sat_data = utils.get_abi_data(
                                dt,
                                ch[2:],
                                datadir,
                                startY=startY,
                                endY=endY,
                                startX=startX,
                                endX=endX,
                            )
                        elif imgr == "AHI":  # Himawari
                            sat_data = scn[chan].compute().data
                            if ch == "CH02" or ch == "CH05":
                                sat_data /= 100  # ** satpy values are in % **
                        elif imgr == "SEVIRI":  # meteosat 2nd gen
                            sat_data = scn[chan].compute().data
                            if ch == "CH02" or ch == "CH05":
                                sat_data /= 100  # ** satpy values are in % **
                        elif imgr == "FCI":  # meteosat 3rd gen
                            sat_data = scn[chan].compute().data
                            if ch == "CH02" or ch == "CH05":
                                sat_data /= 100  # ** satpy values are in % **
                        else:
                            logging.critical(f"Imager {imgr} is not supported.")
                            sys.exit(1)
                        logging.info(f"got {chan}..., {sat_data.shape}")
                    except (IndexError, ValueError) as err:
                        logging.critical(traceback.format_exc())
                        logging.critical("Can't get input data for " + ch)
                        sys.exit(1)

                    # Ensure each input has the correct resolutions and dims (div. by 4, 8, 16) for the model
                    if imgr in idict["nongoes"]:
                        sat_data = spatial_align(sat_data, mc, imgr, scn, ch, idict)
                    
                    filled_data = ltg_utils.clean_data(sat_data, idict, ch)

                # MRMS predictors
                elif ch == "REF10":  # get ref10 data and remap to geos projection
                    filled_data = get_grib2_data(
                        dt, ch, mrms_patt, target_area_def=abi_1km_def
                    )
                    filled_data = filled_data[
                        0 : endY * 2 - startY * 2, 0 : endX * 2 - startX * 2
                    ]  # truncating in case of extra elements (*2 is b/c it's 1km)
                    logging.info(f"got {ch}..., {filled_data.shape}")

                # HRAC_COM_FR is "high-res" daily ltg climatology --> **not currently used**
                # elif(ch == 'HRAC_COM_FR'):
                #  doy = int(dt.strftime('%j'))-1
                #  filled_data = ltg_climo_data[doy, startY//climo_fac:endY//climo_fac, startX//climo_fac:endX//climo_fac]
                #  logging.info(f'got {ch}..., {filled_data.shape}')

                # GLM channels
                elif ch in ["FED", "FCD", "TOE", "MFA", "AFA"]:
                    try:
                        if idict["make_remap_cache"]:
                            idict["remap_cache"] = get_remap_cache(
                                idict["glm_area_def"], idict["abi_area_def"]
                            )  # If new corner point, cache info for remapping
                        glmgrid = ltg_utils.get_fed(
                            dt, glmpatt, glmvar=glmvar, remap_cache=idict["remap_cache"]
                        )
                        glmgrid = glmgrid[
                            0 : (endY - startY), 0 : (endX - startX)
                        ]  # force it to have the same dims as ABI data
                    except (UnboundLocalError, ValueError):
                        logging.critical(
                            "Couldn't find FED to make predictions for "
                            + dt.strftime("%Y%m%d-%H%M%S")
                        )
                        sys.exit(1)
                    filled_data = np.copy(glmgrid)
                    fed_data = np.copy(glmgrid)
                    logging.info(f"got {ch}..., {fed_data.shape}")

                # Scale data and add to our input tensor, X
                if idict["mm_scaling"]:  # Scale from 0 to 255
                    if (
                        ch == "REF10"
                    ):  # Use special encoding to denote inside/outside of radar range
                        scaled_data = (
                            utils.bytescale(
                                filled_data,
                                idict["mm_svals"][ch]["vmin"],
                                idict["mm_svals"][ch]["vmax"],
                            )
                            + 10
                        )
                        scaled_data[filled_data <= -999] = 0
                        scaled_data[filled_data == -99] = 1
                        scaled_data[scaled_data > 255] = 255
                    else:
                        scaled_data = utils.bytescale(
                            filled_data,
                            idict["mm_svals"][ch]["vmin"],
                            idict["mm_svals"][ch]["vmax"],
                        )
                        scaled_data[scaled_data < 0] = 0
                        scaled_data[scaled_data > 255] = 255
                elif (
                    idict["mm_scaling"] and idict["byte_scaling"]
                ):  # Indicates that we used the bitmap methodology to train, and thus we should scale accordingly.
                    scaled_data = (filled_data - idict["mm_svals"][ch]["vmin"]) / (
                        idict["mm_svals"][ch]["vmax"] - idict["mm_svals"][ch]["vmin"]
                    )  # scale to 0 to 1
                    scaled_data[scaled_data < 0] = 0
                    scaled_data[scaled_data > 1] = 1
                    scaled_data = (
                        scaled_data - idict["byte_scaling_vals"][ch]["mean"]
                    ) / idict["byte_scaling_vals"][ch]["std"]
                else:  # Normal mean/std scaling using native-unit data
                    try:
                        scaled_data = (
                            filled_data - idict["svals"].loc[ch, "mean"]
                        ) / idict["svals"].loc[ch, "std"]
                    except AttributeError:
                        scaled_data = (
                            filled_data - idict["svals"][ch]["mean"]
                        ) / idict["svals"][ch]["std"]

                # if(ch == 'CH02' or ch == 'CH05'): scaled_data *=0

                if X is None:
                    X = np.expand_dims(np.expand_dims(scaled_data, axis=0), axis=-1)
                else:
                    X = np.concatenate(
                        (X, np.expand_dims(scaled_data, axis=(0, -1))), axis=-1
                    )

            # Append input X to all_preds.
            all_preds.append(X)

        line_cter += 1

        # get fed_data, if necessary:
        try:
            fed_data = (
                fed_data + 0.0
            )  # we test here, b/c we may already have the data if FED was a predictor
        except (UnboundLocalError, NameError):
            if timeseries or ltg_stats:
                try:
                    if idict["make_remap_cache"]:
                        idict["remap_cache"] = get_remap_cache(
                            idict["glm_area_def"], abi_area_def
                        )  # If new corner point, cache info for remapping
                    fed_data = ltg_utils.get_fed(
                        dt, glmpatt, glmvar=glmvar, remap_cache=idict["remap_cache"]
                    )
                    fed_data = fed_data[
                        0 : (endY - startY), 0 : (endX - startX)
                    ]  # force it to have the same dims as ABI data
                except (UnboundLocalError, ValueError):  # couldn't find fed data
                    logging.warning(
                        "Couldn't find FED for " + dt.strftime("%Y%m%d-%H%M%S")
                    )
                    continue  # move on to next line / datetime

        # if saving preds/labels, get accumulation target/truth data
        if pickle_preds_labs:
            if line_cter == 1:  # hard-coded dimensions for container #FIXME?
                preds_container = np.zeros(
                    (nlines, 1500 // stride, 2500 // stride), dtype=np.float32
                )
                labs_container = np.copy(preds_container)
                datetimes = []
                filenames = []
            try:  # fed_accum_data is 60-min accumulation of FED, e.g.
                if idict["make_remap_cache"]:
                    idict["remap_cache"] = get_remap_cache(
                        idict["glm_area_def"], idict["abi_area_def"]
                    )  # If new corner point, cache info for remapping
                # Doing this so that we can use the glmpatt with "agg" for --ltg_stats
                glmpatt2 = (
                    glmpatt.replace("/agg/", "/FED_accum_60min_2km/")
                    if ("accum" not in glmpatt)
                    else glmpatt
                )
                accum_dt = dt + timedelta(
                    minutes=60
                )  # N.B. accumulation dt is *end time* of the accumulation.
                fed_accum_data = ltg_utils.get_fed(
                    accum_dt,
                    glmpatt2,
                    glmvar="FED_accum_60min_2km",
                    remap_cache=idict["remap_cache"],
                )
                fed_accum_data = fed_accum_data[
                    0 : (endY - startY), 0 : (endX - startX)
                ]  # force it to have the same dims as ABI data
            except (UnboundLocalError, ValueError):
                logging.error(
                    f"Couldn't find any fed_accum_data near {accum_dt}. Moving on to next datetime."
                )
                continue

        # if saving ltg stats, instantiate dictionary
        if ltg_stats:
            try:
                ltg_dbase
            except NameError:
                ltg_dbase = ltg_utils.make_dbase(
                    irlons,
                    pixbuff=pixel_buff_and_stride[0],
                    stride=pixel_buff_and_stride[1],
                )
            else:
                pass

        if is_pytorch:  # need channels first
            for iii in range(len(all_preds)):
                all_preds[iii] = np.transpose(
                    all_preds[iii], (0, 3, 1, 2)
                )  # assumes data is 4-dim

        logging.info("making predictions...")
        t0 = time.time()
        input_dims = (" ").join([str(iii.shape) for iii in all_preds])
        logging.info("input dims: " + input_dims)
        try:
            if is_tensorflow:
                new_preds = conv_model.predict(all_preds, verbose=1)
                new_preds = np.squeeze(new_preds) #[...,0])
            else:
                with torch.set_grad_enabled(False):
                    for i in range(len(all_preds)):  # move each input to the device
                        all_preds[i] = torch.from_numpy(
                            all_preds[i].astype(np.float32)
                        ).to(device, dtype=torch.float)
                    new_preds = (
                        conv_model(all_preds).data.cpu().numpy()
                    )  # making the predictions
                    new_preds = np.squeeze(
                        new_preds, axis=(0, 1)
                    )  # remove sample and channel dims

        except ValueError as err:
            print(
                err
            )  # if 'infinite' is set, this is an error possibly caused by certain input data not being available
            # will just move on to next scene
        else:
            logging.info(
                "predict time: " + "{:.3f}".format(time.time() - t0) + " seconds"
            )

            # Error fixing
            new_preds = np.ma.fix_invalid(new_preds, fill_value=0)

            # QC high-satzen pixels. We get FAs due to gradient at edge of disk.
            satzen_thresh = 80
            logging.info(
                f"Using satzen thresh. of {satzen_thresh} deg. to screen out large viewing-angle predictions."
            )
            new_preds[idict["satzen"] >= satzen_thresh] = 0


            # if(ref10): new_preds *= 0
            logging.info("max new_preds: " + "{:.3f}".format(np.max(new_preds)))

            if make_json: # This supports make_json as either a bool or int value
                if not (type(make_json) == int and make_json == 2):
                    logging.info("making contours and json...")

                    json_outdir = f"{outdir}/{dt.strftime('%Y%m%d')}/{slot.lower()}/{idict['sector_outname']}/json/"

                    _ = write_json(
                        new_preds,
                        idict,
                        meta,
                        json_outdir,
                        prefix="LtgCast",
                        values=values,
                    )

                if infinite and re_upload:
                    # If in infinite mode at CIMSS, copy outjson to RE-compatible filename and upload. Then remove the copied file.
                    upload_re_json(dt, json_outdir, idict, re_upload)

                # This supports make_json as either a bool or int value
                if plax_hgt > 0 and idict["plaxcorr_success"] and not (type(make_json) == int and make_json == 1):
                    # Use parallax correction info and write out geojson
                    # Only do if we successfully got the plax_lats and plax_lons

                    json_outdir = f"{outdir}/{dt.strftime('%Y%m%d')}/{slot.lower()}/{idict['sector_outname']}/json_plax/"

                    _ = write_json(
                        new_preds,
                        idict,
                        meta,
                        json_outdir,
                        prefix="LtgCast-pc",
                        values=values,
                    )

            if grplacefile:
                logging.info(
                    f"Creating GRPlacefile for {slot} {idict['sector_outname']}"
                )
                endtime = dt.strftime("%Y%m%d-%H%M%S")
                starttime = (dt - timedelta(minutes=60)).strftime("%Y%m%d-%H%M%S")
                try:
                    timeRangeGRFile(
                        starttime,
                        endtime,
                        datapatt=f"{outdir}/%Y%m%d/{slot.lower()}/{idict['sector_outname']}/json_plax/1fl-60min/",
                        outdirpatt=f"{outdir}/%Y%m%d/{slot.lower()}/{idict['sector_outname']}/placefile/",
                        realtime=infinite,
                    )
                except (KeyError, IndexError, OSError) as err:
                    logging.warning(str(err))
                    logging.warning(
                        "Something is wrong with creating or transmitting the GR placefile"
                    )

            if geotiff:
                make_geotiff(
                    scn[irCH],
                    new_preds,
                    dt,
                    projection,
                    stride,
                    slot,
                    sector,
                    outdir,
                    re_upload=re_upload,
                )

            if awips_nc:
                outnetcdf, save_status = save_netcdf(
                    scn[irCH],
                    new_preds,
                    dt,
                    meta,
                    outdir,
                    idict,
                    stride=stride,
                    savecode=awips_nc,
                    awips=True,
                )
                if save_status and extra_script:
                    status = subprocess.Popen(
                        ["bash", extra_script, outnetcdf]
                    )  # to send via LDM

            if netcdf:
                geo_nc, save_status = save_netcdf(
                    scn[irCH],
                    new_preds,
                    dt,
                    meta,
                    outdir,
                    idict,
                    stride=stride,
                    savecode=netcdf,
                )

            if lightning_meteograms:
                if infinite:
                    eni_patt = (
                        "/data/PLTG/lightning/realtime/ENI/flash-%Y-%m-%dT%H-%M-.csv"
                    )
                else:
                    eni_patt = "/ships19/grain/lightning/%Y/%Y_%m_%d_%j/%H/entln_%Y%j_%H%M*.txt"

                if plax_hgt > 0 and idict["plaxcorr_success"]:
                    try:
                        plaxcorr_preds = np.zeros(new_preds.shape, dtype=np.float32)
                        plaxcorr_preds[idict["PLAX_inds_nonstatic"]] = np.ravel(new_preds)
                    except ValueError:
                        shape = new_preds.shape
                        plaxcorr_preds = np.zeros((shape[0],shape[1]), dtype=np.float32)
                        plaxcorr_preds[idict["PLAX_inds_nonstatic"]] = np.ravel(new_preds[...,0])

                    # if(not(infinite)):
                    # fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(16,9))
                    # im = ax[0].imshow(new_preds[200:300,700:900]); ax[0].set_title('new_preds'); plt.colorbar(im,ax=ax[0],shrink=0.5); ax[0].grid()
                    # im = ax[1].imshow(plaxcorr_preds[200:300,700:900]); ax[1].set:_title('plaxcorr_preds'); plt.colorbar(im,ax=ax[1],shrink=0.5); ax[1].grid()
                    # plt.savefig('tmppreds.png',bbox_inches='tight')
                    # sys.exit(1)

                else:
                    logging.warning(
                        "Using un-parallax-corrected predictions in meteograms."
                    )
                    # The input probs to ltg_utils.lightning_meteograms_inmem (prob_grid) must be 2D
                    if len(new_preds.shape) == 3:
                        plaxcorr_preds = new_preds[...,0]
                    else:
                        plaxcorr_preds = new_preds

                # TAF sites

                if sector == "RadF" and slot == "GOES_East":
                    # Hard-coding Central and South American  airports that are only covered by RadF. For OPC/TAFB sector only.
                    code_includes = ["S", "MROC", "MPTO", "MNMG"]
                else:
                    code_includes = []

                try:
                    ltg_loc_dicts = ltg_utils.lightning_meteograms_inmem(
                        prob_grid=plaxcorr_preds,
                        irx=projx,
                        iry=projy,
                        dt=dt,
                        projection=projection,
                        locations_file=f"{pltg}/lightningcast/static/locations.txt",
                        glm_L2_patt=f"/satbuf2_data/goes/grb/{satname}/%Y/%Y_%m_%d_%j/glm/L2/LCFA/*_s%Y%j%H%M*",
                        meta=meta,
                        eni_patt=eni_patt,
                        code_includes=code_includes,  # only include location codes that begin with these strings
                        plax_hgt=plax_hgt,
                    )
                    if len(ltg_loc_dicts) > 0:
                        ltg_utils.insert_grafana_records(
                            ltg_loc_dicts, dbname="lightning_meteograms"
                        )
                except mysql.connector.Error as err:
                    logging.error(str(err))

                # Stadiums and on-demand sites
                try:
                    ltg_loc_dicts = ltg_utils.lightning_meteograms_inmem(
                        prob_grid=plaxcorr_preds,
                        irx=projx,
                        iry=projy,
                        dt=dt,
                        projection=projection,
                        locations_file=f"{pltg}/lightningcast/static/stadiums-geocoded.csv",
                        glm_L2_patt=f"/satbuf2_data/goes/grb/{satname}/%Y/%Y_%m_%d_%j/glm/L2/LCFA/*_s%Y%j%H%M*",
                        meta=meta,
                        ondemand_sites=True,
                        eni_patt=eni_patt,
                        code_includes=code_includes,  # only include location codes that begin with these strings
                        plax_hgt=plax_hgt,
                    )
                    if len(ltg_loc_dicts) > 0:
                        # PW, HOST, and USERNAME should be same as "lightning_meteograms" db
                        ltg_utils.insert_grafana_records(
                            ltg_loc_dicts, dbname="lightning_dss"
                        )
                except mysql.connector.Error as err:
                    logging.error(str(err))

            if fire_event_ts and awips_nc:  # need awips outnetcdf
                sat_and_sector = (
                    os.path.basename(outnetcdf).split("Cast_")[1].split("_2")[0]
                )
                logging.info("Farming out job to fire_event_timeseries.py")
                status = subprocess.Popen(
                    [
                        "python",
                        f"{pltg}/lightningcast/fire_event_timeseries.py",
                        "--fire_event_geojson",
                        "/ships19/grain/jcintineo/DixieFire/Dixie.geojson",  #'/apollo/grain/common/NIFC/Current-Fire-Location.geojson',
                        "--fire_event_pickle",
                        f"{pltg_data}/fire_ts/Current_Fire_Timeseries_{sat_and_sector}.pkl",
                        "--glm_L2_patt",
                        f"/arcdata/goes/grb/{satname}/%Y/%Y_%m_%d_%j/glm/L2/LCFA/*_s%Y%j%H%M*",
                        "--outdir",
                        f"{pltg_data}/fire_ts/",
                        "--prob_netcdf",
                        outnetcdf,
                    ]
                )

            if ltg_stats:
                solzen, _ = solar_angles(dt, lat=irlats, lon=irlons)
                ltg_utils.update_ltg_info(
                    ltg_dbase,
                    new_preds,
                    glmpatt,
                    dt,
                    solzen,
                    pixbuff=pixel_buff_and_stride[0],
                    fed=fed_data,
                )

            if len(timeseries):  # create a time series image
                pt_x, pt_y = projection(timeseries[1], timeseries[0])
                tsX, _ = utils.find_nearest(projx, pt_x)  # returns the ind and value
                tsY, _ = utils.find_nearest(projy, pt_y)
                ltg_utils.update_ltg_ts(
                    new_preds,
                    tsX,
                    tsY,
                    ts_pltg,
                    ts_pltg_radius,
                    ts_dts,
                    line_cter,
                    timeseries,
                    outdir,
                    fed_data,
                    ltg_obs,
                    ltg_obs_radius,
                    everytime=ts_every,
                )  # fed_data should be cropped

            if pickle_preds_labs:
                preds_container[line_cter - 1] = new_preds[::stride, ::stride]
                labs_container[line_cter - 1] = np.where(fed_accum_data >= 0.1, 1, 0)[
                    ::stride, ::stride
                ]  # hard-coded binarize threshold
                datetimes.append(dt)
                indices = [
                    i for i, s in enumerate(abi_filenames) if "C13_" in s
                ]  # assume we'll always use Ch13
                filenames.append(abi_filenames[indices[0]])

            if make_img:  # make an image
                write_image(scn, dt, new_preds, projx, projy, idict, meta, files_for_ir_only_cloud_phase, glm_aggregate_time)

        # clean up
        try:
            del scn, orig_scn
        except:
            logging.warning("Couldn't delete scn or orig_scn variables")
        if ltg_stats or timeseries:
            del fed_data
        if pickle_preds_labs:
            del fed_accum_data

        logging.info(
            "total time: " + "{:.3f}".format(time.time() - tstart) + " seconds"
        )

    if ltg_stats:
        logging.info("Computing and saving lightning scoring stats")
        ltg_score_dict = ltg_utils.score_ltg_pts(ltg_dbase)
        ltg_score_dict["ll_bbox"] = ll_bbox
        ltg_utils.write_ltg_stats(ltg_score_dict, ltg_dbase, outdir)
        logging.info("Done")

    if pickle_preds_labs:  # save all data at the end
        preds_labs_dict = {
            "preds": preds_container,
            "labels": labs_container,
            "files": filenames,
            "datetimes": datetimes,
            "stride": stride,
        }
        pickle.dump(preds_labs_dict, open(f"{outdir}/all_preds_labs.pkl", "wb"))
        logging.info(f"saved {outdir}/all_preds_labs.pkl")

    logging.info("Process ended.")


def set_num_threads(num_threads):
    if num_threads and num_threads > 0:
        os.environ['NUMEXPR_MAX_THREADS'] = f"{num_threads}"
        os.environ["OMP_NUM_THREADS"] = f"{num_threads}"
        os.environ["TF_NUM_INTRAOP_THREADS"] = f"{num_threads}"
        os.environ["TF_NUM_INTEROP_THREADS"] = f"{num_threads}"
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.set_soft_device_placement(enabled=False)


########################################################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Takes a list of datetimes or rabbitMQ real-time listing of GOES scans and \
                                                will make predictions of lightning in the next 60 min. Can make images, jsons, or netCDFs."
    )
    parser.add_argument(
        "filelist",
        help="A list of files. Each line should be a separate file with unique datetime.\
                                 If infinite is set, then this argument is the listened log output from amqpfind.",
        type=str,
    )
    parser.add_argument("model", help="CNN model used for predictions.", type=str)
    # parser.add_argument("sector", help="RadC, RadM1, RadM2, JP, FLDK, etc.", type=str)
    parser.add_argument(
        "-mc",
        "--model_config_file",
        help="This file contains mean and SD of channels (a pickle file). \
                      Default is to use the model_config.pkl in os.path.dirname(model)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-ddp",
        "--datadir_patt",
        help="Pattern for datadir. Assumes all needed L1b data files (one per channel) \
                                           will be in that pattern. Not used if --infinite=True. \
                                           Default = None",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sector",
        help="Default is None. Hard override sector instead of retrieving dynamically. Use should generally be avoided. \
              Supported sectors: GOES: RadC, RadF, RadM1, RadM2; Himawari: FLDK, JP",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-ss",
        "--sector_suffix",
        help="What to append to the --sector for writing files. E.g., 'AL' will generate "
        + "output files like so: {outdir}/RadC_AL/{filename} (assuming --sector is 'RadC')",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-gp",
        "--glmpatt",
        help='GLM file pattern. E.g.,: /ships19/grain/probsevere/lightning/%%Y/%%Y%%m%%d/GLM/goes_east/agg/%%Y%%m%%d-%%H%%M*.netcdf. For segments of FD scenes, outside of CONUS or PACUS, use (e.g.), "https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/%%Y/%%Y%%j/OR_GLM-L3-GLMF-M3_G16_e%%Y%%m%%d%%H%%M00.nc". Default=None.',
        type=str,
        default=None,
    )
    # FD goes_west example: /ships19/grain/jcintineo/GLM/goes_west/FD/%Y/%Y%m%d/5min/*e%Y%m%d%H%M*.nc
    parser.add_argument(
        "-gv",
        "--glmvar",
        help='GLM variable name. Default: "flash_extent_density". \
                                            Could use "Flash_extent_density_window" for UMD archive files.',
        default="flash_extent_density",
        type=str,
    )
    parser.add_argument(
        "-ep",
        "--eni_path",
        help='Path to global ENI point data. Generally used for Himawari. Default = "/ships19/grain/lightning/".',
        default="/ships19/grain/lightning/",
        type=str,
    )
    parser.add_argument(
        "-mp",
        "--mrms_patt",
        help="Pattern for MRMS grib2 files. {ch} is channel, such as 'Reflectivity_-10C'. Only used if"
        " model has MRMS predictors or if --ref10 is True (i.e., plotting radar data). Default = "
        "'/ships19/grain/probsevere/radar/%%Y/%%Y%%m%%d/grib2/{ch}/MRMS_{ch}_00.50_%%Y%%m%%d-%%H%%M*.grib2'",
        type=str,
        default="/ships19/grain/probsevere/radar/%Y/%Y%m%d/grib2/{ch}/MRMS_{ch}_00.50_%Y%m%d-%H%M*.grib2",
    )
    parser.add_argument(
        "-llbb",
        "--ll_bbox",
        help="Lat/lon bounding box of image. SWlon, NElon, SWlat, NElat. Default is entire image.",
        default=[-999],
        type=float,
        nargs=4,
    )
    parser.add_argument(
        "-ibb",
        "--ind_bbox",
        help="data ind bounding box of image. xmin, xmax, ymin, ymax. Note that ymin is towards top of the image."
        "Also, these are data indices with respect to 2-km channels (e.g., C13)...not projection coords. This is"
        "generally only used for RadF, where you might have space-looks. If you don't want or expect space-looks, just "
        "use --ll_bbox",
        default=[-999],
        type=int,
        nargs=4,
    )
    parser.add_argument(
        "-o",
        "--outdir",
        help="Root output directory. Default=$PWD/OUTPUT/",
        default=os.path.join(os.environ["PWD"], "OUTPUT/"),
        type=str,
    )
    parser.add_argument(
        "-ru",
        "--re_upload",
        help="Full path to re_upload executable. Will execute after json is written.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-x",
        "--extra_script",
        help="Full path to script to execute after netcdf creation (usu. for LDM send).",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-f", "--ftp", help="Post json files to this FTP path", default=None, type=str
    )
    parser.add_argument(
        "-ts",
        "--timeseries",
        help="A lat/lon pair to create a time series with. E.g., 43.82 -76.40."
        + " The optional 3rd arg is flag for if you want an image at every time. E.g., 43.82 -76.40 1",
        default=[],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--awips_nc",
        help="Write a netcdf of the LC probs in AWIPS-compatible format. 1 = w/o parallax correction; 2 = w/ parallax correction \
                      using a constant cloud height of --plax_hgt meters; 3 = save both plax-corrected and uncorrected grids. \
                      Default = 0 (i.e., don't write netcdf).",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--netcdf",
        help="Write a netcdf of the LC probs in native geostationary format. 1 = w/o parallax correction; 2 = w/ parallax correction \
                      using a constant cloud height of --plax_hgt meters; 3 = save both plax-corrected and uncorrected grids. \
                      Default = 0 (i.e., don't write netcdf).",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-fi",
        "--fire_event_ts",
        help="Post-processing that grabs probs and GLM flashes and associates them with \
                      current active fire events",
        action="store_true",
    )
    parser.add_argument(
        "--lightning_meteograms",
        help="Create meteograms of LightningCast probs and GLM and ENI flashes for \
                      static locations. Upload the updated meteogram data to a web database.",
        action="store_true",
    )
    parser.add_argument(
        "-gt", "--geotiff", help="Save as a geotiff", action="store_true"
    )
    parser.add_argument(
        "-mj", "--make_json", help="Save a json to outdir.", action="store_true"
    )
    parser.add_argument(
        "-mi",
        "--make_img",
        help="Make images. 1=DayLandCloud, 2=VIS, 3=DCC, 4=DCP, 5=sandwich, 0=none (default). \
                      Makes grayscale CH13 at night.",
        default=0,
        type=int,
    )
    parser.add_argument(
        "-pv",
        "--prob_values",
        help="Probability levels. Up to 8. Default= 0.1 0.25 0.5 0.75.",
        default=[0.1, 0.25, 0.5, 0.75],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "-c",
        "--colors",
        help="List of colors for prob. contours. Up to 8. Can be hex or standard. Default = blue cyan green magenta. \
                      Ignored if --contour_cmap is set.",
        type=str,
        nargs="+",
        default=["#0000FF", "#00FFFF", "#00FF00", "#CC00FF"],
    )
    parser.add_argument(
        "-ccm",
        "--contour_cmap",
        help="matplotlib colormap string for the probability contours.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-ppl",
        "--pickle_preds_labs",
        help="Save the predictions and labels in a pickle file for later evaluation.",
        action="store_true",
    )
    parser.add_argument(
        "-cm",
        "--county_map",
        help="Use county map in image creation.",
        action="store_true",
    )
    parser.add_argument(
        "--plot_points",
        help="Plot red dots for these points. E.g., -100.1 36.45 -98.22 44.87",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "-pe",
        "--plot_engln",
        help="Plot ENGLN flashes. This is True if imager is non-GOES.",
        action="store_true",
    )
    parser.add_argument(
        "-ref",
        "--ref10",
        help="Make images with MRMS Ref -10C overlaid. Assumes you want sat. RGB background and no contours.",
        action="store_true",
    )
    parser.add_argument(
        "-gr", "--grplacefile", help="Make GR placefile.", action="store_true"
    )
    parser.add_argument(
        "-ls",
        "--ltg_stats",
        help="Save/create ltg statistics for points in your domain. Typically used for lead-time.",
        action="store_true",
    )
    parser.add_argument(
        "-pbs",
        "--pixel_buff_and_stride",
        help="Used in conjuction with --ltg_stats. Provide two ints, correspoinding to \
                      the number of 2-km pixels to buffer a point and number of 2-km pixels to stride the image. Default = 5 40. If you want stats \
                      for every pixel without any buffer (generally not recommended), you would do --pixel_buff_and_stride 0 1",
        default=[5, 40],
        type=int,
        nargs=2,
    )
    parser.add_argument(
        "-ph",
        "--plax_hgt",
        help="Height above the earth's surface for parallax correction, in meters. \
                      We will use this constant height to do a parallax correction. Default = 0, which means no correction. \
                      If > 0, a parallax-corrected field will be written out in the netCDF (if --awips_nc ≥ 2 or --netcdf ≥ 2), \
                      or a parallax-corrected geoJSON will be written out in addition to the non-corrected geoJSON (if --make_json).",
        default=0,
        type=float,
    )
    parser.add_argument(
        "-i",
        "--infinite",
        help="Set for infinite processing. filelist arg is the watched file from ampqfind. --sector must also be set.",
        action="store_true",
    )

    parser.add_argument(
        "--num_threads",
        help="Limit the number of threads that lightningcast can use.",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    set_num_threads(args.num_threads)

    logDict = logging_def(logfile=None)
    # do this regardless of infinite/offline, so we get formatted stdout messages
    dictConfig(logDict)
    # send stderr to logger
    stderr_logger = logging.getLogger("STDERR")
    sl = StreamToLogger(stderr_logger, logging.ERROR)

    ll_bbox = None if (args.ll_bbox[0] == -999) else args.ll_bbox
    ind_bbox = None if (args.ind_bbox[0] == -999) else args.ind_bbox

    if args.model_config_file:
        model_config_file = args.model_config_file
    else:
        model_config_file = os.path.dirname(args.model) + "/model_config.pkl"
    assert os.path.isfile(model_config_file)
    assert os.path.isfile(args.model)

    predict_ltg(
        args.filelist,
        args.model,
        model_config_file,
        outdir=args.outdir,
        ll_bbox=ll_bbox,
        ind_bbox=ind_bbox,
        datadir_patt=args.datadir_patt,
        values=args.prob_values,
        glmpatt=args.glmpatt,
        glmvar=args.glmvar,
        eni_path=args.eni_path,
        mrms_patt=args.mrms_patt,
        sector=args.sector,
        sector_suffix=args.sector_suffix,
        infinite=args.infinite,
        re_upload=args.re_upload,
        make_img=args.make_img,
        colors=args.colors,
        contour_cmap=args.contour_cmap,
        county_map=args.county_map,
        plot_points=args.plot_points,
        lightning_meteograms=args.lightning_meteograms,
        geotiff=args.geotiff,
        awips_nc=args.awips_nc,
        netcdf=args.netcdf,
        fire_event_ts=args.fire_event_ts,
        make_json=args.make_json,
        pickle_preds_labs=args.pickle_preds_labs,
        ltg_stats=args.ltg_stats,
        pixel_buff_and_stride=args.pixel_buff_and_stride,
        timeseries=args.timeseries,
        grplacefile=args.grplacefile,
        plax_hgt=args.plax_hgt,
        extra_script=args.extra_script,
        ftp=args.ftp,
        ref10=args.ref10,
        plot_engln=args.plot_engln,
    )
