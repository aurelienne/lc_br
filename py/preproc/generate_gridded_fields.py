# conda activate glmval #

import netCDF4
import xarray as xr
import sys
import numpy as np
import configparser
from glmtools.io.glm import GLMDataset
from matplotlib import pyplot as plt
import plotly.express as px
import glob
from datetime import datetime, timedelta
import csv
import os
import pandas
#import generate_glm_stats as glm_stats
from multiprocessing import Pool

# Receive args
config_file = "config.ini"
start_dt_str = sys.argv[1]
end_dt_str = sys.argv[2]
delta_min = sys.argv[3]

# Parameters from Config File
config = configparser.ConfigParser()
config.read(config_file)
glm_path = config['PATH']['glm_path']
temp_path = config['PATH']['temp_path']
glm_1min_path = config['PATH']['glm_1min_path']
makeGLMgrids_path =  config['PATH']['makeGLMgrids_path']
ctr_lon = config['GEO']['ctr_lon']
ctr_lat = config['GEO']['ctr_lat']
width_km = config['GEO']['width_km']
height_km = config['GEO']['height_km']


def make_grid_deltamin(dt_str):
    dt = datetime.strptime(dt_str, '%Y%j%H%M000')
    end_dt = dt + timedelta(minutes=5)

    glm_path_dt = glm_path.replace('<YYYY>', YYYY)
    glm_path_dt = glm_path_dt.replace('<mm>', mm)
    glm_path_dt = glm_path_dt.replace('<dd>', dd)
    glm_path_dt = glm_path_dt.replace('<jd>', jd)
            
    files_list = ' '
    while dt < end_dt:
        dt_str = datetime.strftime(dt, '%Y%j%H%M000')
        file1 = glob.glob(glm_path_dt + 'OR_GLM-L2-LCFA_G16_s' + dt_str + '*.nc')[0]
        prefix_file2 = os.path.basename(file1)[0:31]+"200"
        prefix_file3 = os.path.basename(file1)[0:31]+"400"
        input_path = os.path.dirname(file1)
        file2 = glob.glob(input_path + '/' + prefix_file2 + '*')[0]
        file3 = glob.glob(input_path + '/' + prefix_file3 + '*')[0]
        files_list = files_list + ' ' + file1 + ' ' + file2 + ' ' + file3
        dt = dt + timedelta(minutes=1)

    # Make grid
    #try:
    os.system(f'python {makeGLMgrids_path} --fixed_grid --split_events --goes_position east --dx=2.0 --dy=2.0 --ctr_lon={ctr_lon} --ctr_lat={ctr_lat} --width={width_km} --height={height_km} -o {glm_1min_path}/grids/' + '{start_time:%Y/%b/%d}/{dataset_name}' + files_list)
    #except:
    #    pass


if __name__ == "__main__":
    start_dt = datetime.strptime(start_dt_str, '%Y%m%d%H%M')
    end_dt = datetime.strptime(end_dt_str, '%Y%m%d%H%M')
    dt = start_dt
    dt_list = []
    while dt <= end_dt:
        YYYY = datetime.strftime(dt, '%Y')
        mm = datetime.strftime(dt, '%m')
        dd = datetime.strftime(dt, '%d')
        jd = datetime.strftime(dt, '%j')
        dt_str = datetime.strftime(dt, '%Y%j%H%M000') 
        dt_list.append(dt_str)

        dt = dt + timedelta(minutes=5)

    with Pool(6) as p:
        p.map(make_grid_deltamin, dt_list)
