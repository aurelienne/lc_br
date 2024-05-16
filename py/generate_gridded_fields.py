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
from datetime import datetime
import csv
import os
import pandas
import generate_glm_stats as glm_stats
from multiprocessing import Pool

# Receive args
config_file = sys.argv[1]
files_list = sys.argv[2]

# Parameters from Config File
config = configparser.ConfigParser()
config.read(config_file)
glm_path = config['PATHS']['glm_path']
temp_path = config['PATHS']['temp_path']
output_path = config['PATHS']['output_path']
makeGLMgrids_path =  config['PATHS']['makeGLMgrids_path']


def make_grid_1min(filename):
    file1 = filename.strip()
    print(file1)
            
    # Make grid
    input_path = os.path.dirname(file1)
    prefix_file2 = os.path.basename(file1)[0:31]+"200"
    prefix_file3 = os.path.basename(file1)[0:31]+"400"
    try:
        file2 = glob.glob(input_path + '/' + prefix_file2 + '*')[0]
        file3 = glob.glob(input_path + '/' + prefix_file3 + '*')[0]
        os.system('python ' + makeGLMgrids_path + ' --fixed_grid --split_events --goes_position east --dx=2.0 --dy=2.0 --ctr_lon=-54.5 --ctr_lat=-14.5 --width="4500.0" --height="4500.0" -o '+ output_path + '/grids/{start_time:%Y/%b/%d}/{dataset_name} ' + file1 + ' ' + file2 + ' ' + file3)
    except:
        pass


if __name__ == "__main__":
    with open(files_list, 'r') as f:
        lines = f.readlines()

        files_list = [filename.strip() for filename in lines]
        
        with Pool(6) as p:
            p.map(make_grid_1min, files_list)
