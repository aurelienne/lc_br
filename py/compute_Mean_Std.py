import tensorflow as tf
from sklearn import preprocessing
import joblib
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import netCDF4
import xarray as xr
from pathlib import Path
from pickle import dump
import sys
import csv
import random


def select_truth_files():
    selected_glm = '/home/ajorge/data/glm_filtered.csv'
    f = open(selected_glm, 'r')
    filenames = f.readlines()
    return filenames


if __name__ == '__main__':

    var = sys.argv[1]
    print(var)

    # Indexes to clip channels (calculated using code: ~/py/aux/test2.py
    if var == 'glm':
        Y_ch, X_ch = 0,0
        NY,NX = 2100,2100 # grid size
    elif var == 'ch02':
        Y_ch, X_ch = 9503, 10709
        NY,NX = 2100*4,2100*4 # grid size
    elif var == 'ch05':
        Y_ch, X_ch = 4753, 5300
        NY,NX = 2100*2,2100*2 # grid size
    elif var == 'ch13' or var == 'ch15':
        Y_ch, X_ch = 2365, 2681
        NY,NX = 2100,2100 # grid size
    else:
        print('ERROR!! Invalid argument: Var options = (ch02/ch05/ch13/ch15/glm)')
        sys.exit()


    data_path = '/home/ajorge/data/'
    glm_path = '/ships22/grain/ajorge/data/glm_grids_60min/agg/'
    glm_pattern = '%Y%m%d-%H%M00.netcdf'
    abi_path = '/arcdata/goes/grb/goes16/%Y/%Y_%m_%d_%j/abi/L1b/RadF/'
    glmvar = "flash_extent_density"

    sector = 'Full'
    #ltgthresh = 3 #should be in byte-scaled space.
              #We set a minimum threshold of 3 flashes, so that we
              #don't get any patches with very sparse lightning.

    out_path = '/ships22/grain/ajorge/data/tfrecs/%Y/%Y%m%d/'


    #truth_files = np.sort(glob.glob(glm_path + '/*.netcdf'))
    truth_files = select_truth_files()
    num_files = len(truth_files)
    ts_len = int(num_files*0.25) # Enough samples to get a representative distribution
    print('ts_len='+str(ts_len))
    #ts_len = 1   ## For tests only

    # Get Sample
    sample_files = random.sample(truth_files, ts_len)
    print(sample_files)

    var_train = np.zeros((ts_len, NY, NX), dtype=np.float32)
    print(var_train.size)
    print(var_train.itemsize)
    print(var_train.size*var_train.itemsize)
    # For each target/truth file, find the corresponding ABI files and create
    # the TFRecord for that time
    t = 0
    for sample_file in sample_files:
        sample_file = sample_file.strip()
        print(sample_file)

        #if t > 5:
        if t >= ts_len: # Enough samples to get a representative distribution
            break

        if var == 'glm':
            # Now get the truth/target data
            nc = netCDF4.Dataset(os.path.join(glm_path, sample_file))
            nc_var = nc.variables[glmvar][:]
            nc.close()
            var_train[t,:,:] = nc_var[Y_ch:NY, X_ch:NX] 
            del nc, nc_var
            print(np.mean(var_train), np.std(var_train))
        else:
            # This datetime is the *end* dt of the accumulation. We need to subtract an hour
            # to get the correct ABI time.
            glmdt = datetime.strptime(sample_file, glm_pattern)
            print(glmdt)
            abidt = glmdt - timedelta(hours=1)
            abidt_str = abidt.strftime('%Y%j%H%M')[:-1] + '0'

            ch_num = var.replace('ch','')    
            file_location = glob.glob(abidt.strftime(os.path.join(abi_path, '*C'+ch_num+'*_s' + abidt_str + '*.nc')))
            print(file_location)
            ds = xr.open_mfdataset(file_location,combine='nested',concat_dim='time')
            ch = ds['Rad'][0].data.compute()
            ch = ch[Y_ch:Y_ch+NY, X_ch:X_ch+NX]

            if var == 'ch02' or var == 'ch05':
                kappa = ds['kappa0'].data[0]
                var_train[t] = ch * kappa # Convert to reflectance
                del ch, ds
            else: # Channels 13 and 15
                # Convert to brightness temperature
                # First get some constants
                planck_fk1 = ds['planck_fk1'].data[0]
                print(planck_fk1)
                planck_fk2 = ds['planck_fk2'].data[0]
                print(planck_fk2)
                planck_bc1 = ds['planck_bc1'].data[0]
                print(planck_bc1)
                planck_bc2 = ds['planck_bc2'].data[0]
                print(planck_bc2)
                var_train[t,::] = (planck_fk2 / (np.log((planck_fk1 / ch) + 1)) - planck_bc1) / planck_bc2
                del ch, ds

        t = t + 1


    #scaler = preprocessing.StandardScaler().fit(var_train.reshape([1,-1]))

    # Dump Scalers
    #dump(scaler, open(os.path.join(data_path, 'scaler_'+var+'.pkl'), 'wb'), protocol=4)
    #joblib.dump(scaler, os.path.join(data_path, 'scaler_'+var+'.gz'))

    # Compute Mean and Standard Deviation 
    var_mean = np.nanmean(var_train)
    var_std = np.nanstd(var_train)
    print(var_mean, var_std)
    
    stats_file = '/home/ajorge/data/train_mean_std.csv'
    if os.path.isfile(stats_file):
        with open(stats_file, 'a') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow([var, var_mean, var_std])
    else:
        with open(stats_file, 'w') as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(['Based on ' + str(num_files) + ' files - At ' + str(datetime.now()),None,None])
            csvwriter.writerow(['var', 'mean', 'std'])
            csvwriter.writerow([var, var_mean, var_std]) 

