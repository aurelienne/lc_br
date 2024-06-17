import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import glob
import os
from datetime import datetime, timedelta
import netCDF4
import xarray as xr
from pathlib import Path
from pickle import dump
import sys
from matplotlib import pyplot as plt

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    #return tf.train.Feature(
    #    bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    #)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def create_example(channels):
    feature = {
        "CH02": image_feature(channels["CH02"]),
        "CH05": image_feature(channels["CH05"]),
        "CH13": image_feature(channels["CH13"]),
        "CH15": image_feature(channels["CH15"]),
        "FED_accum_60min_2km": image_feature(channels["FED_accum_60min_2km"]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def select_truth_files(glm_filtered_list):
    f = open(glm_filtered_list, 'r')
    filenames = f.readlines()
    return filenames

def get_mean_std(standard_file):
    with open(standard_file, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            cols = line.split(',')
            print(cols[0])
            if cols[0] == 'glm':
                glm_mean = cols[1]
                glm_std  = cols[2]
            elif cols[0] == 'ch02':
                ch02_mean = cols[1]
                ch02_std  = cols[2]
            elif cols[0] == 'ch05':
                ch05_mean = cols[1]
                ch05_std  = cols[2]
            elif cols[0] == 'ch13':
                ch13_mean = cols[1]
                ch13_std  = cols[2]
            elif cols[0] == 'ch15':
                ch15_mean = cols[1]
                ch15_std  = cols[2]
            else:
                print('ERRO: Opcao invalida = ' + cols[0])
                sys.exit()
    return glm_mean, glm_std, ch02_mean, ch02_std, ch05_mean, ch05_std, \
           ch13_mean, ch13_std, ch15_mean, ch15_std

def standardize_dist(values, mean, std):
    """ Standardize distribution according to full dataset patterns (mean and standard deviation) """
    new_val = (values - float(mean)) * 1.0/float(std)
    return new_val

if __name__ == '__main__':

    #glm_path = '/ships22/grain/ajorge/data/glm_grids_60min/agg/'
    glm_path = '/ships22/grain/ajorge/data/glm_grids_60min_sum/agg/'
    #out_path = '/ships22/grain/ajorge/data/tfrecs_sumglm/test/%Y/%Y%m%d/'
    out_path = '/ships22/grain/ajorge/data/tfrecs_sumglm_FED5min/%Y/%Y%m%d/'

    abi_path = '/arcdata/goes/grb/goes16/%Y/%Y_%m_%d_%j/abi/L1b/RadF/'
    glm_pattern = '%Y%m%d-%H%M00.netcdf'
    glmvar = "flash_extent_density"
    
    #standard_file = '/home/ajorge/lc_br/data/train_mean_std_aure.csv'
    standard_file = '/home/ajorge/lc_br/data/train_mean_std_LCcontrol.csv'
    glm_filtered_list = '/home/ajorge/lc_br/data/glm_filtered_60sum_5fde.csv'

    sector = 'Full'
    #ltgthresh = 3 #should be in byte-scaled space.
              #We set a minimum threshold of 3 flashes, so that we
              #don't get any patches with very sparse lightning.


    NY,NX = 2100,2100 # grid size
    ny,nx = 700,700   # patch size
    Y_glm, X_glm = 0, 0
    Y_ch02, X_ch02 = 9503, 10709
    Y_ch05, X_ch05 = 4753, 5300
    Y_ch13, X_ch13 = 2365, 2681
    Y_ch15, X_ch15 = Y_ch13, X_ch13


    # Load Scalers
    glm_mean, glm_std, ch02_mean, ch02_std, ch05_mean, ch05_std, \
           ch13_mean, ch13_std, ch15_mean, ch15_std = get_mean_std(standard_file)

    # Select Truth Files
    truth_files = select_truth_files(glm_filtered_list)
    num_files = len(truth_files)

    for truth_file in truth_files:
        truth_file = os.path.basename(truth_file.strip())
        
        glmdt = datetime.strptime(truth_file, glm_pattern)
        abidt = glmdt - timedelta(hours=1)

        # Now get the truth/target data
        nc = netCDF4.Dataset(os.path.join(glm_path, truth_file))
        glm_agg = nc.variables[glmvar][:]
        nc.close()
        glm_agg = np.expand_dims(glm_agg[Y_glm:Y_glm+NY, X_glm:X_glm+NX], axis=-1)

        abidt_str = abidt.strftime('%Y%j%H%M')[:-1] + '0'

        #CH02
        file_location = glob.glob(abidt.strftime(os.path.join(abi_path, '*C02*_s' + abidt_str + '*.nc')))
        ds = xr.open_mfdataset(file_location,combine='nested',concat_dim='time')
        ch02 = ds['Rad'][0].data.compute()
        ch02 = ch02[Y_ch02:Y_ch02+NY*4, X_ch02:X_ch02+NX*4]
        ch02 = ch02 * ds['kappa0'].data[0] # Convert to reflectance
        ch02 = np.expand_dims(ch02, axis=-1)

        #CH05
        file_location = glob.glob(abidt.strftime(os.path.join(abi_path, '*C05*_s' + abidt_str + '*.nc')))
        ds = xr.open_mfdataset(file_location,combine='nested',concat_dim='time')
        ch05 = ds['Rad'][0].data.compute()
        ch05 = ch05[Y_ch05:Y_ch05+NY*2, X_ch05:X_ch05+NX*2]
        ch05 = ch05 * ds['kappa0'].data[0] # Convert to reflectance
        ch05 = np.expand_dims(ch05, axis=-1)
  
        #CH13
        file_location = glob.glob(abidt.strftime(os.path.join(abi_path, '*C13*_s' + abidt_str + '*.nc')))
        ds = xr.open_mfdataset(file_location,combine='nested',concat_dim='time')
        ch13 = ds['Rad'][0].data.compute()
        ch13 = ch13[Y_ch13:Y_ch13+NY, X_ch13:X_ch13+NX]
        # Convert to brightness temperature
        # First get some constants
        planck_fk1 = ds['planck_fk1'].data[0]; planck_fk2 = ds['planck_fk2'].data[0]; planck_bc1 = ds['planck_bc1'].data[0]; planck_bc2 = ds['planck_bc2'].data[0]
        ch13 = (planck_fk2 / (np.log((planck_fk1 / ch13) + 1)) - planck_bc1) / planck_bc2
        #plt.imshow(ch13)
        #plt.colorbar()
        #plt.show()
        ch13 = np.expand_dims(ch13, axis=-1)


        #CH15
        file_location = glob.glob(abidt.strftime(os.path.join(abi_path, '*C15*_s' + abidt_str + '*.nc')))
        ds = xr.open_mfdataset(file_location,combine='nested',concat_dim='time')
        ch15 = ds['Rad'][0].data.compute()
        ch15 = ch15[Y_ch15:Y_ch15+NY, X_ch15:X_ch15+NX]
        # Convert to brightness temperature
        # First get some constants
        planck_fk1 = ds['planck_fk1'].data[0]; planck_fk2 = ds['planck_fk2'].data[0]; planck_bc1 = ds['planck_bc1'].data[0]; planck_bc2 = ds['planck_bc2'].data[0]
        ch15 = (planck_fk2 / (np.log((planck_fk1 / ch15) + 1)) - planck_bc1) / planck_bc2
        ch15 = np.expand_dims(ch15, axis=-1)
        
        # Now make patches out of the data and write TFRecords.
        # Recall that GOES-16 ABI and GOES-16 GLM are on the same grids.
        # However, CH02 has 0.5-km resolution, CH05 has 1-km resolution, and
        # CH13 and CH15 have 2-km spatial resolution. That is why the patch sizes are different.
        for Y in range(0, NY, ny):
            for X in range(0, NX, nx):
                print(Y,X)

                outdir = abidt.strftime(out_path)
                Path(outdir).mkdir(parents=True, exist_ok=True)
                outfile = f"{outdir}/goes16_{abidt.strftime('%Y%m%d-%H%M%S')}_Y{Y}_X{X}.tfrec"

                with tf.io.TFRecordWriter(outfile) as writer:
                    channels = {}

                    channels['CH02'] = tf.io.serialize_tensor(tf.convert_to_tensor(standardize_dist(ch02[Y*4:(Y+ny)*4,X*4:(X+nx)*4], ch02_mean, ch02_std), dtype=tf.float32))
                    channels['CH05'] = tf.io.serialize_tensor(tf.convert_to_tensor(standardize_dist(ch05[Y*2:(Y+ny)*2,X*2:(X+nx)*2], ch05_mean, ch05_std), dtype=tf.float32))
                    channels['CH13'] = tf.io.serialize_tensor(tf.convert_to_tensor(standardize_dist(ch13[Y:Y+ny,X:X+nx], ch13_mean, ch13_std), dtype=tf.float32))
                    channels['CH15'] = tf.io.serialize_tensor(tf.convert_to_tensor(standardize_dist(ch15[Y:Y+ny,X:X+nx], ch15_mean, ch15_std), dtype=tf.float32))
                    #channels['FED_accum_60min_2km'] = tf.io.serialize_tensor(tf.convert_to_tensor(standardize_dist(glm_agg[Y:Y+ny,X:X+nx], glm_mean, glm_std), dtype=tf.float32)) 
                    channels['FED_accum_60min_2km'] = tf.io.serialize_tensor(tf.convert_to_tensor(glm_agg[Y:Y+ny,X:X+nx], dtype=tf.float32)) # There is no need to standardize GLM, as it will be binarized later

                    example = create_example(channels)
                    writer.write(example.SerializeToString())
                    del channels

        del ch02, ch05, ch13, ch15
