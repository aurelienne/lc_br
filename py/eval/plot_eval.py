import os
import pickle
import sys
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import glob
import netCDF4
import xarray as xr


tfrecs_path = '/ships22/grain/ajorge/data/tfrecs_sumglm/test/'
figs_path = '/home/ajorge/lc_br/figs/'
ny, nx = 700, 700

def parse_tfrecord_fn(example):
    feature_description = {
        "CH02": tf.io.FixedLenFeature([], tf.string),
        "CH05": tf.io.FixedLenFeature([], tf.string),
        "CH13": tf.io.FixedLenFeature([], tf.string),
        "CH15": tf.io.FixedLenFeature([], tf.string),
        "FED_accum_60min_2km": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)

    example = {}

    image = tf.io.parse_tensor(features["CH02"], tf.float32)
    image = tf.reshape(image, [ny*4,nx*4,1])
    example['CH02'] = image

    image = tf.io.parse_tensor(features["CH05"], tf.float32)
    image = tf.reshape(image, [ny*2,nx*2,1])
    example['CH05'] = image

    image = tf.io.parse_tensor(features["CH13"], tf.float32)
    image = tf.reshape(image, [ny,nx,1])
    example['CH13'] = image

    image = tf.io.parse_tensor(features["CH15"], tf.float32)
    image = tf.reshape(image, [ny,nx,1])
    example['CH15'] = image

    image = tf.io.parse_tensor(features["FED_accum_60min_2km"], tf.float32)
    image = tf.reshape(image, [ny,nx,1])
    example['FED_accum_60min_2km'] = image

    return example

def get_tfrec(dt_ref, var, y, x):
    yyyy = str(dt_ref.year)
    mm = datetime.strftime(dt_ref, "%m")
    dd = str(dt_ref.day)
    tfrecs_path_dt = os.path.join(tfrecs_path, yyyy, yyyy+mm+dd)
    filename = glob.glob(tfrecs_path_dt + '/*' + 
                         datetime.strftime(dt_ref, '%Y%m%d-%H%M' + 
                                           '*Y'+str(y)+'_X'+str(x)+'.tfrec'))[0]
    raw_dataset = tf.data.TFRecordDataset(os.path.join(tfrecs_path_dt, filename))
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
    for features in parsed_dataset.take(1):
        var_data = features[var]
    return var_data

def get_glm_agg(dt_ref, period, y, x):
    dt_ref = dt_ref + timedelta(minutes=period)
    yyyy = dt_ref.year
    month = dt_ref.strftime("%b")
    dd = dt_ref.day
    jd = dt_ref.strftime("%j")
    if period == 0:
        glm_path = f'/ships22/grain/ajorge/data/glm_grids_1min/{yyyy}/{month}/{dd}/'
        dt_ref_str = datetime.strftime(dt_ref, '%Y%j%H%M')
    else:
        glm_path = f'/ships22/grain/ajorge/data/glm_grids_{period}min_sum/agg/'
        dt_ref_str = datetime.strftime(dt_ref, '%Y%m%d-%H%M')
    filename = glob.glob(glm_path + '*' + dt_ref_str + '*')[0]
    ds = xr.open_dataset(filename)
    vals = ds.flash_extent_density.data[y:y+ny, x:x+nx] # Get patch 
    return vals

if __name__ == "__main__":

    pickle_file = str(sys.argv[1])
    dt_req = sys.argv[2] #YYYYmmddHH"
    p = pickle.load(open(pickle_file, 'rb'))
    stride = p['stride']
    nx_s = int(nx / stride)
    ny_s = int(ny / stride)

    i = 0
    for filename in p['files']:
        dt_ref = p['datetimes'][i]
        dt_hour = datetime.strftime(dt_ref, '%Y%m%d%H')
        if dt_hour != dt_req:
            i = i + 1
            continue
        ystr = filename.split('_')[2]
        y = int(ystr.replace("Y",""))
        xstr = filename.split('_')[3]
        x = int(xstr.replace("X","").replace(".tfrec",""))
        strideY = p['strideY'][i]
        strideX = p['strideX'][i]

        fig = plt.figure(figsize=(14, 8))
        plt.suptitle('CH13 + Accumulated GLM')

        # Get GLM aggregations
        l, pos = 1, 1
        periods = [0, 10, 20, 30, 40, 50, 60]
        for period in periods:
            glm_acc = get_glm_agg(dt_ref, period, x, y)
            ax = fig.add_subplot(2, 4, pos)

            # Get ABI Channels from TFRecords
            ch_dt = dt_ref + timedelta(minutes=period)
            ch02 = get_tfrec(ch_dt, 'CH02', y, x)
            ch13 = get_tfrec(ch_dt, 'CH13', y, x)

            ax.imshow(ch13, zorder=0, alpha=0.5)
            ax.contour(glm_acc, zorder=10, levels=[1, 10])
            ax.set_title(ch_dt)
            pos += 1
            if pos == 4:
                l += 1
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figs_path, 'agg60_steps.png'))
        plt.close()

        fig = plt.figure(figsize=(14, 8))
        plt.suptitle('Inputs, Target and Prediction')
        ch02 = get_tfrec(dt_ref, 'CH02', y, x)
        ch05 = get_tfrec(dt_ref, 'CH05', y, x)
        ch13 = get_tfrec(dt_ref, 'CH13', y, x)
        ch15 = get_tfrec(dt_ref, 'CH15', y, x)
        ax = fig.add_subplot(2, 3, 1)
        ax.imshow(ch02)
        ax.set_title('CH02')
        ax = fig.add_subplot(2, 3, 2)
        ax.imshow(ch05)
        ax.set_title('CH05')
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(ch13)
        ax.set_title('CH13')
        ax = fig.add_subplot(2, 3, 4)
        ax.imshow(ch15)
        ax.set_title('CH15')
        ax = fig.add_subplot(2, 3, 5)
        ax.imshow(p['labels'][i])
        ax.set_title('Target')
        ax = fig.add_subplot(2, 3, 6)
        ax.imshow(p['preds'][i])
        ax.set_title('Prediction')
        #plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(figs_path, 'inputs_target_pred.png'))
        break
        #i += 1

