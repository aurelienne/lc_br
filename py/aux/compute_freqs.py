import sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import basemap
import matplotlib.gridspec as gridspec
import os
import glob
import netCDF4
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


NY, NX = 2100, 2100
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

def compute_monthly_freqs(log_file):
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.split(',')
            dt = cols[0]
            freq = cols[1]
    """
    df = pd.read_csv(log_file, delimiter=',')
    df['date'] = pd.to_datetime(df['dt'])
    grouped = df.groupby([df['date'].dt.month]).mean()
    print(grouped)

#def compute_daily_freqs():

def calc_file_freqs():
    f = open(out_log, 'w')

    filenames = glob.glob(os.path.join(tfrec_dir, '*/*.tfrec'))
    total_pixels = nx*ny
    freqs = []
    for filename in filenames:
        print(filename)
        basename_ = os.path.basename(filename)
        dt = basename_.split('_')[1]
        raw_dataset = tf.data.TFRecordDataset(filename)
        parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
        for features in parsed_dataset.take(1):
            data = features['FED_accum_60min_2km'].numpy()
            data = np.where(data>=1, 1, 0)
            count = np.sum(data)
            freq = count/total_pixels
            print(f"{dt},{freq}\n")
            f.write(f"{dt},{freq}\n")
            #freqs.append(freq)
    #median = np.median(np.array(freqs))
    #mean = np.mean(np.array(freqs))
    #print(f"Median = {median}")
    #print(f"Mean = {mean}")
    f.close()
if __name__ == '__main__':
    #tfrec_dir = '/ships22/grain/ajorge/data/tfrecs_sumglm/train/2020/'
    tfrec_dir = '/ships22/grain/ajorge/data/tfrecs_sumglm/test/2021/'
    out_log = '/home/ajorge/lc_br/data/test_freqs.log'

    #calc_file_freqs()
    compute_monthly_freqs(out_log)

