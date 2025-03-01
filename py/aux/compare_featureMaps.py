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
import aux
import pyproj
from skimage.metrics import structural_similarity
from datetime import datetime
import pandas as  pd


NY, NX = 2100, 2100
ny, nx = 700, 700
nyf, nxf = 175, 175
nfilters = 256

figs_path = '/home/ajorge/lc_br/figs/'

state_borders = cfeature.STATES.with_scale("50m")

def parse_tfrecord_fn(example):
    feature_description = {
        "input": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)

    example = {}

    image = tf.io.parse_tensor(features["input"], tf.float32)
    image = tf.reshape(image, [nyf,nxf,nfilters])
    example['input'] = image

    image = tf.io.parse_tensor(features["target"], tf.float32)
    image = tf.reshape(image, [ny,nx])
    example['target'] = image

    return example


def compare(tfrec1, tfrec2):
    raw_dataset = tf.data.TFRecordDataset(tfrec1)
    parsed_dataset1 = raw_dataset.map(parse_tfrecord_fn)

    raw_dataset2 = tf.data.TFRecordDataset(tfrec2)
    parsed_dataset2 = raw_dataset2.map(parse_tfrecord_fn)

    l = 0
    features1 = [x for x in parsed_dataset1.take(1)][0]
    features2 = [x for x in parsed_dataset2.take(1)][0]
    nfilters = features1['input'].numpy().shape[2]
    ssim, nonzero1, nonzero2 = 0, 0, 0
    for id_filter in range(nfilters):
        ssim = structural_similarity(features1['input'].numpy()[:,:,id_filter], features2['input'].numpy()[:,:,id_filter])
        print(ssim)
        ssim = ssim + ssim 
        nonzero1 = nonzero1 + np.count_nonzero(features1['input'].numpy()[:,:,id_filter])
        nonzero2 = nonzero2 + np.count_nonzero(features2['input'].numpy()[:,:,id_filter])
    return ssim/nfilters


def get_proj(X, Y, NX, NY):
    georef_file = '/home/ajorge/lc_br/data/GLM_BR_finaldomain.nc'
    nc = netCDF4.Dataset(georef_file, "r")
    gip = nc.variables["goes_imager_projection"]
    x = nc.variables["x"] #[Y:Y+NY, X:X+NX]
    y = nc.variables["y"] #[Y:Y+NY, X:X+NX]
    print(x.shape, y.shape)
    x *= gip.perspective_point_height
    y *= gip.perspective_point_height

    geoproj = ccrs.Geostationary(
        central_longitude=gip.longitude_of_projection_origin,
        sweep_axis="x",
        satellite_height=gip.perspective_point_height,
    )

    return x, y, geoproj


def compute_stats(list1, list2):
    with open(list1, 'r') as f:
        filenames1 = f.readlines()
    with open(list2, 'r') as f:
        filenames2 = f.readlines()

    df = pd.DataFrame(columns=('ssim',))
    for i in range(len(filenames1)):
        tfrec1 = filenames1[i]
        tfrec2 = filenames2[i]
        fname = os.path.basename(filenames1[i])
        dt_str = fname.split('_')[3]
        dt = datetime.strptime(dt_str, '%Y%m%d-%H%M%S')
        ssim = compare(tfrec1.strip(), tfrec2.strip())
        df[dt] = [ssim]
    df["Month"] = df.index.strftime("%b")
    print(df)

if __name__ == '__main__':
    tfrec1_list = sys.argv[1]
    tfrec2_list = sys.argv[2]
   
    compute_stats(tfrec1_list, tfrec2_list)
