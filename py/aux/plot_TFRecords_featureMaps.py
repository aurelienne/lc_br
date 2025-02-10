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


NY, NX = 2100, 2100
ny, nx = 700, 700

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
    image = tf.reshape(image, [ny,nx])
    example['input'] = image

    image = tf.io.parse_tensor(features["target"], tf.float32)
    image = tf.reshape(image, [ny,nx])
    example['target'] = image

    return example

def inv_transform(values, var):
    """ Invert Standardization of distribution according to full dataset patterns (mean and standard deviation) """

    #with open('/home/ajorge/lc_br/data/train_mean_std_aure.csv', 'r') as f:
    with open('/home/ajorge/lc_br/data/train_mean_std_LCcontrol.csv', 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            cols = line.split(',')
            if cols[0] == var:
                var_mean = float(cols[1])
                var_std  = float(cols[2])

    new_val = (values * var_std) + var_mean
    return new_val


def plot_single_patch():
    raw_dataset = tf.data.TFRecordDataset(tfrec1)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    for features in parsed_dataset.take(1):

        fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(6, 12))
        #for ii,ax in enumerate(axes.ravel()):
        #    ax.axis('off')
        im = axes[l,0].imshow(features['input'].numpy(), interpolation='none', cmap=plt.get_cmap('Greys_r'))
        #im = axes[l,1].imshow(features['input'].numpy(), interpolation='none', cmap=plt.get_cmap('Greys_r'))
        l = l + 1

    plt.tight_layout()
    plt.savefig(os.path.join(figs_path, 'patch_channels.png'))
    plt.clf()
    plt.close()

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



if __name__ == '__main__':
    tfrec1 = sys.argv[1]
    tfrec2 = sys.argv[2]
    plot_single_patch()
   
