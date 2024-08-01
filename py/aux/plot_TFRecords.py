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


tfrec = sys.argv[1]

NY, NX = 2100, 2100
ny, nx = 700, 700

figs_path = '/home/ajorge/lc_br/figs/'

state_borders = cfeature.STATES.with_scale("50m")

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
    raw_dataset = tf.data.TFRecordDataset(tfrec)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

    # Load Scalers
    #scaler_glm = joblib.load(os.path.join(data_path, 'scaler_glm.gz'))
    #scaler_ch02 = joblib.load(os.path.join(data_path, 'scaler_ch02.gz'))
    #scaler_ch05 = joblib.load(os.path.join(data_path, 'scaler_ch05.gz'))
    #scaler_ch13 = joblib.load(os.path.join(data_path, 'scaler_ch13.gz'))
    #scaler_ch15 = joblib.load(os.path.join(data_path, 'scaler_ch15.gz'))

    for features in parsed_dataset.take(1):

        fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(10, 6))
        for ii,ax in enumerate(axes.ravel()):
            ax.axis('off')
            if(ii==0):
                im = ax.imshow(inv_transform(features['CH02'].numpy(), 'ch02'), interpolation='none', cmap=plt.get_cmap('Greys_r'))
                plt.colorbar(im,ax=ax,orientation='horizontal',pad=0.03)
                ax.set_title('CH02 reflectance')
            elif(ii==1):
                im = ax.imshow(inv_transform(features['CH13'].numpy(), 'ch13'), interpolation='none', cmap=plt.get_cmap('inferno_r'))
                plt.colorbar(im,ax=ax,orientation='horizontal',pad=0.03)
                ax.set_title('CH13 brightness temperature')
            elif(ii==2):
                im = ax.imshow(inv_transform(features['CH05'].numpy(), 'ch05'), interpolation='none', cmap=plt.get_cmap('Greys_r'))
                plt.colorbar(im,ax=ax,orientation='horizontal',pad=0.03)
                ax.set_title('CH05 reflectance')
            elif(ii==3):
                im = ax.imshow(inv_transform(features['CH15'].numpy(), 'ch15'), interpolation='none', cmap=plt.get_cmap('inferno_r'))
                plt.colorbar(im,ax=ax,orientation='horizontal',pad=0.03)
                ax.set_title('CH15 brightness temperature')
            elif(ii==5):
                im = ax.imshow(features['FED_accum_60min_2km'].numpy(), interpolation='none')
                plt.colorbar(im,ax=ax,orientation='horizontal',pad=0.03)
                ax.set_title('Flash-extent density 60-min accumulation')

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


def plot_entire_grid(ch):
    tfrec_basename = os.path.basename(tfrec)
    tfrec_prefix = tfrec_basename.split('_')[1]
    tfrec_dir = os.path.dirname(tfrec)
    if ch == 'GLM':
        ch = 'FED_accum_60min_2km'
        var = 'glm'
        ch_cmap = 'viridis'
        var_label = 'Flash Extent Density'
    else:
        var = ch.lower()
        if ch == 'CH02' or ch == 'CH05':
            ch_cmap = 'Greys_r'
            var_label = 'Reflectance'
        else:
            ch_cmap = 'inferno_r'
            var_label = 'Brightness Temperature'

    if var == 'glm' or var == 'ch13' or var == 'ch15':
        m = 1
    elif var == 'ch02':
        m = 4
    elif var == 'ch05':
        m = 2

    NY, NX = 2100, 2100
    ny, nx = 700, 700

    xx, yy, geoproj = get_proj(0, 0, NX, NY)
    #fig,axes = plt.subplots(nrows=3,ncols=3, gridspec_kw = {'wspace':0, 'hspace':0},figsize=(9,9))
    #axes.set_extent([xx.min(), xx.max(), yy.min(), yy.max()], crs=geoproj)

    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_axes([0, 0, 0.95, 1], projection=geoproj)
    extent = [xx.min(), xx.max(), yy.min(), yy.max()]
    ax.set_extent(extent, crs=geoproj)

    full_grid = np.zeros((NY*m, NX*m))
    x_coords = np.linspace(xx.min(), xx.max(), NX*m)
    y_coords = np.linspace(yy.min(), yy.max(), NY*m)

    l = 0
    for Y in range(0, NY, ny):
        c = 0
        for X in range(0, NX, nx):
            #filename = glob.glob(os.path.join(tfrec_dir, '*' + tfrec_prefix + '_Y' + str(Y).zfill(4) + '_X'+str(X).zfill(4) + '*'))[0]
            filename = glob.glob(os.path.join(tfrec_dir, '*' + tfrec_prefix + '_Y' + str(Y) + '_X'+str(X) + '*'))[0]
            #print(filename)
            raw_dataset = tf.data.TFRecordDataset(filename)
            parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
            for features in parsed_dataset.take(1):
                #print(var)
                if var != 'glm':
                    data = inv_transform(features[ch].numpy(), var)
                else:
                    data = features[ch].numpy()
                    data = np.where(data>=1, 1, 0)


            full_grid[Y*m:Y*m+ny*m,X*m:X*m+nx*m] = np.squeeze(data)

            plt.axvline(x=x_coords[X*m], color='yellow')
            c = c + 1

        plt.axhline(y=y_coords[Y*m], color='yellow')
        l = l + 1

    im = plt.imshow(full_grid, interpolation=None, cmap=plt.get_cmap(ch_cmap), zorder=1, extent=extent)
    ax.add_feature(state_borders, edgecolor='white', linewidth=1.0, facecolor="none", zorder=20)
    ax.coastlines(color='white', linewidth=1.0, zorder=30)
    #ax.set_xticks(np.linspace(xx.min(), xx.max(), 10))
    #ax.set_yticks(np.linspace(yy.min(), yy.max(), 10))

    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.set_label(var_label)
    #plt.suptitle(ch)
    #plt.tight_layout()

    plt.savefig(os.path.join(figs_path, 'tfrec_fullgrid_' + ch + '.eps'), format='eps')
    #plt.show()
    plt.close()


if __name__ == '__main__':
    plot_single_patch()
   
    for var in ('GLM', 'CH02', 'CH05', 'CH13', 'CH15'):
        print(var)
        plot_entire_grid(var)
