import sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import basemap
import matplotlib.gridspec as gridspec
import os
import glob

tfrec = sys.argv[1]

NY, NX = 2100, 2100
ny, nx = 700, 700

figs_path = '/home/ajorge/lc_br/figs/'

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


def plot_entire_grid(ch):
    tfrec_basename = os.path.basename(tfrec)
    tfrec_prefix = tfrec_basename.split('_')[1]
    tfrec_dir = os.path.dirname(tfrec)
    if ch == 'GLM':
        ch = 'FED_accum_60min_2km'
        var = 'glm'
        ch_cmap = 'viridis'
    else:
        var = ch.lower()
        if ch == 'CH02' or ch == 'CH05':
            ch_cmap = 'Greys_r'
        else:
            ch_cmap = 'inferno_r'


    NY, NX = 2100, 2100
    ny, nx = 700, 700

    fig,axes = plt.subplots(nrows=3,ncols=3, gridspec_kw = {'wspace':0, 'hspace':0},figsize=(9,9))

    l = 0
    for Y in range(0, NY, ny):
        c = 0
        for X in range(0, NX, nx):
            #filename = glob.glob(os.path.join(tfrec_dir, '*' + tfrec_prefix + '_Y' + str(Y).zfill(4) + '_X'+str(X).zfill(4) + '*'))[0]
            filename = glob.glob(os.path.join(tfrec_dir, '*' + tfrec_prefix + '_Y' + str(Y) + '_X'+str(X) + '*'))[0]
            print(filename)
            raw_dataset = tf.data.TFRecordDataset(filename)
            parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
            for features in parsed_dataset.take(1):
                print(var)
                if var != 'glm':
                    data = inv_transform(features[ch].numpy(), var)
                else:
                    data = features[ch].numpy()
                    data = np.where(data>=1, 1, 0)

                print(np.max(data))
                print(np.sum(np.nonzero(data)))

            im = axes[l, c].imshow(data, interpolation=None, cmap=plt.get_cmap(ch_cmap))

            axes[l, c].get_xaxis().set_visible(False)
            axes[l, c].get_yaxis().set_visible(False)
            axes[l, c].set_aspect('equal')
            c = c + 1

        l = l + 1

    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.suptitle(ch)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.tight_layout()
    plt.savefig(os.path.join(figs_path, 'tfrec_fullgrid_' + ch + '.png'))
    plt.close()


if __name__ == '__main__':
    plot_single_patch()
   
    for var in ('GLM', 'CH02', 'CH05', 'CH13', 'CH15'):
        print(var)
        plot_entire_grid(var)
