import sys
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import basemap
import matplotlib.gridspec as gridspec
import os
import glob
import csv
import argparse

#input_path = sys.argv[1]
#try:
#    rejected_path = sys.argv[2]
#except:
#    pass

NY, NX = 2100, 2100
ny, nx = 700, 700

figs_path = '/home/ajorge/lc_br/figs/'
logs_path = '/home/ajorge/lc_br/logs/'

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
            if cols[0] == var.lower():
                var_mean = float(cols[1])
                var_std  = float(cols[2])

    new_val = (values * var_std) + var_mean
    return new_val


def discard_nan_patches():
    csvfile = open(os.path.join(logs_path, 'patches_withNAN.csv'), 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['filename', 'nan_count', 'inf_count'])
    for tfrec in sorted(glob.glob(input_path+'/**/*.tfrec', recursive=True)):
        raw_dataset = tf.data.TFRecordDataset(tfrec)
        parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

        for features in parsed_dataset.take(1):
            for var in ('FED_accum_60min_2km', 'CH02', 'CH05', 'CH13', 'CH15'):
                var_data = features[var].numpy()
                var_nan = np.count_nonzero(np.isnan(var_data))
                var_inf = np.count_nonzero(np.isinf(var_data))
                if var_nan > 0 or var_inf > 0:
                    csvwriter.writerow([tfrec, var_nan, var_inf])
                    os.rename(tfrec, os.path.join(rejected_path, os.path.basename(tfrec)))
                    print(tfrec, os.path.join(rejected_path, os.path.basename(tfrec)))
                    break


if __name__ == '__main__':
    parse_desc = """Check TensorflowRecord Files. It can remove patches with nan values."""

    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(
        "-d", "--discard_nan",
        help="Remove patches with nan values from input path. It is required to inform input_path and rejected_path as well.",
        action="store_true",
    )
    parser.add_argument(
        "-i", "--input_dir",
        help="Input Directory from which to read the TFRecord files."
    )
    parser.add_argument(
        "-r", "--rejected_dir",
        help="Output Directory to move the rejected files."
    )

    args = parser.parse_args()
    
    if args.input_dir:
        input_path = args.input_dir
    if args.rejected_dir:
        rejected_path = args.rejected_dir

    if args.discard_nan:
        discard_nan_patches()
