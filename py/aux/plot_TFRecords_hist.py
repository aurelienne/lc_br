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
from datetime import datetime
import plotly.express as px
import pandas as pd
import seaborn as sns

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

#def plot_samples_histogram(var):

def get_date_count(path):
    dates_list = []
    file_list = glob.glob(path+'/**/*.tfrec', recursive=True)
    for tfrec in sorted(file_list):
        filename = os.path.basename(tfrec)
        #month_file = datetime.strptime(filename.split('_')[1][4:6], '%m')
        dt_file = datetime.strptime(filename.split('_')[1][0:8], '%Y%m%d')
        dates_list.append(dt_file)

    return dates_list

def plot_samples_histogram():
    tr_dates = get_date_count(train_dir)
    val_dates = get_date_count(val_dir)
    ts_dates = get_date_count(test_dir)
    months = sorted(set(tr_dates + val_dates + ts_dates))
    months_names = [datetime.strftime(month, '%b') for month in months]

    data = {
    'Date': tr_dates + val_dates + ts_dates,
    'Dataset': ['Training (2020)'] * len(tr_dates) + ['Validation (2020)'] * len(val_dates) + ['Testing (2021)'] * len(ts_dates)
    }
    df = pd.DataFrame(data)
    df['Month'] = df['Date'].dt.strftime('%b')
    df['Month'] = pd.Categorical(df['Month'], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Sep', 'Oct', 'Nov', 'Dec'])

    sns.histplot(data=df, x='Month', hue='Dataset', multiple='stack', palette='muted', element='bars', discrete=False)
    plt.xlabel('Month')
    plt.ylabel('Number of Samples')
    ax = plt.gca()
    ax.grid(axis='y', alpha=0.5)
    plt.setp(ax.get_legend().get_texts(), fontsize='9')
    #plt.show()
    #plt.savefig(os.path.join(figs_path, 'dataset_samples_histogram.png')
    plt.savefig(os.path.join(figs_path, 'dataset_samples_histogram.eps'), format='eps')
    plt.savefig(os.path.join(figs_path, 'dataset_samples_histogram.png'))
    plt.close()

def plot_dates_histogram():
    tr_dates = list(set(get_date_count(train_dir)))
    val_dates = list(set(get_date_count(val_dir)))
    ts_dates = list(set(get_date_count(test_dir)))

    data = {
    'Date': tr_dates + val_dates + ts_dates,
    'Dataset': ['Training'] * len(tr_dates) + ['Validation'] * len(val_dates) + ['Testing'] * len(ts_dates)
    }
    df = pd.DataFrame(data)
    df['Month'] = df['Date'].dt.strftime('%b')
    df['Month'] = pd.Categorical(df['Month'], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Sep', 'Oct', 'Nov', 'Dec'])

    sns.histplot(data=df, x='Month', hue='Dataset', multiple='stack', palette='muted', element='bars', discrete=False)
    plt.xlabel('Month')
    plt.ylabel('Number of Days')
    ax = plt.gca()
    ax.grid(axis='y', alpha=0.5)
    plt.setp(ax.get_legend().get_texts(), fontsize='9')
    #plt.show()
    plt.savefig(os.path.join(figs_path, 'dataset_dates_histogram.eps'), format='eps')
    plt.savefig(os.path.join(figs_path, 'dataset_dates_histogram.png'))
    plt.close()

def plot_values_histogram(var):
    file_list = glob.glob(input_path+'/**/*.tfrec', recursive=True)
    var_ts = np.zeros((len(file_list), ny, nx))
    ts_idx = 0
    for tfrec in file_list:
        print(tfrec)
        raw_dataset = tf.data.TFRecordDataset(tfrec)
        parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

        for features in parsed_dataset.take(1):
            if var == 'CH02':
                stride = 4
            elif var == 'CH05':
                stride = 2
            else:        
                stride = 1
            data = features[var].numpy()[::stride, ::stride, :]
            if args.original_values and var != 'FED_accum_60min_2km':
                trans_data = inv_transform(data, var)
                var_ts[ts_idx] = np.squeeze(trans_data)
            else:
                var_ts[ts_idx] = np.squeeze(data)

        ts_idx += 1

    print(var_ts.flatten().size)
    plt.hist(var_ts.flatten(), weights=np.zeros_like(var_ts.flatten()) + 1. / (var_ts.flatten()).size)
    subset_name = os.path.basename(os.path.basename(input_path))
    if args.original_values:
        plt.title(var + ' - ' + subset_name) 
        format_ = 'original'
    else:
        plt.title(var + " (Normalized) - " + subset_name)
        format_ = 'norm'

    plt.ylabel('Relative Frequency')
    if var == 'CH02' or var == 'CH05':
        plt.xlabel('Reflectance')
    else:
        plt.xlabel('Brightness Temperature')

    #plt.show()
    plt.savefig(os.path.join(figs_path, 'hist_'+var+'-'+subset_name+'_'+format_+'.png'))
    plt.close()
    print("min, mean, max:")
    print(np.min(var_ts), np.mean(var_ts), np.max(var_ts))



if __name__ == '__main__':
    parse_desc = """Plot histograms from TensorflowRecord Files. """

    parser = argparse.ArgumentParser(description=parse_desc)
    
    parser.add_argument(
        "-tr", "--train_dir",
        help="Training Dataset Directory"
    )
    parser.add_argument(
        "-v", "--val_dir",
        help="Validation Dataset Directory"
    )
    parser.add_argument(
        "-ts", "--test_dir",
        help="Test Dataset Directory"
    )
    parser.add_argument(
        "-pd", "--plot_dates_hist",
        help="Plot Dates Histograms by Month",
        action="store_true",
    )
    parser.add_argument(
        "-ps", "--plot_samples_hist",
        help="Plot Histograms regarding number of samples/patches by Month",
        action="store_true",
    )
    parser.add_argument(
        "-pv", "--plot_values_hist",
        help="Plot Histograms by each Variable",
        action="store_true",
    )
    parser.add_argument(
        "-o", "--original_values",
        help="""Whether to plot histograms with original or normalized values. 
               If this option is not set, normalized values will be considered.""",
        action="store_true",
    )

    args = parser.parse_args()
    
    train_dir = args.train_dir
    val_dir = args.val_dir
    test_dir = args.test_dir

    if args.plot_values_hist:
        for var in ('CH02', 'CH05', 'CH13', 'CH15'):
            plot_values_histogram(var)

    if args.plot_dates_hist:
        plot_dates_histogram()

    if args.plot_samples_hist:
        plot_samples_histogram()
