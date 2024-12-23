# from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import sys

import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.dirname(current_dir))

import argparse
import glob
from datetime import datetime, timedelta, timezone
import time
import pickle
import matplotlib.pyplot as plt
import collections
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from pathlib import Path

parse_desc = """Feature extraction from LightningCast to a new dataset"""
parser = argparse.ArgumentParser(description=parse_desc)
parser.add_argument(
    "-m",
    "--modeldir",
    help="Directory containing trained CNN model and config.",
    type=str,
    nargs=1,
)
parser.add_argument(
    "-o",
    "--outdir",
    help="Output directory. Default={PWD}",
    nargs=1,
    type=str,
)
parser.add_argument(
    "-t", "--train_datadir", help="Dataset directory used for fine-tuning the model with training dataset",
    type=str,
    nargs=1,
    required=True,
)
parser.add_argument(
    "-v", "--val_datadir", help="Dataset directory used for fine-tuning the model with validation dataset",
    type=str,
    nargs=1,
    required=True,
)

args = parser.parse_args()

modeldir = args.modeldir[0]
train_dir = args.train_datadir[0]
val_dir = args.val_datadir[0]
if not args.outdir:
    outdir = os.environ["PWD"] + "/"
else:
    outdir = args.outdir[0]

config = pickle.load(open(os.path.join(modeldir, "model_config.pkl"), "rb"))
model_file = f"{modeldir}/fit_conv_model.h5"


#ncpus = args.ncpus[0]
ncpus = -1

BINARIZE = 1
NGPU = 1
batchsize = 2
assert batchsize % NGPU == 0
if NGPU == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#else:
#    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#    os.environ["CUDA_VISIBLE_DEVICES"]=NGPU #hard-coded!

import tensorflow as tf

tf.keras.backend.clear_session()
from lightningcast import losses
from lightningcast import tf_metrics
import tensorflow_addons as tfa

optimizer = (
    tfa.optimizers.AdaBelief(learning_rate=0.0001)
    if (config["optimizer"].lower() == "adabelief")
    else "Adam"
)

if ncpus > 0:
    # limiting CPUs
    num_threads = 30
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)

from tensorflow import keras

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------------------------------------------------------------------------------------------
def make_divisible(X, batch_size, n_gpu):
    """
    This function takes our batch_size and n_gpu we're using and
    will create a new number of samples X that is divisible by both.
    We will then use dataset.take(amount_to_add) to replicate a few
    samples. We do this so that we don't have any remainders in our
    dataset during evaluation.
    """

    amount_to_add = 0
    while X % batch_size != 0 or X % n_gpu != 0:
        X += 1
        amount_to_add += 1

    assert X % batch_size == 0 and X % n_gpu == 0

    return X, amount_to_add


# ---------------------------------------------------------------------------------------------------------------------------------------------------
def get_metrics():

    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(name="aupr", curve="PR"),
        tf.keras.metrics.MeanSquaredError(name="brier_score"),
    ]

    custom_metrics = collections.OrderedDict()
    custom_metrics["loss"] = "foo"
    custom_metrics["accuracy"] = "foo"
    custom_metrics["aupr"] = tf.keras.metrics.AUC(name="aupr", curve="PR")
    custom_metrics["brier_score"] = tf.keras.metrics.MeanSquaredError(name="brier_score")

    ntargets = 1

    for jj in range(ntargets):
        for ii in np.arange(0.05,1,0.05):
    
            met_name = f"csi{str(int(round(ii*100))).zfill(2)}_index{jj}"
            csi_met = losses.csi(
                use_as_loss_function=False,
                use_soft_discretization=False,
                hard_discretization_threshold=ii,
                name=met_name,
                index=jj
            )
            metrics.append(csi_met)
            custom_metrics[met_name] = csi_met
    
            met_name = f"pod{str(int(round(ii*100))).zfill(2)}_index{jj}"
            pod_met = tf_metrics.pod(
                use_soft_discretization=False, hard_discretization_threshold=ii, name=met_name, index=jj
            )
            metrics.append(pod_met)
            custom_metrics[met_name] = pod_met
    
            met_name = f"far{str(int(round(ii*100))).zfill(2)}_index{jj}"
            far_met = tf_metrics.far(
                use_soft_discretization=False, hard_discretization_threshold=ii, name=met_name, index=jj
            )
            metrics.append(far_met)
            custom_metrics[met_name] = far_met

            met_name = f"obsct{str(int(round(ii*100))).zfill(2)}_index{jj}"
            obsct_met = tf_metrics.obs_ct(
                threshold1=ii-0.05, threshold2=ii, name=met_name, index=jj
            )
            metrics.append(obsct_met)
            custom_metrics[met_name] = obsct_met

            met_name = f"fcstct{str(int(round(ii*100))).zfill(2)}_index{jj}"
            fcstct_met = tf_metrics.fcst_ct(
                threshold1=ii-0.05, threshold2=ii, name=met_name, index=jj
            )
            metrics.append(fcstct_met)
            custom_metrics[met_name] = fcstct_met


    return custom_metrics, metrics

# ----------------------------------------------------------------------------------------------------------------
def parse_tfrecord_control(example):

    feature_description = {
        "CH02": tf.io.FixedLenFeature([], tf.string),
        "CH05": tf.io.FixedLenFeature([], tf.string),
        "CH13": tf.io.FixedLenFeature([], tf.string),
        "CH15": tf.io.FixedLenFeature([], tf.string),
        "FED_accum_60min_2km": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)

    ny, nx = 700, 700

    ch02 = tf.reshape(tf.io.parse_tensor(features["CH02"], tf.float32), [ny*4, nx*4, 1])
    features["input_1"] = ch02

    ch05 = tf.reshape(tf.io.parse_tensor(features["CH05"], tf.float32), [ny*2, nx*2, 1])
    features["input_2"] = ch05
    
    ch13 = tf.reshape(tf.io.parse_tensor(features["CH13"], tf.float32), [ny, nx, 1])

    ch15 = tf.reshape(tf.io.parse_tensor(features["CH15"], tf.float32), [ny, nx, 1])
    features["input_3"] = tf.concat([ch13, ch15], axis=-1)

    fed_accum = tf.reshape(
        tf.io.parse_tensor(features["FED_accum_60min_2km"], tf.float32), [ny, nx]
    )  # no last dim here

    # binarize
    zero = tf.zeros_like(fed_accum)
    one = tf.ones_like(fed_accum)
    targets = tf.where(fed_accum >= BINARIZE, x=one, y=zero)

    return (features, targets)


################################################################################################################
def get_base_model():
    custom_objs, metrics = get_metrics()
    conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)
    pool_2d_layer = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=None)
    pool_2d_layer._name = "MaxPool_Output"
    base_model_out = pool_2d_layer(conv_model.layers[-9].output)
    base_model = keras.Model(inputs = [layer_.input for layer_ in conv_model.layers[:3]], outputs = base_model_out)
    base_model.compile(optimizer=optimizer, loss=config["loss_fcn"], metrics=metrics)
    return base_model

################################################################################################################
def extract_features(ds, outdir):
    ny, nx = 700, 700
    #nfilters = 16 # Filters of Conv2D_17
    nfilters = 32 # Filters of Conv2D_17

    cter = 0
    for inputs,targets in ds:
        features = base_model.predict(inputs)
        batchsize = features.shape[0]
        for i in range(batchsize):
            print(features[i].shape)
            print(targets[i].shape)
            write_tfrecords(features[i], targets[i], cter+i, outdir)
        del features
        cter += batchsize
    #return features, labels

def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))
    #return tf.train.Feature(float_list=tf.train.FloatList(value=[value.numpy()]))

def write_tfrecords(features, labels, idx, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    outfile = f"{outdir}/lc_br_{idx}.tfrec"

    with tf.io.TFRecordWriter(outfile) as writer:
        inputs = tf.io.serialize_tensor(tf.convert_to_tensor(features))
        targets = tf.io.serialize_tensor(tf.convert_to_tensor(labels))

        data = {
        "input": image_feature(inputs),
        "target": image_feature(targets)
        }
        example = tf.train.Example(features=tf.train.Features(feature=data))

        writer.write(example.SerializeToString())
################################################################################################################
## MAIN ##


# Load model and set up GPUs
if NGPU > 1:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        #custom_objs, metrics = get_metrics()
        #conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)
        base_model = get_base_model()
else:
    base_model = get_base_model()

print(base_model.summary())


AUTOTUNE = tf.data.AUTOTUNE

# Trainining Dataset
train_filenames = glob.glob(train_dir + "/*/*.tfrec", recursive=True)
n_samples = len(train_filenames)
print("\nNumber of training samples:", n_samples, "\n")
train_ds = (tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
               .shuffle(10)
               .map(parse_tfrecord_control, num_parallel_calls=AUTOTUNE)
               .batch(batchsize)
               .prefetch(AUTOTUNE)
)

# Validation Dataset
val_filenames = glob.glob(val_dir + "/*/*.tfrec", recursive=True)
n_vsamples = len(val_filenames)
print("\nNumber of validation samples:", n_vsamples, "\n")
val_ds = (tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTOTUNE)
       .map(parse_tfrecord_control, num_parallel_calls=AUTOTUNE)
       .batch(batchsize)
       .prefetch(AUTOTUNE)
)


# Get remainder dataset
#_, amt_to_add = make_divisible(n_samples, batchsize, NGPU)
##    assert (
##        amt_to_add / n_samples < 0.01
##    )  # make sure we're not duplicating too many samples
#remainder_ds = train_ds.take(amt_to_add)
# Concatenate to test_ds
#train_ds = train_ds.concatenate(remainder_ds)

#train_ds = (
#    train_ds.map(parse_tfrecord_control, num_parallel_calls=AUTOTUNE)
#    .batch(batchsize)  # Don't batch if making numpy arrays
#    .prefetch(AUTOTUNE)
#)

# Disable AutoShard.
#options = tf.data.Options()
#options.experimental_distribute.auto_shard_policy = (
#    tf.data.experimental.AutoShardPolicy.OFF
#)
#train_ds = train_ds.with_options(options)

# Make predictions to extract features
train_outdir = os.path.join(outdir, 'train')
val_outdir = os.path.join(outdir, 'val')
extract_features(train_ds, train_outdir)
extract_features(val_ds, val_outdir)
