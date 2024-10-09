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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import  Model, load_model
import pathlib
import tensorflow as tf
tf.keras.backend.clear_session()
from lightningcast import losses
from lightningcast import tf_metrics
import tensorflow_addons as tfa
import tf_train
import configparser

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
def get_model_without_weights():
    custom_objs, metrics = tf_train.get_metrics()

    if os.path.basename(model_file) == 'fit_conv_model.h5': # Remove Squeeze layer
        conv_model_squeeze = load_model(model_file, custom_objects=custom_objs, compile=False)
        conv_model = Model(conv_model_squeeze.input, conv_model_squeeze.layers[-2].output)
    else:
        conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)

    model_json = conv_model.to_json()
    new_model = tf.keras.models.model_from_json(model_json)
    print(new_model.summary())
    del conv_model
    return new_model


################################################################################################################
if __name__ == "__main__":

    # Receive args
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
    )
    parser.add_argument(
        "-v", "--val_datadir", help="Dataset directory used for fine-tuning the model with validation dataset",
        type=str,
        nargs=1,
        required=True,
    )
    parser.add_argument(
        "-tf", "--train_files_list", help="List of sample files for traning",
        type=str,
        nargs=1,
    )
    parser.add_argument(
        "-c", "--continue_fit", help="Indicate if it is a continuation of fitting (in order to not compile the model).",
        action="store_false",
    )
    parser.add_argument(
        "-lr", "--learning_rate", help="Customized learning rate to start fitting.",
        type=str,
    )
    
    args = parser.parse_args()

    modeldir = args.modeldir[0]
    val_dir = args.val_datadir[0]
    if not args.outdir:
        outdir = os.environ["PWD"] + "/"
    else:
        outdir = args.outdir[0]
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.train_files_list:
        train_files_list = args.train_files_list[0]
    else:
        if args.train_datadir:
            train_dir = args.train_datadir[0]
        else:
            print("ERROR: Please, inform one of: '--train_datadir' or '--train_files_list'")
            sys.exit()

    if os.path.isdir(modeldir):
        model_file = f"{modeldir}/fit_conv_model.h5"
    else:
        model_file = modeldir
    print(f"Model : {model_file}")

    # Read Config
    config = configparser.ConfigParser()
    config.read('config.ini')
    NGPU = int(config['FIT']['ngpu'])
    batchsize = int(config['FIT']['batchsize'])
    lr = args.learning_rate if args.learning_rate else config['FIT']['learning_rate']
    loss = config['FIT']['loss']
    BINARIZE = float(config['FIT']['binarize'])
    POS_WEIGHT = float(config['FIT']['pos_weight'])

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    AUTOTUNE = tf.data.AUTOTUNE

    #ncpus = args.ncpus[0]
    ncpus = -1

    #BINARIZE = 1
    #NGPU = 1
    #batchsize = 4
    #assert batchsize % NGPU == 0
    if NGPU == 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    #else:
    #    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #    os.environ["CUDA_VISIBLE_DEVICES"]=NGPU #hard-coded!

    if ncpus > 0:
        # limiting CPUs
        num_threads = 30
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)



    optimizer=Adam(learning_rate=lr)
    custom_objs, metrics = tf_train.get_metrics()

    # Load model and set up GPUs
    if NGPU > 1:
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            new_model = get_model_without_weights()
            if not args.continue_fit:
                new_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
                print("Compiling model...")
            else:
                new_model.compile(loss="binary_crossentropy", metrics=metrics)
    else:
        new_model = get_model_without_weights()
        if not args.continue_fit:
            new_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
        else:
            new_model.compile(loss="binary_crossentropy", metrics=metrics)
            print("Compiling model...")

    # Prepare train and validation datasets
    if 'train_files_list' in globals():
        with open(train_files_list, 'r') as f:
            filenames = f.readlines()
            train_filenames = [filename.strip() for filename in filenames]
    else:
        train_filenames = glob.glob(train_dir + "/**/*.tfrec", recursive=True)

    val_filenames = glob.glob(val_dir + "/202001*/*.tfrec", recursive=True)

    train_ds, val_ds = tf_train.prepare_dataset(train_filenames, val_filenames, pos_weight_=POS_WEIGHT, batchsize=batchsize)

    # Fit Model
    tf_train.train(new_model, train_ds, val_ds, outdir)
