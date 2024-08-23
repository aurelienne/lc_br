# Model fitting using Cross-Validation
# from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import sys
import pathlib
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
from pathlib import Path
import tensorflow as tf
tf.keras.backend.clear_session()
from lightningcast import losses
from lightningcast import tf_metrics
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
import tf_train

parse_desc = """Fine-tune LightningCast using Cross-Validation"""
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
    "-k", "--k_folds", help="Number of folds to perform Cross-Validation",
    type=int,
    nargs=1,
    required=True,
)
parser.add_argument(
    "-l", "--layer_name",
    help="Layer name from which the fine tuning process starts (All layers before will be kept frozen). Set 'full' to train the entire model.",
    type=str,
    nargs=1,
    required=True,
)


args = parser.parse_args()

modeldir = args.modeldir[0]
train_dir = args.train_datadir[0]
if not args.outdir:
    outdir = os.environ["PWD"] + "/"
else:
    outdir = args.outdir[0]

kfold_splits = args.k_folds[0]
layername = args.layer_name[0]

#config = pickle.load(open(os.path.join(modeldir, "model_config.pkl"), "rb"))
model_file = f"{modeldir}/fit_conv_model.h5"

#ncpus = args.ncpus[0]
ncpus = -1

BINARIZE = 1
NGPU = 1
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


gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


################################################################################################################
def get_pretrained_model():
    custom_objs, metrics = tf_train.get_metrics()

    if os.path.basename(model_file) == 'fit_conv_model.h5': # Remove Squeeze layer
        conv_model_squeeze = load_model(model_file, custom_objects=custom_objs, compile=False)
        conv_model = Model(conv_model_squeeze.input, conv_model_squeeze.layers[-2].output)
    else:
        conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)

    print(conv_model.summary())

    return conv_model

################################################################################################################
## MAIN ##

optimizer=Adam(learning_rate=1e-4)
custom_objs, metrics = tf_train.get_metrics()

AUTOTUNE = tf.data.AUTOTUNE

# Prepare train and validation datasets
train_filenames = glob.glob(train_dir + "/*/*.tfrec", recursive=True)
train_idxs = np.arange(len(train_filenames))
print(train_idxs)

# Instantiate the cross validator
kf = KFold(n_splits=kfold_splits, shuffle=True)

# Loop through the indices the split() method returns
for index, (train_idxs, val_idxs) in enumerate(kf.split(train_idxs)):
    print(f'Fold = {index}')
    print(f'Train size = {len(train_idxs)}, Validation size = {len(val_idxs)}')

    # Prepare Training and Validation Datasets for each fold
    fold_train_files = [train_filenames[t_idx] for t_idx in train_idxs]
    fold_val_files = [train_filenames[v_idx] for v_idx in val_idxs]
    train_ds, val_ds = tf_train.prepare_dataset(fold_train_files, fold_val_files, pos_weight_=1.0)

    # Load Model and Compile
    conv_model = get_pretrained_model()
    conv_model = tf_train.set_trainable_layers(conv_model, layername)
    optimizer=Adam(learning_rate=1e-4)
    conv_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

    # Create outdir for fold
    fold_outdir = os.path.join(outdir, layername, f'fold{index}')
    pathlib.Path(fold_outdir).mkdir(parents=True, exist_ok=True)

    # Fit Model
    tf_train.train(conv_model, train_ds, val_ds, fold_outdir)
    del conv_model
