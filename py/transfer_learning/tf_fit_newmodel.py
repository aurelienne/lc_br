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
from tensorflow.keras.optimizers import Adam

parse_desc = """Feature extraction from LightningCast to a new dataset"""
parser = argparse.ArgumentParser(description=parse_desc)
parser.add_argument(
    "-o",
    "--outdir",
    help="Output directory. Default={PWD}",
    nargs=1,
    type=str,
)
parser.add_argument(
    "-t", "--train_datadir", help="Dataset directory used for fitting new model with training dataset",
    type=str,
    nargs=1,
    required=True,
)
parser.add_argument(
    "-v", "--val_datadir", help="Dataset directory used for fitting new model with validation dataset",
    type=str,
    nargs=1,
    required=True,
)

args = parser.parse_args()

train_dir = args.train_datadir[0]
val_dir = args.val_datadir[0]
if not args.outdir:
    outdir = os.environ["PWD"] + "/"
else:
    outdir = args.outdir[0]


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
        "input": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)

    ny, nx = 700, 700

    feature_map = tf.reshape(tf.io.parse_tensor(features["input"], tf.float32), [ny, nx, nfilters])
    targets = tf.reshape(tf.io.parse_tensor(features["target"], tf.float32), [ny, nx])

    # binarize
    zero = tf.zeros_like(targets)
    one = tf.ones_like(targets)
    targets_bin = tf.where(targets >= BINARIZE, x=one, y=zero)

    return (feature_map, targets_bin)

def get_model():
    # New Model 
    new_input = tf.keras.Input(shape=(700,700,nfilters))
    conv2d = tf.keras.layers.Conv2D(1, (1,1), activation = 'sigmoid', padding='same')(new_input)
    new_model = tf.keras.Model(inputs = new_input, outputs = conv2d)
    custom_objs, metrics = get_metrics()
    optimizer=Adam(learning_rate=1e-4)
    new_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
    print(new_model.summary())
    return new_model

def training_history_figs(history,outdir):
  '''
  This function simply makes a couple of figures for how the accuracy and loss
  changed during training.
  '''

  # summarize history for accuracy
  try:
    plt.plot(history['binary_accuracy'])
    if('val_binary_accuracy' in history): plt.plot(history['val_binary_accuracy'])
    plt.title('model binary accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if('val_binary_accuracy' in history):
      plt.legend(['train', 'val'], loc='upper left')
    else:
      plt.legend(['train'], loc='upper left')
    plt.savefig(os.path.join(outdir,'binary_accuracy_history.png'))
    plt.close()
  except KeyError:
    plt.plot(history['accuracy'])
    if('val_accuracy' in history): plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if('val_accuracy' in history):
      plt.legend(['train', 'val'], loc='upper left')
    else:
      plt.legend(['train'], loc='upper left')
    plt.savefig(os.path.join(outdir,'accuracy_history.png'))
    plt.close()

  # summarize history for loss
  plt.plot(history['loss'])
  if('val_loss' in history): plt.plot(history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  if('val_loss' in history):
    plt.legend(['train', 'val'], loc='upper left')
  else:
    plt.legend(['train'], loc='upper left')
  plt.savefig(os.path.join(outdir,'loss_history.png'))

################################################################################################################
## MAIN ##

nfilters = 16 # NUmber of filters of last Conv2D layer during Feature Extraction

# Load model and set up GPUs
if NGPU > 1:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        #custom_objs, metrics = get_metrics()
        #conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)
        model = get_model()
else:
    model = get_model()

print(model.summary())


AUTOTUNE = tf.data.AUTOTUNE

# Trainining Dataset
train_filenames = glob.glob(train_dir + "/*.tfrec", recursive=True)
n_samples = len(train_filenames)
print("\nNumber of training samples:", n_samples, "\n")
train_ds = (tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
               .shuffle(10)
               .map(parse_tfrecord_control, num_parallel_calls=AUTOTUNE)
               .batch(batchsize)
               .prefetch(AUTOTUNE)
)

# Validation Dataset
val_filenames = glob.glob(val_dir + "/*.tfrec", recursive=True)
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

# Training New Model
# Callbacks
#csvlogger = CSVLogger(f"{outdir}/log.csv", append=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=4) # If no improvement after {patience} epochs, the stop early
mcp_save = ModelCheckpoint(os.path.join(outdir,'model-{epoch:02d}-{val_loss:03f}.h5'),save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', cooldown=1, verbose=1, min_delta=0.00001,factor=0.1, patience=2, mode='min')
callbacks = [early_stopping, mcp_save, reduce_lr_loss] #, csvlogger]
history = model.fit(train_ds,
          verbose=1,
          epochs=100,
          validation_data=val_ds,
          callbacks=callbacks)

print('Saving model and training history...')
##copy the model with the best val_loss
try:
    best_model = get_best_model(outdir)
except IndexError:
    print("Something is wrong with finding the best model. Exiting.")
    sys.exit()
else:
    shutil.copy(best_model,f"{os.path.dirname(best_model)}/fit_conv_model.h5")

# Save training history and make figs
pickle.dump(history.history,open(os.path.join(outdir,"training_history.pkl"),"wb"),protocol=4)
training_history_figs(history.history,outdir)

