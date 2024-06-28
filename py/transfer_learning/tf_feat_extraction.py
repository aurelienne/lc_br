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
from tensorflow.keras.optimizers import Adam, RMSprop
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
        #tf.keras.metrics.AUC(name="aupr", curve="PR"),
        #tf.keras.metrics.MeanSquaredError(name="brier_score"),
    ]

    custom_metrics = collections.OrderedDict()
    custom_metrics["loss"] = "foo"
    custom_metrics["accuracy"] = "foo"
    #custom_metrics["aupr"] = tf.keras.metrics.AUC(name="aupr", curve="PR")
    #custom_metrics["brier_score"] = tf.keras.metrics.MeanSquaredError(name="brier_score")

    ntargets = 1

    for jj in range(ntargets):
        for ii in np.arange(0.05,1,0.05):
    
            met_name = f"csi{str(int(round(ii*100))).zfill(2)}"
            csi_met = losses.csi(
                use_as_loss_function=False,
                use_soft_discretization=False,
                hard_discretization_threshold=ii,
                name=met_name,
                index=jj
            )
            metrics.append(csi_met)
            custom_metrics[met_name] = csi_met
    
            met_name = f"pod{str(int(round(ii*100))).zfill(2)}"
            pod_met = tf_metrics.pod(
                use_soft_discretization=False, hard_discretization_threshold=ii, name=met_name, index=jj
            )
            metrics.append(pod_met)
            custom_metrics[met_name] = pod_met
    
            met_name = f"far{str(int(round(ii*100))).zfill(2)}"
            far_met = tf_metrics.far(
                use_soft_discretization=False, hard_discretization_threshold=ii, name=met_name, index=jj
            )
            metrics.append(far_met)
            custom_metrics[met_name] = far_met
 
            """
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
            """


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

    inputs = {}
    ch02 = tf.reshape(tf.io.parse_tensor(features["CH02"], tf.float32), [ny*4, nx*4, 1])
    inputs["input_1"] = ch02

    ch05 = tf.reshape(tf.io.parse_tensor(features["CH05"], tf.float32), [ny*2, nx*2, 1])
    inputs["input_2"] = ch05
    
    ch13 = tf.reshape(tf.io.parse_tensor(features["CH13"], tf.float32), [ny, nx, 1])

    ch15 = tf.reshape(tf.io.parse_tensor(features["CH15"], tf.float32), [ny, nx, 1])
    inputs["input_3"] = tf.concat([ch13, ch15], axis=-1)

    fed_accum = tf.reshape(
        tf.io.parse_tensor(features["FED_accum_60min_2km"], tf.float32), [ny, nx]
    )  # no last dim here

    # binarize
    zero = tf.zeros_like(fed_accum)
    one = tf.ones_like(fed_accum)
    targets = tf.where(fed_accum >= BINARIZE, x=one, y=zero)

    return (inputs, targets)


################################################################################################################
def get_base_model():
    custom_objs, metrics = get_metrics()
    conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)
    print("ORIGINAL MODEL:")
    print(conv_model.summary())
    print("")
    #for layer_ in conv_model.layers:
    #    print(layer_.get_config())

    base_model_out = conv_model.layers[-9].output
    base_model = keras.Model(inputs = [layer_.input for layer_ in conv_model.layers[:3]], outputs = base_model_out)
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
        if('val_accuracy' in history): 
            plt.plot(history['val_accuracy'])
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
        if('val_loss' in history): 
            plt.plot(history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
        if('val_loss' in history):
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
            plt.savefig(os.path.join(outdir,'loss_history.png'))

def add_scores_to_file(outdir,scores={'val_aupr':{'best_score':0},'val_brier_score':{'best_score':1}}):
  '''
  Will append to {outdir}/verification_scores.txt or create it if it does not exist.
  If scores['score']['best_score'] = 0, then "higher_is_better" = True.
  If scores['score']['best_score'] = 1, then "higher_is_better" = False (i.e., lower is better).
  '''

  of = open(f'{outdir}/log.csv', 'r')
  lines = of.readlines()
  of.close()

  for ii,line in enumerate(lines):
    parts = line.split(',')
    if(ii == 0): #first line
      #for score in scores:
      for score in list(scores):
        try:
          scores[score]['idx'] = parts.index(score) #set the score index
        except ValueError:
          del scores[score]  #remove score if it doesn't exist in the log.csv
        else:
          #set 'best_epoch' to initial epoch and if higher or lower is better
          scores[score]['best_epoch'] = 0
          if(scores[score]['best_score'] == 1): #indicates lower is better
            scores[score]['higher_is_better'] = False
          else:
            scores[score]['higher_is_better'] = True
    else: #All other lines
      for score in scores:
        idx = scores[score]['idx']
        best_val_score = scores[score]['best_score']
        val_score = float(parts[idx]) #the validation metric for this epoch
        if(scores[score]['higher_is_better']):
          if(val_score > best_val_score):
            scores[score]['best_score'] = val_score
            scores[score]['best_epoch'] = int(parts[0]) + 1
        else: #lower is better
          if(val_score < best_val_score):
            scores[score]['best_score'] = val_score
            scores[score]['best_epoch'] = int(parts[0]) + 1

  of = open(f'{outdir}/verification_scores.txt','a')
  for score in scores:
    best_score = scores[score]['best_score']
    best_epoch = scores[score]['best_epoch']
    of.write(f'Best {score}: {np.round(best_score,5)}; at epoch {best_epoch}\n')
  of.close()

def get_best_model(outdir):

  best_model = 'foo'; best_loss = 999999

  models = glob.glob(f"{outdir}/model-*.h5")
  for mod in models:
    loss = float(os.path.basename(mod).split('-')[-1].split('.h5')[0])
    if(loss < best_loss):
      best_loss = loss
      best_model = mod

  if(best_model == 'foo'):
    print("Couldn't find best model. Returning last model.")
    try:
      return np.sort(models)[-1]
    except IndexError:
      raise
  else:
    return best_model

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

# Training Dataset
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

# Appending Layers to the Base Model
base_model.trainable = False
conv = tf.keras.layers.Conv2D(16, (3,3), padding='same', name='conv2D_16')(base_model.output)
conv = tf.keras.layers.Activation('relu', name='Activation_16')(conv)
conv = tf.keras.layers.Conv2D(16, (3,3), padding='same', name='conv2D_17')(conv)
#conv = tf.keras.layers.BatchNormalization()(conv)
conv = tf.keras.layers.Activation('relu', name='Activation_17')(conv)
conv = tf.keras.layers.Conv2D(1, (1,1), padding='valid',  name='conv2D_18', bias_initializer='zeros')(conv)
conv = tf.keras.layers.Activation('sigmoid', name='Activation_18')(conv)
pool_2d_layer = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=None, name="MaxPool_Output")(conv)
pool_2d_layer._name = "MaxPool_Output"
new_model = tf.keras.Model(base_model.input, pool_2d_layer)
custom_objs, metrics = get_metrics()
#optimizer=Adam(learning_rate=1e-4)
optimizer = RMSprop(lr=2e-5)
print("Optimizer: "+str(optimizer))
new_model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
print(new_model.summary(show_trainable=True))


early_stopping = EarlyStopping(monitor='val_loss', patience=3) # If no improvement after {patience} epochs, the stop early
mcp_save = ModelCheckpoint(os.path.join(outdir,'model-{epoch:02d}-{val_loss:03f}.h5'),save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', cooldown=1, verbose=1, min_delta=0.00001,factor=0.1, patience=2, mode='min')
callbacks = [early_stopping, mcp_save, reduce_lr_loss] #, csvlogger]
history = new_model.fit(train_ds,
          verbose=1,
          epochs=100,
          validation_data=val_ds,
          callbacks=callbacks)

training_history_figs(history.history,outdir)

# Delete objects in RAM
del train_ds, val_ds, history

# Create scores file
scores = {}
scores['val_aupr'] = {'best_score':0}
scores['val_brier_score'] = {'best_score':1}
scores['val_csi05'] = {'best_score':0}
scores['val_csi10'] = {'best_score':0}
scores['val_csi15'] = {'best_score':0}
scores['val_csi20'] = {'best_score':0}
scores['val_csi25'] = {'best_score':0}
scores['val_csi30'] = {'best_score':0}
scores['val_csi35'] = {'best_score':0}
scores['val_csi40'] = {'best_score':0}
scores['val_csi45'] = {'best_score':0}
scores['val_csi50'] = {'best_score':0}

add_scores_to_file(outdir,scores=scores)

