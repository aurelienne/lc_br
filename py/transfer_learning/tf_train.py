from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryFocalCrossentropy
import tensorflow as tf
import numpy as np
import sys
import pathlib
import os
import glob
import shutil
from matplotlib import pyplot as plt
import pickle
import time
import collections
from lightningcast import losses
from lightningcast import tf_metrics
import argparse
import configparser

"""
def random_crop(features):
    IMG_HEIGHT, IMG_WIDTH = 480, 480
    stacked_image_ch02 = tf.stack([features], axis=0)
    cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 1])
    inputs = cropped_image[0:3]
    target = cropped_image[3]

  return inputs, target
"""


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

            #met_name = f"csi{str(int(round(ii*100))).zfill(2)}_index{jj}"
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

            #met_name = f"pod{str(int(round(ii*100))).zfill(2)}_index{jj}"
            met_name = f"pod{str(int(round(ii*100))).zfill(2)}"
            pod_met = tf_metrics.pod(
                use_soft_discretization=False, hard_discretization_threshold=ii, name=met_name, index=jj
            )
            metrics.append(pod_met)
            custom_metrics[met_name] = pod_met

            #met_name = f"far{str(int(round(ii*100))).zfill(2)}_index{jj}"
            met_name = f"far{str(int(round(ii*100))).zfill(2)}"
            far_met = tf_metrics.far(
                use_soft_discretization=False, hard_discretization_threshold=ii, name=met_name, index=jj
            )
            metrics.append(far_met)
            custom_metrics[met_name] = far_met
            """
            #met_name = f"obsct{str(int(round(ii*100))).zfill(2)}_index{jj}"
            met_name = f"obsct{str(int(round(ii*100))).zfill(2)}"
            obsct_met = tf_metrics.obs_ct(
                threshold1=ii-0.05, threshold2=ii, name=met_name, index=jj
            )
            metrics.append(obsct_met)
            custom_metrics[met_name] = obsct_met

            #met_name = f"fcstct{str(int(round(ii*100))).zfill(2)}_index{jj}"
            met_name = f"fcstct{str(int(round(ii*100))).zfill(2)}"
            fcstct_met = tf_metrics.fcst_ct(
                threshold1=ii-0.05, threshold2=ii, name=met_name, index=jj
            )
            metrics.append(fcstct_met)
            custom_metrics[met_name] = fcstct_met
            """


    return custom_metrics, metrics


def parse_tfrecord_fn(example):
    config = configparser.ConfigParser()
    config.read('config.ini')
    ny = int(config['DATA']['ny'])
    nx = int(config['DATA']['nx'])

    feature_description = {
        "CH02": tf.io.FixedLenFeature([], tf.string),
        "CH05": tf.io.FixedLenFeature([], tf.string),
        "CH13": tf.io.FixedLenFeature([], tf.string),
        "CH15": tf.io.FixedLenFeature([], tf.string),
        "FED_accum_60min_2km": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)

    example = {}
    
    #CH02
    image = tf.io.parse_tensor(features["CH02"], tf.float32)
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny*4,nx*4,1])
    example['CH02'] = image

    #CH05
    image = tf.io.parse_tensor(features["CH05"], tf.float32)
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny*2,nx*2,1])
    example['CH05'] = image

    #CH13
    image = tf.io.parse_tensor(features["CH13"], tf.float32)
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny,nx,1])
    example['CH13'] = image

    #CH15
    image = tf.io.parse_tensor(features["CH15"], tf.float32)
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny,nx,1])
    example['CH15'] = image

    image = tf.io.parse_tensor(features["FED_accum_60min_2km"], tf.float32)
    #image = tf.reshape(image, [ny,nx])
    image = tf.reshape(image, [ny,nx, 1])
    example['FED_accum_60min_2km'] = image

    print(example['CH02'].shape,  example['CH05'].shape, example['CH13'].shape, example['CH15'].shape)
    return example

def prepare_sample(features):
  global POS_WEIGHT
  '''
  This function binarizes our truth data and gets the truth/target and
  ABI predictor data for this sample. We set up the inputs dict appropriately
  for how the model expects the data (i.e., three inputs).
  '''
  #nx, ny = 480, 480

  #image = tf.image.resize(features['FED_accum_60min_2km'], [ny, nx])
  # Squeeze 3rd dimension (needs to be done here (and not inside parsing function) bacause resize (above) demands a 3-D tensor
  targetImage_int = features['FED_accum_60min_2km']

  # Binarize... A flash count of "1" is scaled to "1". So, we are making
  # anything ≥ 1 to equal 1, and anything less than 1 to equal 0.
  zero = tf.zeros_like(targetImage_int)
  one = tf.ones_like(targetImage_int)
  targetImage = tf.where(targetImage_int >= 1, x=one, y=zero)

  inputs = {}

  #inputs['input_1'] = tf.image.resize(features['CH02'], [ny*4, nx*4])
  #inputs['input_2'] = tf.image.resize(features['CH05'], [ny*2, nx*2])
  #ch13 = tf.image.resize(features['CH13'], [ny, nx])
  #ch15 = tf.image.resize(features['CH15'], [ny, nx])
  inputs['input_1'] = features['CH02']
  inputs['input_2'] = features['CH05']
  ch13 = features['CH13']
  ch15 = features['CH15']
  inputs['input_3'] = tf.concat([ch13,ch15], axis=2)

  if POS_WEIGHT > 0: #multiplier of the positive class
      class_weight = tf.multiply(one, POS_WEIGHT) 
      sample_weights = tf.where(targetImage_int >= 1, x=class_weight, y=one)
      return inputs, targetImage, sample_weights
  else:
      return inputs, targetImage

#-----------------------------------------------------------------------------------------------
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
  plt.close()

#--------------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------------
class TimeHistory(Callback):
    def on_train_begin(self, logs):
        self.times = []

    def on_epoch_begin(self, batch, logs):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs):
        self.times.append(time.time() - self.epoch_time_start)
        logs['time'] = self.times[-1]

#------------------------------------------------------------------------------------------------
def prepare_dataset(train_filenames, val_filenames, pos_weight_, batchsize):
    print('\nBuilding training Dataset')
    global POS_WEIGHT
    POS_WEIGHT = pos_weight_
    AUTOTUNE=tf.data.AUTOTUNE

    n_tsamples = len(train_filenames)
    print(f"Training samples: {n_tsamples}")
    train_ds = (tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
               .shuffle(10)
               .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
               .map(prepare_sample, num_parallel_calls=AUTOTUNE)
               .batch(batchsize)
               .prefetch(AUTOTUNE)
    )

    print('built training dataset')

    print('\nBuilding validation Dataset')

    n_vsamples = len(val_filenames)
    print(f"Validation samples: {n_vsamples}")
    val_ds = (tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTOTUNE)
           .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
           .map(prepare_sample, num_parallel_calls=AUTOTUNE)
           .batch(batchsize)
           .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds

#------------------------------------------------------------------------------------------------
def set_trainable_layers(model, layername):

    # Unfreeze first encoder block and last decoder block
    if layername == '1stEnc_LastDec':
        t = True
        for layer in model.layers:
            if layer.name == 'conv2d_transpose_3':
                t = True
            layer.trainable = t
            if layer.name == 'max_pooling2d':
                t = False
    elif layername == 'Enc_Bot': # Encoder Block + Bottleneck
        t = True
        for layer in model.layers:
            if layer.name == 'conv2d_transpose':
                t = False
            layer.trainable = t
    elif layername == 'Bot': # Bottleneck
        t = False
        for layer in model.layers:
            if layer.name == 'conv2d_8':
                t = True
            if layer.name == 'conv2d_transpose':
                t = False
            layer.trainable = t
    else:
        t = False
        for layer in model.layers:
            if layer.name == layername or layername == "full":
                t = True
            layer.trainable = t

    return model

#------------------------------------------------------------------------------------------------
def train(input_model, train_ds, val_ds, outdir, NGPU=1):
    
    print(input_model.summary(show_trainable=True))
    sys.exit()

    # Callbacks
    csvlogger = CSVLogger(f"{outdir}/log.csv", append=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3) # If no improvement after {patience} epochs, the stop early
    mcp_save = ModelCheckpoint(os.path.join(outdir,'model-{epoch:02d}-{val_loss:03f}.h5'),save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', cooldown=1, verbose=1, min_delta=0.00001,factor=0.1, patience=2, mode='min')
    time_callback = TimeHistory()
    callbacks = [early_stopping, mcp_save, reduce_lr_loss, time_callback, csvlogger]

    start_time = time.time()

    # FIT THE MODEL
    history = input_model.fit(
              x=train_ds,
              verbose=1,
              epochs=100,
              validation_data=val_ds,
              callbacks=callbacks)

    # Compute time and save to log file
    total_time = time.time() - start_time
    time_log = os.path.join(outdir, 'time.csv')
    with open(time_log, 'w') as tl:
        tl.write(str(total_time))

    print('Saving model and training history...')
    ##copy the model with the best val_loss
    try:
        best_model = get_best_model(outdir)
    except IndexError:
        print("Something is wrong with finding the best model.")
        #sys.exit()
        pass
    else:
        shutil.copy(best_model,f"{os.path.dirname(best_model)}/fit_conv_model.h5")
        del best_model

    # Save training history and make figs
    pickle.dump(history.history,open(os.path.join(outdir,"training_history.pkl"),"wb"),protocol=4)
    training_history_figs(history.history,outdir)

    # Delete objects in RAM
    del train_ds, history, mcp_save, early_stopping, reduce_lr_loss, callbacks

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
    del input_model
