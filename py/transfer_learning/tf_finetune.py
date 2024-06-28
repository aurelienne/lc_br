from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
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

#input_model = '/home/ajorge/src/lightningcast-master/lightningcast/static/fit_conv_model.h5'
NY, NX = 2100, 2100
ny, nx = 700, 700
NGPU = 2

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
    feature_description = {
        "CH02": tf.io.FixedLenFeature([], tf.string),
        "CH05": tf.io.FixedLenFeature([], tf.string),
        "CH13": tf.io.FixedLenFeature([], tf.string),
        "CH15": tf.io.FixedLenFeature([], tf.string),
        "FED_accum_60min_2km": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)

    example = {}
    
    # Mean and Standard Deviation from Aure's Scaling process
    """
    glm_mean_aure = tf.constant(0.037830263, dtype=tf.float32)
    glm_std_aure = tf.constant(0.4442353, dtype=tf.float32)
    ch02_mean_aure = tf.constant(0.15861996, dtype=tf.float32)
    ch02_std_aure = tf.constant(0.16588475, dtype=tf.float32)
    ch05_mean_aure = tf.constant(0.12517355, dtype=tf.float32)
    ch05_std_aure = tf.constant(0.09750033, dtype=tf.float32)
    ch13_mean_aure = tf.constant(275.44543, dtype=tf.float32)
    ch13_std_aure = tf.constant(25.964622, dtype=tf.float32)
    ch15_mean_aure = tf.constant(271.02097, dtype=tf.float32)
    ch15_std_aure = tf.constant(25.430159, dtype=tf.float32)
    """

    #CH02
    image = tf.io.parse_tensor(features["CH02"], tf.float32)
    """
    # Unscale data and Scale again according to Control LC patterns
    image = image * ch02_std_aure + ch02_mean_aure
    mean = tf.constant(0.08713, dtype=tf.float32)
    std = tf.constant(0.15314, dtype=tf.float32)
    image = (image - mean) / std
    """
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny*4,nx*4,1])
    example['CH02'] = image

    #CH05
    image = tf.io.parse_tensor(features["CH05"], tf.float32)
    """
    # Uscale data and Scale again according to Control LC patterns
    image = image * ch05_std_aure + ch05_mean_aure
    mean = tf.constant(0.047332, dtype=tf.float32)
    std = tf.constant(0.0834376, dtype=tf.float32)
    image = (image - mean) / std
    """
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny*2,nx*2,1])
    example['CH05'] = image

    #CH13
    image = tf.io.parse_tensor(features["CH13"], tf.float32)
    """
    # Uscale data and Scale again according to Control LC patterns
    image = image * ch13_std_aure + ch13_mean_aure
    mean = tf.constant(270.93, dtype=tf.float32)
    std = tf.constant(25.9126, dtype=tf.float32)
    image = (image - mean) / std
    """
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny,nx,1])
    example['CH13'] = image

    #CH15
    image = tf.io.parse_tensor(features["CH15"], tf.float32)
    """
    # Uscale data and Scale again according to Control LC patterns
    image = image * ch15_std_aure + ch15_mean_aure
    mean = tf.constant(273.168, dtype=tf.float32)
    std = tf.constant(19.9895, dtype=tf.float32)
    image = (image - mean) / std
    """
    # Reshape to add channel dimension
    image = tf.reshape(image, [ny,nx,1])
    example['CH15'] = image

    image = tf.io.parse_tensor(features["FED_accum_60min_2km"], tf.float32)
    #image = tf.reshape(image, [ny,nx])
    image = tf.reshape(image, [ny,nx, 1])
    example['FED_accum_60min_2km'] = image

    return example

def prepare_sample(features):
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

def set_trainable_layers(model):
    global layername
    
    # Unfreeze first encoder block and last decoder block
    if layername == '1stEnc_LastDec':
        t = True
        for layer in model.layers:
            print(layer.name)
            if layer.name == 'conv2d_transpose_3':
                t = True
            layer.trainable = t
            if layer.name == 'max_pooling2d':
                t = False
    else:
        t = False
        for layer in model.layers:
            if layer.name == layername or layername == "full":
                t = True
            layer.trainable = t

    return model


def train():
    print('\nBuilding training Dataset')

    AUTOTUNE=tf.data.AUTOTUNE

    ##train_filenames = tf.io.gfile.glob(train_dir + "/*.tfrec")
    train_filenames = glob.glob(train_dir + "/**/*.tfrec", recursive=True)
    #train_filenames = glob.glob(train_dir + "/202001*/*.tfrec", recursive=True)
    n_tsamples = len(train_filenames)
    train_ds = (tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
               .shuffle(10)
               .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
               .map(prepare_sample, num_parallel_calls=AUTOTUNE)
               .batch(BATCHSIZE)
               .prefetch(AUTOTUNE)
    )


    print('built training dataset')

    print('\nBuilding validation Dataset')

    #val_filenames = tf.io.gfile.glob(val_dir + "/*.tfrec")
    val_filenames = glob.glob(val_dir + "/**/*.tfrec", recursive=True)
    #val_filenames = glob.glob(val_dir + "/202001*/*.tfrec", recursive=True)
    print(val_filenames)
    n_vsamples = len(val_filenames)
    val_ds = (tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTOTUNE)
           .map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
           .map(prepare_sample, num_parallel_calls=AUTOTUNE)
           .batch(BATCHSIZE)
           .prefetch(AUTOTUNE)
    )

    # Callbacks
    csvlogger = CSVLogger(f"{outdir}/log.csv", append=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3) # If no improvement after {patience} epochs, the stop early
    mcp_save = ModelCheckpoint(os.path.join(outdir,'model-{epoch:02d}-{val_loss:03f}.h5'),save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', cooldown=1, verbose=1, min_delta=0.00001,factor=0.1, patience=2, mode='min')
    callbacks = [early_stopping, mcp_save, reduce_lr_loss, csvlogger]

    #conv_model = unet_2D(config)
    custom_objs, metrics = get_metrics()

    # Load model and set up GPUs
    if NGPU > 1:
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            custom_objs, metrics = get_metrics()

            if os.path.basename(input_model) == 'fit_conv_model.h5':
                conv_model_squeeze = load_model(input_model, custom_objects=custom_objs, compile=False)
                conv_model = Model(conv_model_squeeze.input, conv_model_squeeze.layers[-2].output)
            else:
                conv_model = load_model(input_model, custom_objects=custom_objs, compile=False)

            conv_model = set_trainable_layers(conv_model)
            #conv_model.trainable = True
            # Compile the Model
            conv_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5),metrics=metrics)
    else:
        custom_objs, metrics = get_metrics()
        if os.path.basename(input_model) == 'fit_conv_model.h5':
            conv_model_squeeze = load_model(input_model, custom_objects=custom_objs, compile=False)
            conv_model = Model(conv_model_squeeze.input, conv_model_squeeze.layers[-2].output)
        else:
            conv_model = load_model(input_model, custom_objects=custom_objs, compile=False)
        conv_model = set_trainable_layers(conv_model)
        #conv_model.trainable = True
        # Compile the Model
        conv_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-5),metrics=metrics)
        
    print(conv_model.summary(show_trainable=True))


    # FIT THE MODEL
    history = conv_model.fit(
              x=train_ds,
              verbose=1,
              epochs=100,
              validation_data=val_ds,
              callbacks=callbacks)
    #shuffle has no effect if generator OR a tf.data.Dataset

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

    # Delete objects in RAM
    del train_ds, history

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


#-----------------------------------------------------------------------------------------------
#-------------------------------------DRIVING CODE----------------------------------------------
#-----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Fine-Tuning...")
    parse_desc = """Fine-Tune LightningCast to a new dataset"""

    parser = argparse.ArgumentParser(description=parse_desc)
    parser.add_argument(
        "-m", "--model_file",
        help="Pre-trained model to be loaded.",
        type=str,
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
    parser.add_argument(
        "-o", "--outdir", help="Output directory. Default = ${PWD}", 
        type=str, 
        nargs=1,
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

    input_model = args.model_file[0]
    print("Model: "+input_model)
    layername = args.layer_name[0]
    train_dir = args.train_datadir[0]
    val_dir = args.val_datadir[0]
    if not args.outdir:
        outdir = os.environ["PWD"] + "/"
    else:
        outdir = args.outdir[0]


    BATCHSIZE = 4

    outdir = os.path.join(outdir, layername)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    train()
