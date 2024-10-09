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
import tf_train
import configparser


gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_model(input_model, lr):
    # Load model and set up GPUs
    if NGPU > 1:
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            custom_objs, metrics = tf_train.get_metrics()

            if os.path.basename(input_model) == 'fit_conv_model.h5':
                conv_model_squeeze = load_model(input_model, custom_objects=custom_objs, compile=False)
                conv_model = Model(conv_model_squeeze.input, conv_model_squeeze.layers[-2].output)
            else:
                conv_model = load_model(input_model, custom_objects=custom_objs, compile=False)

            conv_model = tf_train.set_trainable_layers(conv_model, layername)
            #conv_model.trainable = True
            # Compile the Model
            if POS_WEIGHT > 0:
                conv_model.compile(loss=loss, optimizer=Adam(learning_rate=lr),weighted_metrics=metrics)
            else:
                conv_model.compile(loss=loss, optimizer=Adam(learning_rate=lr),metrics=metrics)
    else:
        custom_objs, metrics = tf_train.get_metrics()
        if os.path.basename(input_model) == 'fit_conv_model.h5':
            conv_model_squeeze = load_model(input_model, custom_objects=custom_objs, compile=False)
            conv_model = Model(conv_model_squeeze.input, conv_model_squeeze.layers[-2].output)
        else:
            conv_model = load_model(input_model, custom_objects=custom_objs, compile=False)
        conv_model = tf_train.set_trainable_layers(conv_model, layername)
        #conv_model.trainable = True
        # Compile the Model
        if POS_WEIGHT > 0:
            conv_model.compile(loss=loss, optimizer=Adam(learning_rate=lr),weighted_metrics=metrics)
        else:
            conv_model.compile(loss=loss, optimizer=Adam(learning_rate=lr),metrics=metrics)
        
    return conv_model


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
        "-lr", "--learning_rate", help="Customized learning rate to start fitting.",
        type=str,
    )

    args = parser.parse_args()

    input_model = args.model_file[0]
    print("Model: "+input_model)
    layername = args.layer_name[0]
    if args.train_files_list:
        train_files_list = args.train_files_list[0]
    else:
        if args.train_datadir:
            train_dir = args.train_datadir[0]
        else: 
            print("ERROR: Please, inform one of: '--train_datadir' or '--train_files_list'")
            sys.exit()

    val_dir = args.val_datadir[0]
    if not args.outdir:
        outdir = os.environ["PWD"] + "/"
    else:
        outdir = args.outdir[0]

    # Read Config
    config = configparser.ConfigParser()
    config.read('config.ini')
    NGPU = int(config['FIT']['ngpu'])
    batchsize = int(config['FIT']['batchsize'])
    lr = args.learning_rate if args.learning_rate else float(config['FIT']['learning_rate'])
    loss = config['FIT']['loss']
    BINARIZE = float(config['FIT']['binarize'])
    POS_WEIGHT = float(config['FIT']['pos_weight'])

    print("POS_WEIGHT= "+str(POS_WEIGHT))

    # Create output directory
    i = 1
    outdir_ = os.path.join(outdir, layername)
    # Check if folder exists and is not empty
    while os.path.exists(outdir_) and len(os.listdir(outdir_)) != 0:
        i = i + 1
        outdir_ = os.path.join(outdir, layername + '_'+str(i))
    pathlib.Path(outdir_).mkdir(parents=True, exist_ok=True)
    outdir = outdir_
    print(f"Output path: {outdir}")

    # Prepare train and validation datasets
    if 'train_files_list' in globals():
        with open(train_files_list, 'r') as f:
            filenames = f.readlines()
            train_filenames = [filename.strip() for filename in filenames]
    else:
        train_filenames = glob.glob(train_dir + "/**/*.tfrec", recursive=True)

    val_filenames = glob.glob(val_dir + "/**/*.tfrec", recursive=True)

    train_ds, val_ds = tf_train.prepare_dataset(train_filenames, val_filenames, pos_weight_=POS_WEIGHT, batchsize=batchsize)

    # Load and compile model
    conv_model = get_model(input_model, lr)

    # Fit model
    start_time = time.time()
    tf_train.train(conv_model, train_ds, val_ds, outdir)
    total_time = time.time() - start_time

    time_log = os.path.join(outdir, 'time.csv')
    with open(time_log, 'w') as tl:
        tl.write(str(total_time))

