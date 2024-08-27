import glob
import random
import os
import numpy as np
import sys
import inspect
import argparse
import pickle
import collections


BINARIZE = 1
NGPU = 1
BATCHSIZE = 4

import tensorflow as tf
tf.keras.backend.clear_session()
from lightningcast import losses
from lightningcast import tf_metrics
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model

AUTOTUNE = tf.data.AUTOTUNE

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--outdir",
    help="Output directory for model and figures. Default=args.indir + '/TEST/'",
    nargs=1,
    type=str,
    required=True,
)
parser.add_argument(
    "-m",
    "--model",
    help="Use this model to evaluate", 
    type=str,
    nargs=1,
    required=True,
)
parser.add_argument(
    "-t",
    "--test_dataset",
    help="Path of test dataset samples.",
    type=str,
    nargs=1,
    required=True,
)
parser.add_argument(
    "-nt",
    "--num_trials",
    help="Number of bootstrapping trials.",
    type=int,
    required=True,
)

args = parser.parse_args()
model_file = args.model[0]
outdir = args.outdir[0]
trials = args.num_trials
test_path = args.test_dataset[0]
print(args.test_dataset)


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


def get_model(model_file):
    custom_objs, metrics = get_metrics()
    conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)
    conv_model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=metrics)
    return conv_model, custom_objs


def parse_tfrecord_control(example):

    feature_description = {
        "CH02": tf.io.FixedLenFeature([], tf.string),
        "CH05": tf.io.FixedLenFeature([], tf.string),
        "CH13": tf.io.FixedLenFeature([], tf.string),
        "CH15": tf.io.FixedLenFeature([], tf.string),
        "FED_accum_60min_2km": tf.io.FixedLenFeature([], tf.string),
    }
    features = tf.io.parse_single_example(example, feature_description)

    #AURE:
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
    # Unscaling GLM (Only for TFRecs with max)
    #fed_accum = fed_accum * glm_std_aure + glm_mean_aure

    # binarize
    zero = tf.zeros_like(fed_accum)
    one = tf.ones_like(fed_accum)
    targets = tf.where(fed_accum >= BINARIZE, x=one, y=zero)

    print(features['input_1'].shape)
    print(features['input_2'].shape)
    print(features['input_3'].shape)
    print(targets.shape)

    return (features, targets)


def eval(conv_model, custom_objs, test_filenames, t):

    n_samples = len(test_filenames)
    print("\nNumber of samples:", n_samples, "\n")

    # Create the TFRecordDataset
    test_ds = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
    test_ds = (
        test_ds.map(parse_tfrecord_control, num_parallel_calls=AUTOTUNE)
        .batch(BATCHSIZE)  # Don't batch if making numpy arrays
        .prefetch(AUTOTUNE)
    )

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    test_ds = test_ds.with_options(options)

    # Perform eval
    eval_results = conv_model.evaluate(test_ds)
    cter = 0
    dict_results = collections.OrderedDict(
        {
            "n_gpu": NGPU,
            "batch_size": BATCHSIZE,
            "n_samples": n_samples,
            #"n_duplicates": amt_to_add,
        }
    )
    for key in custom_objs:
        dict_results[key]= np.round(eval_results[cter], 5)
        cter += 1
    del eval_results

    final_outdir = outdir.strip()+str(t)

    os.makedirs(final_outdir, exist_ok=True)
    pkl_file = "eval_results_sumglm_JScaler.pkl"
    pickle.dump(dict_results, open(f"{final_outdir}/"+pkl_file, "wb"))
    del dict_results
    print(f"Saved {final_outdir}/" + pkl_file)


#---- MAIN ---------#
test_filenames = glob.glob(f"{test_path}/*/*.tfrec")
num_samples = len(test_filenames)
conv_model, custom_objs = get_model(model_file)

for t in range(497, 497+trials):
    bootstp_list = []
    for s in range(num_samples):
        idx = random.randint(0,num_samples-1)
        bootstp_list.append(test_filenames[idx])
    eval(conv_model, custom_objs, bootstp_list, t)
