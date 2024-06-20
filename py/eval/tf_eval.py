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

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--indir",
    help="Directory containing trained CNN model and config.",
    type=str,
    nargs=1,
)
parser.add_argument(
    "-o",
    "--outdir",
    help="Output directory for model and figures. Default=args.indir + '/TEST/'",
    nargs=1,
    type=str,
)
parser.add_argument(
    "-m",
    "--model",
    help="Use this model instead of fit_conv_model.h5 in --indir.",
    type=str,
    nargs=1,
)
parser.add_argument(
    "-n",
    "--ncpus",
    help="Limit TF to this number of CPUs. Default is to use all available.",
    type=int,
    default=[-1],
    nargs=1,
)

args = parser.parse_args()

indir = args.indir[0]

config = pickle.load(open(os.path.join(indir, "model_config.pkl"), "rb"))
model_file = args.model[0] if (args.model) else f"{indir}/fit_conv_model.h5"

outdir = args.outdir[0] if (args.outdir) else os.path.join(indir, "TEST")


ncpus = args.ncpus[0]

BINARIZE = 1

NGPU = 1
batchsize = 4
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

    #AURE:
    ny, nx = 700, 700
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

    #ch02 = tf.reshape(tf.io.parse_tensor(features["CH02"], tf.uint8), [1920, 1920, 1])
    ch02 = tf.reshape(tf.io.parse_tensor(features["CH02"], tf.float32), [ny*4, nx*4, 1])
    features["input_1"] = ch02
    # Unscaling the TensorFlow tensor to native units. Then applying mean/std scaling (according to previous dataset), since the model was trained thusly.
    """
    ch02 = tf.cast(ch02, dtype=tf.float32)
    vmax = tf.constant(1.0, dtype=tf.float32)  # Assuming vmax and vmin are constants
    vmin = tf.constant(
        0.0, dtype=tf.float32
    )  # These are values for how the tfrecords were scaled
    two56 = tf.constant(255.9999, dtype=tf.float32)
    ch02 = ch02 / two56 * (vmax - vmin) + vmin

    ch02 = ch02 * ch02_std_aure + ch02_mean_aure
    mean = tf.constant(0.08713, dtype=tf.float32)
    std = tf.constant(0.15314, dtype=tf.float32)
    features["input_1"] = (ch02 - mean) / std
    """

    ch05 = tf.reshape(tf.io.parse_tensor(features["CH05"], tf.float32), [ny*2, nx*2, 1])
    features["input_2"] = ch05
    """
    # Unscaling the TensorFlow tensor to native units. Then applying mean/std scaling, since the model was trained thusly.
    ch05 = tf.cast(ch05, dtype=tf.float32)
    vmax = tf.constant(0.75, dtype=tf.float32)  # Assuming vmax and vmin are constants
    vmin = tf.constant(
        0.0, dtype=tf.float32
    )  # These are values for how the tfrecords were scaled
    ch05 = ch05 / two56 * (vmax - vmin) + vmin
    ch05 = ch05 * ch05_std_aure + ch05_mean_aure
    mean = tf.constant(0.047332, dtype=tf.float32)
    std = tf.constant(0.0834376, dtype=tf.float32)
    features["input_2"] = (ch05 - mean) / std
    """
    
    ch13 = tf.reshape(tf.io.parse_tensor(features["CH13"], tf.float32), [ny, nx, 1])
    # Unscaling the TensorFlow tensor to native units. Then applying mean/std scaling, since the model was trained thusly.
    """
    ch13 = tf.cast(ch13, dtype=tf.float32)
    vmax = tf.constant(320, dtype=tf.float32)  # Assuming vmax and vmin are constants
    vmin = tf.constant(
        190, dtype=tf.float32
    )  # These are values for how the tfrecords were scaled
    ch13 = ch13 / two56 * (vmax - vmin) + vmin
    ch13 = ch13 * ch13_std_aure + ch13_mean_aure
    mean = tf.constant(270.93, dtype=tf.float32)
    std = tf.constant(25.9126, dtype=tf.float32)
    ch13 = (ch13 - mean) / std
    """

    ch15 = tf.reshape(tf.io.parse_tensor(features["CH15"], tf.float32), [ny, nx, 1])
    # Unscaling the TensorFlow tensor to native units. Then applying mean/std scaling, since the model was trained thusly.
    """
    ch15 = tf.cast(ch15, dtype=tf.float32)
    vmax = tf.constant(320, dtype=tf.float32)  # Assuming vmax and vmin are constants
    vmin = tf.constant(
        190, dtype=tf.float32
    )  # These are values for how the tfrecords were scaled
    ch15 = ch15 / two56 * (vmax - vmin) + vmin
    ch15 = ch15 * ch15_std_aure + ch15_mean_aure
    mean = tf.constant(273.168, dtype=tf.float32)
    std = tf.constant(19.9895, dtype=tf.float32)
    ch15 = (ch15 - mean) / std
    """
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

    return (features, targets)


################################################################################################################

def spatial_eval(test_ds, outdir):
    ny, nx = 700, 700

    ##preallocate
    #stride = 4
    stride = 7
    all_preds = np.empty((n_samples,ny//stride,nx//stride),dtype=np.float32) #Striding to save space and time.
    #all_labels = np.zeros((n_samples,ny//stride,nx//stride),dtype=np.byte)
    all_labels = np.zeros((n_samples,ny//stride,nx//stride),dtype=np.float32)
    ##float32 precision needed for validation routines (e.g., MSE)
    #
    cter = 0
    for inputs,labels in test_ds:
        preds = conv_model.predict(inputs)
        batchsize = preds.shape[0]
        if cter+batchsize >= n_samples:
            batchsize = n_samples - cter 
        all_preds[cter:cter+batchsize] = preds[:batchsize,::stride,::stride]
        all_labels[cter:cter+batchsize] = labels[:batchsize,::stride,::stride]
        cter+=batchsize

    basenames = [os.path.basename(vf) for vf in test_filenames]
    datetimes = [datetime.strptime(x.split('_')[1],'%Y%m%d-%H%M%S') for x in basenames]
    midptY = [int(x.split('_Y')[1].split('_X')[0]) for x in basenames]
    midptX = [int(x.split('_X')[1].split('.')[0]) for x in basenames]
    strideY = np.array(midptY) // stride
    strideX = np.array(midptX) // stride

    ##save the predictions / labels first
    preds_labs_dict1 = {'preds':all_preds,'labels':all_labels,'files':basenames,
                        'datetimes':datetimes,
                        'midptX':midptX, 'midptY':midptY,
                        'strideX':strideX, 'strideY':strideY, 'stride':stride}
    pickle.dump(preds_labs_dict1, open(f"{outdir}/all_preds_labs.pkl","wb"))
    del test_ds, preds, all_preds, all_labels, preds_labs_dict1


################################################################################################################
## MAIN ##

# Load model and set up GPUs
if NGPU > 1:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        custom_objs, metrics = get_metrics()
        conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)
        conv_model.compile(
            optimizer=optimizer, loss=config["loss_fcn"], metrics=metrics
        )
else:
    custom_objs, metrics = get_metrics()
    conv_model = load_model(model_file, custom_objects=custom_objs, compile=False)
    conv_model.compile(optimizer=optimizer, loss=config["loss_fcn"], metrics=metrics)

print(conv_model.summary())
#for layer in conv_model.layers:
#    print(layer.name)


AUTOTUNE = tf.data.AUTOTUNE

# Assuming 'static' means control model (LCv1)
#parse_fn = parse_tfrecord_control if ("static" in model_file) else parse_tfrecord_fn
parse_fn = parse_tfrecord_control 

# Create the test dataset for running the permutation
sat = "goes16"
year = "2021"
#inroot = f"/ships22/grain/ajorge/data/tfrecs_sumglm/test/{year}/"
inroot = f"/ships22/grain/probsevere/LC/tfrecs3/goes16/{year}"

# leave empty if you only want one run
#subdirs = ['01','02','03','04','09','10','11','12']
subdirs = ['01']

if len(subdirs) > 0:
    val_lists = []
    for sub in subdirs:
        val_lists.append(f"{inroot}/{year}{sub}*/*tfrec")
else:
    val_lists = [f"{inroot}/20210620/*.tfrec"]  # this is your "one-run" case

for ii, val_list in enumerate(val_lists):
    test_filenames = sorted(glob.glob(val_list))  # sort to ensure reproducibility
    n_samples = len(test_filenames)
    print("\nNumber of samples:", n_samples, "\n")

    # Create the TFRecordDataset
    test_ds = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
    #features, target = test_ds.map(parse_fn)
    #for features in target.take(1):
    #    print(type(features))
    #    print(tf.reduce_max(features))
    #sys.exit()

    # Get remainder dataset
    _, amt_to_add = make_divisible(n_samples, batchsize, NGPU)
#    assert (
#        amt_to_add / n_samples < 0.01
#    )  # make sure we're not duplicating too many samples
    remainder_ds = test_ds.take(amt_to_add)
    # Concatenate to test_ds
    test_ds = test_ds.concatenate(remainder_ds)

    test_ds = (
        test_ds.map(parse_fn, num_parallel_calls=AUTOTUNE)
        .batch(batchsize)  # Don't batch if making numpy arrays
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
    
    # Plot prediction Sample
    preds = conv_model.predict(test_ds)
    fig,axes = plt.subplots(nrows=3,ncols=3, gridspec_kw = {'wspace':0, 'hspace':0},figsize=(9,9))
    i = 0
    for l in range(3):
        for c in range(3):

            im = axes[l, c].imshow(preds[i,:,:], interpolation=None, cmap='viridis')

            axes[l, c].get_xaxis().set_visible(False)
            axes[l, c].get_yaxis().set_visible(False)
            axes[l, c].set_aspect('equal')
            i = i + 1

    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.tight_layout()
    plt.savefig('/home/ajorge/output/sample_pred.png')
    plt.close()

    cter = 0
    dict_results = collections.OrderedDict(
        {
            "n_gpu": NGPU,
            "batch_size": batchsize,
            "n_samples": n_samples,
            "n_duplicates": amt_to_add,
        }
    )
    for key in custom_objs:
        dict_results[key]= np.round(eval_results[cter], 5)
        cter += 1
    del eval_results 

    # Append outsubdir, if necessary
    try:
        final_outdir = f"{outdir}/month{subdirs[ii]}"
    except IndexError:
        final_outdir = outdir

    os.makedirs(final_outdir, exist_ok=True)
    pkl_file = "eval_results_sumglm_JScaler.pkl"
    pickle.dump(dict_results, open(f"{final_outdir}/"+pkl_file, "wb"))
    del dict_results
    print(f"Saved {final_outdir}/" + pkl_file)

    spatial_eval(test_ds, final_outdir)

