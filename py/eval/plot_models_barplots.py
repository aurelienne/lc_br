import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import argparse
import os
import numpy as np
import sys
import pickle
import csv

figs_path = '/home/ajorge/lc_br/figs/'
models_path = '/home/ajorge/lc_br/data/results/lr10-4/'

originalLC_eval = '/home/ajorge/lc_br/data/results/eval/original_LC_valDataset/eval_results_sumglm_JScaler.pkl' # Evaluation over validation dataset in order to have the same comparison basis

def read_pkl_eval(metric, model):
    if 'val_' in metric:
        metric = metric.split('val_')[1]
    pf = pickle.load(open(model, 'rb')) 
    try:
        value = pf[metric]
    except:
        metric = metric + '_index0'
        value = pf[metric]
    return value

def read_log(metric, model):
    log_file = 'log.csv'
    log_file = os.path.join(models_path, model, log_file)
    with open(log_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        metric_values = [float(row[metric]) for row in reader]
    return metric_values

def get_model_groups(models, metric, plot_type, dataset):
    # Get metric value from original model
    orig_value = read_pkl_eval(metric, originalLC_eval)

    # Get and plot metric from other models
    percents, model_labels, values = [], [], []
    for model in models:
        if dataset == 'val': # Read from training log
            value = read_log(metric, model)[-1] # Get only metric value from last epoch
        elif dataset == 'test':
            value = read_pkl_eval(metric, os.path.join(models_path, model))

        """
        if '/' in model:
            model = model.split('/')[0]
        if model.startswith('fine_tune'):
            model_label = 'FT_'
        else:
            model_label = 'TS_'
        """

        if plot_type == 'ts':
            try:
                model_sufix = float(model.split('subset')[1])
            except:
                model_sufix = 1.0
            model_sufix = (model_sufix*100)
            model_sufix = "{:.0f}".format(model_sufix)
            #model_label = model_label + model_sufix
            model_labels.append(model_sufix)
        elif plot_type == 'ft':
            model_sufix = model.split('/')[1]
            model_labels.append(model_sufix)

        percent = (value / orig_value - 1) * 100
        percents.append(percent)
        values.append(value)

    return model_labels, percents, values

def adjust_model_labels(labels):
    new_labels = []
    for label in labels:
        print(label)
        match label:
            case 'conv2d_16':
                new_label = 'LDec'
            case '1stEnc_LastDec':
                new_label = 'FEnc_LDec'
            case 'Bot':
                new_label = 'Btn'
            case 'Enc_Bot':
                new_label = 'Enc_Btn'
            case 'conv2d_8':
                new_label = 'Btn_Dec'
            case _:
                new_label = label
        new_labels.append(new_label)
    return new_labels


def plot_bars_TS_vs_FT(metric, fname, ylabel, dataset):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_zorder(ax1.get_zorder()+10)
    ax1.patch.set_visible(False)

    # Get metric value from original model
    orig_value = read_pkl_eval(metric)
    ax2.axhline(orig_value, label='Original Model', color='red', linestyle='dashed', linewidth=2)

    ts_labels, ts_perc, ts_values = get_model_groups(ts_models, metric, 'ts', dataset)
    ft_labels, ft_perc, ft_values = get_model_groups(ft_models, metric, 'ft', dataset)

    values = ts_values + ft_values
    ts_idxs = np.arange(1, len(ts_values)+1)+0.2
    ft_idxs = np.arange(1, len(ft_values)+1)-0.2
    ax2.plot(ts_idxs, ts_values, marker='o', color='green')
    ax2.plot(ft_idxs, ft_values, marker='o', color='blue')

    # Legend from bars
    lblue_patch = mpatches.Patch(color='lightgreen', label='Train from Scratch')
    blue_patch = mpatches.Patch(color='lightblue', label='Fine-Tuning')
    labels1 = ['Train from Scratch', 'Fine-Tuning']

    ax1.bar(ts_idxs, height=ts_perc, width=0.5, color='lightgreen')
    ax1.bar(ft_idxs, height=ft_perc, width=0.5, color='lightblue')
    plt.xticks(range(1, len(ts_labels)+1), ts_labels) 
    ax1.set_ylabel(f"% of Improvement [{ylabel}]\n(bars)")
    ax2.set_ylabel(f"{ylabel}\n(lines)")

    #ax1.tick_params(axis='x', labelrotation=90)
    #ax1.set_ylim(bottom=-1.2)
    # Adjust the margins
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.87)
    plt.grid(axis='y', alpha=0.5)

    # Get the handles and labels from both axes
    handles1 = [lblue_patch, blue_patch]
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2

    ax1.set_xlabel('Amount of Data (%)')
    plt.legend(handles, labels)
    outdir = os.path.join(figs_path, 'ts_vs_ft')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    plt.savefig(os.path.join(outdir, fname + '.png'))
    plt.savefig(os.path.join(outdir, fname + '.eps'), format='eps')
    plt.close()


def plot_bars_FT_opts(metric, fname, ylabel, dataset):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_zorder(ax1.get_zorder()+10)
    ax1.patch.set_visible(False)

    # Get metric value from original model
    orig_value = read_pkl_eval(metric)
    ax2.axhline(orig_value, label='Original Model', color='red', linestyle='dashed', linewidth=2)

    ft_labels, ft_perc, ft_values = get_model_groups(ft_models, metric, 'ft', dataset)
    ft_labels = adjust_model_labels(ft_labels)

    ft_idxs = np.arange(1, len(ft_values)+1)
    ax2.plot(ft_idxs, ft_values, marker='o', color='blue')

    # Legend from bars
    #blue_patch = mpatches.Patch(color='lightblue', label='Fine-Tuning')
    #labels1 = ['Train from Scratch', 'Fine-Tuning']

    ax1.bar(ft_idxs, height=ft_perc, width=0.5, color='lightblue')

    plt.xticks(range(1, len(ft_labels)+1), ft_labels) 
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set_ylabel(f"% of Improvement [{ylabel}]\n(bars)")
    ax2.set_ylabel(f"{ylabel}\n(lines)")

    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.87)
    plt.grid(axis='y', alpha=0.5)

    # Get the handles and labels from both axes
    #handles1 = [lblue_patch, blue_patch]
    #handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    #handles = handles1 + handles2
    #labels = labels1 + labels2

    #ax1.set_xlabel('Fine-Tuning Options')
    #plt.legend(handles, labels)
    outdir = os.path.join(figs_path, 'ft_options')
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    plt.savefig(os.path.join(outdir, fname + '.png'))
    plt.savefig(os.path.join(outdir, fname + '.eps'), format='eps')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ts",
        "--ts_vs_ft",
        help="Training from Scratch vs Fine-Tuning Models, with different amount of data.",
        action="store_true",
    )
    parser.add_argument(
        "-ft",
        "--ft_options",
        help="Different options of Fine-Tuning Models.",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--use_test_evaluation",
        help="Whether to use evaluation over test dataset or Validation metrics. To use validation, just omit this option.",
        action="store_true",
    )
    args = parser.parse_args()

    if args.use_test_evaluation:
        dataset = 'test'
    else:
        dataset = 'val'

    if args.ts_vs_ft:
        ts_models = ['fit_full_subset0.10', 'fit_full_subset0.25', 'fit_full_subset0.50', 'fit_full_subset0.75', 'fit_full']
        ft_models = ['fine_tune_subset0.10/full', 'fine_tune_subset0.25/full', 'fine_tune_subset0.50/full', 'fine_tune_subset0.75/full', 'fine_tune/full']
        plot_bars_TS_vs_FT('val_aupr', 'val_aucpr', 'AUC-PR', dataset)
        plot_bars_TS_vs_FT('val_csi35', 'val_csi35', 'CSI-35%', dataset)

    if args.ft_options:
        ft_models = ['fine_tune/conv2d_16', 'fine_tune/1stEnc_LastDec', 'fine_tune/Bot', 'fine_tune/Enc_Bot', 'fine_tune/conv2d_8', 'fine_tune/full']
        plot_bars_FT_opts('val_aupr', 'val_aucpr', 'AUC-PR', dataset)
        plot_bars_FT_opts('val_csi35', 'val_csi35', 'CSI-35%', dataset)

