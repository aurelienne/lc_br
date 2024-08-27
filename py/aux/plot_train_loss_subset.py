import sys
import matplotlib
matplotlib.use("Agg")

import os
from matplotlib import pyplot as plt
import csv

prefix_path = '/home/ajorge/lc_br/data/results/lr10-4/'
figs_path = '/home/ajorge/lc_br/figs/'
#models = ['fine_tune_subset0.10', 'fine_tune_subset0.25', 'fine_tune_subset0.50', 'fine_tune_subset0.75', 'fine_tune_w1.0/full', 'fit_full_subset0.10', 'fit_full_subset0.25', 'fit_full_subset0.50', 'fit_full_subset0.75','fit_full']
#models = ['fine_tune_subset0.10/full', 'fine_tune_subset0.25/full', 'fine_tune_subset0.50/full', 'fine_tune_subset0.75/full', 'fine_tune', 'fit_full_subset0.10', 'fit_full_subset0.25', 'fit_full_subset0.50', 'fit_full_subset0.75']
#models = ['fine_tune_subset0.10/full', 'fine_tune_w1.5_subset0.10/full', 'fine_tune_w2.0_subset0.10/full', 'fine_tune_w15.0_subset0.10/full', 'fit_full_subset0.10']
models = ['fine_tune_w1.0_subset0.10/full', 'fine_tune_w1.0_subset0.25/full', 'fine_tune_w1.0_subset0.50/full', 'fine_tune', 'fit_full_subset0.10', 'fit_full_subset0.25', 'fit_full_subset0.50', 'fit_full_subset0.75']

def read_log(metric, model):
    log_file = 'log.csv'
    log_file = os.path.join(prefix_path, model, log_file)
    with open(log_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        metric_values = [float(row[metric]) for row in reader]
    return metric_values


def plot_loss(metric):
    for model in models:
        loss = read_log(metric, model)

        if '/' in model:
            model = model.split('/')[0]
        if model.startswith('fine_tune'):
            model_label = 'FT_'
        else:
            model_label = 'TS_'

        try:
            model_sufix = model.split('subset')[1]
        except:
            model_sufix = '1.0'

        model_label = model_label + model_sufix

        if '_w' in model:
            weight = (model.split('_w')[1]).split('_')[0]
            model_label = model_label + f"_W{weight}"

        plt.plot(loss, label=model_label)

    if metric == 'loss':
        fname = 'train_loss_lr10-4_2.png'
        label = 'Training Loss'
    elif metric == 'val_loss':
        fname = 'val_loss_lr10-4_2.png'
        label = 'Validation Loss'
    elif metric == 'val_aupr':
        fname = 'val_aupr_lr10-4_2.png'
        label = 'Validation AUC-PR'
    elif metric == 'val_csi35':
        fname = 'val_csi35_lr10-4_2.png'
        label = 'Validation CSI at 0.35'
    #plt.ylim([0.070, 0.125])  
    #plt.gca().set_ylim(top=0.12)
    plt.ylabel(label)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(axis='y',alpha=0.5)
    plt.savefig(os.path.join(figs_path, fname))
    plt.close()


plot_loss('loss')
plot_loss('val_loss')
plot_loss('val_aupr')
plot_loss('val_csi35')
