import sys
import matplotlib
matplotlib.use("Agg")

import os
from matplotlib import pyplot as plt
import csv

prefix_path = '/home/ajorge/lc_br/data/results/'
models = ['fine_tune_w1.0/1stEnc_LastDec', 'fine_tune_w1.0/Bot', 'fine_tune_w1.0/Enc_Bot', 'fine_tune_w1.0/conv2d_16', 'fine_tune_w1.0/conv2d_8', 'fine_tune_w1.0/full', 'fit_full']

def read_log(metric, model):
    if model == 'fit_full':
        log_file = 'merged_log.csv'
    else:
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
            model  = model.split('/')[1]
        if model == 'Bot':
            model = 'FT-Btn'
        elif model == 'conv2d_16':
            model = 'FT-LDec'
        elif model == 'conv2d_8':
            model = 'FT-DecBtn'
        elif model == '1stEnc_LastDec':
            model = 'FT-FEncLDec'
        elif model == 'Enc_Bot':
            model = 'FT-EncBtn'
        elif model == 'full':
            model = 'FT-Full'
        elif model == 'fit_full':
            model = 'TS'

        plt.plot(loss, label=model)

    if metric == 'loss':
        fname = 'train_loss.png'
        label = 'Training Loss'
    elif metric == 'val_loss':
        fname = 'val_loss.png'
        label = 'Validation Loss'
    plt.ylim([0.070, 0.125])  
    #plt.gca().set_ylim(top=0.12)
    plt.ylabel(label)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(axis='y',alpha=0.5)
    plt.savefig(fname)
    plt.close()


plot_loss('loss')
plot_loss('val_loss')
