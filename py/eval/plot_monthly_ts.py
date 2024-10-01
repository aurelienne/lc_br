import pickle
import glob
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

eval_prefix = '/home/ajorge/lc_br/data/results/eval/' 
data_dir = '/ships22/grain/ajorge/data/tfrecs_sumglm/test/2021/'
figs_dir = '/home/ajorge/lc_br/figs/'

#models = ['original_LC', 'tuned_LC_w1.0_1stEncLastDec', 'tuned_LC_w1.0_Bot', 'tuned_LC_w1.0_BotDec', 'tuned_LC_w1.0_EncBot', 'tuned_LC_w1.0_LastDec', 'tuned_LC_w1.0_Full']
models = ['original_LC', 'fine_tune/full']
colors = ['red', 'green']
#model_labels = ['Original', '1stEncLastDec', 'Bottleneck', 'BotDec', 'EncBot', 'LastDec', 'Full']
model_labels = ['Original', 'Full-FT']
months = ['01', '02', '03', '04', '09', '10', '11', '12']
months_labels = ['Jan/21', 'Feb/21', 'Mar/21', 'Apr/21', 'Sep/21', 'Oct/21', 'Nov/21', 'Dec/21']


def get_num_samples(mm):
    num_samples = len(glob.glob(data_dir + f'????{mm}*/*.tfrec'))
    return num_samples

def get_metric_values(model, metric, month):
    eval_pkl = os.path.join(eval_prefix, model, 'month' + month, 'eval_results_sumglm_JScaler.pkl')
    p = pickle.load(open(eval_pkl, 'rb'))
    return float(p[metric])

def plot_monthly_metric(metric, metric_ext, metric2=None, metric2_ext=None, metric3=None, metric3_ext=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    for i, model in enumerate(models):
        color = colors[i]
        metric_values = []
        metric2_values = []
        metric3_values = []
        num_samples = []

        for mm in months:
            metric_values.append(get_metric_values(model, metric, mm))
            num_samples.append(get_num_samples(mm))

            if metric2 != None:
                metric2_values.append(get_metric_values(model, metric2, mm))
            if metric3 != None:
                metric3_values.append(get_metric_values(model, metric3, mm))

        ax1.plot(metric_values, color=color, label=f'{model_labels[i]} - {metric_ext}', marker='o', linestyle='-', markersize=5)
        ax2.bar(range(len(months)), height=num_samples, color='lightgrey', alpha=0.3, label='Number of Samples')

        if metric2 != None:
            ax1.plot(metric2_values, color=color, label=f'{model_labels[i]} - {metric2_ext}', marker='o', linestyle='-.', markersize=5)

        if metric3 != None:
            ax1.plot(metric3_values, color=color, label=f'{model_labels[i]} - {metric3_ext}', marker='o', linestyle='--', markersize=5)

    ax1.legend()
    #ax1.set_ylabel(metric_ext + '(lines)')
    ax1.set_ylabel('Metric values (lines)')
    ax2.set_ylabel('Number of Samples (bars)')
    plt.grid(axis='y', alpha=0.5)
    plt.xticks(range(len(months)), months_labels)
    plt.savefig(os.path.join(figs_dir, f'monthly_{metric}.png'))
    plt.close()

plot_monthly_metric('aupr', 'AUC - Precision/Recall')
plot_monthly_metric('csi35_index0', 'CSI35', 'far35_index0', 'FAR35', 'pod35_index0', 'POD35')
