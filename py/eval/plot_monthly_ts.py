import pickle
import glob
import os
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

eval_prefix = '/home/ajorge/lc_br/data/results/eval/' 
data_dir = '/ships22/grain/ajorge/data/tfrecs_sumglm/test/2021/'

models = ['original_LC', 'tuned_LC_w1.0_1stEncLastDec', 'tuned_LC_w1.0_Bot', 'tuned_LC_w1.0_BotDec', 'tuned_LC_w1.0_EncBot', 'tuned_LC_w1.0_LastDec', 'tuned_LC_w1.0_Full']
model_labels = ['Original', '1stEncLastDec', 'Bottleneck', 'BotDec', 'EncBot', 'LastDec', 'Full']
months = ['01', '02', '03', '04', '09', '10', '11', '12']
months_labels = ['Jan/20', 'Feb/20', 'Mar/20', 'Apr/20', 'Sep/20', 'Oct/20', 'Nov/20', 'Dec/20']


def get_num_samples(mm):
    num_samples = len(glob.glob(data_dir + f'????{mm}*/*.tfrec'))
    return num_samples

def plot_monthly_metric(metric, metric_ext):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder()+1)
    ax1.patch.set_visible(False)
    for i, model in enumerate(models):
        aupr_values = []
        num_samples = []
        for mm in months:
            eval_pkl = os.path.join(eval_prefix, model, 'month' + mm, 'eval_results_sumglm_JScaler.pkl')
            p = pickle.load(open(eval_pkl, 'rb'))
            aupr_values.append(float(p[metric]))
            num_samples.append(get_num_samples(mm))

        ax1.plot(aupr_values, label=model_labels[i], marker='o', linestyle='-', markersize=5)
        ax2.bar(range(len(months)), height=num_samples, color='lightgrey', alpha=0.3, label='Number of Samples')


    ax1.legend()
    ax1.set_ylabel(metric_ext)
    ax2.set_ylabel('Number of Samples')
    plt.grid(axis='y', alpha=0.5)
    plt.xticks(range(len(months)), months_labels)
    plt.savefig(f'monthly_{metric}.png')
    plt.close()

plot_monthly_metric('aupr', 'AUC - Precision/Recall')
plot_monthly_metric('csi35_index0', 'CSI35')
