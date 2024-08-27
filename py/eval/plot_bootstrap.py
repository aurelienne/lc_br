import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import os
import numpy as np
import sys
import pickle

prefix_path = '/home/ajorge/lc_br/data/results/eval/'
models = ['original_LC', 'tuned_LC_w1.0_Full']
metric = 'aupr'

boxplots = []
for model in models:
    model_values = []
    for i in range(500):
        folder = os.path.join(prefix_path, model, str(i))
        if not os.path.exists(folder):
            continue
        with open(os.path.join(folder, 'eval_results_sumglm_JScaler.pkl'), 'rb') as pf:
            p = pickle.load(pf)
            model_values.append(float(p[metric]))
    boxplots.append(np.array(model_values))

fig, ax = plt.subplots()
ax.boxplot(boxplots, patch_artist=True) #, tick_labels=models)
plt.xticks(range(1, len(models)+1), models)
plt.ylabel(metric)
plt.savefig('boxplots.png')
plt.close()
