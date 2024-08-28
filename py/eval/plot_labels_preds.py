import pickle
import sys
from matplotlib import pyplot as plt
import numpy as np
import os

pkl_file = sys.argv[1]
figs_path = '/home/ajorge/lc_br/figs/samples/'

p = pickle.load(open(pkl_file, 'rb'))
for idx, preds in enumerate(p['preds']):
    fig, axs = plt.subplots(1,2)
    #print(p['labels'][idx])
    axs[0].imshow(p['labels'][idx])
    axs[0].set_title('Target')
    axs[1].imshow(preds.astype(np.float32), vmin=0, vmax=1)
    axs[1].set_title('Prediction')
    plt.savefig(os.path.join(figs_path, f"{idx}.png")) 
