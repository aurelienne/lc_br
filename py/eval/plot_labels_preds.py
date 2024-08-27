import pickle
import sys
from matplotlib import pyplot as plt

pkl_file = sys.argv[1]

p = pickle.load(open(pkl_file, 'rb'))
for idx, preds in enumerate(p['preds']):
    fig, axs = plt.subplots(1,2)
    #print(preds)
    #print(p['labels'][idx])
    ax[0].imshow(p['labels'][idx])
    ax[1].imshow(preds)
    plt.show()
    break

