import pickle
import sys
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import os
import argparse
from matplotlib import gridspec
from datetime import datetime


def plot_one_model(pkl_file):
    p = pickle.load(open(pkl_file, 'rb'))
    for idx, preds in enumerate(p['preds']):
        fig, axs = plt.subplots(1,2, figsize=(8, 4))

        # Target
        im1 = axs[0].imshow(p['labels'][idx], cmap='inferno')
        axs[0].set_title('Target')
        # Model Prediction
        im2 = axs[1].imshow(preds.astype(np.float32), vmin=0, vmax=1, cmap='inferno')
        axs[1].set_title('Prediction')

        # Remove ticks
        axs[0].tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)
        axs[1].tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)

        #cbar = fig.colorbar(im, ax=axs.ravel().tolist())
        cbar1 = fig.colorbar(im1, ax=axs[0])
        cbar1.set_label('Flash Extent Density')
        cbar1.ax.set_position([0.42, 0.2, 0.05, 0.6])  # [left, bottom, width, height]

        cbar2 = fig.colorbar(im2, ax=axs[1])
        cbar2.set_label('Prediction Probability')
        cbar2.ax.set_position([0.84, 0.2, 0.05, 0.6])  # [left, bottom, width, height]

        plt.savefig(os.path.join(out_dir, f"{idx}.png"))
        break

def plot_two_models(pkl_file1, pkl_file2, label1, label2):
    p1 = pickle.load(open(pkl_file1, 'rb'))
    p2 = pickle.load(open(pkl_file2, 'rb'))

    for idx, preds1 in enumerate(p1['preds']):
        preds2 = p2['preds'][idx]
        fig, axs = plt.subplots(1,3, figsize=(10, 4))

        # Target
        im = axs[0].imshow(p1['labels'][idx], cmap='plasma')
        axs[0].set_title('Target')

        # Prediction of Model 1 
        im2 = axs[1].imshow(preds1.astype(np.float32), vmin=0, vmax=1, cmap='plasma')
        axs[1].set_title(label1)

        # Prediction of Model 2
        im2 = axs[2].imshow(preds2.astype(np.float32), vmin=0, vmax=1, cmap='plasma')
        axs[2].set_title(label2)

        axs[0].tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)
        axs[1].tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)
        axs[2].tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)

        cbar1 = fig.colorbar(im, ax=axs[0])
        cbar1.set_label('Flash Extent Density')
        cbar1.ax.set_position([0.32, 0.2, 0.05, 0.6])  # [left, bottom, width, height]

        cbar2 = fig.colorbar(im2, ax=axs[1:3].ravel().tolist())
        cbar2.set_label('Prediction Probability')
        cbar2.ax.set_position([0.82, 0.2, 0.05, 0.6])  # [left, bottom, width, height]
        plt.savefig(os.path.join(out_dir, f"{idx}.png"))



def plot_models_contours(pkl_file1, pkl_file2, label1, label2, use_dates=False):
    p1 = pickle.load(open(pkl_file1, 'rb'))
    p2 = pickle.load(open(pkl_file2, 'rb'))

    for idx, preds1 in enumerate(p1['preds']):
        preds2 = p2['preds'][idx]
        fig, axs = plt.subplots(1,2, figsize=(10, 5))

        # Target
        im = axs[0].imshow(p1['labels'][idx], cmap='Blues')
        # Prediction of Model 1 
        im2 = axs[0].contour(preds1.astype(np.float32), vmin=0, vmax=1, levels=[0.25, 0.50, 0.75, 1.0], alpha=0.6, cmap='jet')
        axs[0].set_title(label1)

        # Target
        im = axs[1].imshow(p2['labels'][idx], cmap='Blues')
        # Prediction of Model 2 
        contour = axs[1].contour(preds2.astype(np.float32), vmin=0, vmax=1, levels=[0.25, 0.50, 0.75, 1.0], alpha=0.6, cmap='jet')
        axs[1].set_title(label2)

        axs[0].tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)
        axs[1].tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)

        cbar1 = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
        cbar1.set_label('Flash Extent Density')
        cbar1.ax.set_position([0.15, 0.20, 0.3, 0.1])  # [left, bottom, width, height]

        # Extract the colormap used by the contour plot
        cmap = contour.cmap
        norm = contour.norm
        handles = [plt.Line2D([0], [0], color=cmap(norm(level)), lw=2) for level in contour.levels]
        labels = [f'{level:.0f} %' for level in contour.levels*100]
        plt.legend(handles, labels, title='Probability Level', loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels))
        #plt.text(0.5, -0.25, "Probability Levels", ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)

        if use_dates:
            dt = datetime.strftime(p2['datetimes'][idx], '%Y%m%d%H%M')
            plt.savefig(os.path.join(out_dir, f"{dt}.png"))
            if idx == 0 or idx == 1002:
                plt.savefig(os.path.join(out_dir, f"{dt}.eps"), format='eps')
        else:
            plt.savefig(os.path.join(out_dir, f"{idx}.png"))
            if idx == 0 or idx == 1002:
                plt.savefig(os.path.join(out_dir, f"{idx}.eps"), format='eps')

def plot_combined_samples(pkl_file1, pkl_file2, label1, label2, samples_ids):
    p1 = pickle.load(open(pkl_file1, 'rb'))
    p2 = pickle.load(open(pkl_file2, 'rb'))

    #fig, axs = plt.subplots(len(samples_ids), 2, figsize=(8, 8))
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(len(samples_ids), 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.1)
    #plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0.1, wspace=0.1, hspace=0.2)
    for i, sample_id in enumerate(samples_ids):
        preds1 = p1['preds'][sample_id]
        preds2 = p2['preds'][sample_id]
        target = p1['labels'][sample_id]

        # Target
        ax1 = plt.subplot(gs[i, 0])
        ax1.set_ylabel(f'Sample {i}')
        im = ax1.imshow(target, cmap='Blues')
        # Prediction of Model 1
        im2 = ax1.contour(preds1.astype(np.float32), vmin=0, vmax=1, levels=[0.25, 0.50, 0.75, 1.0], alpha=0.6, cmap='jet')
        # Target
        ax2 = plt.subplot(gs[i, 1])
        im = ax2.imshow(target, cmap='Blues')
        # Prediction of Model 2
        contour = ax2.contour(preds2.astype(np.float32), vmin=0, vmax=1, levels=[0.25, 0.50, 0.75, 1.0], alpha=0.6, cmap='jet')

        ax1.tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)
        ax2.tick_params(left = False, right = False , labelleft = False,
                labelbottom = False, bottom = False)

        if i == 0:
            ax1.set_title(label1)
            ax2.set_title(label2)

    cbar1 = fig.colorbar(im, ax=[ax1, ax2], orientation='horizontal')
    cbar1.set_label('Flash Extent Density')
    cbar1.ax.set_position([0.1, 0.00, 0.3, 0.1])  # [left, bottom, width, height]

    # Extract the colormap used by the contour plot
    cmap = contour.cmap
    norm = contour.norm
    handles = [plt.Line2D([0], [0], color=cmap(norm(level)), lw=2) for level in contour.levels]
    labels = [f'{level:.0f} %' for level in contour.levels*100]
    plt.legend(handles, labels, title='Probability Level', loc='upper center', bbox_to_anchor=(0.45, -0.1), ncol=len(labels))
    plt.subplots_adjust(bottom=0.15, top=0.95)

    plt.savefig(os.path.join(out_dir, "merged_samples.png"))
    plt.savefig(os.path.join(out_dir, "merged_samples.eps"), format='eps')

if __name__ == "__main__":

    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Invalid number of arguments. Please, inform one of: \n")
        print(f'{sys.argv[0]} <pkl_file1> <out_dir> [pkl_file2] [list_of_samples]')
        sys.exit(1)

    pkl_file = sys.argv[1]
    out_dir = sys.argv[2]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    elif len(sys.argv) == 3:
        plot_one_model(pkl_file)
    elif len(sys.argv) == 4:
        pkl_file2 = sys.argv[3]
        #plot_two_models(pkl_file, pkl_file2, 'Prediction of \nOriginal Model', 'Prediction of\nFine-Tuned Model')
        plot_models_contours(pkl_file, pkl_file2, 'Prediction of \nOriginal Model', 'Prediction of\nFine-Tuned Model', True)
    elif len(sys.argv) == 5:
        pkl_file2 = sys.argv[3]
        samples_ids = [int(val) for val in sys.argv[4].split(',')]
        plot_combined_samples(pkl_file, pkl_file2, 'Prediction of \nOriginal Model', 'Prediction of\nFine-Tuned Model', samples_ids)

