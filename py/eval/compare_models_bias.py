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
import mult_perf_diagrams 


def merge_pkl_files(pkl_list_file):
    dicts_list = []
    pkl_list = open(pkl_list_file, 'r')
    for pkl_file in pkl_list.readlines():
        pkl_dict = pickle.load(open(pkl_file.strip(), 'rb'))
        dicts_list.append(pkl_dict)

    all_dicts = {}
    i = 0
    for d in dicts_list:
        for key, values in d.items():
            if key == 'stride':
                continue
            for value in values:
                if key == 'preds' or key == 'labels':
                    if i == 0:
                        all_dicts[key] = value
                    else:
                        all_dicts[key] = np.concatenate([all_dicts[key], value])
                else:
                    if i == 0:
                        all_dicts[key] = [value,]
                    else:
                        all_dicts[key].append(value)
        i = i + 1

    return all_dicts

def compare_models(pkl_dict1, pkl_dict2):

    difs = []
    for idx, preds1 in enumerate(pkl_dict1['preds']):
        preds2 = pkl_dict2['preds'][idx]

        difs.append(np.mean(preds1 - preds2))

    avg_dif = np.mean(difs)
    return avg_dif


def subtract_dif(pkl_dict1, dif):

    for idx, preds1 in enumerate(pkl_dict1['preds']):
        print(pkl_dict1['preds'][idx])
        pkl_dict1['preds'][idx] = pkl_dict1['preds'][idx] - dif
        np.where(pkl_dict1['preds'][idx] < 0, 0, pkl_dict1['preds'][idx])
        print(pkl_dict1['preds'][idx])

    return pkl_dict1['preds']



if __name__ == "__main__":

    list_pkls1 = sys.argv[1]
    list_pkls2 = sys.argv[2]

    all_dicts1 = merge_pkl_files(list_pkls1)
    all_dicts2 = merge_pkl_files(list_pkls2)
    dif = compare_models(all_dicts1, all_dicts2)
    dicts1_sub = subtract_dif(all_dicts1, dif)

    all_preds1 = dicts1_sub.flatten()
    all_preds1 = np.expand_dims(all_preds1, axis=0)
    all_labels1 = all_dicts1["labels"].flatten()
    all_labels1 = np.expand_dims(all_labels1, axis=0)
    all_preds2 = all_dicts2["preds"].flatten()
    all_labels2 = all_dicts2["labels"].flatten()
    all_preds2 = np.expand_dims(all_preds2, axis=0)
    all_labels2 = np.expand_dims(all_labels2, axis=0)

    all_preds = np.concatenate([all_preds1, all_preds2], axis=0)
    all_labels = np.concatenate([all_labels1, all_labels2])

    scores_dict = mult_perf_diagrams.verification(all_preds, all_labels, tmpdir)

