import glob
import random
import os
import numpy as np
import sys
import argparse
import math

percent = float(sys.argv[1])
num_lists = int(sys.argv[2])
train_path = "/ships22/grain/ajorge/data/tfrecs_sumglm/train/2020/"
out_path = "/home/ajorge/lc_br/data/"


def create_list(i):
    subset_list = []
    while True:
        idx = random.randint(0,num_samples-1)
        if train_filenames[idx] in subset_list:
            continue
        subset_list.append(train_filenames[idx])
        if len(subset_list) == size_subset:
            break
    
    with open(os.path.join(out_path, f"subset_train_list_{percent}_{i}.txt"), 'w') as f:
        for item in subset_list:
            f.write(f"{item}\n")


if __name__ == "__main__":
    train_filenames = glob.glob(f"{train_path}/*/*.tfrec")
    num_samples = len(train_filenames)
    size_subset = math.trunc(percent * num_samples)
    for i in range(1, num_lists+1):
        create_list(i)
