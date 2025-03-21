#!/bin/bash

# example using singularity

# you need permission to submit jobs to the salvador partition.
# alternately, use "all" partition for testing, this will work due to 
# the 'gres=gpu:1'. Salvador partition jobs can preempt "All" partition

# If you have priority to salvador, you can remove a comment from the next line
##SBATCH -w r740-105-15
#SBATCH --partition=salvador 
##SBATCH -w gustav
#SBATCH --nodes=1-1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=20000
#SBATCH --time=23:59:00 
#SBATCH --output=/home/%u/logs/sb_ft_%j.log
##output directory must exist or the job will silently fail
#SBATCH --mail-type=END
#SBATCH --mail-user=souzajorge@wisc.edu
#SBATCH --job-name=finetune

#NOTE - cpu-per-gpu and mem-per-gpu defaults are 16G of ram per GPU and 6 cores.
# these are reasonable defaults, but can be changed with --cpus-per-gpu and --mem-per-gpu

#point at container in your home directory
#CONTAINER=/home/shared/containers/cuda-11.4_cudnn-8.2.2.26_tensorflow-2.4.3.sif
CONTAINER=/home/shared/containers/tensorflow_23.04-tf2-py3.sif

source /etc/profile

modeldir=/ships22/grain/probsevere/LC/tests/2019-2020/c02051315_poswt5/

# Run script
#/usr/bin/time -v
#singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/.local --nv $CONTAINER python tf_finetune.py -m /home/ajorge/src/lightningcast-master/lightningcast/static/fit_conv_model.h5 -l full -o /home/ajorge/lc_br/data/results/lr10-4/fine_tune/ -t /ships22/grain/ajorge/data/tfrecs_sumglm/train/2020/ -v /ships22/grain/ajorge/data/tfrecs_sumglm/val/2020/

singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/.local --nv $CONTAINER python tf_finetune.py -m /home/ajorge/src/lightningcast-master/lightningcast/static/fit_conv_model.h5 -l conv2d_8 -o /home/ajorge/lc_br/data/results/fine_tune_subset0.75/ -tf /home/ajorge/lc_br/data/subset_train_list_0.75_2.txt -v /ships22/grain/ajorge/data/tfrecs_sumglm/val/2020/
