#!/bin/bash

# example using singularity

# you need permission to submit jobs to the salvador partition.
# alternately, use "all" partition for testing, this will work due to 
# the 'gres=gpu:1'. Salvador partition jobs can preempt "All" partition

# If you have priority to salvador, you can remove a comment from the next line
##SBATCH -w r740-105-19
#SBATCH --partition=salvador #--exclude gustav
##SBATCH -w gustav
#SBATCH --nodes=1-1
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=20000
#SBATCH --time=23:59:00 
#SBATCH --output=/home/%u/logs/sb_ft_fe_%j.log
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
singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/.local --nv $CONTAINER python tf_finetune.py -m /home/ajorge/lc_br/models/feat_ext.087555.h5 -l full -o /home/ajorge/lc_br/data/results/fineTune_full_featExt_2/ -t /ships22/grain/ajorge/data/tfrecs_sumglm/train/2020/ -v /ships22/grain/ajorge/data/tfrecs_sumglm/val/2020/
