#!/bin/bash

# example using singularity

# you need permission to submit jobs to the salvador partition.
# alternately, use "all" partition for testing, this will work due to 
# the 'gres=gpu:1'. Salvador partition jobs can preempt "All" partition

# If you have priority to salvador, you can remove a comment from the next line
##SBATCH -w r740-105-19
#SBATCH --partition=salvador #--exclude gustav
#SBATCH -w gustav
##SBATCH --nodes=1-1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-gpu=32000
##SBATCH --mem=40000
#SBATCH --time=23:59:00 
#SBATCH --output=/home/%u/logs/sb_%j.log
##output directory must exist or the job will silently fail
#SBATCH --mail-type=END
#SBATCH --mail-user=souzajorge@wisc.edu
#SBATCH --job-name=fit_full

#NOTE - cpu-per-gpu and mem-per-gpu defaults are 16G of ram per GPU and 6 cores.
# these are reasonable defaults, but can be changed with --cpus-per-gpu and --mem-per-gpu

#point at container in your home directory
#CONTAINER=/home/shared/containers/cuda-11.4_cudnn-8.2.2.26_tensorflow-2.4.3.sif
CONTAINER=/home/shared/containers/tensorflow_22.04-tf2-py3.sif

source /etc/profile


# Run script
#/usr/bin/time -v
#singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_fit_LCfull.py -m /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/data/results/lr10-4/fit_full_3/ -t /ships22/grain/ajorge/data/tfrecs_sumglm/train/2020/ -v /ships22/grain/ajorge/data/tfrecs_sumglm/val/2020/
singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_fit_LCfull.py -m /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/data/results/lr10-4/fit_full_subset0.75_3/ -v /ships22/grain/ajorge/data/tfrecs_sumglm/val/2020/ -tf /home/ajorge/lc_br/data/subset_train_list_0.75_1.txt
#singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_finetune.py -m /home/ajorge/lc_br/data/results/fit_full_subset1.0/model-48-0.069863.h5 -o /home/ajorge/lc_br/data/results/fit_full_subset1.0_cont/ -t /ships22/grain/ajorge/data/tfrecs_sumglm/train/2020/ -v /ships22/grain/ajorge/data/tfrecs_sumglm/val/2020/ -lr 0.000000001 -l full  # Continuation of Full Fit must be done with finetuning code
