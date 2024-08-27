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
#SBATCH --job-name=lc_eval

#NOTE - cpu-per-gpu and mem-per-gpu defaults are 16G of ram per GPU and 6 cores.
# these are reasonable defaults, but can be changed with --cpus-per-gpu and --mem-per-gpu

#point at container in your home directory
#CONTAINER=/home/shared/containers/cuda-11.4_cudnn-8.2.2.26_tensorflow-2.4.3.sif
CONTAINER=/home/shared/containers/tensorflow_22.04-tf2-py3.sif

source /etc/profile


# Run script
#/usr/bin/time -v
#singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_eval.py -m /home/ajorge/lc_br/models/ft_w1.0_1stEncLastDec.h5 -i /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/results/eval_2021_finetuned_w1.0_1stEncLastDec/
#singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_eval.py -m /home/ajorge/lc_br/data/results/fine_tune_w1.0/Enc_Bot/fit_conv_model.h5 -i /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/data/results/eval/tuned_LC_w1.0_EncBot/
#singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_eval.py -m /home/ajorge/lc_br/data/results/lr10-4/fine_tune/conv2d_16/fit_conv_model.h5 -i /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/data/results/eval/tuned_LC_subset0.10_w1.0_LastDec/ -s
#singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_eval.py -m /home/ajorge/lc_br/data/results/fine_tune_w1.0/full/fit_conv_model.h5 -i /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/data/results/eval/tuned_LC_w1.0_Full/
singularity run -B /ships22 -B /ships19 -B $HOME/local-TF:$HOME/miniconda3 --nv $CONTAINER python tf_eval.py -m /home/ajorge/lc_br/data/results/lr10-4/fine_tune/full/fit_conv_model.h5 -i /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/data/results/eval/tuned_LC_w1.0_Full_20211110/ -s
