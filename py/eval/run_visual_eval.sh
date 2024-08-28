#!/bin/bash

folder=$1

outdir_base="/home/ajorge/lc_br/data/results/eval/"
outdir="${outdir_base}${folder}"

months=(01 02 03 04 09 10 11 12)
for month in ${months[@]}; do
    /home/ajorge/envs/lc/bin/python /home/ajorge/lc_br/py/eval/visualize_validation.py -o ${outdir} -g /home/ajorge/lc_br/data/GLM_BR_finaldomain.nc -sv -msv -s 7 -plp "${outdir}/month${month}/all_preds_labs.pkl"
done
/home/ajorge/envs/lc/bin/python /home/ajorge/lc_br/py/eval/visualize_validation.py -o ${outdir} -g /home/ajorge/lc_br/data/GLM_BR_finaldomain.nc -sv -ssv -s 7
