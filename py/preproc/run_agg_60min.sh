#!/bin/bash -x

input_path='/ships22/grain/ajorge/data/glm_grids_1min/'
output_path='/ships22/grain/ajorge/data/glm_grids_60minsum_prevCode/'
YY=$1
mm=$2
dd=(15)

start_hour=1900
end_hour=2200

#for dd in {10..13}; do
#for dd in {10..24}; do
for dd in ${dd[*]}; do
    dt=$(date -d "$YY$mm$dd ${start_hour}")
    dt_hour=$(date -d "${dt}" "+%H%M")
    while [ $dt_hour -le $end_hour ]; do
	dt_str=$(date -d "$dt" '+%Y%j%H%M')
        python aggregate_glm_grids.py $dt_str $input_path -a 60 -o $output_path
        dt=$(date -d "$dt +10 min")
        dt_hour=$(date -d "${dt}" "+%H%M")
    done
done
