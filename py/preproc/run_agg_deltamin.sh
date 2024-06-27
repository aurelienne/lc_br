#!/bin/bash -x

input_path='/ships22/grain/ajorge/data/glm_grids_1min/'
output_path='/ships22/grain/ajorge/data/glm_grids_'
YY=$1
mm=$2
dd=15
deltas_min=(10 20 30 40 50)

start_hour=1810
end_hour=2200

for delta_min in ${deltas_min[@]}; do
    output_path_dt="${output_path}${delta_min}sum"
    dt=$(date -d "$YY$mm$dd ${start_hour}")
    dt_hour=$(date -d "${dt}" "+%H%M")
    while [ $dt_hour -le $end_hour ]; do
	dt_str=$(date -d "$dt" '+%Y%j%H%M')
        python aggregate_glm_grids.py $dt_str $input_path -a ${delta_min} -o $output_path_dt
        dt=$(date -d "$dt +10 min")
        dt_hour=$(date -d "${dt}" "+%H%M")
    done
done
