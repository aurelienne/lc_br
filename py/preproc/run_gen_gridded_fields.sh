#!/bin/bash

delta_min=5 # to calculate extent density / 5 min
year=2020
#year=2021
#months=(01 02 03 12)
#days=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24) #Training and Validation (2020)
#days=(10 11 12 13 14 15)  # Test Dataset (2021)

months=(04 05 06 07 08 09 10 11)
days=(10 11 12 13 14 15 16 17 18 19) #Training and Validation (2020)
#days=(10 11 12 13) # Test Dataset (2021)

for month in ${months[@]}; do
        for day in ${days[@]}; do
		start_dt="${year}${month}${day}1800"
		end_dt="${year}${month}${day}2200"
		python generate_gridded_fields.py /home/ajorge/lc_br/py/config.ini ${start_dt} ${end_dt} ${delta_min}
	done
done

