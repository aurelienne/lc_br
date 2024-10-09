#!/bin/bash -x

delta_min=1 # to calculate extent density 
#year=2020
year=2021
#months=(01 02 03 12)
#days=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24) #Training and Validation (2020)
#days=(10 11 12 13 14 15)  # Test Dataset (2021)

months=(04 05 06 07 08 09 10 11)
#days=(10 11 12 13 14 15 16 17 18 19) #Training and Validation (2020)
#days=(10 11 12 13) # Test Dataset (2021)
days=(14 15 16 17 18 19) # Test Dataset (2021)

start_time="1800"
end_time="2200"

for month in ${months[@]}; do
        for day in ${days[@]}; do
		start_dt="${year}${month}${day}${start_time}"
		end_dt="${year}${month}${day}${end_time}"
		python generate_gridded_fields.py /home/ajorge/lc_br/py/config.ini ${start_dt} ${end_dt} ${delta_min}
		break
	done
done

