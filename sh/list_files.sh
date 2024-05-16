#!/bin/bash -x

prefix_path=/arcdata/goes/grb/goes16/
output_file=/home/ajorge/data/files_list_test_dataset2.txt

#year=2020
year=2021
months=(01 02 03 12)
#days=(10 11 12 13 14 15 16 17 18 19 20 21 22 23 24) #Training and Validation (2020)
days=(10 11 12 13 14 15)  # Test Dataset (2021)

#months=(04 05 06 07 08 09 10 11)
#days=(10 11 12 13 14 15 16 17 18 19) #Training and Validation (2020)
#days=(10 11 12 13) # Test Dataset (2021)

hours=(18 19 20 21)

for month in ${months[@]}; do
	for day in ${days[@]}; do
		for hour in ${hours[@]}; do
			jd=$(date -d "${year}${month}${day}" "+%j")
			prefix_file="${prefix_path}/${year}/${year}_${month}_${day}_${jd}/glm/L2/LCFA/OR_GLM-L2-LCFA_G16_s${year}${jd}${hour}"
			file=$(ls ${prefix_file}??000*)
			printf "${file}\n" >> ${output_file}
		done
	done
done
