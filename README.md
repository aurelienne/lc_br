# Fine-Tuning LightningCast Model

## Description
This project includes code for adapting the [Probsevere LightningCast Model](https://gitlab.ssec.wisc.edu/jcintineo/lightningcast/) (LC), an artificial neural network (ANN) designed for lightning prediction, to a new spatial domain. The adaptation is achieved through fine-tuning, which involves unfreezing part or all of the model's layers to retrain it using a new dataset.
The project comprises codes for the following steps:
- Data pre-processing
- Training
- Evaluation

Originally developed for fine-tuning LC for the Brazilian territory, this project can be easily adapted to other spatial domains. The code is written in Python, with most tasks utilizing the TensorFlow library.

## Required Dataset
The model uses data from four different channels from GOES-16 ABI as input, and from the Geostationary Lightning Mapper (GLM) as target/truth.
- GOES-ABI channels: GOES-16 ABI 0.64-μm Reflectance (CH02) 0.5 km 10 minutes
                     GOES-16 ABI 1.6-μm Reflectance (CH05) 1 km 10 minutes
                     GOES-16 ABI 10.3-μm Brightness Temp. (CH13) 2 km 10 minutes
                     GOES-16 ABI 12.3-μm Brightness Temp. (CH15)
- GLM-Level2
The data sources can be freely obtained from NOAA’s Comprehensive Large Array Data Stewardship System (https://www.class.noaa.gov/CLASS).

## Environment Installation
- Install [glmtools](https://github.com/deeplycloudy/glmtools);
- Activate the created conda environment;
- Install additional packages: `bzip2`, `cartopy`, `hdf4`, `hdf5`, `matplotlib`, `netcdf4`, `numpy`, `pandas`, `openssl`, `scikit-learn`, `tensorflow`, `xarray`, `yaml`.
   
## How to Use
### Data Pre-processing
1. Adjust paths and other parameters at the `PREPROC` section in the `config.ini` file.
2. Activate `glmtools` environment which was previously created.
3. Create gridded fields of “flash-extent density” aggregating the GLM files into 1-minute intervals, and cropping the data into the spatial domain of interest:
```
cd py/preproc
python generate_gridded_fields.py <start_date(YYYYmmddHHMM)> <end_date(YYYYmmddHHMM)> <delta_minutes>
```
OR, to run multiple intervals, adapt the bash script `run_gen_gridded_fields.sh` and run it:
```
cd py/preproc
./run_gen_gridded_fields.sh
```
4. For each timestep (according to the temporal resolution of the dataset - e.g.: 10 minutes if using GOES-16 Full Scan), aggregate flash extent densities for the next 60 minutes.
```
python aggregate_glm_grids.py <start_date(yyyyjjjhhmm)> <input_path> -a 60 -o <output_path>
```
**OR, to run multiple intervals**, adapt the bash script `run_agg_60min.sh` and run it:
```
./run_agg_60min.sh <yyyy> <mm>
```
5. Select files with minimum number of flashes to be used by the network. Adapt parameters inside `FILTER` section in the `config.ini` file, and also parameters `num_patches` and `patch_size` inside `GEO` Section. Then, run:
```
python select_glm_files.py <input_path> <file_pattern> <output_file>
```
Example: 
```
python select_glm_files.py /ships22/grain/ajorge/data/glm_grids_60min_extra/ *.netcdf glm_filtered_60sum_extra.csv
```
6. Write final dataset as TFRecords, standardizing variables (standard deviation = 1 and mean = 0) to fit the same distribution as used by the Control Version of LightningCast.
```
python generate_TFRecords.py <glm_60min_path> <output_path> <list_filtered_GLMfiles>
```
7. Discard patches with space-look pixels (NaN values).
```
python check_TFRecords.py
```
8. Plot sample TFRecords to check data.
```
python plot_TFRecords.py
```
### Fine-Tuning
The fine-tuning process can use training samples specified either through a text file listing the full path of each sample (`-tf`) or by pointing to an entire training directory (`-t`):
```
python tf_finetune.py -m <h5_model_file> -l <layer_name> -o <output_directory> -tf <train_file_list> -v <validation_folder>
```
OR
```
python tf_finetune.py -m <h5_model_file> -l <layer_name> -o <output_directory> -t <train_dir> -v <validation_folder>
```
Example:
```
python tf_finetune.py -m /home/ajorge/src/lightningcast-master/lightningcast/static/fit_conv_model.h5 -l conv2d_8 -o /home/ajorge/lc_br/data/results/fine_tune_subset0.75/ -tf /home/ajorge/lc_br/data/subset_train_list_0.75_2.txt -v /ships22/grain/ajorge/data/tfrecs_sumglm/val/2020/
```

### Evaluation
For evaluation, one should indicate the model file (H5 format; option `-m`), the input directory with the config file (option `-i`), and the output directory (option `-o`):
```
python tf_eval.py -m <model_file> -i <config_path> -o <output_dir>
```
Example:
```
python tf_eval.py -m /home/ajorge/lc_br/data/results/lr10-4/fit_full/fit_conv_model.h5 -i /home/ajorge/src/lightningcast-master/lightningcast/static -o /home/ajorge/lc_br/data/results/eval/fitFull_w1.0/
```

## Credits
This project reuses and adapt part of the code of the original [LC Model](https://gitlab.ssec.wisc.edu/jcintineo/lightningcast/).

## License
