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
These datasets are available at:
- GOES-ABI channels: ...
- GLM-Level2: ...

## Environment Installation
- Python libs (yaml file for conda)...
- Install glmtools...

## How to Use
### Data Pre-processing
1. Adjust paths and other parameters at the config.ini file.
2. Activate glmtools environment which was previously created.
3. Create gridded fields of “flash-extent density” aggregating the GLM files into 1-minute intervals, and cropping the data into the spatial domain of interest:
```
cd py/preproc
python generate_gridded_fields.py [config_file_path] [start_date(YYYYmmddHHMM)] [end_date(YYYYmmddHHMM)] [delta_minutes]
```
OR, to run multiple intervals, adapt the script run_gen_gridded_fields.sh and run it:
```
cd py/preproc
./run_gen_gridded_fields.sh
```


