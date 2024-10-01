import netCDF4
import numpy as np
import sys
import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
import glob
import configparser


def get_flashes_per_patch(values):
    num_flashes = []
    for y_pos in range(0, values.shape[0], patch_size[0]):
        for x_pos in range(0, values.shape[1], patch_size[1]):
            y_pos_end = y_pos + patch_size[0]
            x_pos_end = x_pos + patch_size[1]
            #num_flashes.append(np.sum(np.count_nonzero(values[y_pos:y_pos_end, x_pos:x_pos_end])))
            num_flashes.append(np.max(values[y_pos:y_pos_end, x_pos:x_pos_end]))
    return num_flashes

def write_csv():
    with open(os.path.join(data_path, 'grid_percents.csv'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        patches_header = ['patch_'+str(i+1) for i in range(num_patches)]
        csvwriter.writerow(['filename', 'percentage'] + patches_header) 

        for filename in sorted(glob.glob(os.path.join(input_path, pattern_list))):
            print(filename)
            nc = netCDF4.Dataset(os.path.join(input_path, filename), 'r')
            flash_extent_density = nc.variables['flash_extent_density'][:]
            #values = np.ma.getdata(flash_extent_density)
            # Discarding extra grid:
            values = np.ma.getdata(flash_extent_density)[0:2100,0:2100]
            total_pixels = len(values.flatten())
            num_pixels = len(values[values>0])
            percent = num_pixels/total_pixels
            patch_flashes = get_flashes_per_patch(values)
            csvwriter.writerow([filename, percent] + patch_flashes)

def write_filtered_csv(df):
    with open(os.path.join(data_path, outfile), 'w', newline='') as csvfile:
        for index, row in df.iterrows():
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([row['filename'],])
    

def create_dataframe(min_percent=0, max_percent=100, min_flashes=0):
    df = pd.read_csv(os.path.join(data_path, 'grid_percents.csv'))
    df['100_percent'] = df['percentage'] * 100
    cols = df.columns[2:-1]
    df['min_flashes_patch'] = df[cols].astype('float32').min(axis=1)
    
    df['100_percent'].hist(bins=np.arange(0,10), grid=False, rwidth=0.9)
    plt.title('Percentage of Non-Zero Pixels')
    plt.ylabel('Absolute Frequency')
    plt.xlabel('Percentage (%)')
    plt.savefig(os.path.join(figs_path, 'percent_nonzero.png'))

    df['min_flashes_patch'].hist(grid=False, rwidth=0.9)
    plt.title('Max number of flashes per patch')
    plt.ylabel('Absolute Frequency')
    plt.xlabel('Number of flashes')
    #plt.xscale('log')
    plt.savefig(os.path.join(figs_path, 'max_flashes_patch.png'))

    print('Total number of files: ' + str(df.shape[0]))
    print('Number of filtered files (' + str(min_percent) + '% - ' + str(max_percent) + '%  & ' +
            'Min. flashes per patch = ' + str(min_flashes) + '): ')
    df_filt = df[(df['min_flashes_patch'] >= min_flashes) & 
        (df['100_percent'] >= min_percent) & (df['100_percent'] <= max_percent)]
    print(df_filt.shape[0])

    write_filtered_csv(df_filt)


if __name__ == "__main__":
    input_path = sys.argv[1]
    #pattern_list = '*.netcdf'
    pattern_list = sys.argv[2]
    print(pattern_list)
    #outfile = 'glm_filtered_60sum_extra.csv'
    outfile = sys.argv[3]
    config_file = 'config.ini'

    # Parameters from Config File
    config = configparser.ConfigParser()
    config.read(config_file)
    patch_size = [int(size) for size in config['GEO']['patch_size'].split(',')]
    num_patches = int(config['GEO']['num_patches'])
    figs_path = config['PATH']['figs_path']
    data_path = config['PATH']['data_path']
    min_perc_flashes = float(config['FILTER']['min_percent_flashes'])
    max_perc_flashes = float(config['FILTER']['max_percent_flashes'])
    min_flashes = int(config['FILTER']['min_flashes_patch'])

    write_csv()
    #create_dataframe(min_percent=2, max_percent=10, min_flashes=3)
    create_dataframe(min_percent=min_perc_flashes, max_percent=max_perc_flashes, min_flashes=min_flashes)
