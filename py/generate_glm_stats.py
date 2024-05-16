# conda activate glmval #

import netCDF4
import xarray as xr
import sys
import numpy as np
import configparser
from glmtools.io.glm import GLMDataset
from matplotlib import pyplot as plt
import plotly.express as px
import glob
from datetime import datetime, timedelta
import csv
import os
import pandas
import logging
import multiprocessing


class GLM_stats:

    def __init__(self, config_file):
        # Parameters from Config File
        config = configparser.ConfigParser()
        config.read(config_file)
        self.log_path = config['PATHS']['log_path']
        self.glm_path = config['PATHS']['glm_path']
        self.output_path = config['PATHS']['output_path']
        self.stats_perfile_prefix = config['PATHS']['stats_perfile_prefix']
        self.stats_perday_prefix = config['PATHS']['stats_perday_prefix']
        self.lon_range = [float(x) for x in (config['FILTERS']['lon_range']).split(',')]
        self.lat_range = [float(x) for x in (config['FILTERS']['lat_range']).split(',')]

    def get_spatial_subset(self, filename):
        try:
            glm =  GLMDataset(filename)
            ds = glm.dataset

            # The approach below is commented because it did not work so well!!
            #fl_idx = (ds['flash_lat'] > -34) & (ds['flash_lat'] < 5) & (ds['flash_lon'] > -75) & (ds['flash_lon'] < -34)
            #flash_lats = ds[{glm.fl_dim: fl_idx}].flash_lat.data
            #flash_lons = ds[{glm.fl_dim: fl_idx}].flash_lon.data
            #flash_ids = ds[{glm.fl_dim: fl_idx}].flash_id.data
            #smaller_dataset = glm.get_flashes(flash_ids)

            subset = glm.subset_flashes(lon_range=self.lon_range, lat_range=self.lat_range)
            return subset
        except Exception as e:
            logging.error(filename + getattr(e, 'message', repr(e)))
            return None

    def get_num_flashes(self, dataset):
        flash_ids = dataset.flash_id.data
        num_flashes = len(flash_ids)
        return num_flashes

    def export_csv_stats(self, output_file, filename_st_date, num_flashes, header=False):
        with open(output_file,'a') as csvfile: 
            csvwriter = csv.writer(csvfile)
            if header == True:
                csvwriter.writerow(['datetime', 'num_flashes'])
            csvwriter.writerow([filename_st_date, num_flashes])
        return output_file

    def process_daily_stats(self, input_file, output_file):
        df = pandas.read_csv(input_file)
        #df['num_flashes'] = df['num_flashes'].apply(lambda x: int(x))
        #df['day'] = [str(date)[0:7] for date in df['datetime'].values]
        df['day_hour'] = [str(date)[0:9] for date in df['datetime'].values]

        #grouped_df = df.groupby(['day']).sum()
        grouped_df = df.groupby(['day_hour']).sum()

        days = []
        totals = []
        with open(output_file,'w') as outfile:
            csvwriter = csv.writer(outfile)
            #csvwriter.writerow(['day', 'total_flashes'])
            csvwriter.writerow(['day_hour', 'total_flashes'])
    
            for index, row in grouped_df.iterrows():
                day = index
                total_flashes = row['num_flashes']
                csvwriter.writerow([day, total_flashes])
                days.append(day)
                totals.append(total_flashes)

        #plt.bar(np.arange(len(days)), totals)
        #plt.xticks(np.arange(len(days)), labels=days)
        #plt.title('Total Flashes / Day')
        #plt.show()
        #plt.savefig(os.path.join(self.output_path, 'daily_flashes.png'))


if __name__ == "__main__":
    # Receive args
    config_file = sys.argv[1]
    start_date_str = sys.argv[2]
    end_date_str = sys.argv[3]
   
    stats = GLM_stats(config_file)

    # Initialize Logging
    log_file = (os.path.basename(__file__)).replace('.py','.log')
    logging.basicConfig(filename=os.path.join(stats.log_path, log_file), level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)
    
    # List files within time period
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    # Definition of Stats files
    stats_perfile_filename = stats.stats_perfile_prefix + start_date_str + '_' + end_date_str + '.csv'
    stats_perday_filename = stats.stats_perday_prefix + start_date_str + '_' + end_date_str + '.csv'
    stats_perfile_path = os.path.join(stats.output_path, stats_perfile_filename)
    stats_perday_path = os.path.join(stats.output_path, stats_perday_filename)

    # Remove stats file from previous execution
    try:
        os.remove(stats_perfile_path)
        os.remove(stats_perday_path)
    except:
        pass

    while start_date < end_date:
        YYYY = start_date.strftime('%Y')
        mm = start_date.strftime('%m')
        dd = start_date.strftime('%d')
        jd = start_date.strftime('%j')
        glm_path = stats.glm_path
        glm_path = glm_path.replace("<YYYY>", YYYY)
        glm_path = glm_path.replace("<mm>", mm)
        glm_path = glm_path.replace("<dd>", dd)
        glm_path = glm_path.replace("<jd>", jd)

        i = 0
        for filename in sorted(glob.glob(glm_path + '/*_s'+YYYY+jd+'*.nc')):
            print(filename)
            basename = os.path.basename(filename)
            filename_st_date = basename.split('_')[3].replace('s','')

            # Spatial Subset
            br_dataset = stats.get_spatial_subset(filename)
            if br_dataset is None:
                continue

            # Get Flashes
            num_flashes = stats.get_num_flashes(br_dataset)

            # Stats
            if i == 0:
                stats.export_csv_stats(stats_perfile_path, filename_st_date, num_flashes, header=True)
            else:
                stats.export_csv_stats(stats_perfile_path, filename_st_date, num_flashes)

            i = i + 1
            #if i == 20:
            #    break

        start_date = start_date + timedelta(days=1)

    stats.process_daily_stats(stats_perfile_path, stats_perday_path)
