# Source: https://colab.research.google.com/drive/1PKTltCfGqE_UjeRRfA1k0vxztiYPfpsg?usp=sharing#scrollTo=xPFA-EoINDLN

from datetime import datetime, timedelta
import glob
import numpy as np
import netCDF4
import sys
import os
import collections
from pathlib import Path

#------------------------------------------------------------
def write_netcdf(output_file,datasets,dims,atts={},**kwargs):
  print('Process started')

  try:
    #mkdir_p(os.path.dirname(output_file))
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
  except (OSError,IOError) as err:
    print(str(err))
    return False
  else:
    ncfile = netCDF4.Dataset(output_file,'w')

  #dimensions
  for dim in dims:
    ncfile.createDimension(dim,dims[dim])
  #variables
  for varname in datasets:
    try:
      dtype = datasets[varname]['data'].dtype.str
    except AttributeError:
      if(isinstance(datasets[varname]['data'],int)):
        dtype = 'int'
      elif(isinstance(datasets[varname]['data'],float)):
        dtype = 'float'

    if('_FillValue' in datasets[varname]['atts']):
      dat = ncfile.createVariable(varname,dtype,datasets[varname]['dims'],fill_value=datasets[varname]['atts']['_FillValue'])
    else:
      dat = ncfile.createVariable(varname,dtype,datasets[varname]['dims'])
    #dat[:] = datasets[varname]['data']
    dat[:] = np.copy(datasets[varname]['data'])
    if varname == 'flash_extent_density':
      values7 = dat[:]
      print(len(values7[values7>0]))
      values8 =  datasets[varname]['data']
      print(len(values8[values8>0]))
    #variable attributes
    if('atts' in datasets[varname]):
      for att in datasets[varname]['atts']:
        if(att != '_FillValue'): dat.__setattr__(att,datasets[varname]['atts'][att]) #_FillValue is made in 'createVariable'
    del dat
  #global attributes
  for key in atts:
    ncfile.__setattr__(key,atts[key])
  ncfile.close()
  print('Wrote out ' + output_file)
  #values7 = datasets['flash_extent_density']['data']

  return True

#------------------------------------------------------
def aggregate_glm_grids(lastdatetime,
                     datadir,
                     accumPeriod=10,
                     outdir=os.environ['PWD']+'/'):


  if(datadir[-1] != '/'): datadir += '/'

  print('Process started')

  try:
    last_dt = datetime.strptime(lastdatetime,'%Y%j%H%M')
  except ValueError as err:
    return False

  #aggregate data for the accumPeriod
  accumMins = 0
  for tt in range(accumPeriod):
    curr_dt = last_dt - timedelta(minutes=tt)
    if(tt==0): ts_dt = curr_dt

    filepattern = datadir + curr_dt.strftime('%Y/%b/%d/OR_GLM-L2-*s%Y%j%H%M*.nc*')

    ncfile = np.sort(glob.glob(filepattern))
    print(ncfile)

    if(len(ncfile)):
      #print(f'received {ncfile[0]}')
      if(tt == 0): #get end datetime of first file in accumulation, since we're working back in time.
        ts_dt = datetime.strptime(os.path.basename(ncfile[0]).split('_e')[1].split('_')[0][0:13],'%Y%j%H%M%S')
      accumMins += 1
      ncfile = ncfile[0]
      nc = netCDF4.Dataset(ncfile,'r')


      flash_extent_density_TMP = nc.variables['flash_extent_density'][:]

      average_area_perflash = nc.variables['average_flash_area'][:]
      average_area_TMP = np.multiply(average_area_perflash,flash_extent_density_TMP)

      group_extent_density_TMP = nc.variables['group_extent_density'][:]

      average_area_pergroup = nc.variables['average_group_area'][:]
      average_garea_TMP = np.multiply(average_area_pergroup,group_extent_density_TMP)

      flash_centroid_density_TMP = nc.variables['flash_centroid_density'][:]

      total_energy_TMP = nc.variables['total_energy'][:] * 1e6 #in nJ...put into fJ

      group_centroid_density_TMP = nc.variables['group_centroid_density'][:]


      #first see if the vars exist (i.e., it's the first iteration if they do not)
      try:
        flash_extent_density
      except NameError:
        #if they do not exist, copy fields
        flash_extent_density = np.copy(flash_extent_density_TMP)
        total_energy = np.copy(total_energy_TMP)
        average_area_AGG = np.copy(average_area_TMP)
        average_garea_AGG = np.copy(average_garea_TMP)
        flash_centroid_density = np.copy(flash_centroid_density_TMP)
        group_extent_density = np.copy(group_extent_density_TMP)
        group_centroid_density = np.copy(group_centroid_density_TMP)
        ydim = nc.dimensions['y'].size
        xdim = nc.dimensions['x'].size
        min_flash_area_grids = np.zeros((accumPeriod,ydim,xdim))
        min_flash_area_grids[tt] = nc.variables['minimum_flash_area'][:]
        #and grab x and y
        x = np.array(nc['x'][:])
        y = np.array(nc['y'][:])
        gip = nc['goes_imager_projection'][:]

        pos = np.nonzero(flash_extent_density)
        print(pos)
        print(flash_extent_density[pos][0:20])
        
      else:
        #if they do exist, just aggregate fields
        try:
          flash_extent_density += flash_extent_density_TMP
          total_energy += total_energy_TMP
          average_area_AGG += average_area_TMP
          average_garea_AGG += average_garea_TMP
          flash_centroid_density += flash_centroid_density_TMP
          group_extent_density += group_extent_density_TMP
          group_centroid_density += group_centroid_density_TMP

          min_flash_area_grids[tt] = nc.variables['minimum_flash_area'][:]
        except ValueError as err:
          ogging.error(traceback.format_exc())
          nc.close()
          return False

      #get attributes
      try:
        vatts
        gatts
      except NameError:
        vatts = collections.OrderedDict()
        vatts['x'] = {}
        vatts['y'] = {}
        vatts['goes_imager_projection'] = {}
        vatts['flash_extent_density'] = {}
        vatts['flash_centroid_density'] = {}
        vatts['total_energy'] = {}
        vatts['average_flash_area'] = {}
        vatts['group_extent_density'] ={}
        vatts['group_centroid_density'] = {}
        vatts['average_group_area'] = {}
        vatts['minimum_flash_area'] = {}
        for var in vatts:
          for att in nc.variables[var].ncattrs():
            vatts[var][att] = nc.variables[var].getncattr(att)
        vatts['total_energy']['units'] = 'fJ'
        vatts['total_energy']['scale_factor'] = 1
        vatts['total_energy']['add_offset'] = 0
        vatts['average_flash_area']['units'] = 'km^2'
        vatts['average_group_area']['units'] = 'km^2'

        #global atts
        gatts = collections.OrderedDict()
        gatts['spatial_resolution'] = getattr(nc,'spatial_resolution')
        gatts['subsatellite_longitude'] = vatts['goes_imager_projection']['longitude_of_projection_origin']
        gatts['orbital_slot'] = getattr(nc,'orbital_slot')
        gatts['platform_ID'] = getattr(nc,'platform_ID')
        gatts['scene_id'] = gatts['orbital_slot'][5] + getattr(nc,'scene_id') #gatts['orbital_slot'][5] is "E" or "W"
        gatts['time_coverage_end'] = getattr(nc,'time_coverage_end')

      #do every time
      gatts['time_coverage_start'] = getattr(nc,'time_coverage_start')

      #close ncfile
      nc.close()


  if(accumMins == 0): return False
  gatts['aggregation'] = str(accumMins) + ' min'
  gatts['dataset_name'] = "GLM_gridded_data"

  for var in ['flash_extent_density','flash_centroid_density','group_extent_density','group_centroid_density']:
    attparts = vatts[var]['units'].split(' ')
    attparts[-2] = str(accumMins)
    vatts[var]['units'] = (' ').join(attparts)

  #Make final min area
  fillv = 999999
  min_flash_area_grids[min_flash_area_grids == 0] = fillv
  minimum_flash_area = np.min(min_flash_area_grids,axis=0)
  minimum_flash_area[minimum_flash_area == fillv] = 0

  # Make average areas.
  # These steps are necessary so we don't divide by 0 or some small fraction
  average_flash_area = np.zeros((np.shape(average_area_AGG)))
  good_ind = np.where(flash_extent_density >= 1)
  average_flash_area[good_ind] = average_area_AGG[good_ind] / flash_extent_density[good_ind]
  average_flash_area = average_flash_area.astype('f4')

  average_group_area = np.zeros((np.shape(average_garea_AGG)))
  good_ind = np.where(group_extent_density >= 1)
  average_group_area[good_ind] = average_garea_AGG[good_ind] / group_extent_density[good_ind]
  average_group_area = average_group_area.astype('f4')

  ny,nx = np.shape(total_energy)

  #write out data into one file
  #first pack the data using scale_factor and add_offset
  print("Before Write...")
  values4 = flash_extent_density
  print(len(values4[values4>0]))

  xpacked = ((x - vatts['x']['add_offset'])/vatts['x']['scale_factor']).astype(np.int16)
  ypacked = ((y - vatts['y']['add_offset'])/vatts['y']['scale_factor']).astype(np.int16)
  FEDpacked = ((flash_extent_density - vatts['flash_extent_density']['add_offset'])/vatts['flash_extent_density']['scale_factor']).astype(np.int16)
  FCDpacked = ((flash_centroid_density - vatts['flash_centroid_density']['add_offset'])/vatts['flash_centroid_density']['scale_factor']).astype(np.int16)
  TOEpacked = ((total_energy - vatts['total_energy']['add_offset'])/vatts['total_energy']['scale_factor']).astype(np.int16)
  AFApacked = ((average_flash_area - vatts['average_flash_area']['add_offset'])/vatts['average_flash_area']['scale_factor']).astype(np.int16)
  GEDpacked = ((group_extent_density - vatts['group_extent_density']['add_offset'])/vatts['group_extent_density']['scale_factor']).astype(np.int16)
  GCDpacked = ((group_centroid_density - vatts['group_centroid_density']['add_offset'])/vatts['group_centroid_density']['scale_factor']).astype(np.int16)
  AGApacked = ((average_group_area - vatts['average_group_area']['add_offset'])/vatts['average_group_area']['scale_factor']).astype(np.int16)
  MFApacked = ((minimum_flash_area - vatts['minimum_flash_area']['add_offset'])/vatts['minimum_flash_area']['scale_factor']).astype(np.int16)

  print("After Scaling...")
  #print(vatts['flash_extent_density']['add_offset'])
  #print(vatts['flash_extent_density']['scale_factor'])
  values5 = FEDpacked
  print(len(values5[values5>0]))

  dims = collections.OrderedDict(); dims['y'] = ny; dims['x'] = nx
  datasets = collections.OrderedDict()
  datasets['x'] = {'data':xpacked,'dims':('x'),'atts':vatts['x']}
  datasets['y'] = {'data':ypacked,'dims':('y'),'atts':vatts['y']}
  datasets['goes_imager_projection'] = {'data':gip,'dims':(),'atts':vatts['goes_imager_projection']}
  datasets['flash_extent_density'] = {'data':FEDpacked,'dims':('y','x'),'atts':vatts['flash_extent_density']}
  datasets['flash_centroid_density'] = {'data':FCDpacked,'dims':('y','x'),'atts':vatts['flash_centroid_density']}
  datasets['total_energy'] = {'data':TOEpacked,'dims':('y','x'),'atts':vatts['total_energy']}
  datasets['average_flash_area'] = {'data':AFApacked,'dims':('y','x'),'atts':vatts['average_flash_area']}
  datasets['group_extent_density'] = {'data':GEDpacked,'dims':('y','x'),'atts':vatts['group_extent_density']}
  datasets['group_centroid_density'] = {'data':GCDpacked,'dims':('y','x'),'atts':vatts['group_centroid_density']}
  datasets['average_group_area'] = {'data':AGApacked,'dims':('y','x'),'atts':vatts['average_group_area']}
  datasets['minimum_flash_area'] = {'data':MFApacked,'dims':('y','x'),'atts':vatts['minimum_flash_area']}

  grid_outdir = outdir + '/agg/'

  timestamp = ts_dt.strftime('%Y%m%d-%H%M%S')
  epoch_secs = int((ts_dt - datetime.utcfromtimestamp(0)).total_seconds())
  outfile = grid_outdir + timestamp + '.netcdf'

  values6 = datasets['flash_extent_density']['data']
  print(len(values6[values6>0]))

  status = write_netcdf(outfile,datasets,dims,atts=gatts)


############################################################
input_path = '/ships22/grain/ajorge/data/glm_grids_1min/'
output_path = '/ships22/grain/ajorge/data/glm_grids_60min_sum/'
year = 2020

# Summer Period
#months = [1, 2, 3, 12]
#days = range(10,25)
#days = range(10,16)
months = [sys.argv[1]]

# Other seasons
#months = [4, 5, 6, 7, 8, 9, 10, 11]
#days = range(10,20)
days = range(10, 14)

start_hour = 19
end_hour = 22

for month in months:
    for day in days:
        dt = datetime(year, int(month), day, start_hour, 0)
        while dt.hour < end_hour:
            dt_str = dt.strftime('%Y%j%H%M')
            aggregate_glm_grids(dt_str, input_path, accumPeriod=60, outdir=output_path)
            dt += timedelta(minutes=10)
            sys.exit()
