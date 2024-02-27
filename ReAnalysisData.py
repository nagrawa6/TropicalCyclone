import pandas as pd
import numpy as np
import random
import datetime
import os
from datetime import timedelta
import cdsapi
import datetime
import os
import cdsapi

import ecmwflibs
import eccodes
from eccodes import *

from statistics import mean 

RAD_ROOT_DIR = "data/rad/"
cdsApi = cdsapi.Client() 

#Function to get u, v, z, r, t at three pressure levels  for a grid centerd at cyclone location. 
def getPressureData(cdsApi, year, month, date, time, lati, long, fileName):
  spread = 10
  #print('fileName  ', fileName)
  if os.path.isfile(fileName):
    #print("file already exist")
    return
  cdsApi.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'geopotential', 'relative_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind',
        ],
        'pressure_level': [
             '300', '500', '700', '850',
        ],
        'year': [
            year,
        ],
        'month': [
            month,
        ],
        'day': [
            date,
        ],
        'time': [
            time,
        ],
        'format': 'grib',
        'area': [
            lati+spread, long-spread, lati-spread,
            long+spread,
        ],
    },
    fileName)

def read_grib_file(pressureFile, datac, CHECK):  #, colums
    id = codes_index_new_from_file(pressureFile, ["shortName",'level'])
    names = codes_index_get(id, "shortName")
    levels = codes_index_get(id, "level")

    #print("names  ", names)
    #print("levels  ",levels)
    for level in levels:
        #Getting index for each leval 225, 500, 700
        codes_index_select(id, "level", level)
        for name in names:
          #Getting index for each name (u, v, z)
          codes_index_select(id, "shortName", name)
          gid = codes_new_from_index(id)
          values = codes_get_values(gid)
          length = len(values)
          #print(f"gid:{gid}, length:{length}")
          datac.append(mean(values))
          length = len(datac)
          #print(f"datac length:{length}")
          # Appening values of field for each location of grid. 
          '''
          for i in range(length):
                val = np.float32(values[i])
                datac.append(val)
                #Creating dataset columnds name
                #if CHECK:
                #    colums.append(name+str(level)+'_'+str(i))
          '''
    return id, datac  #, colums

#Position is the index position in the _coordinates dataframe which is to keep track of 
#which cyclone position data is being fetched for
def get_prev_current_features(_strTime, _pos, _crow, nrow):
    _data = []    
    _year = str(_strTime)[0:4]
    _month = _strTime[5:7]
    _day = _strTime[8:10]
    _hour = _strTime[11:13] + ":00"

    #Important: _crow is the previous row which is used to get the variables for the first position
    _clat = _crow['lat']
    _clon = _crow['lon']
    _fileName = RAD_ROOT_DIR + str(_pos).zfill(6) + '.grib'
    #_coordinates.append([index, _llat, _llon, _fileName])
    #print(_year, _month, _day, _hour, _clat, _clon, _fileName)
    getPressureData(cdsApi, _year, _month, _day, _hour, _clat, _clon, _fileName)

    datac = []
    CHECK = False
    _, datac = read_grib_file(_fileName, datac, CHECK)
    _data.append(datac)

     #Important: nrow is the next row which is used to get the variables for the second position
    _nlat = nrow['lat']
    _nlon = nrow['lon']
    _fileName =  RAD_ROOT_DIR + str(_pos+1).zfill(6) + '.grib'

    getPressureData(cdsApi, _year, _month, _day, _hour, _nlat, _nlon, _fileName)
    datac = []
    CHECK = False
    _, datac = read_grib_file(_fileName, datac, CHECK)
    _data.append(datac)
   
    _cycl_move_vector = [_nlat- _clat, _nlon-_clon]
    return _data, _cycl_move_vector
    
def generate_reanalysis_ACyclone_differential(_df):
    times = []
    row_iterator = _df.iterrows()
    _crow = _df.iloc[0]
    #Skip the first row
    _currentRow=next(row_iterator)
    _currentTime = _currentRow[0]

    _coordinates = []
     #Store this row for the next iteration. We use the information in lat and lon to get data values
    #use the next row to get the data values then do 2-1
    for i, _nrow in row_iterator:
     
        _strTime = _currentTime.strftime("%Y-%m-%d %H:%M:%S")
        times.append(_strTime)
        #print(_currentTime)                             

        _data, _cycl_move_vector = get_prev_current_features(_strTime, len(_coordinates), _crow, _nrow)
            
        #_data has 2 rows one for currnet position and second for the next position
        # We first convert this into 2x4x5 matrix and take deltas for each individual reading
        _a = np.array(_data)
        _radata = _a.reshape(2,4,5)
        _delta_1_2 = _radata[1][:][:] - _radata[0][:][:]
        #This gives us 1x4x5 matrix
        #next ew flatten to create a 1x20 array
        _delta_1_2 = _delta_1_2.flatten() 

        #Append delta change in position of cyclone by taking the difference. Could be thought as vector movement.
        _sample = np.append(_delta_1_2, _cycl_move_vector)
        _coordinates.append(_sample)

        #Store this row and timestamp for the next iteration. 
        #We use the information in lat and lon to get data values for this timestamp
        #then use the next row to get the data values then do 2-1
        _crow = _nrow
        _currentTime = i
    _dfc = pd.DataFrame(data=_coordinates)  
    _dfc['CY'] = _df['CY'][0]
    _dfc['Date'] = pd.to_datetime(times)
    _dfc.set_index('Date', inplace=True)
    return _dfc

#This function generates cyclones feature(variables) differences for 2 consecutive best track locations
#We first loop through year, then loop through cyclone
#For the best track of each cyclone, we first take 2 consecutive data points, obtain atmospheric variables
#Then take a difference. Next we flatten the matrix which gives X. For Y, we take delta change in position
# of cyclone for Lat and Lon 

def generate_reanalysis_Cyclones_differential(_df, RAD_PICKLE_FILE="", load_from_pickle=False):

    #Check if data/rad folder exists. else create it.
    if not os.path.exists(RAD_ROOT_DIR):
        os.makedirs(RAD_ROOT_DIR) 
    #Check if the pickle file exists. If yes, read from it.
    if load_from_pickle:
        if os.path.exists(RAD_PICKLE_FILE):
            _dfc = pd.read_pickle(RAD_PICKLE_FILE) 
            return _dfc
        else:
            print("File Not Found:", RAD_PICKLE_FILE) 
            return None
    #This is a master dataframe which stores flattened sample points which contain difference 
    # in atmospeheric variables in 2 consecutive location of cyclones
    _df_master_rad = pd.DataFrame()
    #For a given year
    for i in _df.index.year.unique():
        #Get best tracks in a given year
        df_best1 = _df.loc[_df.index.year==i]  
        #For a given cyclone
        for j in df_best1['CY'].unique():
           # print(j)
            #Obtain the best track for the cyclone
            df_best2 = df_best1.loc[df_best1['CY']==j] 
           # print(i, j, df_best2[['lat', 'lon']])
            #We call the function below to generate differentials in atmospheric variables 
            #in 2 consecutive locations.
            _dfc= generate_reanalysis_ACyclone_differential(df_best2)
            _df_master_rad=_df_master_rad.append(_dfc, ignore_index=False)

    #Store the master datafile in a pickle so that it can be accessed later.
    if(RAD_PICKLE_FILE == ""):
        num = int(random.random()*10000)
        os.makedirs(RAD_ROOT_DIR+str(num)) 
        RAD_PICKLE_FILE = RAD_ROOT_DIR + str(num) + "/" + str(num) + "rad.pkl"
    _df_master_rad.to_pickle(RAD_PICKLE_FILE)

    return _df_master_rad

def download_data(_df):
    cdsApi = cdsapi.Client()    
    for index, row in _df.iterrows():
        #print(row['lat'], row['lon'])
        #date TimeStamp is of the format 2010-06-25 00:00:00. 
        #Extract each component to generate fileName for grib file
        _strTime = index.strftime("%Y-%m-%d %H:%M:%S")
       # print(_strTime)                             
        _year = str(index)[0:4]
        _month = _strTime[5:7]
        _day = _strTime[8:10]
        _hour = _strTime[11:13] + ":00"
        _lat = row['lat']
        _lon = row['lon']
      #  print(_year, _month, _day, _hour, _lat, _lon, _path)
        time1 = _hour.replace(':','_')
        fileName = path+ date + _month+ _year+'_'+time1+'.grib'
        getPressureData(cdsApi, _year, _month, _day, _hour, _lat, _lon, fileName)
        
        
def generate_reanalysis_data_matrix(_df):
    datac = []
        
    for index, row in _df.iterrows():
        #print(row['lat'], row['lon'])
        _strTime = index.strftime("%Y-%m-%d %H:%M:%S")
        #_strTime = str(index)  #datetime.strptime(str(index), '%Y-%m-%d %H:%M:%S')
        #print(_strTime)                             
        _year = _strTime[0:4]
        _month = _strTime[5:7]
        _day = _strTime[8:10]
        _hour = _strTime[11:13] + "_00"
        _lat = row['lat']
        _lon = row['lon']
        
        _fileName =  RAD_ROOT_DIR + _day + _month+ _year+'_'+ _hour+'.grib'
        CHECK = False
        read_grib_file(_fileName, datac, CHECK)
        
    
    _ncycl = _df.shape[0]
    _a = np.array(datac)
    _radata = _a.reshape(_ncycl,4,5,1681)
    return _radata