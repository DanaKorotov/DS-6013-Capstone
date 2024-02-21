#%% Import packages
# Basics
import numpy as np
import pandas as pd
# Dask
import dask.dataframe as dd
# Distance and time(speed stuff)
from geopy.distance import geodesic
from datetime import datetime
from shapely.wkt import loads
from shapely.geometry import Point

#%%
# Read data into dask
dtypes = {'FID': 'object', 'mmsi': 'int64', 'source_id': 'int64', 'imo': 'float64', 
          'vessel_name': 'object', 'callsign': 'object', 'vessel_type': 'object', 
          'vessel_type_code': 'float64', 'vessel_type_cargo': 'object', 'vessel_class': 'object',
          'length': 'float64', 'width': 'float64', 'flag_country': 'object', 'flag_code': 'float64',
          'destination': 'object', 'eta': 'int64', 'draught': 'float64', 'position': 'object',
          'latitude': 'float64', 'longitude': 'float64', 'sog': 'float64', 'cog': 'float64',
          'rot': 'float64', 'heading': 'int64', 'nav_status': 'object', 'nav_status_code': 'int64',
          'source': 'object', 'ts_pos_utc': 'int64', 'ts_static_utc': 'int64', 
          'dt_pos_utc': 'object', 'dt_static_utc': 'object', 'vessel_type_main': 'float64',
          'vessel_type_sub': 'float64', 'message_type': 'int64', 'eeid': 'float64', 'dtg': 'object'}
oct_data = dd.read_csv("Sample data/ais_east_asia_20231001/ais_east_asia_oct2024_20231001.csv", 
                       dtype = dtypes)
# Make dtg column in datetime form
oct_data['dtg'] = dd.to_datetime(oct_data['dtg'])
# Make mmsi_length column
oct_data['mmsi_length'] = oct_data['mmsi'].map_partitions(lambda s: s.astype(str).str.len(), 
                                                          meta = ('mmsi_length', 'int'))
# Remake latitude and longitude columns
oct_data['latitude'] = oct_data['position'].apply(lambda geom: loads(geom).y if geom else None, 
                                                  meta = ('latitude', 'float64'))
oct_data['longitude'] = oct_data['position'].apply(lambda geom: loads(geom).x if geom else None, 
                                                   meta = ('longitude', 'float64'))
# Calculate the time difference between consecutive rows for each mmsi
oct_data['time_diff'] = oct_data.groupby('mmsi').apply(lambda group: group.sort_values('dtg')['dtg'].diff(), 
                                                       meta = ('diff_result', 'timedelta64[ns]'))
# Create a new column called first_occurrence based on the identified conditions
oct_data['first_occurrence'] = ((oct_data['time_diff'].isnull()) | (oct_data['time_diff'] > pd.Timedelta(hours=10))).astype(int)
# Add occurences together
oct_data['occ_num'] = oct_data.groupby('mmsi')['first_occurrence'].cumsum()
# Shift lat, long, dtg back by one per mmsi group and find distance traveled, speed in KPH

# Get distance and spped from consecutive readings
# Get rid of extraneous columns
oct_data = oct_data.drop(['length', 'width', 'draught', 
                          'vessel_type_code', 'vessel_type_cargo',
                          'nav_status_code', 'ts_pos_utc',
                          'ts_static_utc', 'dt_pos_utc', 
                          'dt_static_utc', 'position', 'time_diff',
                          #'old_lat', 'old_long'
                          ], 
                          axis = 1)

#%%

