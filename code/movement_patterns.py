#%% Tells you what's up
# Hey! This .py file contains code to find vessels that started and ended together,
# vessels that floated for a while, and 


#%% Import packages
# Basics
import numpy as np
import pandas as pd
# Distance and time(speed stuff)
from geopy.distance import geodesic
from datetime import datetime
from shapely.wkt import loads
# Packages to map the ships
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

#%% Data formatting 
oct_data = pd.read_csv("Sample data/ais_east_asia_20231001/ais_east_asia_oct2024_20231001.csv")
oct_data['dtg'] = pd.to_datetime(oct_data['dtg'])
oct_data = oct_data.sort_values(['mmsi', 'dtg'])
# Make mmsi_length column
oct_data['mmsi_length'] = oct_data['mmsi'].astype('str').str.len()
# Remake latitude and longitude columns
oct_data['position'] = oct_data['position'].apply(loads)
oct_data['latitude'] = oct_data['position'].apply(lambda geom: geom.y if geom else None)
oct_data['longitude'] = oct_data['position'].apply(lambda geom: geom.x if geom else None)
# Calculate the time difference between consecutive rows for each mmsi
oct_data['time_diff'] = oct_data.groupby('mmsi')['dtg'].diff()
# Create a new column called first_occurrence based on the identified conditions
oct_data['first_occurrence'] = ((oct_data['time_diff'].isnull()) | (oct_data['time_diff'] > pd.Timedelta(hours=10))).astype(int)
# Add occurences together
oct_data['occ_num'] = oct_data.groupby('mmsi').agg(occ_num = ('first_occurrence', 'cumsum'))
# Shift lat, long, dtg back by one per mmsi group and find distance traveled, speed in KPH
oct_data['old_lat'] = oct_data.groupby(['mmsi', 'occ_num'])['latitude'].shift(1)
oct_data['old_long'] = oct_data.groupby(['mmsi', 'occ_num'])['longitude'].shift(1)
# Get distance and spped from consecutive readings
def calculate_distance(row):
    if pd.notna(row['old_lat']) and pd.notna(row['old_long']) and pd.notna(row['latitude']) and pd.notna(row['longitude']):
        return geodesic((row['old_lat'], row['old_long']), (row['latitude'], row['longitude'])).nautical
    else:
        return pd.NA

oct_data['distance'] = oct_data.apply(calculate_distance, axis = 1)
oct_data['speed'] = oct_data['distance'] / (oct_data['time_diff'].dt.seconds.replace(0, pd.NA) / 3600)
# Get rid of extraneous columns
oct_data = oct_data.drop(['length', 'width', 'draught', 
                          'vessel_type_code', 'vessel_type_cargo',
                          'nav_status_code', 'ts_pos_utc',
                          'ts_static_utc', 'dt_pos_utc', 
                          'dt_static_utc', 'position', 'time_diff',
                          'old_lat', 'old_long'], 
                          axis = 1)

# Sum distance traveled in nautical miles
ship_info = oct_data\
    .groupby('mmsi')\
    .agg(
        total_distance = ('distance', 'sum'),
        num_rows = ('mmsi', 'count'),
        start_time = ('dtg', 'min'),
        end_time = ('dtg', 'max'),
        start_lat = ('latitude', lambda x: x.iloc[0]),
        end_lat = ('latitude', lambda x: x.iloc[-1]),
        start_long = ('longitude', lambda x: x.iloc[0]),
        end_long = ('longitude', lambda x: x.iloc[-1])
        )\
    .reset_index()\
    .sort_values('total_distance', ascending = False)\
    .pipe(lambda df: df.assign(
        start_time = pd.to_datetime(df['start_time']),
        end_time = pd.to_datetime(df['end_time'])
    ))


#%% See if vessels traveled together
def check_proximity(row, threshold_distance=0.5, threshold_minutes=10):
    threshold_time = pd.Timedelta(threshold_minutes)
    # Filter rows that match the conditions
    similar_boats = ship_info[
        (abs(ship_info['start_lat'] - row['start_lat']) <= threshold_distance) &
        (abs(ship_info['end_lat'] - row['end_lat']) <= threshold_distance) &
        (abs(ship_info['start_long'] - row['start_long']) <= threshold_distance) &
        (abs(ship_info['end_long'] - row['end_long']) <= threshold_distance) &
        (abs(ship_info['start_time'] - row['start_time']) <= threshold_time) &
        (abs(ship_info['end_time'] - row['end_time']) <= threshold_time) &
        (ship_info['total_distance'] != 0)
    ]
    
    return ((len(similar_boats) > 2) & (row['total_distance'] != 0))

ship_info['proximity'] = ship_info.apply(check_proximity, axis=1)
oct_data['proximity'] = oct_data['mmsi']\
    .map(
        dict(zip(ship_info['mmsi'], ship_info['proximity']))
    )


#%% Check for flotation
def do_it_float(mmsi_list, df):
    slow_prop_list = []

    for mmsi in mmsi_list:
        mmsi_rows = df[df['mmsi'] == mmsi]
        # Check if there are rows for the current MMSI
        if len(mmsi_rows) > 0:
            # Calculate the proportion only when there are rows
            proportion = len(
                mmsi_rows[(mmsi_rows['sog'] >= 0) & 
                (mmsi_rows['sog'] <= .5)]
                ) / len(mmsi_rows)
        else:
            # Handle the case when there are no rows for the current MMSI
            proportion = np.nan

        slow_prop_list.append(proportion)
    
    return dict(zip(mmsi_list, slow_prop_list))

slow_prop_dict = do_it_float(mmsi_list = oct_data.mmsi.unique(), 
                             df = oct_data)
oct_data['floater'] = oct_data['mmsi'].map(slow_prop_dict)