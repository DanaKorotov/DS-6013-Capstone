#%% Import packages
# Basics
import numpy as np
import pandas as pd
# Distance and time
from shapely.wkt import loads
# Packages to map the ships or coastlines
import geopandas as gpd
from geopy.distance import geodesic

#%% Load data
oct_data = pd.read_csv("Sample data/ais_east_asia_20231001/ais_east_asia_oct2024_20231001.csv")
oct_data['dtg'] = pd.to_datetime(oct_data['dtg'])
oct_data = oct_data.sort_values(['mmsi', 'dtg'])
# Remake latitude and longitude columns
oct_data['position'] = oct_data['position'].apply(loads)
oct_data['latitude'] = oct_data['position'].apply(lambda geom: geom.y if geom else None)
oct_data['longitude'] = oct_data['position'].apply(lambda geom: geom.x if geom else None)
# Calculate the time difference between consecutive rows for each 'mmsi'
oct_data['time_diff'] = oct_data.groupby('mmsi')['dtg'].diff()\
# Create a column saying if this is the first we've seen them in a while
oct_data['first_occurrence'] = ((oct_data['time_diff'].isnull()) | (oct_data['time_diff'] > pd.Timedelta(hours=10))).astype(int)
# Get rid of extraneous columns
oct_data = oct_data.drop(['length', 'width', 'draught', 
                          'vessel_type_code', 'vessel_type_cargo',
                          'nav_status_code', 'ts_pos_utc',
                          'ts_static_utc', 'dt_pos_utc', 
                          'dt_static_utc', 'position', 'time_diff'], 
                          axis = 1)
# Make mmsi_length column
oct_data['mmsi_length'] = oct_data['mmsi'].astype(str).apply(len)

#%%
# Read in coastline data from https://www.naturalearthdata.com/downloads/10m-physical-vectors/
coastline = pd.concat([gpd.read_file("ne_10m_land/ne_10m_land.shp"), 
                       gpd.read_file("ne_10m_minor_islands/ne_10m_minor_islands.shp")])
coastline = coastline.cx[min(oct_data['longitude'])-1:max(oct_data['longitude'])+1, 
                         min(oct_data['latitude'])-1:max(oct_data['latitude'])+1].geometry

#%%
# Function to calculate distance from a point to the coastline
from scipy.spatial import KDTree
def distance_to_coast(df, coastline):
    # Get coordinates of all of the ships
    ship_coords = list(df.apply(lambda x: (x.longitude, x.latitude), 
                                axis = 1))
    # Get coordinates of the coastlines
    coast_coords = [] 
    for geom in coastline.geometry:
        if geom.geom_type == 'Polygon':
            coast_coords.extend(list(geom.exterior.coords))
        elif geom.geom_type == 'MultiPolygon':
            for polygon in geom.geoms:
                coast_coords.extend(list(polygon.exterior.coords))
    # Use KDTree to find the nearest points to each ship
    tree = KDTree(coast_coords)
    idx = tree.query(ship_coords, k=1)[1]
    coast_points = pd.Series(coast_coords).iloc[idx]
    # Get the distances in nautical miles
    naut_dist = [geodesic((ship[1], ship[0]), (coast[1], coast[0])).nautical 
                 for ship, coast 
                 in zip(ship_coords, list(coast_points))]
    # Insert into the dataframe
    df['distance_to_coast'] = naut_dist

    return df

test = oct_data.iloc[0:5000, :].copy()
test = distance_to_coast(test, coastline)
#%%
oct_data = distance_to_coast(oct_data, coastline)

#%%
oct_data['spawn_offshore'] = (oct_data['distance_to_coast'] >= 1) & (oct_data['first_occurrence'] == 1)
oct_data
