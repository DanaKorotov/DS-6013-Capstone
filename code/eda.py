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

#%% Load data and make columns
oct_data = pd.read_csv("Sample data/ais_east_asia_20231001/ais_east_asia_oct2024_20231001.csv")
oct_data['dtg'] = pd.to_datetime(oct_data['dtg'])
oct_data = oct_data.sort_values(['mmsi', 'dtg'])
# Make mmsi_length column
oct_data['mmsi_length'] = oct_data['mmsi'].astype(str).apply(len)
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

#%% # Check for bad names
import re
def check_net_name(name):
    name = name.lower()
    net_name = ('%' in name or 
                'buoy' in name or 
                'net' in name or 
                bool(re.search(r"\d+v", name)))
    return net_name

ship_names = oct_data['vessel_name'].unique()
net_names = [check_net_name(name) for name in ship_names]
oct_data['net_name'] = oct_data['vessel_name']\
    .map(dict(zip(ship_names, net_names)))
#%%
# Filter out ships going > 35 mph(allow for measurement error), as those are fake(according to google?)
# Usually means multiple ships, same MMSI, can mean spoofing location
#oct_data['high_speed'] = oct_data['speed'] > 50
#high_speed_mmsis = oct_data.query("speed > 50 | sog > 50")['mmsi'].unique()

# oct_clean = oct_data[~oct_data['mmsi'].isin(high_speed_mmsis)]\
#     .sort_values(by = 'dtg', ascending = True)

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
#.query("mmsi not in @high_speed_mmsis")\
ship_info.head(6)

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

#%% Naming conventions

# Concrete identifiers of Nets (fishing lines or buoys) vs Boats:
# - alot of it is imagery 
# 	- pattern recognition of 'boat' movement 
# 	- speed and how far something has move
# 	- locations on top of each other 
# 	- sometimes there will be alot of nets but no boats
# 	- some ships will do multiple loops to pick up nets
# 	- sometimes an AIS signal will just appear in the middle of the ocean 
# - quality of ship data, is the ship transmitting? (lat and long harder to spoof)
# 	- dont have any data being release, their mimsy (SSN of a vessel) should be valid
# 	- ships should have their own mimsy (mimsy in two different locations)
# 	- weird movement 
# 	- have a percent sign, BUOY, NETS, BEACON
# 	- 8V 9V (indicates a battery)
# 	- UNAVAILABLE (means it is in format that is not readable -- does not always indicate a net)
# 	- Call sign is good sign of it being a ship 
# 	- class is pretty important 
# 	- flag code = first thee of mimsi 
# 	- sog and cog is speed overground, could be useful 

# Look at summary stats of name attributes(percent sign, check email for more)
# Maybe instead of going through and finding nearby ships, check for similar movement patters?


sum(oct_data['net_name'])/len(oct_data)

#%% Check if a vessel doesn't move for a while
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
            proportion = 0  # You can use np.nan if you prefer

        slow_prop_list.append(proportion)
    
    return dict(zip(mmsi_list, slow_prop_list))

slow_prop_dict = do_it_float(mmsi_list = oct_data.mmsi.unique(), 
                             df = oct_data)
oct_data['floater'] = oct_data['mmsi'].map(slow_prop_dict)

#%% Looking at names and MMSIs
oct_data\
    .groupby('mmsi')\
    .agg(num_names = ('vessel_name', 'nunique'),
         num_net_names = ('net_name', 'sum'))\
    .sort_values('num_names', ascending = False)
#%%
test = oct_data.query("mmsi == 200000000")

# %%
# Things to do:
# Find if a boat is dropped in the middle of the water and ends in the water(Voronoi regions?)
# How to find boats traveling in irregular patterns. 
# We want to find the nets and the boats that drop them. 
# Sketchy names in the same area(roughly straight line configuration)
# Turns on in water, doesn't move, turns off in water
# Allow for ~350 meters of drift in a ~6 hour period on average 
# 10-30 knots for a normal vessel 
# Some fishermen turn off AIS then pick up nets
# Super fast speed probably means different boats same MMSI
# Look at spread of speed?
# Last years' project needed Vournoi regions for identifying lying about region where fish was caught

#%% Visualize several ships traveling
# Get top 5 traveled ships
five_ships = ship_info.nlargest(5, 'total_distance')\
    .loc[:, 'mmsi']
five_ships_data = oct_data[oct_data['mmsi'].isin(five_ships)]\
    .sort_values('mmsi')

ship_colors = ['red', 'blue', 'green', 'black', 'purple']
ship_colors = dict(zip(five_ships.tolist(), ship_colors)) 
five_ships_data['ship_color'] = five_ships_data['mmsi']\
    .map(ship_colors)

# Make geopandas data to plot
geometry = [Point(xy) for xy in zip(five_ships_data.longitude, five_ships_data.latitude)]
gdf = GeoDataFrame(five_ships_data, geometry = geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# Create an Axes object and plot the map
ax = world.plot(figsize=(15,15))
gdf.plot(ax = ax, 
         marker = 'o', 
         color = five_ships_data['ship_color'], 
         markersize = 5)
xmin, ymin, xmax, ymax = gdf.total_bounds
pad = 2  # Add a padding around the mapped values
ax.set_xlim(xmin-pad, xmax+pad) 
ax.set_ylim(ymin-pad, ymax+pad) 
# ax.legend(['First ship', 'Second ship', 'Third ship', 'Fourth ship', 'Fifth ship'])
plt.show()
