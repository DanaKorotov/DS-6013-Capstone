#%% Import packages
import numpy as np
import pandas as pd
import geopandas as gpd 
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame

#%% Load data
oct_data = pd.read_csv("/Users/samuelbrown/Documents/School files/MSDS Stuff/Capstone/Sample data/ais_east_asia_20231001/ais_east_asia_oct2024_20231001.csv")
oct_data.head()


#%% GeoPandas
geometry = [Point(xy) for xy in zip(oct_data.longitude, oct_data.latitude)]
gdf = GeoDataFrame(oct_data,geometry = geometry)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# create an Axes object and plot the map
ax = world.plot(figsize=(15,15))
gdf.plot(ax=ax, marker='o', color="red", markersize=5)
xmin, ymin, xmax, ymax = gdf.total_bounds
pad = 2  # Add a padding around the geometry
ax.set_xlim(xmin-pad, xmax+pad)
ax.set_ylim(ymin-pad, ymax+pad)
