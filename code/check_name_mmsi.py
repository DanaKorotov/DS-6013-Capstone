#%% Read in data
import pandas as pd
ais = pd.read_csv("Sample data/ais_east_asia_20231001/ais_east_asia_oct2024_20231001.csv")

#%% Check vessel names
import re
def check_net_name(name):
    name = name.lower()
    net_name = ('%' in name or 
                'buoy' in name or 
                'net' in name or 
                bool(re.search(r"\d+v", name)))
    return net_name

ship_names = ais['vessel_name'].unique()
net_names = [check_net_name(name) for name in ship_names]
ais['net_name'] = ais['vessel_name']\
    .map(dict(zip(ship_names, net_names)))
#%% Check mmsi length
ais['mmsi_length'] = ais['mmsi'].astype('str').str.len()
