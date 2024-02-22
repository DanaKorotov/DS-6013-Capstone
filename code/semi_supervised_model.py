#%% Import packages
# Basics
import numpy as np
import pandas as pd
# To check vessel names
import re
# To get exact lat/long
from shapely.wkt import loads
# To map the ships or coastlines
import geopandas as gpd
from geopy.distance import geodesic
# To find coastline distance
from scipy.spatial import KDTree
# For XgBoosting
import xgboost as xgb
import random
# For upsampling during the modeling
from imblearn.over_sampling import SMOTE

#%% Data formatting 
ais = pd.read_csv("Sample data/ais_east_asia_20231001/ais_east_asia_oct2024_20231001.csv")
ais['dtg'] = pd.to_datetime(ais['dtg'])
ais = ais.sort_values(['mmsi', 'dtg'])
# Calculate the time difference between consecutive rows for each mmsi
ais['time_diff'] = ais.groupby('mmsi')['dtg'].diff()
# Create a new column called first_occurrence based on the identified conditions
ais['first_occurrence'] = ((ais['time_diff'].isnull()) | (ais['time_diff'] > pd.Timedelta(hours=10))).astype(int)
ais['occ_num'] = ais.groupby('mmsi').agg(occ_num = ('first_occurrence', 'cumsum')) # Separate different trips

#%% Get distance traveled
ais['old_lat'] = ais.groupby(['mmsi', 'occ_num'])['latitude'].shift(1)
ais['old_long'] = ais.groupby(['mmsi', 'occ_num'])['longitude'].shift(1)

# Get distance and spped from consecutive readings
def calculate_distance(row):
    if pd.notna(row['old_lat']) and pd.notna(row['old_long']) and pd.notna(row['latitude']) and pd.notna(row['longitude']):
        return geodesic((row['old_lat'], row['old_long']), (row['latitude'], row['longitude'])).nautical
    else:
        return np.nan

ais['distance'] = ais.apply(calculate_distance, axis = 1)

#%% Check naming conventions
# Look for sketchy names
def check_net_name(name):
    name = name.lower()
    net_name = ('%' in name or 
                'buoy' in name or 
                'net' in name or 
                bool(re.search(r"\d+v", name)))
    return net_name
ship_names = ais['vessel_name'].astype('str').unique()
net_names = [check_net_name(name) for name in ship_names]
ais['net_name'] = ais['vessel_name']\
    .map(dict(zip(ship_names, net_names)))
# Look for bad mmsi values
ais['mmsi_length'] = ais['mmsi'].astype('str').str.len() != 9
del ship_names, net_names

#%% Check for starting in water
# Read in coastline data from https://www.naturalearthdata.com/downloads/10m-physical-vectors/
coastline = pd.concat([gpd.read_file("ne_10m_land/ne_10m_land.shp"), 
                       gpd.read_file("ne_10m_minor_islands/ne_10m_minor_islands.shp")])
coastline = coastline.cx[min(ais['longitude'])-1:max(ais['longitude'])+1, 
                         min(ais['latitude'])-1:max(ais['latitude'])+1].geometry
# Make function to find coastline distance
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
# Find distance to shore at start
new_vessels = ais.loc[ais['first_occurrence'] == 1, ['mmsi', 'occ_num', 'latitude', 'longitude']]
new_vessels = distance_to_coast(new_vessels, coastline)
new_vessels['spawn_offshore'] = new_vessels['distance_to_coast'] >= 1

# Make dictionary and map to dataframe
spawn_dict = {}
for index, row in new_vessels.iterrows():
    # Combine mmsi and occ_num to create the key
    key = (row['mmsi'], row['occ_num'])
    # Assign the value of 'spawn_offshore' to the key in the dictionary
    spawn_dict[key] = row['spawn_offshore']
ais['spawn_offshore'] = ais.apply(lambda row: (row['mmsi'], row['occ_num']), axis=1)\
    .map(spawn_dict)
del index, row, key, coastline, spawn_dict, new_vessels

# Add all potential issues together
ais['red_flags'] = ais['net_name'].astype('int') + ais['mmsi_length'].astype('int') + ais['spawn_offshore'].astype('int')

#%% Check movement patterns




#%%
# Get rid of extraneous columns
ais = ais.loc[:, ['mmsi', 'occ_num', 'longitude', 'latitude', 'sog', 'cog', 'rot', 
                  'distance', 'dtg', 'net_name', 'mmsi_length', 'spawn_offshore', 
                  'red_flags']]



#%% Prep data for XgBoost
ais_model_data = ais.groupby(['mmsi', 'occ_num'])\
    .agg(
        net_name = ('net_name', 'max'),
        mmsi_length = ('mmsi_length', 'max'),
        spawn_offshore = ('spawn_offshore', 'max'),
        speed_0 = ('sog', 'min'),
        speed_med = ('sog', 'median'),
        speed_99 = ('sog', 'max'),
        speed_std = ('sog', 'std'),
        dist_med = ('distance', 'median'),
        dist_99 = ('distance', 'max'),
        dist_std = ('distance', 'std'), 
        red_flags = ('red_flags', 'max')
    ) # If this starts failing, maybe use np.median() or other numpy functions
ais_model_data[np.isnan(ais_model_data)] = pd.NA # Set to pd.NA for future reasons

# Find the ones we think are def nets and def not nets
prob_nets = ais_model_data.query("red_flags == 3").copy().reset_index()
prob_nets['net'] = 1
prob_ships = ais_model_data.query("red_flags == 0").copy().reset_index()
prob_ships['net'] = 0
out_of_model = ais_model_data.query("red_flags != 0 & red_flags != 3").copy().reset_index()

# Combine the datasets of probable good/bad
model_data = pd.concat([prob_nets, prob_ships]).reset_index(drop = True)





#%%

































#%% Make a function that can handle multiple rounds
def semi_supervised_xgb(prob_nets, prob_ships, out_of_model, 
                        boost_params = {
                            'objective': 'binary:logistic',  # Binary classification objective 
                            'eval_metric': 'logloss',  # Evaluation metric 
                            'eta': .03 # Learning rate
                            },
                        features = ['speed_0', 'speed_med', 'speed_99', 'speed_std', 
                                    'dist_med', 'dist_99', 'dist_std'], 
                        num_rounds = 5, k_cv = 5, np_seed = 4120, xgb_seed = 34567
                            ):
    # Print out starting numbers:
    print(f"Pre-modeling: {len(prob_nets)} probable nets and {len(prob_ships)} probable ships with {len(out_of_model)} out of model")
    unassigned = len(out_of_model)
    for i in range(1, num_rounds + 1):
        model_data = pd.concat([prob_nets, prob_ships]).reset_index(drop = True)
        # Upsample the data before modeling
        sampling_strategy = {0: len(prob_ships), 1: round(len(prob_ships)/2)}
        smote = SMOTE(sampling_strategy = sampling_strategy, random_state = 52)
        model_data[pd.isna(model_data)] = -9999 # Use as placeholder value
        upsampled_data, upsampled_labels = smote.fit_resample(model_data, model_data['net'])
        upsampled_data[upsampled_data == -9999] = pd.NA # Put NAs instead of placeholder
        np.random.seed(np_seed) # Set seed
        # Get data ready for XgBoost
        upsampled_data['fold'] = np.random.randint(low = 1, high = k_cv + 1, 
                                                   size = len(upsampled_data))
        cv_folds = [(upsampled_data[upsampled_data.fold == i].index, 
                     upsampled_data[upsampled_data.fold != i].index) for i in range(1, k_cv + 1)]
        x_features = upsampled_data[features]
        x_mat = xgb.DMatrix(x_features, label = upsampled_labels)
        # Get that CV in
        cv_results = xgb.cv(params = boost_params, 
                            dtrain = x_mat, 
                            num_boost_round = 1000, 
                            nfold = k_cv, 
                            folds = cv_folds,
                            metrics = 'logloss', 
                            early_stopping_rounds = 12, 
                            seed = xgb_seed)
        # Get the optimal number of boosting rounds
        optimal_rounds = cv_results['test-logloss-mean'].idxmin() + 1
        # Perform the optimized boost 
        real_boost = xgb.train(params = boost_params, dtrain = x_mat, 
                               num_boost_round = optimal_rounds)
        test_features = out_of_model[features]
        test_mat = xgb.DMatrix(test_features)
        # Make predictions 
        predictions = real_boost.predict(test_mat) 
        with_probs = pd.concat([out_of_model, pd.DataFrame(predictions)], axis = 1)\
            .rename(columns = {0: 'prob_net'})
        # Move out-of-model rows into correct DataFrames 
        prob_nets = pd.concat([prob_nets,
                               with_probs.query("prob_net >= .9").drop(['prob_net'], axis = 1)])
        prob_nets['net'] = 1
        prob_ships = pd.concat([prob_ships,
                                with_probs.query("prob_net <= .02").drop(['prob_net'], axis = 1)])
        prob_ships['net'] = 0
        out_of_model = with_probs.query("prob_net > .02 & prob_net < .9")\
            .drop(['prob_net'], axis = 1)
        # Print statement:
        print(f"Round {i}: {len(prob_nets)} probable nets and {len(prob_ships)} probable ships with {len(out_of_model)} out of model")
        if (len(out_of_model) == 0):
            print('Stopping early: Out of unassigned ships')
            break
        if (len(out_of_model) == unassigned):
            print('Stopping early: Progress finished')
            break
        unassigned = len(out_of_model) # To use next loop



#%%
semi_supervised_xgb(prob_nets, prob_ships, out_of_model, 
                    num_rounds = 11, k_cv = 7, np_seed = 4200, 
                    xgb_seed = 98765,
                    boost_params = {
                        'objective': 'binary:logistic',  # Binary classification objective 
                        'eval_metric': 'logloss',  # Evaluation metric 
                        'eta': .03 # Learning rate
                        })



#%%













#%%
sampling_strategy = {0: len(prob_ships), 1: round(len(prob_ships)/2)}
smote = SMOTE(sampling_strategy = sampling_strategy, random_state = 529)
model_data[np.isnan(model_data)] = -9999
upsampled_data, upsampled_labels = smote.fit_resample(model_data, 
                                                      model_data['net'])
upsampled_data[upsampled_data == -9999] = np.NaN

















#%% Get xgboost stuff ready
# Upsample to balance data
smote = SMOTE(random_state = 529)
model_data[pd.isna(model_data)] = -9999
upsampled_data, upsampled_labels = smote.fit_resample(model_data, model_data['net'])
upsampled_data[upsampled_data == -9999] = pd.NA

# Make parameters
params = {
    'objective': 'binary:logistic',  # Binary classification objective 
    'eval_metric': 'logloss',  # Evaluation metric 
    'eta': .03 # Learning rate
}

# Make cv folds
k_cv = 5
np.random.seed(4120)
upsampled_data['fold'] = np.random.randint(low = 1, high = k_cv + 1, 
                                           size = len(upsampled_data))
cv_folds = [(upsampled_data[upsampled_data.fold == i].index, 
             upsampled_data[upsampled_data.fold != i].index) for i in range(1, k_cv + 1)]

# Finalize data to put into training function
features = upsampled_data.drop(['mmsi', 'occ_num', 'net_name', 'mmsi_length', 
                                'spawn_offshore', 'red_flags', 'net', 'fold'], 
                                axis = 1)
x_mat = xgb.DMatrix(features, label = upsampled_labels)

#%% Do cross-validation on XgBoost to find optimal rounds
cv_results = xgb.cv(params = params, 
                    dtrain = x_mat, 
                    num_boost_round = 1000, 
                    nfold = k_cv, 
                    folds = cv_folds,
                    metrics = 'logloss', 
                    early_stopping_rounds = 12, 
                    seed = 31958)

# Get the optimal number of boosting rounds
optimal_rounds = cv_results['test-logloss-mean'].idxmin() + 1

#%% Run optimized boost, get predictions
real_boost = xgb.train(params = params, dtrain = x_mat, 
                       num_boost_round = optimal_rounds)
test_features = out_of_model.drop(['mmsi', 'occ_num', 'net_name', 'mmsi_length', 
                                   'spawn_offshore', 'red_flags'], 
                                   axis = 1)
test_mat = xgb.DMatrix(test_features)

predictions = real_boost.predict(test_mat)
with_probs = pd.concat([out_of_model, pd.DataFrame(predictions)], axis = 1)\
    .rename(columns = {0: 'prob_net'})\
    .sort_values(['prob_net'], ascending = False)
# Move out-of-model rows into correct DataFrames
prob_nets = pd.concat([prob_nets,
                        with_probs.query("prob_net >= .9").drop(['prob_net'], axis = 1)])
prob_nets['net'] = 1
prob_ships = pd.concat([prob_ships,
                        with_probs.query("prob_net <= .02").drop(['prob_net'], axis = 1)])
prob_ships['net'] = 0
out_of_model = with_probs.query("prob_net > .02 & prob_net < .9")\
    .drop(['prob_net'], axis = 1)
del cv_results, real_boost

#%% Redo it with changed data
model_data = pd.concat([prob_nets, prob_ships]).reset_index(drop = True)
np.random.seed(4120)
model_data['fold'] = np.random.randint(low = 1, high = k_cv + 1, 
                                       size = len(model_data))
cv_folds = [(model_data[model_data.fold == i].index, model_data[model_data.fold != i].index) for i in range(1, k_cv + 1)]

# Finalize data to put into training function
labels = model_data['net'] 
features = model_data.drop(['mmsi', 'occ_num', 'net_name', 'mmsi_length', 'spawn_offshore', 
                            'red_flags', 'net', 'fold'], 
                            axis = 1)
x_mat = xgb.DMatrix(features, label = labels)
cv_results = xgb.cv(params = params, 
                    dtrain = x_mat, 
                    num_boost_round = 1000, 
                    nfold = k_cv, 
                    folds = cv_folds,
                    metrics = 'logloss', 
                    early_stopping_rounds = 12, 
                    seed = 31958)

# Get the optimal number of boosting rounds
optimal_rounds = cv_results['test-logloss-mean'].idxmin() + 1
real_boost = xgb.train(params = params, dtrain = x_mat, 
                       num_boost_round = optimal_rounds)
test_features = out_of_model.drop(['mmsi', 'occ_num', 'net_name', 'mmsi_length', 
                                   'spawn_offshore', 'red_flags'], 
                                   axis = 1)
test_mat = xgb.DMatrix(test_features)

predictions = real_boost.predict(test_mat)
with_probs = pd.concat([out_of_model, pd.DataFrame(predictions)], axis = 1)\
    .rename(columns = {0: 'prob_net'})\
    .sort_values(['prob_net'], ascending = False)
# Move out-of-model rows into correct DataFrames
prob_nets = pd.concat([prob_nets,
                        with_probs.query("prob_net >= .9").drop(['prob_net'], axis = 1)])
prob_nets['net'] = 1
prob_ships = pd.concat([prob_ships,
                        with_probs.query("prob_net <= .02").drop(['prob_net'], axis = 1)])
prob_ships['net'] = 0
out_of_model = with_probs.query("prob_net > .02 & prob_net < .9")\
    .drop(['prob_net'], axis = 1)
del cv_results, real_boost