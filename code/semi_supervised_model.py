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

#%% Make a class to perform the xgboosting
class semi_supervised_xgb:
    def __init__(self, class_1, class_0, unknown):
        '''
        Inputs:
            class_1: A dataframe of observations you think are part of class 1, with x and y variables included
            class_2: A dataframe of observations you think are part of class 0, with x and y variables included
            unknown: A dataframe of observations of unknown class, with x and y variables included

        Methods:
            xgboost: Takes in labeled and unlabeled data and returns an XGBoost model
        '''
        # Define important things
        self.class_1_og = class_1
        self.class_0_og = class_0
        self.unknown_og = unknown
        self.class_1 = class_1
        self.class_0 = class_0
        self.unknown = unknown
        self.boost = 0
        # Define to keep track of iterations done
        self.boost_rounds_completed = 0

    def xgboost(self, features, y_name, num_rounds = 11, k_cv = 7, 
                np_seed = 4200, xgb_seed = 98765, 
                boost_params = {
                    'objective': 'binary:logistic',  # Binary classification objective 
                    'eval_metric': 'logloss',  # Evaluation metric
                    },
                learning_rates = [.01, .03, .05], verbose = True, 
                class_1_prob = .95, 
                class_0_prob = .05): 
        '''
        Description: 
            Uses xgboost to self train a final xgboost model and outputs it
        
        Inputs:
            features: A list of the x variables in the given data
            y_name: The name of the y variable in the given data
            num_rounds: How many rounds of reclassification to use
            k_cv: How many cross-validation folds to use
            np_seed: Before setting the cross-validation folds, this sets the seed for replicability
            xgb_seed: The seed to use in the XGBoosts for replicability
            boost_params: Parameters besides learning rates to use in the XGBoost models. defaults to logloss on a binary classification. 
            learning_rates: A list of learning rates to use in the XGBoost models
            verbose: If True, prints the model's progress
            class_1_prob: If this number is met or exceeded by a model's given probability, a given observation is reclassified into class 1
            class_0_prob: If this number is not met by a model's given probability, a given observation is reclassified into class 0

        Outputs:
            xgboost model of class xgboost.core.Booster
        '''
        # Print out starting numbers:
        if verbose:
            print(f"Pre-modeling: {len(self.class_1)} of class 1 and {len(self.class_0)} of class 0 with {len(self.unknown)} out of model")
        unassigned = len(self.unknown)
        for i in range(1, num_rounds + 1):
            self.boost_rounds_completed += 1
            model_data = pd.concat([self.class_1, self.class_0]).reset_index(drop = True)
            # Upsample the data before modeling
            sampling_strategy = {0: len(self.class_0), 1: round(len(self.class_0)/2)}
            smote = SMOTE(sampling_strategy = sampling_strategy, random_state = 52)
            model_data[pd.isna(model_data)] = -999999 # Use as placeholder value
            upsampled_features, upsampled_labels = smote.fit_resample(model_data[features], model_data[y_name])
            upsampled_features[upsampled_features == -999999] = pd.NA # Put NAs instead of placeholder
            np.random.seed(np_seed) # Set seed
            # Get data ready for XgBoost
            upsampled_features['fold'] = np.random.randint(low = 1, high = k_cv + 1, 
                                                       size = len(upsampled_features))
            cv_folds = [(upsampled_features[upsampled_features.fold == i].index, 
                         upsampled_features[upsampled_features.fold != i].index) for i in range(1, k_cv + 1)]
            x_features = upsampled_features[features]
            x_mat = xgb.DMatrix(x_features, label = upsampled_labels)
            # Get that CV in
            best_rmse = float('inf')
            for lr in learning_rates:
                # Update learning rate in parameters
                boost_params['eta'] = lr
            
                # Perform cross-validation
                cv_results = xgb.cv(params = boost_params, 
                                    dtrain = x_mat,  
                                    num_boost_round = 3000, 
                                    nfold = k_cv, 
                                    folds = cv_folds,
                                    metrics = 'logloss', 
                                    early_stopping_rounds = 12, 
                                    seed = xgb_seed)
                
                if cv_results['test-logloss-mean'].min() < best_rmse:
                    best_rmse = cv_results['test-logloss-mean'].min()
                    best_lr = lr
                    optimal_rounds = cv_results['test-logloss-mean'].idxmin() + 1

            # Perform the optimized boost 
            boost_params['eta'] = best_lr
            real_boost = xgb.train(params = boost_params, dtrain = x_mat, 
                                   num_boost_round = optimal_rounds)
            test_features = self.unknown[features]
            test_mat = xgb.DMatrix(test_features)
            # Make predictions 
            predictions = real_boost.predict(test_mat) 
            with_probs = pd.concat([self.unknown, pd.DataFrame(predictions)], axis = 1)\
                .rename(columns = {0: 'prob_1'})
            # Move out-of-model rows into correct DataFrames 
            self.class_1 = pd.concat([self.class_1,
                                      with_probs.query(f"prob_1 >= {class_1_prob}")\
                                        .drop(['prob_1'], axis = 1)])
            self.class_1[y_name] = 1
            self.class_0 = pd.concat([self.class_0,
                                    with_probs.query(f"prob_1 < {class_0_prob}")\
                                        .drop(['prob_1'], axis = 1)])
            self.class_0[y_name] = 0
            self.unknown = with_probs.query(f"prob_1 < {class_1_prob} & prob_1 >= {class_0_prob}")\
                .drop(['prob_1'], axis = 1)
            # Print statement: 
            if verbose:
                print(f'''Round {self.boost_rounds_completed}: {len(self.class_1)} probable nets and {len(self.class_0)} probable ships with {len(self.unknown)} out of model. 
LR of {best_lr} for {optimal_rounds} rounds''')
            if (len(self.unknown) == 0):
                if verbose:
                    print('Stopping early: Out of unassigned ships')
                break
            if (len(self.unknown) == unassigned):
                if verbose:
                    print('Stopping early: Progress finished')
                break
            unassigned = len(self.unknown) # To use next loop 
        self.boost = real_boost
        return real_boost # return best XGBoost model at the end
    
    def reset(self):
        ''' 
        Resets the labels to their original form
        '''
        self.class_0 = self.class_0_og
        self.class_1 = self.class_1_og
        self.unknown = self.unknown_og
        self.boost_rounds_completed = 0

#%% Test out the new class
# Do the semi-supervised model
test = semi_supervised_xgb(prob_nets, prob_ships, out_of_model)
test_boost = test.xgboost(learning_rates = [.01, .03], num_rounds = 5, 
                          features = ['speed_0', 'speed_med', 'speed_99', 'speed_std', 
                                      'dist_med', 'dist_99', 'dist_std'],
                          y_name = 'net', class_0_prob = .015, class_1_prob = .985)

x_mat = xgb.DMatrix(ais_model_data[['speed_0', 'speed_med', 'speed_99', 'speed_std', 
                                    'dist_med', 'dist_99', 'dist_std']])
# Make predictions 
predictions = test_boost.predict(x_mat) 
ais_with_preds = pd.concat([ais_model_data.reset_index(drop = False), 
                            pd.DataFrame(predictions)], 
                            axis = 1)\
    .rename(columns = {0: 'prob_net'})
ais_with_preds.sort_values('prob_net')
