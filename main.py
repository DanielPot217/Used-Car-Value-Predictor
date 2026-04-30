import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import data_prep
import linear_reg
import knn


# This Scipt runs the data preprocessing functions and trains the models: Linear Regression & KNN



# Load the CSV into a dataframe
df_raw = pd.read_csv('data/craigslist_vehicles.csv')

# Data Preprocessing 
df = data_prep.drop_extra_features(df_raw)

df = data_prep.filter_outliers(df)

df = data_prep.encode_categorical_features(df)

df = data_prep.scale_numerical_features(df)


# Trains and evaluates using a  Multiple Linear Regression model
linear_reg.run_linear_regression(df)

# Trains and evaluates using K-Nearest Neighbors (KNN) Regressor
knn.run_knn_regressor(df)