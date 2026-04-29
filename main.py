import pandas as pd
import data_prep



# Load the CSV into a dataframe
df_raw = pd.read_csv('data/craigslist_vehicles.csv')


# Data Preprocessing 
df = data_prep.drop_extra_features(df_raw)

df = data_prep.filter_outliers(df)

df = data_prep.encode_categorical_features(df)

df = data_prep.scale_numerical_features(df)

# Save cleaned dataframe to CSV
df.to_csv('data/cleaned.csv', index=False)