import pandas as pd

from sklearn.preprocessing import StandardScaler



# Drop unneccesary or mostly empty features of the dataset
def drop_extra_features(df):
    
    # Create a copy to avoid the SettingWithCopyWarning and 
    # prevent modifying the original dataframe unintentionally.
    df_cleaned = df.copy()
    
    columns_to_drop = [
        'Unnamed: 0',
        'id', 
        'url', 
        'region', 
        'region_url', 
        'VIN', 
        'drive',
        'size',
        'type',
        'model',
        'image_url', 
        'description',
        'paint_color',
        'county',
        'state',
        'lat',
        'long',
        'posting_date',
        'removal_date'                
    ]
    
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    
    print(f"Successfully dropped {len(columns_to_drop)} columns.")
    print(f"Remaining columns: {list(df_cleaned.columns)}\n")
    
    return df_cleaned



# Filters out the outliers of categoires: price, odometer, year
def filter_outliers(df):
    
    # Create a copy to avoid the pandas SettingWithCopyWarning
    df_filtered = df.copy()
    
    min_price=500
    max_price=100000 
    
    min_odo=1
    max_odo=400000
    
    min_year=1980
    max_year=2026
    
    # 1. Filter Price
    df_filtered = df_filtered[
        (df_filtered['price'] >= min_price) & 
        (df_filtered['price'] <= max_price)
    ]
    
    # 2. Filter Odometer
    df_filtered = df_filtered[
        (df_filtered['odometer'] >= min_odo) & 
        (df_filtered['odometer'] <= max_odo)
    ]
    
    # 3. Filter Year
    df_filtered = df_filtered[
        (df_filtered['year'] >= min_year) & 
        (df_filtered['year'] <= max_year)
    ]
    
    # Calculate how many rows were dropped for your report
    rows_dropped = len(df) - len(df_filtered)
    print(f"Original row count: {len(df)}")
    print(f"New row count: {len(df_filtered)}")
    print(f"Total outliers removed: {rows_dropped}\n")
    
    return df_filtered

# One-Hot Encoding to the specified categorical columns in a DataFrame
def encode_categorical_features(df):

    categorical_cols = ['manufacturer', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission']
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    return df_encoded



def scale_numerical_features(df):

    numerical_cols = ['price', 'year','odometer']
    
    # Create a copy so we don't accidentally modify the original dataframe
    df_scaled = df.copy()
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler to the data and transform it
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    
    return df_scaled