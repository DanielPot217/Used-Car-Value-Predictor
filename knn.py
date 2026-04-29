import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_knn_regressor(final_processed_df):
    """
    Trains and evaluates a K-Nearest Neighbors Regressor model.
    """
    
    target_column='price'
    
    print("--- 1. Setting up Data ---")
    X = final_processed_df.drop(target_column, axis=1)
    y = final_processed_df[target_column]
    
    # 80/20 Train-Test Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split complete. Training model...")

    print("\n--- 2. Training the Model ---")
    # Initialize the model. 
    # n_neighbors=5 means it looks at the 5 most similar cars to guess the price.
    # n_jobs=-1 tells your computer to use all its processors to speed up the math.
    knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    
    # "Fitting" a KNN is instant, because it just memorizes the training data.
    knn_model.fit(X_train, y_train)
    print("Model training complete. (Predicting might take a minute!)")

    print("\n--- 3. Evaluating the Model ---")
    # This is where KNN does the heavy lifting, calculating all the distances.
    y_pred = knn_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared: {r2:.4f}")
    
    return knn_model