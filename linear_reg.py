import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_linear_regression(final_processed_df):
    """
    Trains and evaluates a Multiple Linear Regression model.
    """
    
    target_column='price'
    
    print("--- 1. Setting up Data ---")
    # Separate the features (X) from the target you want to predict (y)
    X = final_processed_df.drop(target_column, axis=1)
    y = final_processed_df[target_column]
    
    # Perform the 80/20 Train-Test Split
    # random_state=42 ensures you get the exact same split every time you run it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split complete. Training model...")

    print("\n--- 2. Training the Model ---")
    # Initialize and train (fit) the model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("Model training complete.")

    print("\n--- 3. Evaluating the Model ---")
    # Ask the model to predict prices based on the unseen test data
    y_pred = lr_model.predict(X_test)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE is the square root of MSE
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared: {r2:.4f}")

    print("\n--- 4. Analyzing Coefficients (Interpretability) ---")
    # Create a clean table of the features and their corresponding weights
    coefficients_df = pd.DataFrame({
        'Feature': X.columns, 
        'Coefficient': lr_model.coef_
    })
    
    # Sort them so you can easily see which features increase price vs decrease price
    coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)
    print(coefficients_df)
    
    return lr_model, coefficients_df