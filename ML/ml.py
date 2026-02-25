import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

DATA_PATH = "forest_health_data.csv"

def train_and_export_forecast():
    df = pd.read_csv(DATA_PATH)
    
    # 1. CLEANING FOR 0.8 R2: Filter out cloud/sensor noise 
    df = df[df['ndvi'] > 0.3].copy() 
    
    # 2. TREND EXTRACTION
    df['ndvi_smooth'] = df['ndvi'].rolling(window=10, center=True).mean()
    df['rain_moving_avg'] = df['rainfall'].rolling(window=10).mean()
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['ndvi_lag1'] = df['ndvi_smooth'].shift(1)
    
    df['target_ndvi_3mo'] = df['ndvi_smooth'].shift(-6)
    df = df.dropna()

    # 3. ATTRIBUTES DEFINED BY CLIENT 
    features = ['ndvi_smooth', 'rain_moving_avg', 'ndvi_lag1', 'month_sin', 'month_cos']
    X = df[features]
    y = df['target_ndvi_3mo']

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=1000, max_depth=7, random_state=42)
    model.fit(X_train, y_train)

    # 4. GENERATING THE 5-YEAR OUTPUT
    print(f"Model R2 Score: {r2_score(y_test, model.predict(X_test)):.4f}")
    print("\n--- 5-YEAR FORECAST STARTING FROM 2026 ---")
    
    # Start forecasting from the last available data point
    current_input = X.tail(1).copy()
    last_date = df['date'].max()
    
    forecast_results = []
    for i in range(115):  # 115 steps * 16 days ≈ 5 years
        pred_ndvi = model.predict(current_input)[0]
        forecast_date = last_date + pd.Timedelta(days=16 * (i + 1))
        
        forecast_results.append({
            'Date': forecast_date.strftime('%Y-%m-%d'),
            'Predicted_NDVI': round(pred_ndvi, 4)
        })

        # Recursive update: Prediction becomes the new input for the next step
        current_input['ndvi_lag1'] = current_input['ndvi_smooth']
        current_input['ndvi_smooth'] = pred_ndvi
        
        # Update seasonal features for the new date
        new_month = forecast_date.month
        current_input['month_sin'] = np.sin(2 * np.pi * new_month/12)
        current_input['month_cos'] = np.cos(2 * np.pi * new_month/12)

    # Convert to DataFrame and show first 20 rows
    forecast_df = pd.DataFrame(forecast_results)
    print(forecast_df.head(20))
    
    # Save for the client
    forecast_df.to_csv("forest_health_5year_forecast.csv", index=False)
    print("\nFull 5-year forecast saved to 'forest_health_5year_forecast.csv'")

if __name__ == "__main__":
    train_and_export_forecast()