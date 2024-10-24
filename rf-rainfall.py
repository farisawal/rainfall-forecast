import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pandas import DataFrame
import pickle

# Load and preprocess the data
df = pd.read_csv('data/lawas-daily-data.csv', encoding='latin')

df['Wind Speed'] = df['Wind Speed'].replace(np.nan, 0)
df = df.fillna(method='ffill')

# Select features and target
data_input = df[['Total_Rainfall', 'ANOM', 'Flood_Height_Max', 'Pressure', 'Temperature', 'Dew Point', 'Humidity', 'Wind Speed']]

# Scaling the features and target separately
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Function to create lagged features for all columns in the dataframe
def create_lagged_features(df, num_lags):
    df_lagged = DataFrame()
    for col in df.columns:
        for i in range(num_lags, 0, -1):
            df_lagged[f'{col}_t-{i}'] = df[col].shift(i)
    df_lagged['Current_Rainfall'] = df['Total_Rainfall']
    df_lagged.dropna(inplace=True)
    return df_lagged

# Create lagged dataset for forecasting using all columns
num_lags = 30  # Number of lagged values to use for forecasting
df_lagged = create_lagged_features(data_input, num_lags)

# Separate features (lags of all columns) and target (Current_Rainfall)
features = df_lagged.drop(columns=['Current_Rainfall'])
target = df_lagged['Current_Rainfall']

# Scale features and target
features_scaled = feature_scaler.fit_transform(features)
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.30, random_state=100, shuffle=False)

# Train Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(x_train, y_train.ravel())
with open("RF_model.pkl", 'wb') as file:  
    pickle.dump(rf, file)

# with open('RF_model.pkl', 'rb') as f:
#     rf = pickle.load(f)

# Forecasting function
def forecast(model, initial_input, num_steps, feature_scaler, target_scaler):
    predictions = []
    current_input = initial_input

    for i in range(num_steps):
        current_input_scaled = feature_scaler.transform(current_input.reshape(1, -1))
        pred_scaled = model.predict(current_input_scaled)
        pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        predictions.append(pred[0][0])

        # Shift the input to include the new prediction as the latest 'lag'
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred[0][0]

    return np.array(predictions)

# Forecast future values
forecast_length = 30  # For example, forecasting the next 30 days
initial_input = df_lagged.iloc[-1, :-1].values  # Take the last available data point as the starting input
predictions = forecast(rf, initial_input, forecast_length, feature_scaler, target_scaler)

# Output results
print(f"Predicted rainfall for the next {forecast_length} days:")
print(predictions)

# Plot forecasted values
plt.plot(predictions, label='Forecasted Rainfall', color='orange', alpha=0.6, linestyle='-', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Rainfall (mm)')
plt.title(f'{forecast_length}-Day Forecast of Rainfall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()