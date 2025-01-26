import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def prepare_data(df, target, lookback=7, forecast_horizon=30):
    # Prepare features and target
    features = [col for col in df.columns if col != target]
    X = df[features].values
    y = df[target].values
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale only AFTER splitting
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create time series generators
    train_generator = TimeseriesGenerator(X_train_scaled, y_train_scaled, 
                                          length=lookback, 
                                          batch_size=32)
    test_generator = TimeseriesGenerator(X_test_scaled, y_test_scaled, 
                                         length=lookback, 
                                         batch_size=32)
    
    return train_generator, test_generator, feature_scaler, target_scaler

def create_lstm_model(input_shape, output_shape=1):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
        LSTM(25, activation='relu'),
        Dense(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def forecast_next_30_days(model, last_lookback_data, feature_scaler, target_scaler, forecast_steps=30):
    current_batch = last_lookback_data
    forecasts = []
    
    for _ in range(forecast_steps):
        # Predict next step
        predicted_scaled = model.predict(current_batch)
        predicted = target_scaler.inverse_transform(predicted_scaled)
        
        forecasts.append(predicted[0][0])
        
        # Update batch for next prediction
        next_batch = np.roll(current_batch[0], -1, axis=0)
        next_batch[-1] = np.concatenate([feature_scaler.transform(predicted_scaled), 
                                          np.zeros((1, feature_scaler.transform(predicted_scaled).shape[1]-1))], axis=1)
        current_batch = next_batch.reshape(1, *next_batch.shape)
    
    return forecasts

# Main execution
df = pd.read_csv('../data/mulu/mulu-rainfall-daily.csv', encoding='latin')
df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d")
df.set_index("DateTime", inplace=True)
df.drop(columns=["Latitude", "Longitude"], inplace=True)

# Feature interaction code (from your original script)
source1 = ['Rainfall']
source2 = ['TOTAL', 'ClimAdjust', 'ANOM']
source3 = ['Temperature', 'DewPoint', 'Humidity', 'Visibility', 'WindSpeed', 'Pressure']
source4 = ['Elevation']

# Interaction feature generation
for s1 in source1:
    for s2 in source2:
        df[f'{s1}_{s2}'] = df[s1] * df[s2]        
    for s3 in source3:
        df[f'{s1}_{s3}'] = df[s1] * df[s3]
        
for s2 in source2:
    for s3 in source3:
        df[f'{s2}_{s3}'] = df[s2] * df[s3]

for i in range(len(source3)):
    for j in range(i + 1, len(source3)):
        df[f'{source3[i]}_{source3[j]}'] = df[source3[i]] * df[source3[j]]

for s4 in source4:
    for s1 in source1:
        df[f'{s4}_{s1}'] = df[s4] * df[s1]
    for s2 in source2:
        df[f'{s4}_{s2}'] = df[s4] * df[s2]        
    for s3 in source3:
        df[f'{s4}_{s3}'] = df[s4] * df[s3]

df['Rainfall_Temperature_Humidity'] = df['Rainfall'] * df['Temperature'] * df['Humidity']

# Rolling statistics
for window in [7,14,30]:
    df[f"Rainfall_{window}d_mean"] = df["Rainfall"].rolling(window).mean()
    df[f"Rainfall_{window}d_std"] = df["Rainfall"].rolling(window).std()
    df[f"Rainfall_{window}d_sum"] = df["Rainfall"].rolling(window).sum()
    df[f"Rainfall_{window}d_min"] = df["Rainfall"].rolling(window).min()
    df[f"Rainfall_{window}d_max"] = df["Rainfall"].rolling(window).max()
    df[f"Rainfall_{window}d_median"] = df["Rainfall"].rolling(window).median()

df.dropna(inplace=True)

# Prepare data
train_generator, test_generator, feature_scaler, target_scaler = prepare_data(df, 'Rainfall')

# Create and train model
n_features = train_generator.data.shape[1]
model = create_lstm_model(input_shape=(7, n_features))
model.fit(train_generator, epochs=50, validation_data=test_generator)

# Forecast next 30 days
last_lookback_data = TimeseriesGenerator(
    feature_scaler.transform(df[df.columns.drop('Rainfall')].values[-7:]),
    df['Rainfall'].values[-7:],
    length=7,
    batch_size=1
)[0][0]

forecast = forecast_next_30_days(model, last_lookback_data, feature_scaler, target_scaler)
print("30-Day Rainfall Forecast:", forecast)