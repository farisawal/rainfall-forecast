import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load and preprocess the data
def prepare_data(df):
    """
    Prepare data for LSTM model with domain-specific feature engineering
    """
    # Convert DateTime to datetime type and set as index
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Feature engineering based on metocean knowledge
    df['Month'] = df['DateTime'].dt.month  # Seasonal patterns
    df['DewPointDelta'] = df['Temperature'] - df['DewPoint']  # Measure of air saturation
    
    # Calculate vapor pressure deficit (VPD) - important for precipitation formation
    # VPD = (1 - RH/100) * SVP, where SVP is saturated vapor pressure
    def calculate_vpd(temp_f, rh):
        temp_c = (temp_f - 32) * 5/9
        svp = 0.61078 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        vpd = (1 - rh/100) * svp
        return vpd
    
    df['VPD'] = calculate_vpd(df['Temperature'], df['Humidity'])
    
    # Create pressure tendency (change in pressure over time)
    df['PressureTendency'] = df['Pressure'].diff()
    
    # Convert wind direction to sine and cosine components
    df['WindDir_Sin'] = np.sin(np.radians(df['WindDir']))
    df['WindDir_Cos'] = np.cos(np.radians(df['WindDir']))
    
    # Calculate wind stress (proportional to wind speed squared)
    df['WindStress'] = df['WindSpeed'] ** 2
    
    # Drop original DateTime and WindDir columns
    features_to_drop = ['DateTime', 'WindDir', 'Rainfall']
    X = df.drop(features_to_drop, axis=1)
    y = df['Rainfall']
    
    return X, y

def create_sequences(X, y, lookback=7):
    """
    Create sequences for LSTM with specified lookback period
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - lookback):
        X_seq.append(X[i:i+lookback])
        y_seq.append(y[i+lookback])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape):
    """
    Build LSTM model architecture
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Main execution
def train_rainfall_model(df, lookback=7, test_size=0.2, validation_split=0.2):
    """
    Train LSTM model with proper scaling and sequence creation
    """
    # Prepare features and target
    X, y = prepare_data(df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Scale features - separate scalers for train and test
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = MinMaxScaler().fit_transform(X_test)  # Separate scaler for test
    
    # Scale target variable
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = MinMaxScaler().fit_transform(y_test.values.reshape(-1, 1))  # Separate scaler
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lookback)
    
    # Build and train model
    model = build_lstm_model((lookback, X_train.shape[1]))
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=validation_split,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history, (X_test_seq, y_test_seq)

# Function to evaluate model
def evaluate_model(model, test_data, target_scaler):
    """
    Evaluate model performance and create visualizations
    """
    X_test_seq, y_test_seq = test_data
    
    # Make predictions
    y_pred = model.predict(X_test_seq)
    
    # Inverse transform predictions and actual values
    y_pred_orig = target_scaler.inverse_transform(y_pred)
    y_test_orig = target_scaler.inverse_transform(y_test_seq)
    
    # Calculate metrics
    mse = np.mean((y_pred_orig - y_test_orig) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_orig - y_test_orig))
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_orig[:100], label='Actual')
    plt.plot(y_pred_orig[:100], label='Predicted')
    plt.title('Rainfall Prediction - First 100 Test Samples')
    plt.xlabel('Time Steps')
    plt.ylabel('Rainfall (mm)')
    plt.legend()
    plt.show()
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

# Load your data
df = pd.read_csv('../data/mulu/mulu-rainfall-daily.csv', encoding='latin')
df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d")
df.set_index("DateTime", inplace=True)
df.drop(columns=["TOTAL", "ClimAdjust", "Latitude", "Longitude", "Elevation"], inplace=True)

# Train the model
model, history, test_data = train_rainfall_model(df)

# Evaluate the model
metrics = evaluate_model(model, test_data, target_scaler = MinMaxScaler())