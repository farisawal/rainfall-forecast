import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class RainfallLSTM:
    def __init__(self, lookback=7):
        """
        Initialize LSTM model for rainfall prediction
        
        Args:
        lookback (int): Number of previous time steps to use
        """
        self.lookback = lookback
        self.model = None
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
    
    def prepare_data(self, df, target='Rainfall', features=None):
        """
        Prepare multivariate time series data for LSTM with proper train/test split
        
        Args:
        df (pd.DataFrame): Input time series data
        target (str): Target variable column name
        features (list): List of feature column names
        
        Returns:
        dict: Dictionary containing train/test data and scalers
        """
        # Use all columns except target if features not specified
        if features is None:
            features = [col for col in df.columns if col != target]
        
        # First split the raw data
        train_size = int(len(df) * 0.7)
        train_data = df[:train_size]
        test_data = df[train_size:]
        
        # Scale features using training data
        self.feature_scaler.fit(train_data[features])
        train_features_scaled = self.feature_scaler.transform(train_data[features])
        test_features_scaled = self.feature_scaler.transform(test_data[features])
        
        # Scale target using training data
        self.target_scaler.fit(train_data[[target]])
        train_target_scaled = self.target_scaler.transform(train_data[[target]])
        test_target_scaled = self.target_scaler.transform(test_data[[target]])
        
        # Create sequences
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        # Training sequences
        for i in range(len(train_features_scaled) - self.lookback):
            X_train.append(train_features_scaled[i:i + self.lookback])
            y_train.append(train_target_scaled[i + self.lookback])
        
        # Test sequences
        for i in range(len(test_features_scaled) - self.lookback):
            X_test.append(test_features_scaled[i:i + self.lookback])
            y_test.append(test_target_scaled[i + self.lookback])
        
        return {
            'X_train': np.array(X_train),
            'X_test': np.array(X_test),
            'y_train': np.array(y_train),
            'y_test': np.array(y_test)
        }
    
    def build_model(self, input_shape):
        """
        Create LSTM model architecture
        
        Args:
        input_shape (tuple): Shape of input data
        """
        self.model = Sequential([
        LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model
    
    def train(self, X_train, y_train, validation_split=0.3):
        """
        Train the LSTM model
        
        Args:
        X_train (np.array): Training input sequences
        y_train (np.array): Training target values
        validation_split (float): Portion of training data for validation
        """
        # Early stopping and learning rate reduction
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=10, 
            min_lr=1e-6
        )
        
        # Build model if not already created
        if self.model is None:
            self.build_model(input_shape=(self.lookback, X_train.shape[2]))
        
        # Train model
        history = self.model.fit(
            X_train, 
            y_train, 
            epochs=200, 
            batch_size=16, 
            validation_split=validation_split, 
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        return history
    
    def predict(self, last_sequence):
        """
        Predict next value
        
        Args:
        last_sequence (np.array): Last observed sequence
        
        Returns:
        prediction (float): Predicted next value
        """
        # Reshape input if needed
        if len(last_sequence.shape) == 2:
            last_sequence = last_sequence.reshape((1, self.lookback, last_sequence.shape[1]))
        
        # Predict and inverse transform
        scaled_prediction = self.model.predict(last_sequence)[0][0]
        return self.target_scaler.inverse_transform([[scaled_prediction]])[0][0]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
        X_test (np.array): Test input sequences
        y_test (np.array): Test target values
        
        Returns:
        dict: Performance metrics
        """
        # Evaluate model
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predict and calculate additional metrics
        y_pred = self.model.predict(X_test)
        y_pred_original = self.target_scaler.inverse_transform(y_pred)
        y_test_original = self.target_scaler.inverse_transform(y_test)
        
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred_original)
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
        
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

    def plot_predictions(self, X_test, y_test, dates=None):
        """
        Plot actual vs predicted values
        
        Args:
        X_test (np.array): Test input sequences
        y_test (np.array): Test target values
        dates (pd.DatetimeIndex): Optional dates for x-axis
        """
        # Get predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        y_pred_orig = self.target_scaler.inverse_transform(y_pred)
        y_test_orig = self.target_scaler.inverse_transform(y_test)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        if dates is not None:
            plt.plot(dates[-len(y_test):], y_test_orig, label='Actual')
            plt.plot(dates[-len(y_test):], y_pred_orig, label='Predicted')
        else:
            plt.plot(y_test_orig, label='Actual')
            plt.plot(y_pred_orig, label='Predicted')
        
        plt.title('Actual vs Predicted Rainfall')
        plt.xlabel('Date')
        plt.ylabel('Rainfall')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return y_pred_orig, y_test_orig

def main():
    # Load data
    df = pd.read_csv('../data/mulu/mulu-rainfall-daily.csv', encoding='latin')
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d")
    df.set_index("DateTime", inplace=True)
    df.drop(columns=["TOTAL", "ClimAdjust", "Latitude", "Longitude", "Elevation"], inplace=True)
    
    # # Define feature groups
    # interaction_features = {
    #     'Rainfall': ['Temperature', 'Humidity', 'Visibility', 'WindDir'],
    #     'Temperature': ['DewPoint', 'Humidity', 'Visibility', 'Pressure'],
    #     'Humidity': ['DewPoint', 'WindSpeed', 'WindDir'],
    #     'Visibility': ['WindSpeed', 'WindDir'],
    #     'WindDir': ['WindSpeed']
    # }

    # # Create interaction features
    # for feature, correlated_features in interaction_features.items():
    #     for correlated_feature in correlated_features:
    #         interaction_term = f'{feature}_{correlated_feature}'
    #         df[interaction_term] = df[feature] * df[correlated_feature]
    
    df['DewPoint_Depression'] = df['Temperature'] - df['DewPoint']
    # df['WindDir_u'] = np.cos(np.radians(df['WindDir']))
    # df['WindDir_v'] = np.sin(np.radians(df['WindDir']))
    df['Humidity_WindSpeed'] = df['Humidity'] * df['WindSpeed']
    df['Humidity_Pressure_Index'] = df['Humidity'] * (1/df['Pressure'])

    # Add rolling statistics
    # for feat in ['Rainfall', 'Temperature', 'DewPoint', 'Humidity', 'Visibility', 'WindSpeed', 'Pressure']:
    #     for window in [3, 7, 14]:
    #         df[f"{feat}_{window}d_avg"] = df[feat].rolling(window).mean()
    #         df[f"{feat}_{window}d_std"] = df[feat].rolling(window).std()
    #         df[f"{feat}_{window}d_sum"] = df[feat].rolling(window).sum()
    #         df[f"{feat}_{window}d_min"] = df[feat].rolling(window).min()
    #         df[f"{feat}_{window}d_max"] = df[feat].rolling(window).max()
    #         df[f"{feat}_{window}d_median"] = df[feat].rolling(window).median()
    
    # Add lagged features
    # for feat in ['Temperature', 'DewPoint', 'Humidity', 'Visibility', 'WindSpeed', 'Pressure']:
    #     for lag in [1, 3, 7]:
    #         df[f"{feat}_{lag}d_lag"] = df[feat].shift(lag)
    
    # Drop NA rows
    df.dropna(inplace=True)
    
    # Prepare features and target
    target = 'Rainfall'
    features = [col for col in df.columns if col != target]
    print(features)

    # Initialize LSTM model
    lstm = RainfallLSTM(lookback=7)

    # Prepare data with built-in train/test split
    prepared_data = lstm.prepare_data(df, target=target, features=features)

    # Access training and test data
    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    
    # Train model
    history = lstm.train(X_train, y_train)
    
    # Evaluate model
    metrics = lstm.evaluate(X_test, y_test)
    print("Model Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # # Example prediction
    # last_sequence = X_test[0]
    # print(f"\nLast Sequence:\n{last_sequence}")
    # prediction = lstm.predict(last_sequence)
    # print(f"\nPredicted Rainfall: {prediction}")

    # In main(), replace the last few lines with:
    # Get the last sequence from test data
    last_sequence = X_test[-1]  # This gets the last sequence
    actual_value = y_test[-1]   # This gets the actual value for comparison

    print(f"\nLast Sequence Shape: {last_sequence.shape}")
    print(f"Last Sequence:\n{last_sequence}")
    prediction = lstm.predict(last_sequence)
    actual = lstm.target_scaler.inverse_transform(actual_value.reshape(1, -1))[0][0]
    print(f"\nPredicted Rainfall: {prediction}")
    print(f"Actual Rainfall: {actual}")

    # Plot predictions
    dates = df.index
    y_pred, y_test = lstm.plot_predictions(X_test, y_test, dates)
    plt.show()

    # Calculate correlation
    correlation = np.corrcoef(y_pred.flatten(), y_test.flatten())[0,1]
    print(f"\nCorrelation between actual and predicted: {correlation:.4f}")

if __name__ == "__main__":
    main()