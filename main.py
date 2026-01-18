import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/google.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)

# ============ METHOD 1: LSTM (Deep Learning) ============
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Prepare data for LSTM
target = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(target)

# Split data (80% train, 20% test)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences
seq_length = 60  # Use 60 days to predict next day
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, 
                    batch_size=32, 
                    epochs=100, 
                    validation_split=0.1,
                    callbacks=[early_stop],
                    verbose=1)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
test_mae = mean_absolute_error(y_test_actual, test_predict)

print(f"LSTM - Train RMSE: ${train_rmse:.2f}")
print(f"LSTM - Test RMSE: ${test_rmse:.2f}")
print(f"LSTM - Test MAE: ${test_mae:.2f}")

# ============ METHOD 2: ARIMA ============
from statsmodels.tsa.arima.model import ARIMA

# Use adjusted close price
train_arima = df['close'][:train_size]
test_arima = df['close'][train_size:]

# Fit ARIMA model (p=5, d=1, q=0)
model_arima = ARIMA(train_arima, order=(5, 1, 0))
fitted_arima = model_arima.fit()

# Forecast
forecast_arima = fitted_arima.forecast(steps=len(test_arima))
arima_rmse = np.sqrt(mean_squared_error(test_arima, forecast_arima))
arima_mae = mean_absolute_error(test_arima, forecast_arima)

print(f"\nARIMA - Test RMSE: ${arima_rmse:.2f}")
print(f"ARIMA - Test MAE: ${arima_mae:.2f}")

# ============ METHOD 3: Prophet ============
from prophet import Prophet

# Prepare data for Prophet
df_prophet = df.reset_index()[['date', 'close']]
df_prophet.columns = ['ds', 'y']

train_prophet = df_prophet[:train_size]
test_prophet = df_prophet[train_size:]

# Fit Prophet model
prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
prophet_model.fit(train_prophet)

# Make predictions
future = prophet_model.make_future_dataframe(periods=len(test_prophet))
forecast_prophet = prophet_model.predict(future)

# Evaluate
prophet_pred = forecast_prophet['yhat'][-len(test_prophet):].values
prophet_rmse = np.sqrt(mean_squared_error(test_prophet['y'], prophet_pred))
prophet_mae = mean_absolute_error(test_prophet['y'], prophet_pred)

print(f"\nProphet - Test RMSE: ${prophet_rmse:.2f}")
print(f"Prophet - Test MAE: ${prophet_mae:.2f}")

# ============ VISUALIZATION ============
plt.figure(figsize=(15, 10))

# Plot 1: LSTM predictions
plt.subplot(3, 1, 1)
plt.plot(df.index[train_size+seq_length:], y_test_actual, label='Actual', linewidth=2)
plt.plot(df.index[train_size+seq_length:], test_predict, label='LSTM Prediction', linewidth=2)
plt.title('LSTM Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)

# Plot 2: ARIMA predictions
plt.subplot(3, 1, 2)
plt.plot(test_arima.index, test_arima.values, label='Actual', linewidth=2)
plt.plot(test_arima.index, forecast_arima.values, label='ARIMA Prediction', linewidth=2)
plt.title('ARIMA Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)

# Plot 3: Prophet predictions
plt.subplot(3, 1, 3)
plt.plot(test_prophet['ds'], test_prophet['y'], label='Actual', linewidth=2)
plt.plot(test_prophet['ds'], prophet_pred, label='Prophet Prediction', linewidth=2)
plt.title('Prophet Time Series Forecasting')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('forecasting_results.png', dpi=300)
plt.show()

# ============ FUTURE PREDICTIONS (30 days) ============
# Using LSTM
last_sequence = scaled_data[-seq_length:]
future_predictions = []

for _ in range(30):
    pred = model.predict(last_sequence.reshape(1, seq_length, 1))
    future_predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Plot future predictions
plt.figure(figsize=(12, 6))
plt.plot(df.index[-100:], df['close'][-100:], label='Historical', linewidth=2)
plt.plot(future_dates, future_predictions, label='30-Day Forecast', linewidth=2, linestyle='--')
plt.title('Google Stock Price - 30 Day Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.savefig('future_forecast.png', dpi=300)
plt.show()

print("\n30-Day Forecast:")
for date, price in zip(future_dates, future_predictions):
    print(f"{date.strftime('%Y-%m-%d')}: ${price[0]:.2f}")