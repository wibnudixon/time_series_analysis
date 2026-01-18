# Google Stock Price Time Series Analysis

A comprehensive time series forecasting project that predicts Google stock prices using three different approaches: LSTM (Deep Learning), ARIMA (Statistical), and Prophet (Facebook's forecasting tool).

## 📊 Project Overview

This project demonstrates multiple time series forecasting techniques to predict stock prices:

- **LSTM (Long Short-Term Memory)**: Deep learning approach for capturing complex patterns
- **ARIMA (AutoRegressive Integrated Moving Average)**: Classical statistical method
- **Prophet**: Facebook's robust forecasting tool designed for business time series

## 🗂️ Project Structure

```
google_stockdata_time_series_analysis/
│
├── data/
│   └── google.csv                    # Historical Google stock data
│
├── time_series_analysis.ipynb        # Main Jupyter notebook with analysis
├── main.py                            # Python script version (legacy)
├── Pipfile                            # Python dependencies
├── Pipfile.lock                       # Locked dependencies
└── README.md                          # Project documentation
```

## 🚀 Features

- **Three ML/Statistical Models**: Compare LSTM, ARIMA, and Prophet models
- **Performance Metrics**: RMSE and MAE evaluation for each model
- **Visualizations**: Comprehensive plots comparing predictions vs actual values
- **30-Day Forecast**: Future stock price predictions
- **Well-Documented Notebook**: Step-by-step explanations with markdown cells

## 📋 Requirements

- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- statsmodels
- prophet

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/wibnudixon/google_stockdata_time_series_analysis.git
cd google_stockdata_time_series_analysis
```

### 2. Set Up Virtual Environment with pipenv

Install pipenv if you haven't already:
```bash
pip install pipenv
```

Install dependencies from Pipfile:
```bash
pipenv install
```

Activate the virtual environment:
```bash
py -m pipenv shell
```

### 3. Alternative: Using pip

If you prefer pip, install the required packages:
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow statsmodels prophet
```

## 💻 Usage

### Running the Jupyter Notebook (Recommended)

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `time_series_analysis.ipynb` and run cells sequentially

### Running the Python Script

```bash
python main.py
```

## 📈 Model Performance

The notebook compares three models using:
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error

Results are displayed in a comparison table to identify the best-performing model.

## 📊 Output

The analysis generates:
- `forecasting_results.png`: Comparison of all three models' predictions
- `future_forecast.png`: 30-day future stock price forecast
- Console output with detailed metrics and predictions

## 🧪 Data

The project uses historical Google stock data with the following columns:
- `date`: Trading date
- `close`: Closing price
- Other OHLC (Open, High, Low, Close) data

## 📝 Notebook Structure

1. **Introduction**: Project overview and objectives
2. **Data Loading**: Import and prepare the dataset
3. **LSTM Model**: Deep learning approach with neural networks
4. **ARIMA Model**: Statistical time series analysis
5. **Prophet Model**: Facebook's forecasting algorithm
6. **Model Comparison**: Performance metrics and evaluation
7. **Visualization**: Graphical comparison of predictions
8. **Future Forecast**: 30-day prediction with LSTM
9. **Conclusion**: Summary and insights

## 🛠️ Troubleshooting

### Python Package Issues

If you encounter issues with specific packages:

```bash
# Using pipenv
py -m pipenv install ipykernel
py -m pipenv install pandas
py -m pipenv install tensorflow

# Using pip
pip install --upgrade tensorflow
pip install --upgrade prophet
```

### Virtual Environment Issues

If the virtual environment isn't activating:
```bash
# Find virtual environment location
pipenv --venv

# Manually activate (Windows)
<path-from-above>\Scripts\activate.bat

# Or recreate environment
pipenv --rm
pipenv install
```

### TensorFlow/Keras Issues

If you encounter TensorFlow warnings or errors:
- Ensure you have Python 3.8-3.11 (TensorFlow compatibility)
- Update to the latest TensorFlow: `pip install --upgrade tensorflow`

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Stock price predictions should not be used for actual trading decisions. Past performance does not guarantee future results.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Wibnu Dixon**
- GitHub: [@wibnudixon](https://github.com/wibnudixon)

## 🙏 Acknowledgments

- TensorFlow/Keras for LSTM implementation
- Statsmodels for ARIMA
- Facebook's Prophet team for the Prophet library
- scikit-learn for preprocessing and metrics
