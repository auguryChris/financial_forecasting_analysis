# Financial Forecasting and Backtesting with Machine Learning
### An empirical survey of financial time series forecasting techniques using S&P 500 data
By Allie Bergmann, Armand Khachatourian, Chris Westendorf
## Abstract
We conducted an empirical survey of these distinct approaches to forecasting stock prices on a daily scale. Specifically we forecasted using: XGBoost Regression, AR, ARIMA, SARIMA, Linear Regression, Random Forests, and various Deep Neural Network LSTM ensembles. We also explored training these models using 363 features that leverage fundamental and technical information. The motivation for this effort is to be able to articulate in layman’s terms the nuanced trade-offs and impact that decisions about feature engineering, signal processing, model complexity, model interpretability have on numerical accuracy and the resulting forecast's potential to create value.

## <a href="https://augurychris.github.io/financial_forecasting_analysis/">Link to paper</a>

<a href="https://github.com/auguryChris/financial_forecasting_analysis/blob/main/data/SAMPLE_DATA_MSFT.csv">Sample Data for 1 stock including all funamental and technical features</a>

<a href="https://github.com/auguryChris/financial_forecasting_analysis/blob/main/requirements.txt">Module Requirements to run all scripts and notebooks</a>

## Notebooks & Scripts : Referenced in the Report

### 2.Data Acquisition
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/blob/main/scripts/data_retrieval.py">Data Retrieval</a>

### 3.2 Feature Engineering
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/blob/main/scripts/feature_engineering.py">Technical Feature Engineering</a>
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/blob/main/scripts/feature_engineering.py">Signal Processing: Decomposition Features</a>

### 3.3 Feature Importance & Dimensionality Reduction
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/tree/main/notebooks/03_XGBoost/xgboost_final_tuned.ipynb">Shapley Additive Explanation Values for Tree Ensembles</a>
### 3.4 Unsupervised Learning Stock Picks
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/blob/main/notebooks/00_Decomposition_4_Clustering/00_120d_CEEMDAN.ipynb">Clustering Stocks</a>

### 3.5 Forecasting Models
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/tree/main/notebooks/03_XGBoost">XGBoost Models</a>
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/blob/main/notebooks/04_Linear_Models/Linear%20Prediction%20Models.ipynb">Linear Models & Random Forest</a>
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/tree/main/notebooks/02_DNN">LSTM Models</a>

### 4.2 Backtesting Results
* <a href="https://github.com/auguryChris/financial_forecasting_analysis/tree/main/notebooks/Backtesting">Backtests with full Trading Logs</a>
