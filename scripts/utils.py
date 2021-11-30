#%%
import warnings
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

#%%

warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

#ticker_names = ['MSFT', 'HD', 'UNH', 'XOM', 'ADSK', 'WAT']

class DataSplit:
    """
    Standardized class object for dataset loading, splitting, normalizing, etc.
    At single-ticker level. One object per ticker.

    INPUT: Financial data for a ticker.
    OUTPUT: Train/Test
    """

    def __init__(self, ticker='', original_target_name='', num_forecasts='all'):
        self.ticker = ticker
        self.all_data = self.load_data(ticker)
        self.reduced_data = self.load_reduced_data(ticker)
        self.original_target_name=original_target_name
        self.num_forecasts=num_forecasts

        # Init train/test sets
        self.X_train, self.y_train, self.X_test, \
        self.y_test, self.X_hold, self.y_hold = self.train_test_split_data(self.all_data, n_forecasts=self.num_forecasts)

        # Init train/test sets for reduced feature data
        self.X_train_reduced, self.y_train_reduced, self.X_test_reduced, \
        self.y_test_reduced, self.X_hold_reduced, self.y_hold_reduced = self.train_test_split_data(self.reduced_data, n_forecasts=self.num_forecasts)


    # Armand K code
    def normalize_target(self, df=pd.DataFrame(), target_name=None, timeframe=-1, normalization_type='simple_return'):
        """
        Normalizes target based on normalzation normilization_type param
        ** DataFrame is shifted using pd.shift(timeframe) to accurately represent target. Note that a shift of -1 will
            shift the data forward by one cell, allowing for prediction of next timeframe price.
        """
        print(
            f'normalization_type: {normalization_type} \ntarget: {target_name} {timeframe *-1} day shifted')


        # Close over close
        if self.original_target_name == 'Close':
            if normalization_type == 'simple_return':
                return df[target_name].shift(timeframe) / df[target_name]

            elif normalization_type == 'return':
                return  (df[target_name].shift(timeframe) / df[target_name]) - 1

            elif normalization_type == 'log_return':
                return np.log(df[target_name].shift(timeframe) / df[target_name])

        # indicator return from last close value
        else:
            if normalization_type == 'simple_return':
                return df[target_name].shift(timeframe) / df['Close']

            elif normalization_type == 'return':
                return (df[target_name].shift(timeframe) / df['Close']) - 1

            elif normalization_type == 'log_return':
                return np.log(df[target_name].shift(timeframe) / df['Close'])


        print('WARNING: NO NORMALIZATION USED, target must be predifined')
        return df[target_name]

    def unnormalze_target(self, df=pd.DataFrame(), target_name=None, timeframe=-1, normalization_type='simple_return'):
        """
        Un-normalizes target based on normalzation normalization_type param
        """
        if self.scaler_dict != None:
            scaler = self.scaler_dict[self.scale_type]
            df[target_name] = scaler.inverse_transform(df[target_name].values.reshape(-1,1))

        # Close over close
        if self.original_target_name == 'Close':
            if normalization_type == 'simple_return':
                return df[self.original_target_name] * (df[target_name])

            elif normalization_type == 'return':
                return df[self.original_target_name] * (df[target_name]+1)

            elif normalization_type == 'log_return':
                return np.exp(df[target_name]) * df[self.original_target_name]

        # indicator return from last close value
        if normalization_type == 'simple_return':
            return df['close'] * (df[target_name])

        elif normalization_type == 'return':
            return df['close'] * (df[target_name]+1)

        elif normalization_type == 'log_return':
            return np.exp(df[target_name]) * df['close']

        else:
            print('WARNING: NO NORMILIZATION USED')



    def define_target(self, df):
        # Define target as pct change between current close and next day high
        # *** Update 'load_reduced_data()' added columns if changes made ***
#        df['target'] = (df['High'].shift(-1) / df['Close']) - 1
        df['target'] = self.normalize_target(df, target_name='High', normalization_type='log_return')

        return df


    def load_data(self, ticker=''):
        if len(ticker) > 0:
            # Read in data
            ticker = ticker.upper()
#            self.df = pd.read_csv(f'../data/ticker_data/{ticker}_full_data.csv')
            df = pd.read_csv(f'/Users/Allie/financial_forecasting_analysis/data/ticker_data/{ticker}_full_data.csv')
            print(f"Data for {ticker} loaded.")

        else:
            df = pd.DataFrame()
            print("No financial data loaded.")

        return df


    def load_reduced_data(self, ticker=''):
        if len(ticker) > 0:
            # Read in data
            ticker = ticker.upper()
#            self.df = pd.read_csv(f'../data/ticker_data/{ticker}_reduced50.csv')
            df = pd.read_csv(f'/Users/Allie/financial_forecasting_analysis/data/ticker_data/{ticker}_reduced50.csv')

            # add High, Close to reduced data, as they are eliminated
            # This may need to be updated if defined target changes
            df = df.join(self.all_data[['reportperiod','High', 'Close']])
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            print(f"Reduced data for {ticker} loaded.")

        else:
            df = pd.DataFrame()
            print("No financial data loaded.")

        return df


    def train_test_split_data(self, df, test_size=0.5, n_forecasts='all'):

        ## define target/features
        df = self.define_target(df)

        # Define feature columns
        features = [col for col in df.columns if col not in ['target']]

        # Set Index
#        df.reset_index(inplace=True)
#        # Set index to datetime
        df.set_index(df['reportperiod'], inplace=True, drop=False)
#        df.index = pd.DatetimeIndex(df.index).to_period('D')

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            df[features], df['target'], test_size=test_size, shuffle=False)

        # Cut by forecast amount
        if n_forecasts=='all':
            pass
        else:
            n_forecasts = n_forecasts * 2
            X_test = X_test[:n_forecasts]
            y_test = y_test[:n_forecasts]

        # Test-Holdout Split
        X_test, X_hold, y_test, y_hold = train_test_split(
            X_test, y_test, test_size=0.50, shuffle=False)



        return X_train, y_train, X_test, y_test, X_hold, y_hold

#%%

class Strategy:

    def __init__(self, ticker='', original_target_name='', num_forecasts='all', valuation_metric='rmse'):

        self.ticker = ticker
        self.data = DataSplit(ticker, original_target_name, num_forecasts)
        self.original_target_name=self.data.original_target_name
        self.num_forecasts=len(self.data.y_test)
        self.valuation_metric=valuation_metric
        self.arima_forecasts, self.arima_results, self.arima_best_params = self.arima(data_type='all_data')
        self.ar_forecasts, self.ar_results = self.ar(data_type='all_data', params=self.arima_best_params)
        self.linreg_forecasts, self.linreg_results = self.linreg(data_type='reduced_data')
        self.rf_reg_forecasts, self.rf_reg_results = self.randomforestreg(data_type='reduced_data')

    ###########################################################################
    ############################ HELPER FUNCTIONS #############################
    ###########################################################################

    def scale_data(self, df, scale_type=None):
        # SCALING
        if scale_type=='minmax':
            scaler = MinMaxScaler()
            df = scaler.fit_transform(df.copy().values.reshape(-1, 1))
            scaler_dict = {scale_type: scaler}
        elif scale_type=='standard':
            scaler = StandardScaler()
            df = scaler.fit_transform(df.copy().values.reshape(-1, 1))
            scaler_dict = {scale_type: scaler}
        else:
            scaler_dict = None

        return scaler_dict, df


    # Armand K code
    def unnormalize_target(self, df=pd.DataFrame(), target_name=None, timeframe=-1, normalization_type='simple_return', scale_type=None):
        """
        Un-normalizes target based on normalzation normalization_type param
        """

        df = df.copy() ## AB add 11/21
        df = df.reset_index() ## AB add 11/21

        scaler_dict, _ = self.scale_data(df, scale_type=scale_type) ## AB add 11/21

        if scaler_dict != None:
            scaler = scaler_dict[scale_type]
            df[target_name] = scaler.inverse_transform(df[target_name].values.reshape(-1,1))

        # Close over close
        if self.original_target_name == 'Close':
            if normalization_type == 'simple_return':
                return df[self.original_target_name] * (df[target_name])

            elif normalization_type == 'return':
                return df[self.original_target_name] * (df[target_name]+1)

            elif normalization_type == 'log_return':
                return np.exp(df[target_name]) * df[self.original_target_name]

        # indicator return from last close value
        if normalization_type == 'simple_return':
            return df['Close'] * (df[target_name])

        elif normalization_type == 'return':
            return df['Close'] * (df[target_name]+1)

        elif normalization_type == 'log_return':
            return np.exp(df[target_name]) * df['Close']

        else:
            print('WARNING: NO NORMILIZATION USED')


    def data_plots(self, data_type='all_data'):
        """
        Helper funciton
        """
        if data_type == 'all_data':
            data = self.data.all_data['target'].dropna()
        elif data_type == 'reduced_data':
            data = self.data.reduced_data['target'].dropna()

        plt.figure(figsize=[15, 7.5])
        plt.plot(data)
        plt.title("Original Target Data")
        plt.show()

    def acf_pacf_plots(self, data_type='all_data'):

        if data_type == 'all_data':
            data = self.data.all_data['target'].dropna()
        elif data_type == 'reduced_data':
            data = self.data.reduced_data['target'].dropna()

        # Check appropriateness of AR via partial autocorrelation graph
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        plot_pacf(data, ax, lags=40)
        ax.set_xlabel(r"Lag")
        ax.set_ylabel(r"Correlation")
        del fig, ax #, plot_pacf

        # Check appropriateness of MA via autocorrelation graph
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        plot_acf(data, ax, lags=40)
        ax.set_xlabel(r"Lag")
        ax.set_ylabel(r"Correlation")
        del fig, ax


    def dickey_fuller_test(self, ser): #data_type='all_data'):

        ad_fuller_result = adfuller(ser, autolag='AIC')

#        output = pd.Series(ad_fuller_result[0:4], index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
#        for key,values in ad_fuller_result[4].items():
#            output['critical value (%s)'%key] = values
#        print('\n')
#        print(output)

        return ad_fuller_result


    ## Armand K code
    def plot_res(self, predictions, actual, y_train):
        """
        Used for plotting residuals during training and testing
        """

        y_hat = predictions
        y_test = actual
        residuals = y_test - y_hat
        rolling_window=7

        y_hat_rolling = pd.DataFrame(y_hat).rolling(rolling_window).mean()
        y_test_rolling = pd.DataFrame(y_test).rolling(rolling_window).mean()
        n = range(len(y_hat_rolling))
        plt.figure(figsize=(15, 6))
        plt.plot(n, y_test_rolling,
                 color='red', label='y_test', alpha=0.5)
        plt.plot(y_hat_rolling,
                 color='black', label='y_pred', alpha=0.8)
        plt.plot(np.mean(y_train)*np.ones((len(n),)),
                 color='blue', label='Training Mean', alpha=0.5)
        plt.legend()
        plt.show()


    def seas_decompose(self, data, model='additive'):

        result = seasonal_decompose(data, model=model)

        print("Performing Seasonal Decomposition")
        fig = plt.figure()
        fig = result.plot()
        fig.set_size_inches(16,9)

        return result


    def calculate_results(self, forecast, y_test, forecast_target='', y_test_target='', model='', scale_type=None):

        results = {}

        forecast = self.unnormalize_target(forecast, target_name=forecast_target, normalization_type='log_return')
        y_test = self.unnormalize_target(y_test, target_name=y_test_target, normalization_type='log_return')

        mse = mean_squared_error(y_test, forecast)
#        print(model + ' MSE: ' + str(mse))

        mae = mean_absolute_error(y_test, forecast)
#        print(model + ' MAE: ' + str(mae))

        rmse = math.sqrt(mean_squared_error(y_test, forecast))
#        print(model + ' RMSE: ' + str(rmse))

        mape = np.mean(np.abs(forecast - y_test)/np.abs(y_test))
#        print(model + ' MAPE: ' + str(mape))

        results['mse'] = mse
        results['mae'] = mae
        results['rmse'] = rmse
        results['mape'] = mape

        return results


#    def eliminate_trend():
#        pass
#
#
#
#
    ###########################################################################
    ############################### ALGORITHMS ################################
    ###########################################################################

    def ar(self, data_type='all_data', params=(None, None, None), q=0, p_params=25):
        if data_type == 'all_data':
            y_train = self.data.y_train
            y_test = self.data.y_test

        elif data_type == 'reduced_data':
            y_train = self.data.y_train_reduced
            y_test = self.data.y_test_reduced


        num_forecasts = self.num_forecasts

        # Define default p,d,q, if any.
        p = params[0]
        if params[1] == None:
            d=1
        else:
            d=params[1]
        q = params[2]

        def arma_first_diff(ser, p, d, q, num_forecasts=num_forecasts):

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)

#            d = 1 # 1st order log difference (log return equivalent)
            arima_model = ARIMA(ser, order=(p,d,q))
            arima_fitted = arima_model.fit()
            forecasts = arima_fitted.forecast(steps=num_forecasts)

            return forecasts

        test_scores = []
        idx = list(y_test.index)

        y_test = y_test.to_frame()
        y_test = y_test.join(self.data.X_test[['Close']])
        idx = list(y_test.index)

        for i in range(2, p_params):

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)

            forecasts = arma_first_diff(y_train, p=i, d=d, q=q)
            forecasts.index = idx
            forecasts = forecasts.to_frame()
            forecasts['reportperiod'] = idx
            forecasts.set_index('reportperiod', inplace=True)
            forecasts = forecasts.join(self.data.X_test[['Close']])

            res = self.calculate_results(forecasts, y_test, 'predicted_mean', 'target', model='AR')

            test_scores.append(self.valuation_metric)  ### May change this based on group decision for scoring evaluation

        best_index = np.argmin(test_scores)
        best_p = best_index + 1

        forecasts = arma_first_diff(y_train, p=best_p,d=d,q=q)
        forecasts.index = idx
        forecasts = forecasts.to_frame()
        forecasts['reportperiod'] = idx
        forecasts.set_index('reportperiod', inplace=True)
        forecasts = forecasts.join(self.data.X_test[['Close']])

        res = self.calculate_results(forecasts, y_test, 'predicted_mean', 'target', model='AR')

        forecasts = self.unnormalize_target(forecasts, target_name='predicted_mean', normalization_type='log_return')
        forecasts.index = idx

#        self.plot_res(forecasts['predicted_mean'], y_test['target'], y_train['target'])

        return forecasts, res


    def arima(self, data_type='all_data'):

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)

        if data_type == 'all_data':
            y_train = self.data.y_train
            y_train.index = pd.DatetimeIndex(y_train.index).to_period('D')

            y_test = self.data.y_test
            y_test.index = pd.DatetimeIndex(y_test.index).to_period('D')

            X_test = self.data.X_test
            X_test.index = pd.DatetimeIndex(X_test.index).to_period('D')

        elif data_type == 'reduced_data':
            y_train = self.data.y_train_reduced
            y_train.index = pd.DatetimeIndex(y_train.index).to_period('D')

            y_test = self.data.y_test_reduced
            y_test.index = pd.DatetimeIndex(y_test.index).to_period('D')

            X_test = self.data.X_test
            X_test.index = pd.DatetimeIndex(X_test.index).to_period('D')

        num_forecasts = self.num_forecasts

        # Re-aggregate data
#        data = y_train.append(y_test)

        # Perform seasonal decomposition
#        data = self.seas_decompose(data)

        # Separate out data again
#        y_train = data[num_forecasts:]
#        y_test = data[:num_forecasts]

        # Run ARIMA optimization
        model_autoARIMA = auto_arima(y_train, start_p=0, start_q=0,
                                     test='adf', # adf to find optimal 'd'
                                     max_p=20, max_q=60,
                                     m=1,
                                     d=1, #d=None, # let model determine 'd' #d = 1 is first order log difference (log return equivalent)
                                     seasonal=True,
                                     start_P=0,
                                     D=1,
                                     trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
#                                     stewise=True # search stopped when thresholds hit: balance best fit and time constraints
                                     )

        # Print ARIMA optimization model
#        print(model_autoARIMA.summary())
#        model_autoARIMA.plot_diagnostics(figsize=(15,8))
#        plt.show()

        # Get optimal ARIMA model params
        best_order = model_autoARIMA.get_params().get('order')

        # Re-run model with optimal params
        model = ARIMA(y_train, order=best_order)
        model_fit = model.fit()
#        print(model_fit.summary())

        # Forecast
        forecasts = model_fit.forecast(steps=num_forecasts)

        # Add back high/close data for unnormalizing data
        y_test = y_test.to_frame()
        y_test = y_test.join(self.data.X_test[['Close']])
        idx = list(y_test.index)

        forecasts = forecasts.to_frame()
        forecasts['reportperiod'] = idx
        forecasts.set_index('reportperiod', inplace=True)
        forecasts = forecasts.join(self.data.X_test[['Close']])

        # Calculate results
        res = self.calculate_results(forecasts, y_test, 'predicted_mean', 'target', model='ARIMA')

        forecasts = forecasts.reset_index()
        forecasts['reportperiod'] = forecasts['reportperiod'].dt.to_timestamp('d').dt.date
        forecasts.set_index('reportperiod', inplace=True, drop=True)

        y_test = y_test.reset_index()
        y_test['reportperiod'] = y_test['reportperiod'].dt.to_timestamp('d').dt.date
        y_test.set_index('reportperiod', inplace=True, drop=True)

        y_train = y_train.reset_index()
        y_train['reportperiod'] = y_train['reportperiod'].dt.to_timestamp('d').dt.date
        y_train.set_index('reportperiod', inplace=True, drop=True)

#        self.plot_res(forecasts['predicted_mean'], y_test['target'], y_train['target'])
        forecasts = self.unnormalize_target(forecasts, target_name='predicted_mean', normalization_type='log_return')
        forecasts.index = idx

        return forecasts, res, best_order



    def linreg(self, data_type='reduced_data'):
        if data_type == 'all_data':
            y_train = self.data.y_train
            y_test = self.data.y_test
            X_train = self.data.X_train
            X_test = self.data.X_test


        elif data_type == 'reduced_data':
            y_train = self.data.y_train_reduced
            y_test = self.data.y_test_reduced
            X_train = self.data.X_train_reduced
            X_test = self.data.X_test_reduced

        # Do some data cleaning to remove non-numeric columns
        X_train.drop(['reportperiod'], axis=1, inplace=True)
        X_test.drop(['reportperiod'], axis=1, inplace=True)

        reg = LinearRegression()
        reg = reg.fit(X_train, y_train)

        idx = list(y_test.index)

        forecasts = reg.predict(X_test)
        forecasts  = pd.DataFrame(forecasts)
        forecasts.index = idx
        forecasts.rename(columns={0 : 'predicted_mean'}, inplace=True)
        forecasts = forecasts.join(X_test[['Close']])

        y_test = pd.DataFrame(y_test)
        y_test.rename(columns={0 : 'target'}, inplace=True)
        y_test = y_test.join(X_test[['Close']])

        res = self.calculate_results(forecasts, y_test, 'predicted_mean', 'target', model='Linear Regression')

#        self.plot_res(forecasts['predicted_mean'], y_test['target'], y_train['target'])
        forecasts = self.unnormalize_target(forecasts, target_name='predicted_mean', normalization_type='log_return')
        forecasts.index = idx

        return forecasts, res

    def randomforestreg(self, data_type='reduced_data'):
        if data_type == 'all_data':
            y_train = self.data.y_train
            y_test = self.data.y_test
            X_train = self.data.X_train
            X_test = self.data.X_test


        elif data_type == 'reduced_data':
            y_train = self.data.y_train_reduced
            y_test = self.data.y_test_reduced
            X_train = self.data.X_train_reduced
            X_test = self.data.X_test_reduced

#        X_train.drop(['reportperiod'], axis=1, inplace=True)
#        X_test.drop(['reportperiod'], axis=1, inplace=True)

        grid = {'n_estimators' : [200], 'max_depth' : [3], 'max_features' : [4,8], 'random_state' : [0]}
        test_scores = []

        rf_model = RandomForestRegressor()

        for g in ParameterGrid(grid):

            rf_model.set_params(**g)
            rf_model.fit(X_train, y_train)

            test_scores.append(rf_model.score(X_test, y_test))  ### May change this based on group decision for scoring evaluation

        best_index = np.argmax(test_scores) # If .score() changed to a metric, check this for change to argmin
        best_params = ParameterGrid(grid)[best_index]


        rf_model.set_params(**best_params)
        rf_model.fit(X_train, y_train)

        idx = list(y_test.index)

        forecasts = rf_model.predict(X_test)
        forecasts  = pd.DataFrame(forecasts)
        forecasts.index = idx
        forecasts.rename(columns={0 : 'predicted_mean'}, inplace=True)
        forecasts = forecasts.join(X_test[['Close']])

        y_test = pd.DataFrame(y_test)
        y_test.rename(columns={0 : 'target'}, inplace=True)
        y_test = y_test.join(X_test[['Close']])

        res = self.calculate_results(forecasts, y_test, 'predicted_mean', 'target', model='Random Forest Regressor')

        forecasts = self.unnormalize_target(forecasts, target_name='predicted_mean', normalization_type='log_return')
        forecasts.index = idx
#        self.plot_res(forecasts['predicted_mean'], y_test['target'], y_train['target'])

        return forecasts, res


##x = DataSplit('MSFT', original_target_name='High',num_forecasts='all')
x = Strategy('MSFT',original_target_name='High',num_forecasts='all')
##t = x.ticker
##x.dickey_fuller_test()
##x.acf_pacf_plots()
forecasts = x.rf_reg_forecasts
res2 = x.rf_reg_results


