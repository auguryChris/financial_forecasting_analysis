import warnings
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

class TickerXGBRegressor:
    """
    Input: Pandas OHLCV DataFrame with any Technical Indicators
    Fits a model to standard format Pandas OHLCV data + indicators and saves the model with best metrics.  
    """

    def __init__(self, data):
        self.df = data
        self.original_dataset = data

    # HELPER for predict()
    def create_prediction_df(self, df, target, timeframe, not_feature_list, normilization_type):
        """
        Create normalized target for OHLCV data, get tech indicators, prepare for prediction
        """
        self.original_dataset = self.original_dataset.shift(timeframe)
        self.original_target_name = target
        target_name = target + \
            str(timeframe) + '_' + \
            normilization_type  # name of target (adjusted target based on timeframe)
        not_feature_list.append(target_name)
        features = [col for col in df.columns if col not in not_feature_list]
        # encode normalized target within the df
        df[target_name] = self.normilize_target(
            df.copy(), target, timeframe, normilization_type)
        self.target_name = target_name
        df.dropna(inplace=True)  # drop nans
        # get rid of n rows since we shifted targets down
        df = df.iloc[:timeframe]  

        return df, features, target_name
        

            
    # HELPER for predict()
    def normilize_target(self, df=pd.DataFrame(), target_name=None, timeframe=-1, normilization_type='simple_return'):
        """
        Normilzes target based on normilzation normilization_type param
        ** DataFrame is shifted using pd.shift(timeframe) to accurately represent target. Note that a shift of -1 will
            shift the data forward by one cell, allowing for prediction of next timeframe price.
        """
        print(
            f'normilization_type: {normilization_type} \ntarget: {target_name} {timeframe *-1} day shifted')
 

        # Close over close
        if self.original_target_name == 'close':
            if normilization_type == 'simple_return':
                return df[target_name].shift(timeframe) / df[target_name]

            elif normilization_type == 'return':
                return  (df[target_name].shift(timeframe) / df[target_name]) - 1

            elif normilization_type == 'log_return':
                return np.log(df[target_name].shift(timeframe) / df[target_name])
        
        # indicator return from last close value
        else:    
            if normilization_type == 'simple_return':
                return df[target_name].shift(timeframe) / df['close']

            elif normilization_type == 'return':
                return (df[target_name].shift(timeframe) / df['close']) - 1

            elif normilization_type == 'log_return':
                return np.log(df[target_name].shift(timeframe) / df['close'])


        print('WARNING: NO NORMILIZATION USED, target must be predifined')
        return df[target_name]

    def unnormilze_target(self, df=pd.DataFrame(), target_name=None, timeframe=-1, normilization_type='simple_return'):
        """
        Un-normilizes target based on normilzation normilization_type param
        """
        if self.scaler_dict != None:
            scaler = self.scaler_dict[self.scale_type]
            df[target_name] = scaler.inverse_transform(df[target_name].values.reshape(-1,1))
        
        # Close over close
        if self.original_target_name == 'close':
            if normilization_type == 'simple_return':
                return df[self.original_target_name] * (df[target_name])

            elif normilization_type == 'return':
                return df[self.original_target_name] * (df[target_name]+1)

            elif normilization_type == 'log_return':
                return np.exp(df[target_name]) * df[self.original_target_name]
        
        # indicator return from last close value    
        if normilization_type == 'simple_return':
            return df['close'] * (df[target_name])

        elif normilization_type == 'return':
            return df['close'] * (df[target_name]+1)
        
        elif normilization_type == 'log_return':
            return np.exp(df[target_name]) * df['close']
        
        else:
            print('WARNING: NO NORMILIZATION USED')
    
    
    def plot_res(self, predictions, actual, y_train):
        """
        Used for plotting residuals during training and testing
        """

        y_hat = predictions
        y_test = actual
        #residuals = y_test - y_hat
        rolling_window=7
        
        y_hat_rolling = pd.DataFrame(y_hat).rolling(rolling_window).mean()
        y_test_rolling = pd.DataFrame(y_test).rolling(rolling_window).mean()
        n = range(len(y_hat_rolling))
        plt.figure(figsize=(15, 6))
        plt.plot(n, y_test_rolling.iloc[:,0], 
                 color='red', label='y_test', alpha=0.5)
        plt.plot(y_hat_rolling.iloc[:,0],
                 color='black', label='y_pred', alpha=0.8)
        plt.plot(np.mean(y_train)*np.ones((len(n),)),
                 color='blue', label='Training Mean', alpha=0.5)
        plt.legend()
        plt.show()

    def predict_with_xgb(self, target='close', timeframe=-1, not_feature_list=['timestamp', 'ticker'],
                         normilization_type='return', gsearch_params={'max_depth': [3], 'learning_rate': [0.01], 'colsample_bytree': [1],
                                                                               'n_estimators': [500], 'objective': ['reg:squarederror']},
                         get_importances=True, test_size=0.2, scale_type=None):
        """
        Input a OHLCV df with a target and return gridsearched, timesplitcv xgb regressor scores to find best params,
        optionally calls get_shap_importances()
        """
        self.scale_type = scale_type
        
        # Creating prediction df and splitting the data
        df = self.df
        self.timeframe = timeframe
        pred_df, features, target_name = self.create_prediction_df(df, target, timeframe,
                                                                   not_feature_list, normilization_type)      
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            pred_df[features], pred_df[target_name], test_size=test_size, shuffle=False)
        # Test-Holdout Split (50-50)
        X_test, X_hold, y_test, y_hold = train_test_split(
            X_test, y_test, test_size=0.50, shuffle=False)
        
        # SCALING
        if scale_type=='minmax':
            scaler = MinMaxScaler()
            self.scaler_dict = {scale_type: scaler}
        elif scale_type=='standard':
            scaler = StandardScaler()
            self.scaler_dict = {scale_type: scaler}
        else:
            self.scaler_dict = None
        
        if self.scaler_dict != None:
            y_train = pd.Series(scaler.fit_transform(y_train.copy().values.reshape(-1, 1)).reshape(-1,), index=y_train.index)
            y_test = pd.Series(scaler.transform(y_test.copy().values.reshape(-1, 1)).reshape(-1,), index=y_test.index)
            y_hold = pd.Series(scaler.transform(y_hold.copy().values.reshape(-1, 1)).reshape(-1,), index=y_hold.index)
            

        
        # Grid Search, Scores, Plotting Actual vs Predictions
        param_grid = ParameterGrid(gsearch_params)
        best_score, best_score_train = (-10000, -10000)
        best_params, best_params_train = (0, 0)       
        print('START: Model is selected and saved to object based on best test score.', '\n')
        for params in param_grid:  # iterate params until best score is found
            print(f"Testing {params}...")
            # init xgboost regressor on each param set
            xgb_regressor = xgb.XGBRegressor(**params)
            trained_model = xgb_regressor.fit(X_train, y_train)
            test_score = trained_model.score(X_test, y_test)
            train_score = trained_model.score(X_train, y_train)
            print(f"Test Score: {test_score}")
            print(f"Train Score: {train_score}\n")
            if test_score > best_score:
                best_score = test_score
                best_params = params
                best_train = train_score
                best_model = trained_model
        print(f"Best TEST R^2 is {best_score} with params: {best_params}")
        print(f"TRAIN R^2 for best test params is {best_train}")
        xgb_best = best_model
        self.model = xgb_best
        prediction = xgb_best.predict(X_test)

        # Defining some useful instance attributes which can be refrenced from TickerXGBRegressor objects
        self.residuals = self.plot_res(prediction, y_test, y_train)
        self.model = xgb_best
        self.params = best_params
        prediction_df = pd.DataFrame({
            self.original_target_name: X_test[self.original_target_name],
            'close': X_test['close'],
            target_name: prediction})
        true_y_df = pd.DataFrame({
            self.original_target_name: X_test[self.original_target_name],
            'close': X_test['close'],
            target_name: y_test})
        self.prediction = self.unnormilze_target(prediction_df, target_name=target_name, timeframe=timeframe,
                                                 normilization_type=normilization_type)
        self.true_y = self.unnormilze_target(true_y_df, target_name=target_name, timeframe=timeframe,
                                             normilization_type=normilization_type)
        self.test_data = self.df.iloc[X_test.index[0]:]
        
        # only calculate extra metrics if test score was positive to speed up process
        if best_score > 0:
            # Final Holdout test results
            self.holdout_metrics = {}
            y_hold_pred = xgb_best.predict(X_hold)
            self.holdout_metrics['r2_normalized'] = r2_score(y_hold, y_hold_pred)
            # Unnormalize before calculating other metrics
            holdout_prediction_df = pd.DataFrame({
                self.original_target_name: X_hold[self.original_target_name],
                'close': X_hold['close'],
                target_name: y_hold_pred})
            holdout_true_y_df = pd.DataFrame({
                self.original_target_name: X_hold[self.original_target_name],
                'close': X_hold['close'],
                target_name: y_hold})
            self.holdout_prediction = self.unnormilze_target(holdout_prediction_df, target_name=target_name, timeframe=timeframe,
                                                     normilization_type=normilization_type)
            self.holdout_true_y = self.unnormilze_target(holdout_true_y_df, target_name=target_name, timeframe=timeframe,
                                                 normilization_type=normilization_type)
            self.holdout_metrics['mae'] = mean_absolute_error(self.holdout_true_y[1:], self.holdout_prediction[:-1])
            self.holdout_metrics['mape'] = mean_absolute_percentage_error(self.holdout_true_y[1:], self.holdout_prediction[:-1]) * 100
            self.holdout_metrics['rmse'] = mean_squared_error(self.holdout_true_y[1:], self.holdout_prediction[:-1], squared=False)
            print(f'Final Holdout Error metrics saved: R^2: {round(self.holdout_metrics["r2_normalized"],4)}. MAPE Unnormalized:{round(self.holdout_metrics["mape"],4)}%')

            # Adding predictions to original dataframe
            self.prediction.name = 'validation_preds'
            self.holdout_prediction.name = 'holdout_preds'
            self.original_dataset = self.original_dataset.join(self.prediction)
            self.original_dataset = self.original_dataset.join(self.holdout_prediction)


            # Tree SHAP feature importance generation
            def get_shap_importances(xgb_regressor):
                """Generates feature importances based on tree shap"""
                # intialize xgb regressor with best params
                # initialize treeshap explainer with fitted model
                explainer = shap.TreeExplainer(xgb_best)
                # predict test data with training model explainer
                shap_values = explainer.shap_values(X_test)
                self.shap_importance_plot = shap.summary_plot(
                    shap_values, X_test, plot_type="bar", max_display=300)  # create summary feature importance chart
                feature_importance_df = pd.DataFrame(shap_values, columns=features)
                return feature_importance_df

            if get_importances == True:
                print('GENERATING FEATURE IMPORTANCES...')
                feature_importance_df = get_shap_importances(xgb_best)
                self.feature_importance_dict = feature_importance_df.abs(
                ).sum().sort_values(ascending=False).to_dict()

class TickerXGBClassifier:
    """
    Input: Pandas OHLCV DataFrame with any Technical Indicators
    Fits a model to standard format Pandas OHLCV data + indicators and saves the model with best metrics.  
    """

    def __init__(self, data):
        self.df = data

    # HELPER for predict()
    def create_prediction_df(self, df, target, timeframe, not_feature_list, normilization_type):
        """
        Create normalized target for OHLCV data, get tech indicators, prepare for prediction
        """
        self.original_target_name = target
        target_name = target + \
            str(timeframe) + '_' + \
            normilization_type  # name of target (adjusted target based on timeframe)
        not_feature_list.append(target_name)
        features = [col for col in df.columns if col not in not_feature_list]
        # encode normalized target within the df
        df[target_name] = self.normilize_target(
            df.copy(), target, timeframe, normilization_type)
        self.target_name = target_name
        df.dropna(inplace=True)  # drop nans
        # get rid of n rows since we shifted targets down
        df = df.iloc[:timeframe]  

        return df, features, target_name
        

            
    # HELPER for predict()
    def normilize_target(self, df=pd.DataFrame(), target_name=None, timeframe=-1, normilization_type='simple_return'):
        """
        Normilzes target based on normilzation normilization_type param
        ** DataFrame is shifted using pd.shift(timeframe) to accurately represent target. Note that a shift of -1 will
            shift the data forward by one cell, allowing for prediction of next timeframe price.
        """
        print(
            f'normilization_type: {normilization_type} \ntarget: {target_name} {timeframe *-1} day shifted')
 
        # Close over close
        if self.original_target_name == 'close':

            if normilization_type == 'return':
                return  np.where(((df[target_name].shift(timeframe) / df[target_name]) - 1) > 0, 1, 0)

        # indicator return from last close value
        else:    

            if normilization_type == 'return':
                return  np.where(((df[target_name].shift(timeframe) / df['close']) - 1) > 0, 1, 0)


        print('WARNING: NO NORMILIZATION USED, target must be predifined')
        return df[target_name]


    def predict_with_xgb(self, target='close', timeframe=-1, not_feature_list=['timestamp'],
                         normilization_type='return', gsearch_params={'max_depth': [3], 'learning_rate': [0.01], 'colsample_bytree': [1],
                                                                               'n_estimators': [500], 'objective': ['reg:squarederror']},
                         get_importances=True, test_size=0.2, scale_type=None):
        """
        Input a OHLCV df with a target and return gridsearched, timesplitcv xgb Classifier scores to find best params,
        optionally calls get_shap_importances()
        """
        self.scale_type = scale_type
        
        # Creating prediction df and splitting the data
        df = self.df
        self.timeframe = timeframe
        pred_df, features, target_name = self.create_prediction_df(df, target, timeframe,
                                                                   not_feature_list, normilization_type)
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            pred_df[features], pred_df[target_name], test_size=test_size, shuffle=False)
        self.test_data = self.df.iloc[X_test.index[0]:]
        # Test-Holdout Split
        X_test, X_hold, y_test, y_hold = train_test_split(
            X_test, y_test, test_size=0.50, shuffle=False)
        
        # Grid Search, Scores, Plotting Actual vs Predictions
        param_grid = ParameterGrid(gsearch_params)
        best_score, best_score_train = (-10000, -10000)
        best_params, best_params_train = (0, 0)       
        print('SCORE=ACCURACY. START: Model is selected and saved to object based on best test score.', '\n')
        for params in param_grid:  # iterate params until best score is found
            print(f"Testing {params}...")
            # init xgboost Classifier on each param set
            xgb_Classifier = xgb.XGBClassifier(**params)
            trained_model = xgb_Classifier.fit(X_train, y_train)
            test_score = trained_model.score(X_test, y_test)
            train_score = trained_model.score(X_train, y_train)
            print(f"Test Score: {test_score}")
            print(f"Train Score: {train_score}\n")
            if test_score > best_score:
                best_score = test_score
                best_params = params
                best_train = train_score
                best_model = trained_model
        print(f"Best TEST Accuracy is {best_score} with params: {best_params}")
        print(f"TRAIN Accuracy for best test params is {best_train}")
        xgb_best = best_model
        self.model = xgb_best
        prediction = xgb_best.predict(X_test)
        train_pred = xgb_best.predict(X_train)
        train_metrics = classification_report(train_pred, y_train)
        self.train_metrics = train_metrics
        print('TRAIN REPORT:')
        print(train_metrics)

        print('TEST REPORT:')
        test_metrics = classification_report(prediction, y_test)
        self.test_metrics = test_metrics
        print(test_metrics)
        self.prediction = prediction

        # Defining some useful instance attributes which can be refrenced from TickerXGBClassifier objects
        self.model = xgb_best
        self.params = best_params
        prediction_df = pd.DataFrame({
            self.original_target_name: X_test[self.original_target_name],
            'close': X_test['close'],
            target_name: prediction})
        true_y_df = pd.DataFrame({
            self.original_target_name: X_test[self.original_target_name],
            'close': X_test['close'],
            target_name: y_test})
        
        
        y_hold_pred = xgb_best.predict(X_hold)
        holdout_metrics =  classification_report(y_hold, y_hold_pred)
        self.holdout_metrics = holdout_metrics
        print('HOLDOUT REPORT:')
        print(holdout_metrics)
       

        # Tree SHAP feature importance generation
        def get_shap_importances(xgb_Classifier):
            """Generates feature importances based on tree shap"""
            # intialize xgb Classifier with best params
            # initialize treeshap explainer with fitted model
            explainer = shap.TreeExplainer(xgb_best)
            # predict test data with training model explainer
            shap_values = explainer.shap_values(X_test)
            self.shap_importance_plot = shap.summary_plot(
                shap_values, X_test, plot_type="bar", max_display=300)  # create summary feature importance chart
            feature_importance_df = pd.DataFrame(shap_values, columns=features)
            return feature_importance_df

        if get_importances == True:
            print('GENERATING FEATURE IMPORTANCES...')
            feature_importance_df = get_shap_importances(xgb_best)
            self.feature_importance_dict = feature_importance_df.abs(
            ).sum().sort_values(ascending=False).to_dict()