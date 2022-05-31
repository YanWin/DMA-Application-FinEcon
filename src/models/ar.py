import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

import plotly.express as px
from statsmodels.regression.linear_model import OLS

from src.models.preliminaries import Settings
from src.data.data_class import Data
from src import utils


class AR():
    def __init__(self, params, data):
        self.y_dep = data.y_dep
        self.X = data.X
        self.T = data.T
        self.intercept = params.intercept
        self.first_sample_ends = params.first_sample_ends
        self.h_fore = params.h_fore
        self.lags = params.plag


    def fit_predict(self):
        y = self.y_dep
        X = self.X
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(self.T*self.first_sample_ends)
        else:
            first_sample_ends = np.where(y.index == self.first_sample_ends)[0][0]
        y_train = y[:first_sample_ends]
        y_test = y[first_sample_ends:]
        X_train = X.iloc[:first_sample_ends, :]
        X_test = X.iloc[first_sample_ends:, :]
        h_fore = self.h_fore

        # adjust training data for lags
        X_train_lag = X_train.shift(h_fore).iloc[h_fore:, :]
        X_test_lag = X_test.shift(h_fore).iloc[h_fore:, :]
        y_train = y_train[h_fore:]
        ar_model = OLS(endog=y_train, exog=X_train_lag).fit()
        pred = ar_model.predict(X_test)
        self.y_pred_ar = pred
        # self.y_pred_ar = pd.Series(np.zeros(X.shape[0]), index=y.index)
        # for t in range(X.shape[0]):
        #     ar_model = OLS(endog=y[:t+1], exog=X.iloc[:t+1,:]).fit()
        #     pred = ar_model.predict(X.iloc[:t+1,:])
        #     self.y_pred_ar[t] = pred[t]

    def fit_predict_next(self):
        y = self.y_dep
        X = self.X
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(self.T * self.first_sample_ends)
        else:
            first_sample_ends = np.where(y.index == self.first_sample_ends)[0][0]

        h_fore = self.h_fore
        T = X.shape[0]
        X_lag = X.shift(h_fore)
        # X_lag = X_lag.iloc[h_fore:,:]
        # y_dep = y[h_fore:]
        fe = pd.Series(index=y.index)
        y_pred = pd.Series(index=y.index)
        for t in range(y.shape[0]-h_fore):
            if t >= first_sample_ends:
                ar_model = OLS(endog=y[h_fore+h_fore:t+1].values, exog=X_lag.iloc[h_fore:t-h_fore+1,:].values).fit()
                pred = ar_model.params @ X_lag.iloc[t,:]
                fe[t+h_fore] = y[t+h_fore] - pred
                y_pred[t + h_fore] = pred

        self.y_pred_ar = y_pred



    def forecast_statistics(self, unit='decimals', plot_fe=False, plot_y_fe=False, save_stats=True, print_stats=True):
        y_forecast = self.y_pred_ar
        y = self.y_dep
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(self.T*self.first_sample_ends)
        else:
            first_sample_ends = np.where(y.index == self.first_sample_ends)[0][0]

        h_fore = self.h_fore
        y = self.y_dep[first_sample_ends+h_fore:]
        y_forecast = y_forecast[first_sample_ends+h_fore:]
        T = self.T
        fe = y-y_forecast
        MAFE = abs(fe).sum()/T
        MSFE = (np.square(fe)).sum()/T
        BIAS = (fe).sum()/T
        self.fe = fe

        stats_names = ['MAFE', 'MSFE', 'BIAS']
        stats = [MAFE, MSFE, BIAS]
        stats_pd = pd.DataFrame.from_dict(data={'Forecasts stats': stats},
                                orient='index',
                                columns=stats_names)
        if unit == 'percent':
            stats_pd = stats_pd*100
        if print_stats:
            print(stats_pd)

        if save_stats:
            self.MAFE = MAFE
            self.MSFE = MSFE
            self.BIAS = BIAS

        if plot_fe:
            # fig = px.line(x=fe_DMA[first_sample_ends:].index, y=fe_DMA[first_sample_ends:],
            #               title='Forecast errors')
            # fig.show()
            fig = px.line(fe,
                          x=fe.index,
                          y=fe,
                          title='Forecast error')
            fig.show()
        if plot_y_fe:
            y_fe = pd.concat((fe,
                              y,
                              y_forecast), axis=1)
            y_fe.columns = ['fe', 'y', 'y_DMA']
            fig = px.line(y_fe,
                          x=y_fe.index,
                          y=y_fe.columns,
                          title='Forecast errors')
            fig.show()



if __name__ == '__main__':
    params = Settings()  # initialize Settings
    # adjust settings
    params.use_y = ['CPI']  # use as y
    params.use_x = None
    params.tcodey = 1
    params.first_sample_ends = '2012-12-31'
    params.h_fore = 1
    params.plag = 2

    # params.print_setting_options() # print explanation to settings
    # params.print_settings()  # print settings

    data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'processed'))
    # print(data_path)
    # load raw data
    with open(os.path.join(data_path, 'df.pkl'), 'rb') as f:
        df = pickle.load(f)
    data = Data(df, params)

    ar = AR(params, data)

    ar.fit_predict_next()
    ar.forecast_statistics()


    print('Done')