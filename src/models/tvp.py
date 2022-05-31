import pandas as pd
import numpy as np
import numpy.linalg as la
import pandas as pd
import pickle
import os
from pathlib import Path

import plotly.express as px

from src.models.preliminaries import Settings
from src.data.data_class import Data
from src import utils


class TVP():
    def __init__(self, params, data):
        # init settings

        # init data information
        self.y_dep = data.y_dep
        self.X = data.X
        self.N = data.N
        self.T = data.T

        self.prior_theta = params.prior_theta
        self.initial_V_0 = params.initial_V_0
        self.h_fore = params.h_fore
        self.lamda = params.lamda
        self.kappa = params.kappa
        self.first_sample_ends = params.first_sample_ends

    def data_to_numpy(self):
        """ Transform panda data to numpy for computational speed"""
        return self.y_dep.to_numpy(), self.X.to_numpy()

    def set_priors(self):
        """set prior means and variances and initial conditions"""
        if not hasattr(self, "X_np"):
            self.y_dep_np, self.X_np = self.data_to_numpy()
        X = self.X_np
        y = self.y_dep_np
        T = self.T

        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(T*self.first_sample_ends)
        else:
            first_sample_ends = np.where(self.X.index == self.first_sample_ends)[0][0]
        # set prior for thetha
        # Initial values on time-varying parameters - theta[0] ~ N(PRM,PRV x I)
            # 1: Diffuse N(0,4)
            # 2: Data-based prior
        if self.prior_theta == 1:
            theta_0_prmean = np.zeros(X.shape[1])  # init zeros
            theta_0_prvar = 4*np.eye(X.shape[1]) # init matrices with 4 on the diagonal
        elif self.prior_theta == 2:
            theta_0_prmean = np.zeros(X.shape[1])  # init zeros
            theta_0_prvar = 2*utils.diag(np.var(y[:first_sample_ends])/np.var(X[:first_sample_ends, :]))

        self.theta_0_prmean = theta_0_prmean
        self.theta_0_prvar = theta_0_prvar
        # prior for V_0
        if self.initial_V_0 == 1:
            self.V_0 = 1e-3 # diffuse prior
        elif self.initial_V_0 == 2:
            self.V_0 = np.var(y[:first_sample_ends])/4  # data based prior - check /4

    def fit(self):
        """run tvp estimation"""
        T = self.T
        N = self.N
        h_fore = self.h_fore
        lamda = self.lamda
        kappa = self.kappa

        self.set_priors()
        X = self.X_np
        y = self.y_dep_np
        V_0 = self.initial_V_0
        theta_prmean = self.theta_0_prmean
        theta_prvar = self.theta_0_prvar

        theta_pred = np.empty((N, T), dtype=float) * np.nan
        theta_update = np.empty((N, T), dtype=float) * np.nan
        P_pred = np.empty((N, N, T), dtype=float) * np.nan
        P_update = np.empty((N, N, T), dtype=float) * np.nan
        y_pred = np.empty(T, dtype=float) * np.nan
        y_pred_h = np.empty(T+h_fore, dtype=float) * np.nan
        e_t = np.empty(T, dtype=float) * np.nan
        H_pred = np.empty(T, dtype=float) * np.nan

        for t in range(h_fore, T):
            if t == h_fore: first_loop = 1
            else: first_loop = 0
            # prediction step
            if first_loop == 1:
                theta_pred[:, t] = theta_prmean
                P_pred[:, :, t] = (1 / lamda) * theta_prvar
            else:
                theta_pred[:, t] = theta_update[:, t - 1]
                P_pred[:, :, t] = (1 / lamda) * P_update[:, :, t - 1]
            # predict y_t and calculate prediction error
            y_pred[t] = X[t-h_fore, :] @ theta_pred[:, t]
            y_pred_h[t+h_fore] = X[t, :] @ theta_pred[:, t]
            e_t[t] = y[t] - y_pred[t]
            # Estimate H_t
            if first_loop == 1:
                H_t = V_0
                # H_t = (1 / (t + 1)) * (e_t[t] ** 2 - X[t-h_fore, :] @ P_pred[:, :, t] @ X[t-h_fore, :].T)
                H_pred[t] = H_t if H_t > 0 else V_0
            else:
                H_t = kappa * H_pred[t - 1] + (1 - kappa) * (e_t[t - 1] ** 2)
                H_pred[t] = H_t if H_t > 0 else H_pred[t - 1]
            # updating step
            F = H_pred[t] + X[t-h_fore, :] @ P_pred[:, :, t] @ X[t-h_fore, :].T
            theta_update[:, t] = theta_pred[:, t] + P_pred[:, :, t] @ X[t-h_fore, :].T * (1 / F) * e_t[t]
            P_update[:, :, t] = P_pred[:, :, t] - P_pred[:, :, t] @ X[[t-h_fore], :].T @ X[[t-h_fore], :] @ P_pred[:, :, t] * (1 / F)

        time_lab = self.y_dep.index[h_fore:T]
        self.y_tvp = pd.Series(y_pred_h.reshape(-1)[h_fore:T], index=time_lab)

        self.theta_update = theta_update

    def forecast_statistics(self, unit='decimals', plot_fe=False, plot_y_fe=False, save_stats=True, print_stats=True):

        y = self.y_dep
        y_tvp = self.y_tvp
        h_fore = self.h_fore
        T = self.T
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(self.T * self.first_sample_ends)
        else:
            first_sample_ends = np.where(self.X.index == self.first_sample_ends)[0][0]
        # DMA Statistics
        fe_TVP = y[h_fore:T] - y_tvp
        MAFE_TVP = (abs(fe_TVP)[first_sample_ends:]).sum()/T
        MSFE_TVP = (np.square(fe_TVP)[first_sample_ends:]).sum()/T
        BIAS_TVP = (fe_TVP[first_sample_ends:]).sum()/T
        self.fe_TVP = fe_TVP

        stats = ['MAFE', 'MSFE', 'BIAS']
        TVP_stats = [MAFE_TVP, MSFE_TVP, BIAS_TVP]
        stats_pd = pd.DataFrame.from_dict(data={'TVP': TVP_stats},
                                orient='index',
                                columns=stats)
        if unit == 'percent':
            stats_pd = stats_pd*100
        if print_stats:
            print(stats_pd)

        if save_stats:
            self.MAFE = MAFE_TVP
            self.MSFE = MSFE_TVP
            self.BIAS = BIAS_TVP

        if plot_fe:
            # fig = px.line(x=fe_DMA[first_sample_ends:].index, y=fe_DMA[first_sample_ends:],
            #               title='Forecast errors')
            # fig.show()
            df_part = fe_TVP[first_sample_ends:]
            fig = px.line(df_part,
                          x=df_part.index,
                          y=df_part,
                          title='Forecast error')
            fig.show()
        if plot_y_fe:
            y_fe = pd.concat((fe_TVP[first_sample_ends:],
                              y[first_sample_ends+h_fore:],
                              y_tvp[first_sample_ends:]), axis=1)
            y_fe.columns = ['fe_TVP', 'y', 'y_tvp']
            fig = px.line(y_fe,
                          x=y_fe.index,
                          y=y_fe.columns,
                          title='Forecast errors')
            fig.show()


if __name__ == '__main__':
    params = Settings()  # initialize Settings
    # adjust settings
    params.use_y = ['CPI']  # use as y
    params.use_x = ['unemp',  # or employment
                    'cons_private',
                    'invest_private_housing',
                    'GDP',
                    'prod_ind',  # or prod_constr
                    'interest_rate_short',
                    'interest_rate_long',
                    'dax',
                    'M1',
                    'infl_exp_current_year',  # or infl_exp_next_year, infl_exp_2_year_ahead
                    'trade_exp',
                    'CPI_house_energy',  # or PCI_energy_ or HICP_energy
                    'supply_index_global',  # or 'supply_index_eu'
                    # 'cons_gov',
                    'business_conf_manufacturing'
                    ]  # indep vars
    params.tcodesX = [1, 1, 5, 1, 1, 1, 1, 5, 5, 1, 1, 5, 5, 1]
    params.tcodey = 1
    params.first_sample_ends = '2012-12-31'
    # params.restricted_vars = ['intercept', 'CPI']
    params.forgetting_method = 2
    params.h_fore = 1
    params.prior_theta = 1
    params.plag = 2

    # params.print_setting_options() # print explanation to settings
    params.print_settings()  # print settings

    data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'processed'))
    # print(data_path)
    # load raw data
    with open(os.path.join(data_path, 'df.pkl'), 'rb') as f:
        df = pickle.load(f)
    data = Data(df, params)

    tvp = TVP(params, data)
    tvp.fit()
    tvp.forecast_statistics()
    print('Done')