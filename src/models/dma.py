import numpy as np
import numpy.linalg as la
import pandas as pd
import itertools
from tqdm import tqdm
import scipy.stats as stats
from scipy.special import gamma

import plotly
import plotly.express as px
import plotly.graph_objects as go

from src.models.preliminaries import Settings
from src.data.data_class import Data
from src import utils

class DMA():
    def __init__(self, params, data):
        # init settings
        self.lamda = params.lamda
        self.alpha = params.alpha
        self.kappa = params.kappa
        self.forgetting_method = params.forgetting_method
        self.prior_theta = params.prior_theta
        self.initial_V_0 = params.initial_V_0
        self.restricted_vars = params.restricted_vars
        self.initial_DMA_weights = params.initial_DMA_weights
        self.expert_opinion = params.expert_opinion
        self.h_fore = params.h_fore
        self.first_sample_ends = params.first_sample_ends
        self.weighting = params.weighting
        self.degrees = params.degrees

        # init data information
        self.y_dep = data.y_dep
        self.X = data.X
        self.N = data.N
        self.T = data.T

        # init new attributes
        self.K = None
        self.models = None

    def update_restricted(self):
        """Update restricted variables to include the according lags and move them to the start of X"""
        X = self.X

        # make sure restricted vars are at the beginning of the data
        all_restricted_vars = []
        for var in self.restricted_vars:
            all_restricted_vars.extend(X.columns[X.columns.str.startswith(var)].to_list())
        # if len(all_restricted_vars) == len(X.columns):
        #     raise Exception("No unrestricted variables. Use a single TVP model!")
        # Z = matrix with unrestricted vars
        # Z = X.loc[:, ~X.columns.isin(all_restricted_vars)]
        # rearrange X
        X = pd.concat([X.loc[:, all_restricted_vars], X.loc[:, ~X.columns.isin(all_restricted_vars)]], axis=1)

        # update attributes
        return X, all_restricted_vars


    def model_index(self):
        """Form all possible model combinations

            If full model matrix of regressors) has N unrestricted elements, all possible combinations
            are (2^N - 1), i.e. 2^N minus the model with all predictors/constant excluded (y_t = error)"""
        # number of restricted vars
        n_restricted = len(self.restricted_vars)
        # get all possible combinatios
        N = self.X.shape[1] - n_restricted
        # store all possible combinations in a list
        comb = [list(itertools.combinations(range(N), i+1)) for i in range(N)]
        comb = list(itertools.chain(*comb)) # flatten list
        # comb = [list(comb[i]) for i in range(len(comb))] # transform tuples in lists
        comb = [np.array(comb[i]) for i in range(len(comb))] # transform tuples in arrays
        comb = [c + n_restricted for c in comb] # adjust indices for restricted variables
        for r in range(n_restricted):   # add indices of restricted variables
            comb = [np.insert(c, r, r) for c in comb]
        comb.append(np.arange(n_restricted))  # add model with only restricted vars


        return comb, len(comb)  # len comb = 2**N

    def set_priors(self):
        """set prior means and variances and initial conditions"""

        # init
        # outdated functions
        # ols_theta = lambda y, x: utils.inverse(x.T @ x) @ x.T @ y   # calc beta estimate
        # ols_theta_var = lambda y, x, theta: np.diag((1 / len(y)) * ((y - x @ theta).T @ (y - x @ theta)) * utils.inverse(x.T @ x))
        if not hasattr(self, "X_np"):
            self.y_dep_np, self.X_np = self.data_to_numpy()
        X = self.X_np
        y = self.y_dep_np
        K = self.K
        T = self.T
        models = self.models
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(T*self.first_sample_ends)
        else:
            first_sample_ends = np.where(self.X.index == self.first_sample_ends)[0][0]
        # init prior
        theta_0_prmean = np.empty((K, 1), dtype=float)
        theta_0_prvar = np.empty((K, 1), dtype=float)
        # set prior for thetha
        # Initial values on time-varying parameters - theta[0] ~ N(PRM,PRV x I)
            # 1: Diffuse N(0,4)
            # 2: Data-based prior
        if self.prior_theta == 1:
            theta_0_prmean = [np.zeros(len(model)) for model in self.models]  # init zeros
            theta_0_prvar = [np.eye(len(model)) for model in self.models]  # init matrices with 4 on the diagonal
        elif self.prior_theta == 2:
            theta_0_prmean = [np.zeros(len(model)) for model in self.models]  # init zeros
            theta_0_prvar = [2*utils.diag(np.var(y[:first_sample_ends])/np.var(X[:first_sample_ends, model]))
                             for model in models]

        self.theta_0_prmean = theta_0_prmean
        self.theta_0_prvar = theta_0_prvar
        # prior for V_0
        if self.initial_V_0 == 1:
            self.V_0 = 1e-3 # diffuse prior
        elif self.initial_V_0 == 2:
            self.V_0 = np.var(y[:first_sample_ends])/4  # data based prior - check /4
        # initial DMA weights
        if self.initial_DMA_weights == 1:
            self.prob_0_prmean = 1/K    # equal weights
        # define prior model weight
        if self.expert_opinion == 1:
            self.model_weight = 1/K
        elif self.expert_opinion == 2:
            self.model_weight = 0
        if self.forgetting_method == 2: # different weights for exponential forgetting method
            self.model_weight = 1

    def data_to_numpy(self):
        """ Transform panda data to numpy for computational speed"""
        return self.y_dep.to_numpy(), self.X.to_numpy()

    def run_dma_estimation(self):
        """ run dma estimation

            Proccedure:
                For all periods:
                |   For all models:
                |   |   Kalman Filter step:
                |   |   |   - Prediction step for parameters and model probability
                |   |   |   -  h-step ahead forecast
                |   |   |   - Updating step
                |   Update model probabilities
                |   For all models:
                |   |   DMA Forecasting
                (|   Select best model)"""

        # init parameters
        h_fore = self.h_fore
        K = self.K  # number of models
        T = self.T
        offset = 1e-20  # for numerical stability
        inv_lambda = 1/self.lamda
        alpha = self.alpha
        kappa = self.kappa
        model_weight = self.model_weight    # DMA_ECB: expert_weight
        weighting = self.weighting


        y = self.y_dep_np
        X = self.X_np
        models = self.models

        # define short functions
            # Linear forgetting: this is the sum of the K model probabilities (all in multiplied by the forgetting factor 'a')
        sum_prob_linear = lambda z: np.sum(alpha*prob_update[z-1, :] + (1-alpha)*model_weight)
            # exponential forgetting: this is the sum of the K model probabilities (all in the power of the forgetting factor 'a')
        sum_prob_exp = lambda z: np.sum(prob_update[z-1, :]**alpha * model_weight**(1-alpha))


        # save space for matrices
            # objects where matrices of different sizes are saved have to be initalized with object pointers
        # A_t = np.empty(K, dtype=float)  # forecast error variance
        e_t = np.empty((T, K), dtype=float)  # forecast error
        log_PL = np.zeros((T, K), dtype=float)  # in DMA_ECB: reseted after each period
        log_PL_DMA = np.zeros(T, dtype=float)
        # log_PL_BEST = np.zeros(T, dtype=float)
        # P_pred = np.empty(K, dtype=object)   # DMA_ECB R_t
        prob_pred = np.zeros((T, K), dtype=float)    # predicted model probabilities pi_t|t-1
        prob_update = np.zeros((T, K), dtype=float) # updated model probabilities pi_t|t
        P_updated = np.empty(K, dtype=object)  # DMA_ECB: S_t
        theta_pred = np.empty(K, dtype=object)  # predicted theta
        theta_update = np.empty(K, dtype=object)
        variance = np.empty((T,K), dtype=float)
        # var_BEST = np.zeros((T,1), dtype=float)
        var_DMA = np.zeros((T, 1), dtype=float)
        H_t = np.empty((T,K), dtype=float)  # error variance of y_t equation in DMA_ECB: V_t
        w_t = np.empty((T, K), dtype=float)    # predicted model probabilities times density
        y_t_DMA = np.zeros((T, 1), dtype=float)
        # y_t_BEST = np.zeros((T, 1), dtype=float)
        y_t_pred = np.empty((T,K), dtype=float)
        y_t_pred_h = np.empty((T+h_fore,K), dtype=float)  # +h_fore to save forecasts beyond T

        # Kalman filter loop
        for t in tqdm(range(h_fore, T)):
            # The model is: P_t/P_t-h = x_t-h * theta + e_t
                # so start with predicting p_t/p_0 = x_0 * theta + e_t
            i = t - h_fore


            # get sum of all model probabilities
                # used to update the individual model probabilities
            if i > 0:
                if self.forgetting_method == 1: # linear forgetting
                    sum_prob = sum_prob_linear(t)
                elif self.forgetting_method == 2:   # exponential forgetting
                    sum_prob_a = sum_prob_exp(t)

            # reset A_t and R_t, to zero at each iteration to save memory
            A_t = np.empty(K, dtype=object)
            P_pred = np.empty(K, dtype=object)

            for k in range(K):  # for all K models
                x = X[:, models[k]]
                # -------- Prediction Step -----------
                if i == 0:  # first period
                    theta_pred[k] = self.theta_0_prmean[k]  # predict theta[t]
                    P_pred[k] = inv_lambda*self.theta_0_prvar[k]    # predict P_t (P_t|t-1)
                    pi_temp = self.prob_0_prmean**alpha
                    prob_pred[t, k] = pi_temp/(K*pi_temp)
                else:
                    theta_pred[k] = theta_update[k] # predict theta[t]
                    P_pred[k] = inv_lambda*P_updated[k] # predict P_t (P_t|t-1)
                    if self.forgetting_method == 1:
                        prob_pred[t, k] = (alpha*prob_update[t-1, k] + (1-alpha)*model_weight)/sum_prob  # linear forgetting
                    elif self.forgetting_method == 2:
                        prob_pred[t, k] = ((prob_update[t-1, k]**alpha)*(model_weight**(1-alpha))+offset)\
                                          /(sum_prob_a + offset)    # exponential forgetting

                # implement individual-model predictions of the variable of interest
                # one step ahead prediction
                y_t_pred[t, k] = (x[t-h_fore]*theta_pred[k]).sum()    # (predict P_t/P_t-h with x_t-h)
                # h_fore-step ahead prediction
                y_t_pred_h[t+h_fore, k] = (x[t, :]*theta_pred[k]).sum()   # predict t+h with data from t

                # -------- Updating step ---------
                e_t[t, k] = y[t] - y_t_pred[t, k]  # one-step ahead prediction error

                # for computational efficiency define matrix products
                P = P_pred[k]
                xPx = np.dot(np.dot(x[t-h_fore], P), x[t-h_fore].T)

                # Update H_t - measurement error covariance matrix using rolling moments estimator
                if i == 0:
                    H_t[t, k] = self.V_0
                else:
                    A_t[k] = e_t[t-1, k]**2
                    H_t[t, k] = kappa*H_t[t-1, k] + (1-kappa)*A_t[k]    # DMA_ECB calls it V_t

                # Update theta[t] (regression coefficient) and its covariance matrix P[t]
                Px = np.dot(P, x[t-h_fore])
                KV = 1/(H_t[t, k] + xPx)  # resembles
                KG = Px*KV  # Kalman Gain?
                theta_update[k] = theta_pred[k] + KG*e_t[t, k]
                P_updated[k] = P - np.outer(KG, np.dot(x[t-h_fore], P))

                # Update model probability. Feed in the forecast mean and forecast
                # variance and evaluate at the future inflation value a Normal density.
                # This density is called the predictive likelihood (or posterior
                # marginal likelihood). Call this f_l, and use that to update model
                # weight/probability called w_t
                variance[t, k] = H_t[t, k] + xPx    # This is the forecast variance of each model
                if variance[t, k] <= 0: # Sometimes, the x[t]*R[t]*x[t]' quantity might be negative
                    variance[t, k] = abs(variance[t, k])
                mean = x[t-h_fore] @ theta_pred[k]  # This is the forecast mean
                if weighting == 'cauchy':
                    f_l = self.cauchy_pdf(y[t], mean, self.match_peak(variance[t, k]))
                elif weighting == 't':
                    v = self.degrees
                    f_l = self.gen_t_pdf(y[t], v, mean, np.sqrt(variance[t, k]))
                elif weighting == 'average':
                    f_l = 1/K
                elif weighting == 'mean1':
                    fe = y[t] - mean
                    var = np.var(y)
                    f_l = (1 / np.sqrt(2 * np.pi * var)) * np.exp(
                        -0.5 * (fe ** 2 / var))  # normpdf
                elif weighting == 'mean2':
                    fe = y[t] - mean
                    if t < 12:
                        var = np.var(y)
                    else:
                        var = np.var(y[t-8:t])
                    f_l = (1 / np.sqrt(2 * np.pi * var)) * np.exp(
                            -0.5 * (fe ** 2 / var))
                elif weighting == 'mean3':
                    fe = y[t] - mean
                    var = 1
                    f_l = (1 / np.sqrt(2 * np.pi * var)) * np.exp(
                        -0.5 * (fe ** 2 / var))  # normpdf
                else:   # standard weighting
                    f_l = (1 / np.sqrt(2 * np.pi * variance[t, k])) * np.exp(
                        -0.5 * ((y[t] - mean) ** 2 / variance[t, k]))  # normpdf
                # f_l = (1 / np.sqrt(2 * np.pi * variance[t, k])) * np.exp(-0.5 * ((y[t] - mean) ** 2 / variance[t, k]))  # normpdf
                w_t[t, k] = prob_pred[t, k]*f_l
                log_PL[t, k] = np.log(f_l + offset)    # Calculate log predictive likelihood for each model
                # end of Kalman filter Step

            # Update Model Probabilities Pi_t_k
            prob_update[t, :] = (w_t[t, :] + offset)/ (w_t[t, :].sum() + offset)    # Equation (16)

            # Now we have the predictions for each model & the associated model probabilities:
            # Do DMA forecasting
            # mean DMA forecast
            y_t_DMA[t] = np.sum(y_t_pred_h[t, :]*prob_pred[t, :])  # weight predictions by model probability (weight)
            # variance of the DMA forecast
            var_DMA[t] = np.sum(variance[t, :]*prob_pred[t, :])  # sum of weighted variance
            # DMA Predictive Likelihood
            log_PL_DMA[t] = np.sum(log_PL[t, :]*prob_pred[t, :])   # sum of weighted log likelihoods


        print("DMA finished")

        # get best model
        best_model_PL = np.max(prob_pred, axis=1)
        best_model_ind = np.argmax(prob_pred, axis=1)
        y_t_DMS = np.array([y_t_pred_h[i, item] for i, item in enumerate(best_model_ind)])

        # add attributes
        time_lab = self.y_dep.index[h_fore:T]
        self.time_lab = time_lab
        self.y_t_DMA = pd.Series(y_t_DMA.reshape(-1)[h_fore:], index=time_lab)
        self.var_DMA = pd.Series(var_DMA.reshape(-1)[h_fore:], index=time_lab)
        self.log_PL_DMA = pd.Series(log_PL_DMA.reshape(-1)[h_fore:], index=time_lab)

        self.y_t_DMS = pd.Series(y_t_DMS.reshape(-1)[h_fore:], index=time_lab)

        self.prob_update = prob_update[h_fore:, :]
        self.prob_pred = prob_pred[h_fore:, :]

    @staticmethod
    def cauchy_pdf(x, x0, g):
        return 1 / (np.pi * g) * (g ** 2 / ((x - x0) ** 2 + g ** 2))

    @staticmethod
    def gen_t_pdf(x, v, mu, sigma):
        part1 = gamma((v + 1) / 2) / (gamma(v / 2) * np.sqrt(np.pi * v * sigma))
        part2 = (1 + (1 / v) * ((x - mu) / sigma) ** 2) ** (-(v + 1) / 2)
        return part1 * part2

    @staticmethod
    def cauchy_peak(g):
        return 1 / (np.pi * g)

    @staticmethod
    def norm_peak(std):
        return stats.norm.pdf(0, loc=0, scale=std)

    def match_peak(self, var):
        std = np.sqrt(var)
        peak = self.norm_peak(std)
        scale = 1 / (np.pi * peak)
        return scale

    def calc_inclusion_probs(self):
        Xcols = self.X.columns
        time_lab = self.time_lab

        var_ind = list(range(len(Xcols)))
        for i in range(len(self.restricted_vars)):
            var_ind.remove(i)  # do not look at restricted vars

        inclusion_probs = pd.DataFrame(data=None,
                                       index=time_lab,
                                       columns=Xcols[var_ind])

        for i, v in enumerate(var_ind):
            v_name = Xcols[v]
            m_with_v = [i for i, m in enumerate(self.models) if v in m]  # find models that contain variable v
            prob_variable = self.prob_update[:, m_with_v].sum(axis=1).round(2)
            inclusion_probs[v_name] = prob_variable

        self.inclusion_probs = inclusion_probs

    def plot_inclusion_prob(self, seperate_plots=True, plot_vars=['all'], renderer='automatic', return_fig=False):
        """Plot inclusion probabilities

        :param vars: list of variables to plot
        :param seperate_plots: plot in one ore seperate plots
        :param renderer: how to render figure (e.g. 'browser')
        :return: plot of variables
        """
        if renderer != 'automatic':
            plotly.io.renderers.default = renderer  # render figure in browser
        if not hasattr(self, 'inclusion_probs'):
            self.calc_inclusion_probs()
        T = self.T
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(self.T * self.first_sample_ends)
        else:
            first_sample_ends = np.where(self.X.index == self.first_sample_ends)[0][0]
        inclusion_probs = self.inclusion_probs.iloc[first_sample_ends:, :]
        time_lab = self.time_lab[first_sample_ends:]
        Xcols = inclusion_probs.columns

        # find indices of variables to plot
        if plot_vars == ['all']:
            var_ind = list(range(len(Xcols)))
            # for i in range(len(self.restricted_vars)):
            #     var_ind.remove(i)   # do not plot restricted vars
            # var_ind = [vi - len(self.restricted_vars) for vi in var_ind]
        else:
            var_ind = [np.where(Xcols == v)[0].item() for v in plot_vars]

        if seperate_plots:
            # prepare plot attributes
            if len(var_ind) == 1:
                n_cols, n_rows = [1, 1]
                col_row_comb = [(1,1)]
            else:   # adapt number of subplots
                n_vars = len(var_ind)
                n_cols = int(np.ceil(np.sqrt(n_vars)))
                n_rows = int(np.ceil(n_vars/n_cols))
                col_row_comb = [(j+1, i+1) for i in range(n_cols) for j in range(n_rows)]
            # plot variables inclusion probabilities
            fig = plotly.subplots.make_subplots(rows=n_rows, cols=n_cols,
                                                shared_xaxes='columns')
                                                # subplot_titles=Xcols.values[var_ind]
            for i, v in enumerate(var_ind):
                v_name = Xcols[v]
                fig.add_trace(go.Scatter(y=inclusion_probs[v_name],
                                         x=time_lab,
                                         mode='lines',
                                         name=v_name),
                              col=col_row_comb[i][1],
                              row=col_row_comb[i][0])
            fig.update_layout(showlegend=False,
                              title_text="Inclusion probabilities")
            return fig if return_fig else fig.show()
        else:
            fig = px.line(inclusion_probs,
                          x=inclusion_probs.index,
                          y=Xcols[var_ind])
            fig.update_layout(
                title="",
                title_x=0.5,
                yaxis_title='Inclusion Probabilities',
                xaxis_title='',
                legend_title="",
                margin_b=10,
                margin_t=20,
                legend_orientation='v',
                legend_y=0.95
            )
            return fig if return_fig else fig.show()


    def forecast_statistics(self, unit='decimals', print_stats = True,
                            save_stats=True,
                            plot_fe=False,
                            plot_y_fe=False,
                            plot_y_fe_DMS=False,
                            return_fig=False):

        y = self.y_dep
        y_t_DMA = self.y_t_DMA
        y_t_DMS = self.y_t_DMS
        h_fore = self.h_fore
        T = self.T
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(self.T * self.first_sample_ends)
        else:
            first_sample_ends = np.where(self.X.index == self.first_sample_ends)[0][0]
        # DMA Statistics
        fe_DMA = y[h_fore:T] - y_t_DMA
        MAFE_DMA = (abs(fe_DMA)[first_sample_ends:]).sum()/T
        MSFE_DMA = (np.square(fe_DMA)[first_sample_ends:]).sum()/T
        BIAS_DMA = (fe_DMA[first_sample_ends:]).sum()/T
        self.fe_DMA = fe_DMA
        # DMS Statistics
        fe_DMS = y[h_fore:T] - y_t_DMS
        MAFE_DMS = (abs(fe_DMS)[first_sample_ends:]).sum() / T
        MSFE_DMS = (np.square(fe_DMS)[first_sample_ends:]).sum() / T
        BIAS_DMS = (fe_DMS[first_sample_ends:]).sum() / T
        self.fe_DMS = fe_DMS

        stats = ['MAFE', 'MSFE', 'BIAS']
        DMA_stats = [MAFE_DMA, MSFE_DMA, BIAS_DMA]
        DMS_stats = [MAFE_DMS, MSFE_DMS, BIAS_DMS]
        stats_pd = pd.DataFrame.from_dict(data={'DMA': DMA_stats, 'DMS': DMS_stats},
                                orient='index',
                                columns=stats)
        if unit == 'percent':
            stats_pd = stats_pd*100
            [y_t_DMA, y_t_DMS, y, fe_DMA, fe_DMS] = [v*100 for v in [y_t_DMA, y_t_DMS, y, fe_DMA, fe_DMS]]
            y_label = 'Inflation (in %)'
        else:
            y_label = 'Inflation'
        if print_stats:
            print(stats_pd)

        if save_stats:
            self.MAFE_DMA = MAFE_DMA
            self.MSFE_DMA = MSFE_DMA
            self.BIAS_DMA = BIAS_DMA
            self.MAFE_DMS = MAFE_DMS
            self.MSFE_DMS = MSFE_DMS
            self.BIAS_DMS = BIAS_DMS

        if plot_fe:
            # fig = px.line(x=fe_DMA[first_sample_ends:].index, y=fe_DMA[first_sample_ends:],
            #               title='Forecast errors')
            # fig.show()
            df_DMA_part = fe_DMA[first_sample_ends:]
            fig = px.line(df_DMA_part,
                          x=df_DMA_part.index,
                          y=df_DMA_part,
                          title='Forecast error')
            return fig if return_fig else fig.show()
        if plot_y_fe:
            y_fe = pd.concat((#fe_DMA[first_sample_ends:],
                              y[first_sample_ends+h_fore:],
                              y_t_DMA[first_sample_ends:]), axis=1)
            y_fe.columns = [r'$y_t$', r'$\hat{y}_t^{DMA}$']
            fig = px.line(y_fe,
                          x=y_fe.index,
                          y=y_fe.columns)
            fig.update_layout(
                title="",
                title_x=0.5,
                yaxis_title=y_label,
                xaxis_title='',
                legend_title="",
                margin_b=10,
                margin_t=10,
                legend_orientation='v',
                legend_y=0.95
                )
            return fig if return_fig else fig.show()
        if plot_y_fe_DMS:
            y_fe = pd.concat((#fe_DMS[first_sample_ends:],
                              y[first_sample_ends + h_fore:],
                              y_t_DMS[first_sample_ends:]), axis=1)
            y_fe.columns = [r'$y_t$', r'$\hat{y}_t^{DMS}$']
            fig = px.line(y_fe,
                          x=y_fe.index,
                          y=y_fe.columns)
            fig.update_layout(
                title="",
                title_x=0.5,
                yaxis_title=y_label,
                xaxis_title='',
                legend_title="",
                margin_b=10,
                margin_t=10,
                legend_orientation='v',
                legend_y=0.95
            )
            return fig if return_fig else fig.show()

            # fig = y_fe.plot(title='Inflation_t/t-h, Forecasts, Forecast errors')
            # # fig = px.line(y=y_fe, x=y_fe.index,
            # #               title='Inflation_t/t-h, Forecasts, Forecast errors')
            # fig.show()

    def calc_E_size(self, out=None, return_fig=False):
        """Calculate number of predictors at each time

            :param out = [None, 'plot', 'print']
            For the caluclation the restricted predictors are ommited.
            The expected size is calculated as Sum(Pi_k,t|t-1 * Size_k,t)"""
        prob_pred = self.prob_pred
        models = self.models
        if isinstance(self.first_sample_ends, float):
            first_sample_ends = round(self.T * self.first_sample_ends)
        else:
            first_sample_ends = np.where(self.X.index == self.first_sample_ends)[0][0]

        models_n_vars = np.array([len(m) for m in models])  # number of predictors in models
        exp_size = (prob_pred @ models_n_vars.reshape(-1, 1)).reshape(-1)

        self.exp_size = pd.Series(data=exp_size,
                                  index=self.time_lab)
        if out == 'plot':
            fig = px.line(x=self.time_lab[first_sample_ends:], y=exp_size[first_sample_ends:],
                          title='Average number of predictors used in DMA')
            fig.update_layout(
                title="",
                title_x=0.5,
                yaxis_title=r'$E[Size]$',
                xaxis_title='',
                legend_title="",
                margin_b=10,
                margin_t=10
            )

            return fig if return_fig else fig.show()


    def run_dma(self):
        self.X, self.restricted_vars = self.update_restricted()  # update restricted vars so they are at the beginning
        self.models, self.K = self.model_index()  # add model indices and number of potential models
        self.set_priors()
        self.run_dma_estimation()




if __name__ == '__main__':
    print('This program is being run by itself')
    from src.data import import_data
    import os
    from pathlib import Path
    import pickle

    # load data from DMA_ECB
    df = import_data.load_dma_ecb()
    # data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'processed'))
    # # print(data_path)
    # # load raw data
    # with open(os.path.join(data_path, 'df.pkl'), 'rb') as f:
    #     df = pickle.load(f)

    # initialize Settings
    params = Settings()
    # adjust settings
    params.use_y = ['HICP_SA']  # use as y
    params.use_x = ['M1', 'USD_EUR', 'OILP', 'STOCKPRICES']  # indep vars
    params.tcodesX = [5, 1, 1, 1]
    params.tcodey = 5
    # params.use_y = ['CPI']  # use as y
    # params.use_x = ['M1', 'dax']  # indep vars
    # params.tcodesX = [1, 1]
    # params.tcodey = 1

    # specify end of training data
    # correct?
    # params.first_sample_ends = '2012-12-31'
    params.first_sample_ends = '1990.Q4'
    # params.restricted_vars = ['intercept', 'CPI']
    params.restricted_vars = ['intercept', 'HICP_SA']
    params.forgetting_method = 2
    params.expert_opinion = 2
    params.initial_V_0 = 1
    params.h_fore = 2
    params.prior_theta = 1
    params.plag = 2
    params.hlag = 2
    params.weighting = 'average'

    # create data class
    data = Data(df, params)

    # prepare data
    # data.prepare_data_all_steps() # now in Data init

    dma = DMA(params, data)
    dma.X, dma.restricted_vars = dma.update_restricted()  # update restricted vars so they are at the start of the df
    # # adjust time indices so it is comparable to DMA_ECB
    # dma.X = dma.X.iloc[1:, :]
    # dma.y_dep = dma.y_dep.iloc[1:]
    dma.models, dma.K = dma.model_index()  # add model indices and number of potential models
    dma.set_priors()
    dma.run_dma_estimation()
    # dma.calc_inclusion_probs()
    # dma.plot_inclusion_prob(seperate_plots=True)
    dma.calc_E_size()
    dma.forecast_statistics()
    print("DMA run")
    print("Stop")
