import numpy as np
import numpy.linalg as la
import pandas as pd

from src.models.preliminaries import Settings

######### THIS DMA FILE USES Y+H/Y as the independent variable. THe new one uses Y-h/Y according to the model

class Data():
    def __init__(self, df, params):
        # set data
        self.df = df
        # init settings for data transformation
        self.intercept = params.intercept
        self.hlag = params.hlag
        self.plag = params.plag
        self.use_x = params.use_x
        self.use_y = params.use_y
        self.tcodesX = params.tcodesX
        self.tcodey = params.tcodey
        self.miss_treatment = params.miss_treatment
        self.h_fore = params.h_fore
        self.first_sample_ends = params.first_sample_ends
        # init other attributes
        self.y_status = 0
        self.X_status = 0
        self.status_codes = {0: "Not assigned yet",
                             1: "Assigned",
                             2: "Transformed",
                             3: "Transformed and shifted",
                             4: "Transformed and lagged",
                             5: "X combined with lags of y",
                             6: "Observations removed",
                             7: "Ready"}

        self.prepare_data_all_steps()

    def set_data(self):
        """ Function to select variables as specified in preliminaries

            Use "use_y" and "use_x" to create attributes with (in)dependent variables"""
        y = self.df[self.use_y[0]].copy()
        self.y_origin = y
        self.y_dep = y
        self.y_indep = y
        self.y_status = 1
        if self.use_x != None:
            X = self.df[self.use_x].copy()
            self.X = X
            self.X_status = 1
        else:
            self.X = None

    def transx_old(self, x, tcode):
        """ Transform x as specified in tcode
           :param x: panda series
           :param tcode:
           :return: y: transformed panda series

           Return Series with same dimension and corresponding dates
           Missing values where not calculated
           -- Tcodes:
                    1 Level
                    2 First Difference
                    3 Second Difference
                    4 Log-Level
                    5 Log-First-Difference
                    6 Log-Second-Difference
                    7 Detrend Log Using 1-sided HP detrending for Monthly data
                    8 Detrend Log Using 1-sided HP detrending for Quarterly data
                   16 Log-Second-Difference
                   17 (1-L)(1-L^12)

         Translated from the Gauss procs of Stock&Watson(2005),'Implications of
         dynamic factor models for VAR analysis'
         Dimitris Korobilis, June 2007.
            """

        small = 1e-06

        # HP params - not implemented yet
        # relvarm = 7.5e-07
        # relvarq = 0.000625
        # .00000075 for monthly data
        # .000625 for quarterly data, see Harvey/Jeager (1993), page 234 @

        x_index = x.index
        x_name = x.name
        x = x.to_numpy()
        n = x.shape[0]
        y = np.zeros((n,))

        if tcode == 1:
            y = x
        elif tcode == 2:
            y[1:n] = x[1:n] - x[0:n - 1]
        elif tcode == 3:
            y[2:n] = x[2:n] - 2 * x[1:n - 1] + x[0:n - 2]
        elif tcode == 4:
            if np.amin(x) < small:
                y = np.nan
                print("log transformation not possible for one variable")
            else:
                y = np.log(x)
        elif tcode == 5:
            if np.amin(x) < small:
                y = np.nan
                print("log transformation not possible for one variable")
            else:
                x = np.log(x)
                y[1:n] = x[1:n] - x[0:n - 1]
        elif tcode == 6:
            if np.amin(x) < small:
                y = np.nan
                print("log transformation not possible for one variable")
            else:
                x = np.log(x)
                y[2:n] = x[2:n] - 2 * x[1:n - 1] + x[0:n - 2]
        elif tcode == 7:
            raise Exception("tcode not implemented yet")
            if np.amin(x) < small:
                y = np.nan
            x = np.log(x)
            y, t1 = detrend1(x, relvarm)  # not translated yet
        elif tcode == 8:
            raise Exception("tcode not implemented yet")
            if np.amin(x) < small:
                y = np.nan
            x = np.log(x)
            y, t1 = detrend1(x, relvarq)  # not translated yet
        elif tcode == 16:
            if np.amin(x) < small:
                y = np.nan
                print("log transformation not possible for one variable")
            else:
                x = np.log(x)
                y[2:n] = x[2:n] - 2 * x[1:n - 1] + x[0:n - 2]
        elif tcode == 17:
            raise Exception("tcode not implemented yet")
            if np.amin(x) < small:
                y = np.nan
            # not implemented yet
        else:
            y = np.nan

        # transform to panda series
        y = pd.Series(y, index=x_index)
        y.name = x_name
        return y

    @staticmethod
    def transx(x, tcode):
        """Transform x as specified in tcode
           :param x: panda series
           :param tcode: transormation code
           :return: y: transformed panda series

           Return Series with same dimension and corresponding dates. Some tranformations induce nan values
            -- Tcodes:
                    1 Level
                    2 First Difference
                    3 Second Difference
                    4 Log-Level
                    5 Log-First-Difference
                    6 Log-Second-Difference"""
        n = x.shape[0]
        # y = pd.Series(np.zeros((n,), index=x.index))
        small = 1e-06

        if tcode == 1:
            y = x
        elif tcode == 2:
            y = x.diff(1)
        elif tcode == 3:
            y = x.diff(1)-x.diff(1).shift(1)
        elif tcode == 4:
            if np.amin(x) < small:
                y = np.nan
                print("log transformation not possible for one variable")
            else:
                y = np.log(x)
        elif tcode == 5:
            if np.amin(x) < small:
                y = np.nan
                print("log transformation not possible for one variable")
            else:
                y = np.log(x).diff(1)
        elif tcode == 6:
            if np.amin(x) < small:
                y = np.nan
                print("log transformation not possible for one variable")
            else:
                y = np.log(x).diff(1) - np.log(x).diff(1).shift(1)
        else:
            y = np.nan
        return y

    def shift_trans_infl_old(self, x=None, tcode=None, nph=None):
        """ Transform data to forecast and shift forward
        :param x: panda series
        :param tcode: transformation code
        :param nph: forecast horizon
        :return yf: transformed x and shifted panda series

        the transformed and shifted variable will be the dependent variable
            e.g. y: y_t = ln(P_t+h/P_(t)) in the case of log transform and first-diff
        compare description in Koop and Korobilis (2012) on p.868 bottom


            -- Tcodes: not all implemented
                    1 Level
                    2 First Difference
                    3 Second Difference
                    4 Log-Level
                    5 100*Log-First-Difference
                    6 100*Log-Second-Difference
                    7 Detrend Log Using 1-sided HP detrending for Monthly data
                    8 Detrend Log Using 1-sided HP detrending for Quarterly data
                   16 Log-Second-Difference
                   17 (1-L)(1-L^12)
            """

        small = 1.0e-06

        x_index = x.index
        x_name = x.name
        x = x.to_numpy()
        n = x.shape[0]
        yf = np.zeros((n,))

        if tcode <= 3:
            y = x
        elif tcode in [4, 5, 6]:
            if np.amin(x) < small:
                y = np.nan
                print("Log transformation not possible")
            else:
                x = np.log(x)
                y = x

        if tcode in [1, 4]:
            yf[0:n - nph] = y[nph:n]
        elif tcode in [2, 5]:
            yf[0:n - nph] = y[nph:n] - y[0:n - nph]
        elif tcode in [3, 6]:
            yf[1:n - nph] = (y[1 + nph:n] - y[1:n - nph]) - nph * (y[1:n - nph] - y[0:n - nph - 1])
        elif tcode in [7, 8]:
            raise Exception("Invalid Transformation Code in trans_infl")
        elif tcode in [16, 17]:
            raise Exception("Not implemented yet")

        # transform to panda series
        yf = pd.Series(yf, index=x_index)
        yf.name = x_name + "_t+" +str(nph) + "/t"
        return yf

    def shift_trans_infl(self, x=None, tcode=None, nph=None):
        """ Transform data to forecast and shift forward
        :param x: panda series
        :param tcode: transformation code
        :param nph: forecast horizon
        :return yf: transformed x and shifted panda series

        the transformed and shifted variable will be the dependent variable
            e.g. y: y_t = ln(P_t/P_(t-h)) in the case of log transform and first-diff
            compare description in Koop and Korobilis (2012) on p.868 bottom
            -- Tcodes: not all implemented
                    1 Level
                    2 First Difference
                    3 Second Difference
                    4 Log-Level
                    5 Log-First-Difference
                    6 Log-Second-Difference
            """

        small = 1.0e-06

        x_name = x.name
        # x = x.to_numpy()
        # n = x.shape[0]

        if tcode <= 3:
            y = x
        elif tcode in [4, 5, 6]:
            if np.amin(x) < small:
                y = np.nan
                print("Log transformation not possible")
            else:
                x = np.log(x)
                y = x

        if tcode in [1, 4]:
            # from the vintage data, we have (CPI_t-CPI_t-1)/CPI_t-1
            # we want (CPI_t-CPI_t-nph)/CPI_t-nph
            # multiply growth rates to get there
            y_temp = y
            for i in range(1,nph):
                y_temp = y_temp.add(1).multiply(y.shift(i).add(1))-1
            y = y_temp
        elif tcode in [2, 5]:
            y = y.diff(nph)
        elif tcode in [3, 6]:
            raise Exception("Not implemented yet")
        elif tcode in [7, 8]:
            raise Exception("Invalid Transformation Code in trans_infl")
        elif tcode in [16, 17]:
            raise Exception("Not implemented yet")

        # transform to panda series
        y.name = x_name + "_t/t-" + str(nph)
        return y

    def check_status(self, var_status, status):
        """ Check if variable status fits to processing step

        :param var: X or y
        :param status: self.var_status
        """
        if not var_status == status:
            raise Exception("wrong status of " +str(var_status) + " : "
                            + str(self.status_codes[0]))

    def transf_data_old(self):
        """ Apply transformations to data and correct for lost observations

            sets forward shifted and transformed y (y_shifted), transformed y (y_lags), tranformed X"""
        self.check_status(self.y_status, 1)  # check if y assinged

        y = self.y_dep
        # transform y series
        y_trans = 4*self.transx_old(y, self.tcodey) # times 4* added
        y_shifted = self.shift_trans_infl_old(y, tcode=self.tcodey, nph=self.h_fore)
        # adjust variable for direct h period ahead forecast
        # needed as e.g. ln(P_t/P_(t-h)) is not in quarterly terms any more
        y_shifted = 4*y_shifted / self.h_fore   # times 4* added
        self.y_status = 3
        self.y_dep = y_shifted
        self.y_indep = y_trans

        # transform vars in X matrix
        # if no X matrix is specified, skip transformation

        if isinstance(self.X, pd.DataFrame):
            self.check_status(self.X_status, 1)  # check if X assinged
            X_trans = self.X.copy()
            if len(self.tcodesX) != X_trans.shape[1]:  # move to checks
                print("Not all transformation statisfied properly, no variable transformed")
                self.tcodesX = np.repeat(1, X_trans.shape[1])
            cols = X_trans.columns
            for i in range(X_trans.shape[1]):
                columnName = cols[i]
                X_trans_i = self.transx(X_trans.loc[:, columnName], self.tcodesX[i])
                if self.tcodesX[i] in [5, 6]:
                    X_trans_i = 4*X_trans_i # times 4* added
                X_trans.loc[:, columnName] = X_trans_i

            # set new X
            self.X = X_trans
            self.X_status = 2

    def transf_data(self):
        """ Apply transformations to data and correct for lost observations

            sets forward shifted and transformed y (y_shifted), transformed y (y_lags), tranformed X"""
        self.check_status(self.y_status, 1)  # check if y assinged

        y = self.y_dep
        # transform y series
        y_trans = 4 * self.transx(y, self.tcodey)  # times 4* added
        y_shifted = self.shift_trans_infl(y, tcode=self.tcodey, nph=self.h_fore)
        # adjust variable for direct h period ahead forecast
        # needed as e.g. ln(P_t/P_(t-h)) is not in quarterly terms any more
        y_shifted = 4 * y_shifted / self.h_fore  # times 4* added
        self.y_status = 3
        self.y_dep = y_shifted
        self.y_indep = y_trans

        # transform vars in X matrix
        # if no X matrix is specified, skip transformation

        if isinstance(self.X, pd.DataFrame):
            self.check_status(self.X_status, 1)  # check if X assinged
            X_trans = self.X.copy()
            if len(self.tcodesX) != X_trans.shape[1]:  # move to checks
                print("Not all transformation statisfied properly, no variable transformed")
                self.tcodesX = np.repeat(1, X_trans.shape[1])
            cols = X_trans.columns
            for i in range(X_trans.shape[1]):
                columnName = cols[i]
                X_trans_i = self.transx(X_trans.loc[:, columnName], self.tcodesX[i])
                if self.tcodesX[i] in [5, 6]:
                    X_trans_i = 4 * X_trans_i  # times 4* added
                X_trans.loc[:, columnName] = X_trans_i

            # set new X
            self.X = X_trans
            self.X_status = 2


    def lagg_matrices(self):
        """ Create matrices that include specified number of lags of variables

            Will create a pd.Dataframe with added lagged variables. Number of lags depends on plag and hlag.
            First max(plag, hlag) observations are discarded.

        :param self.y: panda series of dependet var
        :param self.X: panda matrix of independent vars
        :param self.plag: lags of dependent var
        :param self.hlag: lags of independent var
        :return: self.y_lags, self.X_lags: pd.Dataframes
        """
        self.check_status(self.y_status, 3)  # check if y transformed
        y = self.y_indep
        y.name = y.name
        # lag y variable
        y_lags = pd.DataFrame(y)
        for i in range(self.plag):
            lag_name = y.name + "_t-" + str(i + 1)
            y_lags[lag_name] = y.shift(i + 1)
        # y_lags = y_lags.drop(y.name, axis=1)  # through out y_t
        self.y_indep = y_lags
        self.y_status = 4

        # lag X variables if specified
        if isinstance(self.X, pd.DataFrame):
            self.check_status(self.X_status, 2)  # check if X transformed
            X = self.X
            # prepare empty matrix with ordered colnames
            col_names_lags = []
            for column_name, item in X.iteritems():
                col_names_lags.append(column_name)
                for i in range(self.hlag):
                    lag_name = column_name + "_t-" + str(i + 1)
                    col_names_lags.append(lag_name)
            # prepare empty DataFrame
            X_lags = pd.DataFrame(data=np.nan,
                                  index=X.index,
                                  columns=col_names_lags)
            # fill X_lag with (lagged) data
            for column_name, item in X.iteritems():
                X_lags[column_name] = X[column_name]
                for i in range(self.hlag):
                    lag_name = column_name + "_t-" + str(i + 1)
                    X_lags[lag_name] = X[column_name].shift(i + 1)
            self.X = X_lags
            self.X_status = 4

    def combine_lagged_y_X(self):
        self.check_status(self.y_status, 4) # check if y is lagged
        if self.use_x == None:
            self.X = self.y_indep
        else:
            self.X = pd.concat([self.y_indep, self.X], axis=1)
        # set status to ready
        self.X_status = 5


    def remove_obs(self):
        """ remove observations lost through lagging, transforming and adjusting for forecast horizon

            Initial series will have n observation and returned series n - max(hlag,plag) - h_fore - trans,
                with trans = 0 for level, 1 for first-diff and 2 second-order-diff transformation
        """
        self.check_status(self.X_status, 5)  # check if X transformed and combined
        # periods lost through lagging
        p_lost_lag = max(self.plag, self.hlag)
        # 1 lag for first diff, 2 for second diff
        if any([True for i in [2, 5] if i == self.tcodey or i in self.tcodesX]):
            p_lost_trans = 1
        elif any([True for i in [3, 6, 16] if i == self.tcodey or i in self.tcodesX]):
            p_lost_trans = 2
        else:
            p_lost_trans = 0
        # periods lost at the start of the sample
        p_lost_start = p_lost_lag + p_lost_trans

        # periods lost through forecasting
        p_lost_end = self.h_fore

        # correct matrices
        self.y_dep = self.y_dep.iloc[p_lost_start::, ]
        self.X = self.X.iloc[p_lost_start::, ]
        # self.y_dep = self.y_dep.iloc[p_lost_start:-p_lost_end, ]
        # self.X = self.X.iloc[p_lost_start:-p_lost_end, ]
        # set new status to ready
        self.y_status = 6
        self.X_status = 6


    def deal_with_missing(self):
        """ Deal with missing values

            if self.miss_treatment == 1: Fill missing values with zeros, 2: start at first common obs"""
        # check if previous steps applied
        self.check_status(self.X_status, 6)
        y = self.y_dep
        X = self.X
        if self.miss_treatment == 1:  # Fill missing values with zeros
            X = X.fillna(0)
            y = y.fillna(0)
        elif self.miss_treatment == 2:  # Begin at the earliest common observation among the selected variables
            first_valid_loc = X.apply(lambda col: col.first_valid_index()).max()
            y = y.loc[first_valid_loc:]
            X = X.loc[first_valid_loc:]
            if pd.isnull(X).sum().sum() > 0:
                print("X still has na values")
                print(X.isna().sum())
            if pd.isnull(y).any():
                print("y still has na values")
        elif self.miss_treatment == 3:  # do nothing
            pass
        self.y_dep = y
        self.X = X
        # set status to ready
        self.y_status = 7
        self.X_status = 7

    def add_intercept(self):
        if self.intercept == 1:
            self.X.insert(0, 'intercept', 1)
        else:
            pass


    def prepare_data_all_steps(self):
        """ Prepare data as specified in params

            Apply in order:
                1. set data
                2. transform data
                3. lagg data
                4. combine lagged y with X
                5. remove observations
                6. deal with missings
                7. add intercept
                8. save data information"""

        self.set_data()             # set data
        self.transf_data()          # transform data and shift for forecasting
        # self.transf_data_old()
        self.lagg_matrices()        # create lags
        self.combine_lagged_y_X()   # combine data
        self.remove_obs()           # remove observations
        self.deal_with_missing()    # deal with missings
        self.add_intercept()        # add intercept
        self.save_data_info()       # save data information



    def save_data_info(self):
        """Save index lab, T, N, and variable names"""

        self.T, self.N = self.X.shape
        self.index_lab = self.X.index
        self.y_origin_name = self.y_origin.name
        self.X_names = self.X.columns
        self.y_dep_name = self.y_dep.name


    def data_to_numpy(self):
        """ Transform panda data to numpy for computational speed"""
        self.y_dep_np = self.y_dep.to_numpy()
        self.X_np = self.X.to_numpy()


    def set_priors(self):
        """OUTDATED (Only valid for TVP case) Set priors for theta and V_0

            Initial values on time-varying parameters
            theta[0] ~ N(PRM,PRV x I)
                # 1: Diffuse N(0,10)
                # 2: Data-based prior"""
        if not hasattr(self, "X_np"):
            self.save_data_info()
            self.data_to_numpy()
        X = self.X_np
        y = self.y_dep_np
        # set prior for thetha
        if self.prior_theta == 1:
            self.theta_prmean = np.zeros(self.N)
            self.theta_prvar = 10*np.eye(self.N)
        elif self.prior_theta == 2:
            theta_OLS = la.inv(X.T @ X) @ X.T @ y
            epsi = y - X @ theta_OLS
            self.theta_prmean = theta_OLS
            self.theta_prvar = (1 / self.N) * (epsi.T @ epsi) * la.inv(X.T @ X)
        # prior for V_0
        self.V_0 = np.var(y)/4



if __name__ == '__main__':
    print('This program is being run by itself')
    from src.data import import_data

    # load data from DMA_ECB
    df = import_data.load_dma_ecb()

    # initialize Settings
    params = Settings()
    # adjust settings
    params.use_y = ['HICP_SA']  # use as y
    params.use_x = ['M1', 'USD_EUR', 'OILP', 'STOCKPRICES']  # indep vars
    params.tcodesX = [5, 1, 1, 1]
    params.tcodey = 5
    # specify end of training data
    # correct?
    params.first_sample_ends = '1990.Q4'
    params.restricted_vars = ['intercept', 'HICP_SA']
    params.forgetting_method = 2
    params.expert_opinion = 2
    params.h_fore = 2
    params.prior_theta = 1

    # create data class
    data = Data(df, params)
    data.set_data()  # set data
    data.transf_data()  # transform data and shift for forecasting
    # data.transf_data_2()
    data.lagg_matrices()  # create lags
    data.combine_lagged_y_X()  # combine data
    data.remove_obs()  # remove observations
    data.deal_with_missing()  # deal with missings
    data.add_intercept()  # add intercept
    data.save_data_info()  # save data information

    data2 = Data(df, params)
    data2.prepare_data_all_steps()
    print("Stop")

