#Importing data and functions
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path

import plotly.express as px
from statsmodels.regression.linear_model import OLS
from factor_analyzer.factor_analyzer import FactorAnalyzer

from src.models.preliminaries import Settings
from src.data.data_class import Data


class PCA:

    def initiate(self, data, n_components=2):
        self.n_samples = data.shape[0]
        self.n_components = n_components

        self.newdata = self.standardization(data)

        covmat = self.covariance_matrix()

        eigenvectors = self.eigenvectors(covmat)

        projmat = self.projection(eigenvectors)

        return projmat

    def standardization(self, data):
        # In here we substract the mean and dividing by standard deviation
        z = (data - np.mean(data, axis=0)) / (np.std(data, axis=0))
        return z

    def covariance_matrix(self, ddof=0):
        # We make a dot product with the transposed matrix and the normal matrix,
        # then dividing by the number of samples
        covmat = np.dot(self.newdata.T, self.newdata) / (self.n_samples - ddof)
        return covmat

    def eigenvectors(self, covmat):
        # Calculating eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covmat)

        # Sorting eigenvalues in decending order
        n_cols = np.argsort(eigenvalues)[::-1][:self.n_components]

        # Selecting columns based on the number of components
        selected_Vectors = eigenvectors[:, n_cols]
        return selected_Vectors

    def projection(self, eigenvectors):
        P = np.dot(self.newdata, eigenvectors)
        return P

class FactorAnalysis():
    def __init__(self, df, excl=None, **kwargs):
        """init Factor analysis
            :param excl: list of variables to exclude"""
        self.df = df.drop(excl, axis=1) if not excl == None else df
        fa = FactorAnalyzer(**kwargs)
        fa.fit(df)

    def fit(self, **kwargs):
        fa.fit(self.df)


if __name__ == '__main__':
    # Initialization of Variables
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
                    ]
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
    selection = pd.read_csv(os.path.join(data_path, 'selected_data.csv'))
    # get transformations for variables
    params.tcodesX = Settings.get_tcodesX(params, selection)
    params.intercept = 0
    # prepare data
    data = Data(df, params)
    data.data_to_numpy()

    # Class Initialization
    pca = PCA()

    pca_data = pca.initiate(data.X_np, n_components=2)

    print('Done')


