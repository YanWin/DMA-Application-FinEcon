import numpy as np
import yaml

class Settings():
    # add description
    def __init__(self, **params):
        #  ------------ DATA SETTINGS
        # Estimate intercept
            # 0: no
            # 1: yes (default)
        self.intercept = params.get('intercept', 1)
        # Define lags
        self.plag = params.get('plag', 1) # Lags of dependent variables
        self.hlag = params.get('hlag', 0) # Lags of exogenous variables

        # Specify x variables to use
            # check
        self.use_x = params.get('use_x', None)
        self.use_y = params.get('use_y', ['CPI'])

        # Set transformations for exogenous vars
            # 1: Use levels for all
            # 2: Use specified transformations # check
        self.tcodesX = params.get('tcodesX', np.array(1))
        # Set transformations for dependent vars
        self.tcodey = params.get('tcodey', 1)

        # How to treat missing values (note that these are only in the begining of the sample)
            # 1: Fill in missing values with zeros, let the KF do the rest
            # 2: Trim quarters which are not observed (that way we lose information /
                # for variables which have no missing values)
        self.miss_treatment = params.get('miss_treatment', 2)

        # ----------- MODEL SETTINGS
        # Forgetting Factors
        self.lamda = params.get('lamda', 0.99) # For the time-varying parameters theta
        self.alpha = params.get('alpha', 0.90) # For the model switching
        self.kappa = params.get('kappa', 0.95) # For the error covariance matrix
        # Forgetting method on model switching probabilities
            # 1: Linear Forgetting
            # 2: Exponential Forgetting
        self.forgetting_method = params.get('forgetting_method', 2)

        # Initial values on time-varying parameters
            # theta[0] ~ N(PRM,PRV x I)
            # 1: Diffuse N(0,4)
            # 2: Data-based prior
        self.prior_theta = params.get('prior_theta', 1)
        # Initialize measurement error covariance V[t]
            # 1: a small positive value (but NOT exactly zero)
            # 2: a quarter of the variance of your initial data
        self.initial_V_0 = params.get('initial_V_0', 1)


        # set variables to restrict in DMA (always included)
        self.restricted_vars = params.get('restricted_vars', None)
        # set initial DMA weights (only equal weight option)
        self.initial_DMA_weights = params.get('initial_DMA_weights', 1)
        # Define expert opinion (prior) on model weight
            # 1: Equal weights on all models
            # 2: No prior expert opinion
        self.expert_opinion = params.get('expert_opinion', 2)
        # Define weighting scheme
        self.weighting = params.get('weighting', 'normal')
        self.degrees = params.get('degrees', None)

        # --------- FORECASTING --------------
        # Define forecast horizon (applied to direct forecasts)
        self.h_fore = params.get('h_fore', 1)
        # Define the last observation of the first sample used to start producing forecasts recursively
        self.first_sample_ends = params.get('first_sample_ends', '1990.Q4')

        # change add model check

    def print_settings(self):
        temp = vars(self)
        print("The following preliminary settings are specified:")
        for item in temp:
            print(item, ':', temp[item])

    def print_setting_options(self):
        dict_list_settings = [
            {'intercept':  'Estimate intercept',
                'values': {'0': 'no', '1': 'yes (default)'}},
            {'plag': 'Lags of dependent variable'},
            {'hlag': 'Lags of exogenous variables'},
            {'use_y': 'dependent variable'},
            {'use_x': 'exogenous variables to include'},
            {'tcodesX': 'Specify transformations for exogenous variables',
                'values': {'1': 'no transformations',
                           'list of transformation codes':
                               {'1' : 'Level',
                                '2': 'First Difference',
                                '3': 'Second Difference',
                                '4': 'Log-Level',
                                '5': '100*Log-First-Difference',
                                '6': '100*Log-Second-Difference'}}},

            {'tcodesy': 'Specify transformation for dependent variable'},
            {'miss_treatment': 'How to treat missing values at the beginning of the sample',
                'values': {'1': 'Fill in missing values with zeros, let the KF do the rest',
                           '2': 'Trim quarters which are not observed for all variables',
                           '3': 'Do nothing'}},
            {'lamda': 'Forgetting factor for the time-varying parameters theta'},
            {'alpha': 'Forgetting factor for the model switching'},
            {'kappa': 'Forgetting factor for the error covariance matrix'},
            {'forgetting_method': 'Forgetting method on model switching probabilities',
                'values': {'1': 'Linear Forgetting',
                           '2': 'Exponential Forgetting'}},
            {'prior_theta': 'Prioir on time-varying parameters',
                'values': {'1': 'Diffuse N(0,4)',
                           '2': 'Data-based prior'}},
            {'initial_V_0': 'Initialize measurement error covariance V[t]',
                'values': {'1', 'a small positive value (but NOT exactly zero)',
                           '2', 'a quarter of the variance of your initial data'}},
            {'restricted_vars': 'set variables to restrict in DMA (always included)'},
            {'expert_opinion': ' Define expert opinion (prior) on model weight',
                'values': {'1': 'Equal weights on all models',
                            '2': 'No prior expert opinion'}},
            {'initial_DMA_weights': 'set initial DMA weights (only option 1: equal weight option)'},
            {'weighting': 'define weighting scheme of model forecasts'},
            {'degrees': 'degress of freedom if weighting is done by t-distribution'},
            {'h_fore': 'Define forecast horizon (applied to direct forecasts)'},
            {'first_sample_ends': ' Define the last observation of the first sample used to start producing forecasts recursively'}
        ]
        print(yaml.dump(dict_list_settings, sort_keys=False, default_flow_style=False))

    def get_tcodesX(self, selection):
        tcodesX = []
        for v in self.use_x:
            v_ind = np.where(selection['var code'] == v)[0][0]
            v_tcode = selection['trans_code'][v_ind]
            tcodesX.append(v_tcode)
            # print(f'{v} has tcode {v_tcode}')
        return tcodesX







