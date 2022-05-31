"""Script specifying different variable groups

    used to keep "Replication_Hand_in.ipynb" clean"""

infl_vars = ['CPI',
             'deflator_GDP',
             'HICP_excl_energy'
             ]
labor_market = ['unemp',
                'employment',
                'earnings_hourly']
macro_nrs = ['GDP',
             'cons_private',
             'cons_gov']
trade = ['trade_imp',
         'trade_exp']
investment = ['invest',
              'invest_private_no_housing',
              'invest_private_housing',
              'residential_permits']
prod_indices=['prod_ind',
              'retail',
              'prod_constr']
financial1=['dax']
financial2=['interest_rate_short',
            'interest_rate_long']
money=['M1',
       'M3']
infl_exp = ['infl_exp_current_year',
           'infl_exp_next_year',
           'infl_exp_2_year_ahead',
           'infl_exp_5_year_ahead']
infl_predictors = ['PCI_energy_',
                   'HICP_energy',
                   'CPI_house_energy']
supply_ind = ['supply_index_global',
              'supply_index_eu']
confidence = ['business_conf_manufacturing',
              'business_conf_construct',
              'business_conf_service',
              'business_conf_retail']
confidence2 = ['cons_conf_tendency',
              'business_situation']
var_groups = {'Quarterly Inflation Indicators': infl_vars,
              'Labor Market': labor_market,
              'Macro Variables': macro_nrs,
              'Trade': trade,
              'Investment': investment,
              'Industry Indices': prod_indices,
              'Financial': financial1,
              'Interest Rates': financial2,
              'Money Aggregates': money,
              'Inflation Expectations': infl_exp,
              'Inflation Sub-Indices': infl_predictors,
              'Supply Pressure Indices': supply_ind,
              'Confidence Indicators 1': confidence,
              'Confidence Indicators 2': confidence2}
potential_exo = labor_market + \
                macro_nrs + \
                trade + \
                investment + \
                prod_indices + \
                financial1 + \
                financial2 + \
                money + \
                infl_exp + \
                infl_predictors + \
                supply_ind + \
                confidence + \
                confidence2