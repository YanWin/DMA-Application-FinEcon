import os
import requests as rq
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from statsmodels.tsa import seasonal


def transform_vintage_to_growth(df, m_to_q='no', name='', q_growth=1):
    """transform vintage data matrix into growth series to circumvent rebasing or other method changes

        :param df: pd.Dataframe - vintage dataframe to be transformed
        :param m_to_q: ['no', 'last', 'start', 'mean'] - method of resampling transformation
        :param name: str - name of resulting series
        :param q_growth: int - lags to calculate growth. 1 is QoQ growth, 4 is YoY growth
        :return pd.Series"""
    # df2 = df  # for debug
    if not isinstance(df.index, pd.DatetimeIndex):  # make sure index is Datetime format
        df.index = pd.to_datetime(df.index)
    freq = pd.infer_freq(df.index)  # get frequency, e.g. 'Q' for 'QS-OCT'
    print(f'Frequency of the data is {freq}.')
    if m_to_q != 'no':
        if str(freq)[0] != 'M': # check if frequency is fitting to transformation
            print('Vintage does not seem to have the monthly frequency, which could induce errors when transforming')
        # transform monthly data

        if m_to_q == 'last':
            df = df.resample('M').last()  # preformat last of month date
            q_ind = df.resample('Q').last().index
            q_ind = q_ind[[q in df.index for q in q_ind]]   # only use quarters for which last observation is available
            df = df.loc[q_ind, :]
        elif m_to_q == 'start':
            df = df.resample('M').first()
            q_ind = df.resample('Q').first().index
            q_ind = q_ind[[q in df.index for q in q_ind]]
            df = df.loc[q_ind, :]
        elif m_to_q == 'mean':
            df = df.resample('Q').mean()
        freq = 'Q'
    elif str(freq)[0] not in ['M', 'Q']:
        print('Vintage does not seem to have the quarterly or monthly frequency, which could induce errors')


    # transform vintage data matrix to series

    # returns a series with last observations of columns
        # index of last_obs is the vintage and values the date of last observation
    last_obs = df.apply(pd.Series.last_valid_index)    # returns a series with time index of the last observations

    # if m_to_q != 'no':
    # transform vintages in monthly frequency to quarterly frequency
        # for this detect the publication that published new data first
            # e.g. if the subsequent last observation has the same date, the according vintage is not the one in which the data is
            # first published
        # to verify: pd.concat((last_obs, last_obs.shift(1), last_obs.shift(-1), first_published), axis=1)
    first_published = (last_obs != last_obs.shift(1)) & (last_obs == last_obs.shift(-1))
    if last_obs[-1] != last_obs.shift(1)[-1]:   # check if last observation also contains new information
        first_published[-1] = True
    last_obs = last_obs[first_published]  # now the vintages are also in quarterly frequency

    # transform data into percentage change - to circumvent baseline adjustments
    df = df.pct_change(q_growth)
    # combine last observations to series
    df_series = pd.Series(index=df.index, dtype=float)*np.nan   # init series
    last_obs_vintage = last_obs.index  # series with the vintage names
    for i, ind in enumerate(last_obs):
        df_series[ind] = df.loc[ind, last_obs_vintage[i]]
    df_series = df_series.fillna(df.iloc[:, -1])    # fill data that started before first vintage
    df_series.name = name
    return df_series




if __name__ == '__main__':
    print('This program is being run by itself')

    combine_transform_data = False      # transform vintage data and combine all data sets
    select_data = True                 # select only the series specified in the selection.csv
    seasonal_transform = True           # seasonally transform data if specified
    test = False                         # just for some testing

    if combine_transform_data:
        print('Load raw data, transform vintage data and combine all data to one dataset')
        data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'raw'))
        # print(data_path)
        # load raw data
        with open(os.path.join(data_path, 'data_dict_M.pkl'), 'rb') as f:
            data_dict_M = pickle.load(f)
        with open(os.path.join(data_path, 'data_dict_Q.pkl'), 'rb') as f:
            data_dict_Q = pickle.load(f)
        # with open(os.path.join(data_path, 'data_dict_energy.pkl'), 'rb') as f:
        #     data_dict_energy = pickle.load(f)
        with open(os.path.join(data_path, 'data_dict_MEI_M.pkl'), 'rb') as f:
            data_dict_MEI_M = pickle.load(f)
        with open(os.path.join(data_path, 'data_dict_MEI_Q.pkl'), 'rb') as f:
            data_dict_MEI_Q = pickle.load(f)

        # # to see if changes are present
        # data_m_changes = pd.DataFrame(index=data_dict_M['data'][0].columns, dtype=float)
        # for i,v in enumerate(data_dict_M['subjects']):
        #     data_m_changes[str(v['id'].values)] = data_dict_M['data'][i].diff(1, axis=1).sum(axis=0)

        df_vintages_M = pd.DataFrame()
        for i, v in enumerate(data_dict_M['subjects']):
            v_name = v['name'].values[0]
            print(f'Transform series {v_name}')
            v_series = transform_vintage_to_growth(data_dict_M['data'][i], m_to_q='last', name=v_name, q_growth=1)
            df_vintages_M[v_name] = v_series

        df_vintages_Q = pd.DataFrame()
        for i, v in enumerate(data_dict_Q['subjects']):
            v_name = v['name'].values[0]
            print(f'Transform series {v_name}')

            v_series = transform_vintage_to_growth(data_dict_Q['data'][i], m_to_q='no', name=v_name, q_growth=1)
            df_vintages_Q[v_name] = v_series

        # prepare downloaded data from MEI
        # the downloaded data for these series includes more than one measure
        select_measure = {'ODCNPI03': ('DEU', 'ODCNPI03', 'MLSA'),  # MLSA = monthly level, s.a.
                          'CPGRLE01': ('DEU', 'CPGRLE01', 'IXOB'),  # IXOB = index
                          'CPHPEN01': ('DEU', 'CPHPEN01', 'IXOB'),  # the three strings are due to the multiindex
                          'CPHPLA01': ('DEU', 'CPHPLA01', 'IXOB')}

        df_MEI_M = pd.DataFrame()
        for i, v in enumerate(data_dict_MEI_M['subjects']):
            v_name = v['name'].values[0]
            id = v['id'].values[0]
            # print(f'Variable: {v_name} with code {id} has the measures')
            # print(data_dict_MEI_M['measures'][i])
            # print(data_dict_MEI_M['data'][i].head(3))

            if id in select_measure.keys():  # if data has multiple measures
                measure = select_measure[id]
                v_series = data_dict_MEI_M['data'][i][measure]
            else:
                v_series = data_dict_MEI_M['data'][i]
            df_MEI_M[v_name] = v_series

        if not isinstance(df_MEI_M.index, pd.DatetimeIndex):  # make sure index is Datetime format
            df_MEI_M.index = pd.to_datetime(df_MEI_M.index)
        df_MEI_M = df_MEI_M.resample('M').last()  # preformat last of month date
        q_ind = df_MEI_M.resample('Q').last().index
        q_ind = q_ind[[q in df_MEI_M.index for q in q_ind]]  # only use quarters for which last observation is available
        df_MEI_M = df_MEI_M.loc[q_ind, :]

        # prepare data_dict_MEI_Q
        select_measure = {'CP040000': ('DEU', 'CP040000', 'IXOB'),
                          'PIEAEN02': ('DEU', 'PIEAEN02', 'IXOB'),
                          'PIEAEN01': ('DEU', 'PIEAEN01', 'IXOB')}
        df_MEI_Q = pd.DataFrame()
        for i, v in enumerate(data_dict_MEI_Q['subjects']):
            v_name = v['name'].values[0]
            id = v['id'].values[0]
            if id in select_measure.keys():  # if data has multiple measures
                measure = select_measure[id]
                v_series = data_dict_MEI_Q['data'][i][measure]
            else:
                v_series = data_dict_MEI_Q['data'][i]
            df_MEI_Q[v_name] = v_series

        # load other files
        files = {'Inflation_expectation': os.path.join(data_path, 'Inflation_Exp_GER_1999_2022.xlsx'),
                 'DAX': os.path.join(data_path, '^GDAXI.csv'),
                 'EO_vars': os.path.join(data_path, 'EO_05042022115104422.csv'),
                 'supply_index': os.path.join(data_path, 'LSE_2022_supply-chain-update__benigno_data.xlsx'),
                 'Interest_rates': os.path.join(data_path, 'MEI_FIN_Interest_rates.csv'),
                 'Monetary_aggregates': os.path.join(data_path, 'MEI_FIN_Monetary_aggregates.csv'),
                 # 'MEI_other': os.path.join(data_path, 'MEI_26042022121908751.csv'),   # should now be included in data_dict_MEI_x
                 'HICP_no_vintage': os.path.join(data_path, 'PRICES_HICP_no_vintage.csv')}

        HICP_no_vintage = pd.read_csv(files['HICP_no_vintage'], index_col='TIME'). \
            filter(items=['SUBJECT', 'Value']).\
            pivot(columns='SUBJECT', values='Value')

        inflation_exp = pd.read_excel(files['Inflation_expectation'],
                                      header=0,
                                      index_col='Survey round')
        inflation_exp = inflation_exp[inflation_exp.index.notna()]
        inflation_exp.index = pd.to_datetime(inflation_exp.index.str.replace(' ', ''))

        dax = pd.read_csv(files['DAX'], index_col='Date')
        dax.index = pd.to_datetime(dax.index)

        EO_vars = pd.read_csv(files['EO_vars'], index_col='TIME')
        EO_vars = EO_vars.pivot(columns='VARIABLE', values='Value')

        supply_index = pd.read_excel(files['supply_index'],
                                     sheet_name='SCPI',
                                     skiprows=4,
                                     index_col='Date')
        supply_index = supply_index.add(100)   # to make log transformation possible

        interest_rates = pd.read_csv(files['Interest_rates'], index_col='TIME')
        interest_rates = interest_rates.pivot(columns='SUBJECT', values='Value')

        monetary_aggregates = pd.read_csv(files['Monetary_aggregates'], index_col='TIME')
        monetary_aggregates = monetary_aggregates.pivot(columns='SUBJECT', values='Value')

        # MEI_other = pd.read_csv(files['MEI_other'], index_col='TIME').\
        #     filter(items=['SUBJECT', 'FREQUENCY', 'Value'])
        # MEI_other_M = MEI_other[MEI_other['SUBJECT'].isin(['BSCICP02', 'CSCICP02'])].\
        #     drop('FREQUENCY', axis=1).\
        #     pivot(columns='SUBJECT', values='Value')
        # MEI_other_Q = MEI_other[(MEI_other['SUBJECT'].isin(['PIEAEN01', 'PIEAEN02', 'CP040000'])) &
        #                         (MEI_other['FREQUENCY'] == 'Q')].\
        #     drop('FREQUENCY', axis=1).\
        #     pivot(columns='SUBJECT', values='Value')

        data = [df_vintages_M,
                df_vintages_Q,
                inflation_exp,
                dax,
                EO_vars,
                supply_index,
                interest_rates,
                monetary_aggregates,
                df_MEI_M,
                df_MEI_Q,
                # MEI_other_M,
                # MEI_other_Q,
                HICP_no_vintage]

        # set the same frequency if neccessary
        df_all = pd.DataFrame()
        for d in data:
            if not isinstance(d.index, pd.DatetimeIndex):
                d.index = pd.to_datetime(d.index)
            d = d.resample('Q').last()
            df_all = pd.concat((df_all, d), axis=1)

        save_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'processed', 'combined_data.csv'))
        df_all.to_csv(save_path, index=True)

        print('Done')

    if select_data:
        print('load dataset with all variables and select data')
        data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'processed'))
        selection = pd.read_csv(os.path.join(data_path, 'selected_data.csv'))
        df = pd.read_csv(os.path.join(data_path, 'combined_data.csv'), index_col=0)
        df = df[selection['Variable']]
        df.columns = selection['var code']
        df = df.loc['1991-06-30':'2021-12-31', :]

        with open(os.path.join(data_path, 'df.pkl'), 'wb') as f:
            pickle.dump(df, f)
        print('data selected and saved as df.pkl')

    if seasonal_transform:
        print('load dataset with all variables and seasonally transform data')
        data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'processed'))
        selection = pd.read_csv(os.path.join(data_path, 'selected_data.csv'))
        with open(os.path.join(data_path, 'df.pkl'), 'rb') as f:
            df = pickle.load(f)

        df_sa = df.copy()
        for name, values in df.iteritems():
            v_ind = np.where(selection['var code']==name)[0][0]
            if selection['sa'][v_ind]:
                print(f'seasonally adjust {name}')
                # decomp = seasonal.seasonal_decompose(values.dropna(), period=4)
                decomp = seasonal.STL(values.dropna(), period=4).fit()
                sa_adjust = decomp.seasonal
                series_adjusted = values - sa_adjust
                df_sa[name] = series_adjusted
            else:
                df_sa[name] = values
        with open(os.path.join(data_path, 'df_sa.pkl'), 'wb') as f:
            pickle.dump(df_sa, f)
        print('seasonal adjustment done and saved as df_sa.pkl')
        print('end')

    if test:
        pass