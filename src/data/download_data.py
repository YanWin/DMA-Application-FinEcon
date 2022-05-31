import os
import requests as rq
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path

# Support functions taken and slightly adapted from https://github.com/LenkaV/CIF

# OECD API FUNCTIONS

def makeOECDRequest(dsname, dimensions, params=None, root_dir='http://stats.oecd.org/SDMX-JSON/data'):
    """
    Make URL for the OECD API and return a response.

    Parameters
    -----
    dsname: str
        dataset identifier (e.g., MEI for main economic indicators)
    dimensions: list
        list of 4 dimensions (usually location, subject, measure, frequency)
    params: dict or None
        (optional) dictionary of additional parameters (e.g., startTime)
    root_dir: str
        default OECD API (https://data.oecd.org/api/sdmx-json-documentation/#d.en.330346)

    Returns
    -----
    results: requests.Response
        `Response <Response>` object

    """

    if not params:
        params = {}

    dim_args = ['+'.join(d) for d in dimensions]
    dim_str = '.'.join(dim_args)

    url = root_dir + '/' + dsname + '/' + dim_str + '/all'

    print('Requesting URL ' + url)
    return rq.get(url=url, params=params)

def getOECDJSONStructure(dsname, root_dir='http://stats.oecd.org/SDMX-JSON/dataflow', showValues=[],
                             returnValues=False):
    """
    Check structure of OECD dataset.

    Parameters
    -----
    dsname: str
        dataset identifier (e.g., MEI for main economic indicators)
    root_dir: str
        default OECD API structure uri
    showValues: list
        shows available values of specified variable, accepts list of integers
        which mark position of variable of interest (e.g. 0 for LOCATION)
        e.g. MEI_ARCHIVE ['LOCATION', 'VAR', 'EDI', 'FREQUENCY', 'TIME_PERIOD']
    returnValues: bool
        if True, the observations are returned

    Returns
    -----
    results: list
        list of dictionaries with observations parsed from JSON object, if returnValues = True

    """

    url = root_dir + '/' + dsname + '/all'

    print('Requesting URL ' + url)

    response = rq.get(url=url)

    if (response.status_code == 200):

        responseJson = response.json()

        keyList = [item['id'] for item in responseJson.get('structure').get('dimensions').get('observation')]

        print('\nStructure: ' + ', '.join(keyList))

        for i in showValues:
            print('\n%s values:' % (keyList[i]))
            print('\n'.join(
                [str(j) for j in responseJson.get('structure').get('dimensions').get('observation')[i].get('values')]))

        if returnValues:
            return (responseJson.get('structure').get('dimensions').get('observation'))

    else:

        print('\nError: %s' % response.status_code)


def createOneCountryDataFrameFromOECD(country='CZE', dsname='MEI', subject=[], measure=[], frequency='M',
                                      startDate=None, endDate=None):
    """
    Request data from OECD API and return pandas DataFrame. This works with OECD datasets
    where the first dimension is location (check the structure with getOECDJSONStructure()
    function).

    Parameters
    -----
    country: str
        country code (max 1, use createDataFrameFromOECD() function to download data from more countries),
        list of OECD codes available at http://www.oecd-ilibrary.org/economics/oecd-style-guide/country-names-codes-and-currencies_9789264243439-8-en
    dsname: str
        dataset identifier (default MEI for main economic indicators)
    subject: list
        list of subjects, empty list for all
    measure: list
        list of measures, empty list for all
    frequency: str
        'M' for monthly and 'Q' for quaterly time series
    startDate: str of None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations
    endDate: str or None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations

    Returns
    -----
    data: pandas.DataFrame
        data downloaded from OECD
    subjects: pandas.DataFrame
        subject codes and full names
    measures: pandas.DataFrame
        measure codes and full names

    """

    # Data download

    response = makeOECDRequest(dsname
                               , [[country], subject, measure, [frequency]]
                               ,
                               {'startTime': startDate, 'endTime': endDate, 'dimensionAtObservation': 'AllDimensions'})

    # Data transformation

    if (response.status_code == 200):

        responseJson = response.json()

        obsList = responseJson.get('dataSets')[0].get('observations')

        if (len(obsList) > 0):

            if (len(obsList) >= 999999):
                print('Warning: You are near response limit (1 000 000 observations).')

            print('Data downloaded from %s' % response.url)

            timeList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if
                        item['id'] == 'TIME_PERIOD'][0]['values']
            # subjectList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'SUBJECT'][0]['values']
            # measureList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'MEASURE'][0]['values']
            subjectList = responseJson.get('structure').get('dimensions').get('observation')[1]['values']
            measureList = responseJson.get('structure').get('dimensions').get('observation')[2]['values']

            obs = pd.DataFrame(obsList).transpose()
            obs.rename(columns={0: 'series'}, inplace=True)
            obs['id'] = obs.index
            obs = obs[['id', 'series']]
            obs['dimensions'] = obs.apply(lambda x: re.findall('\d+', x['id']), axis=1)
            obs['subject'] = obs.apply(lambda x: subjectList[int(x['dimensions'][1])]['id'], axis=1)
            obs['measure'] = obs.apply(lambda x: measureList[int(x['dimensions'][2])]['id'], axis=1)
            obs['time'] = obs.apply(lambda x: timeList[int(x['dimensions'][4])]['id'], axis=1)
            # obs['names'] = obs['subject'] + '_' + obs['measure']

            # data = obs.pivot_table(index = 'time', columns = ['names'], values = 'series')

            data = obs.pivot_table(index='time', columns=['subject', 'measure'], values='series')

            return (data, pd.DataFrame(subjectList), pd.DataFrame(measureList))

        else:

            print('Error: No available records, please change parameters')
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    else:

        print('Error: %s' % response.status_code)
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

def createOneCountryDataFrameFromOECD_fromResponse(response_str):
    """
    Request data from OECD API and return pandas DataFrame. This works with OECD datasets
    where the first dimension is location (check the structure with getOECDJSONStructure()
    function).

    Parameters
    -----
    response_str: str
        request string

    Returns
    -----
    data: pandas.DataFrame
        data downloaded from OECD
    subjects: pandas.DataFrame
        subject codes and full names
    measures: pandas.DataFrame
        measure codes and full names

    """

    # Data download

    print('Requesting URL ' + response_str)
    response = rq.get(url=response_str)

    # Data transformation

    if (response.status_code == 200):

        responseJson = response.json()

        obsList = responseJson.get('dataSets')[0].get('observations')

        if (len(obsList) > 0):

            if (len(obsList) >= 999999):
                print('Warning: You are near response limit (1 000 000 observations).')

            print('Data downloaded from %s' % response.url)

            timeList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if
                        item['id'] == 'TIME_PERIOD'][0]['values']
            # subjectList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'SUBJECT'][0]['values']
            # measureList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'MEASURE'][0]['values']
            subjectList = responseJson.get('structure').get('dimensions').get('observation')[1]['values']
            measureList = responseJson.get('structure').get('dimensions').get('observation')[2]['values']

            obs = pd.DataFrame(obsList).transpose()
            obs.rename(columns={0: 'series'}, inplace=True)
            obs['id'] = obs.index
            obs = obs[['id', 'series']]
            obs['dimensions'] = obs.apply(lambda x: re.findall('\d+', x['id']), axis=1)
            obs['subject'] = obs.apply(lambda x: subjectList[int(x['dimensions'][1])]['id'], axis=1)
            obs['measure'] = obs.apply(lambda x: measureList[int(x['dimensions'][2])]['id'], axis=1)
            obs['time'] = obs.apply(lambda x: timeList[int(x['dimensions'][4])]['id'], axis=1)
            # obs['names'] = obs['subject'] + '_' + obs['measure']

            # data = obs.pivot_table(index = 'time', columns = ['names'], values = 'series')

            data = obs.pivot_table(index='time', columns=['subject', 'measure'], values='series')

            return (data, pd.DataFrame(subjectList), pd.DataFrame(measureList))

        else:

            print('Error: No available records, please change parameters')
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    else:

        print('Error: %s' % response.status_code)
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())



def createDataFrameFromOECD(countries=['CZE', 'AUT', 'DEU', 'POL', 'SVK'], dsname='MEI', subject=[], measure=[],
                            frequency='M', startDate=None, endDate=None):
    """
    Request data from OECD API and return pandas DataFrame. This works with OECD datasets
    where the first dimension is location (check the structure with getOECDJSONStructure()
    function).

    Parameters
    -----
    countries: list
        list of country codes, list of OECD codes available at http://www.oecd-ilibrary.org/economics/oecd-style-guide/country-names-codes-and-currencies_9789264243439-8-en
    dsname: str
        dataset identifier (default MEI for main economic indicators)
    subject: list
        list of subjects, empty list for all
    measure: list
        list of measures, empty list for all
    frequency: str
        'M' for monthly and 'Q' for quaterly time series
    startDate: str or None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations
    endDate: str or None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations

    Returns
    -----
    data: pandas.DataFrame
        data downloaded from OECD
    subjects: pandas.DataFrame
        subject codes and full names
    measures: pandas.DataFrame
        measure codes and full names

    """

    dataAll = pd.DataFrame()
    subjectsAll = pd.DataFrame()
    measuresAll = pd.DataFrame()

    for country in countries:

        dataPart, subjectsPart, measuresPart = createOneCountryDataFrameFromOECD(country=country, dsname=dsname,
                                                                                 subject=subject, measure=measure,
                                                                                 frequency=frequency,
                                                                                 startDate=startDate, endDate=endDate)

        if (len(dataPart) > 0):
            dataPart.columns = pd.MultiIndex(levels=[[country], dataPart.columns.levels[0], dataPart.columns.levels[1]],
                                             codes=[np.repeat(0, dataPart.shape[1]), dataPart.columns.codes[0],
                                                    dataPart.columns.codes[1]],
                                             names=['country', dataPart.columns.names[0], dataPart.columns.names[1]])

            dataAll = pd.concat([dataAll, dataPart], axis=1)
            subjectsAll = pd.concat([subjectsAll, subjectsPart], axis=0)
            measuresAll = pd.concat([measuresAll, measuresPart], axis=0)

    if (len(subjectsAll) > 0):
        subjectsAll.drop_duplicates(inplace=True)

    if (len(measuresAll) > 0):
        measuresAll.drop_duplicates(inplace=True)

    return (dataAll, subjectsAll, measuresAll)




if __name__ == '__main__':
    print('This program is being run by itself')

    # For the complete list of available country codes run
    # getOECDJSONStructure(dsname='MEI_ARCHIVE', showValues=[0])

    # 1 Settings
    downloadData = False
    saveData = False  # Save the original data sets if True
    test = True
    country = 'DEU'

    # 2 Loading data from OECD API.
        # example:
        # data_all, subjects_all, measures_all = createDataFrameFromOECD(countries=[country], dsname='MEI_ARCHIVE', frequency='M')

    if downloadData:
        # Vintage data from
        variables_Q = ['101', '102', '103', '104', '105', '106', '108', '502', '503', '701']
        variables_M = ['201', '202', '203', '401', '501', '702', '703']

        data_dict_Q = {"data": [], "subjects": [], "measures": []}
        for v in variables_Q:
            data_temp, subjects_temp, measures_temp = createDataFrameFromOECD(countries=[country], dsname='MEI_ARCHIVE',
                                                                        subject=[v], frequency='Q')
            data_dict_Q['data'].append(data_temp)
            data_dict_Q['subjects'].append(subjects_temp)
            data_dict_Q['measures'].append(measures_temp)
            print('Downloaded variable %s with data set size %d x %d' % (v, data_temp.shape[0], data_temp.shape[1]))
        data_dict_M = {"data": [], "subjects": [], "measures": []}
        for v in variables_M:
            data_temp, subjects_temp, measures_temp = createDataFrameFromOECD(countries=[country], dsname='MEI_ARCHIVE',
                                                                              subject=[v], frequency='M')
            data_dict_M['data'].append(data_temp)
            data_dict_M['subjects'].append(subjects_temp)
            data_dict_M['measures'].append(measures_temp)
            print('Downloaded variable %s with data set size %d x %d' % (v, data_temp.shape[0], data_temp.shape[1]))

        data_rs, subjects_rs, measures_rs = createDataFrameFromOECD(countries=[country], dsname='MEI_ARCHIVE',
                                                                        subject=['201'], frequency='M')

        # # data for for energy prices
        #     # now included in the MEI_M
        # data_dict_energy = {"data": [], "subjects": [], "measures": []}
        # data_energy, subjects_energy, measures_energy = createDataFrameFromOECD(countries=['DEU'], dsname='PRICES_CPI',
        #                                                                         subject=['CPHPEN01'], frequency='M')
        # data_dict_energy['data'].append(data_energy)
        # data_dict_energy['subjects'].append(subjects_energy)
        # data_dict_energy['measures'].append(measures_energy)
        # print(
        #     'Downloaded variable %s with data set size %d x %d' % ('CPHPEN01', data_energy.shape[0], data_energy.shape[1]))

        variables_MEI_M = ['LOCOBSNO',
                           'LOCOBDNO',
                           'LOCOCINO',
                           'LOCOS3NO',
                           'ODCNPI03',
                           'CPGRLE01',
                           'BRCICP02',
                           'BCCICP02',
                           'BVCICP02',
                           'BSCICP02',
                           'CSCICP02',
                           'CSINFT02',
                           'CPHPEN01',
                           'CPHPLA01']
        data_dict_MEI_M = {"data": [], "subjects": [], "measures": []}
        for v in variables_MEI_M:
            data_temp, subjects_temp, measures_temp = createDataFrameFromOECD(countries=[country], dsname='MEI',
                                                                              subject=[v], frequency='M')
            data_dict_MEI_M['data'].append(data_temp)
            data_dict_MEI_M['subjects'].append(subjects_temp)
            data_dict_MEI_M['measures'].append(measures_temp)
            print('Downloaded variable %s with data set size %d x %d' % (v, data_temp.shape[0], data_temp.shape[1]))

        variables_MEI_Q = ['CP040000',
                           'PIEAEN02',
                           'PIEAEN01']
        data_dict_MEI_Q = {"data": [], "subjects": [], "measures": []}
        for v in variables_MEI_Q:
            data_temp, subjects_temp, measures_temp = createDataFrameFromOECD(countries=[country], dsname='MEI',
                                                                              subject=[v], frequency='Q')
            data_dict_MEI_Q['data'].append(data_temp)
            data_dict_MEI_Q['subjects'].append(subjects_temp)
            data_dict_MEI_Q['measures'].append(measures_temp)
            print('Downloaded variable %s with data set size %d x %d' % (v, data_temp.shape[0], data_temp.shape[1]))

    if saveData:
        data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'raw'))
        with open(os.path.join(data_path, 'data_dict_M.pkl'), 'wb') as f:
            pickle.dump(data_dict_M, f)
        with open(os.path.join(data_path, 'data_dict_Q.pkl'), 'wb') as f:
            pickle.dump(data_dict_Q, f)
        # with open(os.path.join(data_path, 'data_dict_energy.pkl'), 'wb') as f:
        #     pickle.dump(data_dict_energy, f)
        with open(os.path.join(data_path, 'data_dict_MEI_M.pkl'), 'wb') as f:
            pickle.dump(data_dict_MEI_M, f)
        with open(os.path.join(data_path, 'data_dict_MEI_Q.pkl'), 'wb') as f:
            pickle.dump(data_dict_MEI_Q, f)
        print('finished')

    if test:


        data_path = os.path.abspath(os.path.join(Path(__file__).parent.parent.parent, 'data', 'raw'))

        print('done')