import sys
import os

import numpy as np
import pandas as pd

from src import utils


def load_files_dma_ecb():
    """ Import the Matlab data avalaible at Korobilis website
    
    The data can be accesed at "https://sites.google.com/site/dimitriskorobilis/matlab/dma".
    It contains quarterly US data from 1959Q1 through 2008Q2.
    """
    root = utils.get_project_root()
    path_data = os.path.join(root, r'data\external\DMA_ECB')
    files = [f for f in os.listdir(path_data) if f.endswith(".dat")]

    dat = []
    for f in files:
        path_temp = os.path.join(path_data, f)
        dat.append(np.genfromtxt(path_temp, dtype=str))
    return files, dat


def dma_ecb_bundle_data(files, dat):
    """ Bundle data to a single panda

    Bundle data from DMA_ECB and create one panda that contains all the relevant information
    """
    data_dict = dict(zip(files, dat))

    namesOTHER = data_dict['namesOTHER.dat']
    namesX = data_dict['namesX.dat']
    ts_index = data_dict['yearlab.dat']

    ts_y = pd.DataFrame(data = data_dict['HICP_SA.dat'].astype(float))
    df_x = pd.DataFrame(data = data_dict['xdata.dat'].astype(float))
    df_other = pd.DataFrame(data=data_dict['otherdata.dat'].astype(float))

    df = pd.concat([ts_y, df_x, df_other], axis=1)
    df.columns = np.concatenate([np.array(['HICP_SA']), namesX, namesOTHER])
    df.index = ts_index

    return df

def load_dma_ecb():
    """ load data from DMA_ECB and bring it into form

    :return: pd.DataFrame
    """
    files, dat = load_files_dma_ecb()
    df = dma_ecb_bundle_data(files, dat)
    return df

if __name__ == '__main__':
    print('This program is being run by itself')
    root = utils.get_project_root()
    # alternative way of getting the root
    # root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    print(root)
    files, dat = load_files_dma_ecb()
    df = dma_ecb_bundle_data(files, dat)
    print("Data loaded")
    print(df.info())

