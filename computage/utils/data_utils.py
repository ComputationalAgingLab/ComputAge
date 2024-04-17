import pandas as pd
import numpy as np
import requests
from urllib.parse import urlencode
import os
from computage.configs.links_config import META_DATASETS_LINK, BASE_DATA_URL

def test_dataset(path: str, compression: str = 'gzip') -> None:
    """
        A function for minimal testing of an assembled omics dataset.
        Currently tests the following:
        - if the data in dataset are float32 type
        - if the data indices coincide with metadata indices

        parameters:
            path: str - path to the dataset in compressed pickle format.

            compression: str - type of compression of the dataset.

        Returns: verdict as a plain string.
    """
    data, meta = pd.read_pickle(f'{path}', compression=compression).values()
    assert all(data.dtypes == np.float32), 'Type of dataframe entries is not float32!'
    assert ~((data > 1) | (data < 0)).any().any(), 'Data contain beta-values beyond [0, 1] interval'
    assert all(data.index == meta.index), 'Indices of data and meta are not equal!'
    # assert not any(meta['Age'] > 130), 'Dataset contains samples with age older than 130 years. Possibly, wrong units of age.'
    for c in ['Title', 'Tissue', 'Age', 'Condition']:
        assert c in meta.columns, f'Metadata does not contain {c} column'
    print('Ok!')


def download_from_storage(public_key, out_path):
    final_url = BASE_DATA_URL + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    download_response = requests.get(download_url)
    with open(out_path, 'wb') as f:
        f.write(download_response.content)


def download_meta(out_path, open_file=True):
    download_from_storage(META_DATASETS_LINK, out_path)
    if open_file:
        return pd.read_excel(out_path)

# standalone function function for dataset download. 
# will be reimplemented within the class
def download_dataset(meta_table, dataset_name, save_dir):
    dataset_idx = meta_table['GSE_ID'].loc[meta_table['GSE_ID']==dataset_name].index[0]
    public_key = meta_table['Link'].iloc[dataset_idx]
    download_from_storage(public_key, os.path.join(save_dir, dataset_name + '.pkl.gz'))
    print(f'Dataset {dataset_name} saved to {save_dir}')


def cond2class(conds: list) -> list:
    """
        Converts condition abbreviations to correspodning class abbreviations.
    """
    classcond = {
        "NDD": ['AD', 'PD', 'MS', 'DLB', 'CJD', 'MCI'],
        "CVD": ['HTN','AS','IHD','CVA','HF',],
        "ISD": ['CD','UC','IBD','IBS','SLE','HIV', 'HIV_TB', 'TB'],
        "MSD": ['SP','OP','OA','RA',],
        "MBD": ['OBS','IR','T1D','T2D','MBS', 'ASD'],
        "LVD": ['NAFLD','NASH','PBC','PSC','LF','HCC',],
        "LUD": ['COPD', 'IPF'],
        "PGS": ['WS', 'HGPS', 'CGL', 'DS', 'aWS', 'MDPS', 'ncLMNA']
    }
    classes = []
    for c in conds:
        for cl, l in classcond.items():
            if c in l:
                classes.append(cl)
    return classes