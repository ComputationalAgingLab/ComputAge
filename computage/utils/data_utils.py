import pandas as pd
import numpy as np
import requests
from urllib.parse import urlencode
import os

from computage.settings import ROOTDIR
from computage.configs.links_config import META_DATASETS_LINK, BASE_DATA_URL

# classcond = {
#     "NDD": ['AD', 'PD', 'MS', 'DLB', 'CJD', 'MCI'],
#     "CVD": ['HTN','AS','IHD','CVA','HF',],
#     "ISD": ['CD','UC','IBD','IBS','SLE','HIV', 'HIV_TB'],
#     "MSD": ['SP','OP','OA','RA'],
#     "MBD": ['OBS','IR','T1D','T2D','MBS', 'ASD', 'XOB'],
#     "LVD": ['NAFLD','NASH','PBC','PSC','LF','HCC',],
#     "RSD": ['COPD', 'IPF', 'TB'],
#     "PGS": ['WS', 'HGPS', 'CGL', 'DS', 'aWS', 'MDPS', 'ncLMNA'],
#     "KDD": ['CKD']  
# }

classcond = {
    'HC':  ['HC'],
    'CVD': ['AS', 'IHD', 'CVA'], 
    'ISD': ['IBD', 'HIV', 'HIV_TB'], 
    'MBD': ['XOB', 'T1D', 'T2D'], 
    'MSD': ['OP', 'RA'], 
    'NDD': ['MS', 'PD', 'AD', 'CJD'], 
    'RSD': ['TB', 'COPD'], 
    'PGS': ['CGL', 'WS', 'HGPS']
}

def check_dataset(path: str, compression: str = 'gzip') -> None:
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


def download_from_storage(public_key: str, out_path: str):
    final_url = BASE_DATA_URL + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']
    download_response = requests.get(download_url)
    with open(out_path, 'wb') as f:
        f.write(download_response.content)


def download_meta(out_path: str, open_file=True) -> None | pd.DataFrame:
    download_from_storage(META_DATASETS_LINK, out_path)
    if open_file:
        return pd.read_excel(out_path)

# standalone function function for dataset download. 
# will be reimplemented within the class
def download_dataset(meta_table: pd.DataFrame, 
                     dataset_name: str, 
                     save_dir: str):
    dataset_idx = meta_table['GSE_ID'].loc[meta_table['GSE_ID']==dataset_name].index[0]
    public_key = meta_table['Link'].iloc[dataset_idx]
    download_from_storage(public_key, os.path.join(save_dir, dataset_name + '.pkl.gz'))
    print(f'Dataset {dataset_name} saved to {save_dir}')


def cond2class(conds: list[str]) -> list:
    """
        Converts condition abbreviations to correspodning class abbreviations.
    """
    classes = []
    for c in conds:
        for cl, l in classcond.items():
            if c in l:
                classes.append(cl)
    return classes


def construct_config(dataset_prefix: str, 
                     datasets_config: dict,
                     split: str = 'benchmark',
                     ):
    """
    Function for rewriting file paths downloaded from hugging face hub.
    params:
        split: str - the name of split, can be from {`benchmark`, `train`}
    """
    assert split in ['benchmark', 'train'], 'Wrong split name, choose from {`benchmark`, `train`}'
    config = datasets_config.copy()
    config_keys = list(datasets_config.keys())
    data_folder = os.path.join(dataset_prefix, f'data/{split}')
    data_files = os.listdir(data_folder)
    for k in config_keys:
        for f in data_files:
             if k in f:
                config[k]['path'] = os.path.join(data_folder, f)
    return config


def prepare_datasets_config(path: str) -> dict:
    """
    Read config from the project folder.
    """
    bsb_table = pd.read_csv(path)
    datasets_config_main = {}
    for _, r in bsb_table.iterrows():
        gse = r['Dataset ID']
        if gse not in datasets_config_main.keys():
            datasets_config_main[gse] = {
			    'path':None,
				'conditions':[r['Condition']],
				'test':'AA2' if r['HC content'] == 'HC' else 'AA1'
			}
        else:
            datasets_config_main[gse]['conditions'].append(r['Condition'])
    return datasets_config_main

#function for fast BSB config retrieval
def get_bsb_config() -> dict:
    config_path = os.path.join(ROOTDIR, "data_library/BenchBSB_table.csv")  
    return prepare_datasets_config(config_path)
