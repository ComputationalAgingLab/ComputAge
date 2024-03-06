"""
The module for getting datasets from the data repository of that project.
"""

import os
import requests
import pandas as pd
from tqdm.notebook import trange

URL = 'http://localhost:9998/'
DATA_DIR = 'datasets'

def download_file(url: str, out_path: str) -> None:
    
    """
    Worker function to download a file

    Parameters:
        url (str): Remote repository link
        out_path (str): Output file path
    """
    
    # Streaming, so we can iterate over the response
    response = requests.get(url, stream=True)

    # Sizes in bytes
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    # Download file if it doesn't exist in the output directory
    if not os.path.exists(out_path) or total_size != os.path.getsize(out_path):
        with trange(1, total=total_size, desc=f'Loading {out_path.split("/")[-1]}', unit='B', unit_scale=True) as progress_bar:
            if '.pkl' in out_path: 
                mode = 'wb'
            else:
                mode = 'w'
            with open(out_path, mode) as f:
                for downloaded in response.iter_content(block_size):
                    progress_bar.update(len(downloaded))
                    f.write(downloaded)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError('Failed to download file. Please check if it exists in the remote repository.')
    
    return

def load_data(dataset_id: str,
              data_dir=DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Downloads data from a remote repository. Downloaded file is saved to data_dir/file_name

    Parameters:
        dataset_id (str): Dataset ID
        data_dir (str): Local directiory containing downloaded data

    Returns:
        tuple(data, meta)
    """

    file_name = f'{dataset_id}.pkl.xz'
    file_url = f'{URL}/{file_name}'

    # Create output directory if it doesn't exist
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    # Output file path
    out_path = f'{data_dir}/{file_name}'

    # Function to download file
    download_file(out_path=out_path, url=file_url)
    
    # Read downloaded pickled dictionary into two data frames
    data, meta = pd.read_pickle(out_path).values()

    return data, meta

def describe_data(data_dir=DATA_DIR) -> pd.DataFrame:

    """
    Downloads description for all datasets contained in a remote repository. Downloaded file is saved to data_dir/datasets_description.csv

    Parameters:
        data_dir (str): Local directiory containing downloaded data

    Returns:
        pd.DataFrame with annotated datasets
    """

    file_name = 'datasets_description.csv'
    file_url = f'{URL}/{file_name}'

    if not os.path.exists(data_dir): os.makedirs(data_dir)
    out_path = f'{data_dir}/{file_name}'
    download_file(out_path=out_path, url=file_url)
        
    data_descr = pd.read_csv(out_path, index_col=0)

    return data_descr