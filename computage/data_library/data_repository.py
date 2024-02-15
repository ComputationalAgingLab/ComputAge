"""
The module for getting datasets from the data repository of that project.
"""

from typing import Any
from tqdm.notebook import trange
import requests
import pickle
import os

def download_data(url: str,
                  out_dir: str,
                  file_name: str,
                  force = False):
    
    """
    Downloads data from a remote repository. Downloaded file is saved to out_dir/file_name

    Parameters:
        url (str): Repository web link
        out_dir (str): Output directiory containing downloaded data
        file_name (str): Output file name
        force (bin): If True, enables overwriting

    Returns:
        None
    """

    # Streaming, so we can iterate over the response
    response = requests.get(url, stream=True)

    # Sizes in bytes
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Output file path
    out_path = f'{out_dir}/{file_name}'

    # Download file if it doesn't exist
    if os.path.exists(out_path) and force == False:
        print('File exists, overwriting disabled. To enable, set "force" = True')
    else:
        with trange(1, total=total_size, desc=f'Loading {file_name}', unit='B', unit_scale=True) as progress_bar:
            with open(out_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    pickle.dump(data, f)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError('Failed to download file. Please check if it exists in the remote repository.')

def import_dataset(dataset_path: str) -> tuple[Any, Any]:
    with open(dataset_path, 'rb') as f:
        object = pickle.load(f)
    return object