"""
The module for getting datasets from the data repository of that project.
"""

import os
import requests
import pandas as pd
from tqdm.notebook import trange

def load_data(url: str,
              out_dir: str,
              dataset_id: str,
              force = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    """
    Downloads data from a remote repository. Downloaded file is saved to out_dir/file_name

    Parameters:
        url (str): Repository web link
        out_dir (str): Output directiory containing downloaded data
        dataset_id (str): Dataset ID
        force (bin): If True, enables overwriting (currently unused)

    Returns:
        tuple(data, meta)
    """

    file_name = f'{dataset_id}.pkl.xz'

    full_url = f'{url}/{file_name}'

    # Streaming, so we can iterate over the response
    response = requests.get(full_url, stream=True)

    # Sizes in bytes
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Output file path
    out_path = f'{out_dir}/{file_name}'

    # Download file if it doesn't exist in the output directory
    if not (os.path.exists(out_path) and total_size == os.path.getsize(out_path)):
        with trange(1, total=total_size, desc=f'Loading {dataset_id}', unit='B', unit_scale=True) as progress_bar:
            with open(out_path, 'wb') as f:
                for downloaded in response.iter_content(block_size):
                    progress_bar.update(len(downloaded))
                    f.write(downloaded)
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError('Failed to download file. Please check if it exists in the remote repository.')
        
    data, meta = pd.read_pickle(out_path).values()

    return data, meta