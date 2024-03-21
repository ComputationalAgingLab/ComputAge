import pandas as pd
import numpy as np

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
    assert all(data.index == meta.index), 'Indices of data and meta are not equal!'
    print('Ok!')