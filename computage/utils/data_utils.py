import pandas as pd
import numpy as np

def test_dataset(path: str, compression: str = 'gzip') -> None:
    data, meta = pd.read_pickle(f'{path}', compression=compression).values()
    assert all(data.dtypes == np.float32), 'Type of dataframe entries is not float32!'
    assert all(data.index == meta.index), 'Indices of data and meta are not equal!'
    print('Ok!')