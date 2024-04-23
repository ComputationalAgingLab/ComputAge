from scipy.stats import linregress
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import mapply

def linear_time_analysis(data: pd.DataFrame, 
                         age: np.ndarray | pd.Series,
                         n_jobs: int | None = 1
                         ) -> pd.DataFrame:
    """
        parameters:
            - n_jobs - number of cores running the analysis in parallel
    """
    def _fit_feature(x):
        y = age.copy()
        idx = np.isfinite(x) #y should always be finite
        y = y[idx] 
        x = x[idx]
        s, i, r, p, serr = linregress(y, x)
        x_cap = y * s + i
        r2 = r2_score(x, x_cap)
        rss = (np.square(x_cap - x)).sum()
        rse = np.sqrt(rss / (x.shape[0] - 2))
        #secondary analysis
        residuals = np.abs(x - x_cap)
        s_res, i_res, r_res, p_res, _ = linregress(y, residuals)
        return s, i, r, p, serr, rse, r2, s_res, i_res, r_res, p_res
    
    if (n_jobs is None) or (n_jobs < 1):
        result = data.apply(_fit_feature, result_type='expand')
    else:
        mapply.init(n_workers=n_jobs, chunk_size=100, max_chunks_per_worker=8, progressbar=False)
        result = data.mapply(_fit_feature, result_type='expand')
    return result.reset_index(drop=True) \
                 .rename(index={
                    0: 'slope', 
                    1: 'intercept', 
                    2: 'rvalue', 
                    3: 'p-value', 
                    4: 'stderr', 
                    5: 'rse',
                    6: 'r2',
                    7: 's_res',
                    8: 'i_res',
                    9: 'r_res',
                    10: 'p_res'
                }).T