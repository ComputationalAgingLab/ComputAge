import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel

def EN_nan_row_predict(X: np.ndarray | pd.DataFrame, 
                       model: LinearModel) -> pd.DataFrame | np.ndarray:
    """
        This function allows a linear model to predict dataset with NaNs
        without any imputations.

        parameters:
            - X: array-like - dataset with NaNs.

            - model: sklearn base LinearModel class. Or any othe model having attributes
            `.coef_` and `.intercept_` for correct working.
    """
    nonanmask = ~np.isnan(X)
    return (model.coef_[nonanmask] * X[nonanmask]).sum() + model.intercept_

def introduce_nans(X: pd.DataFrame, 
                   p: float = 0.1) -> pd.DataFrame:
    """
        Randomly plcaes NaN value to the entry of given data `X` with 
        probability `p`.
    """
    X_nan = X.copy()
    mask = np.random.choice([True, False], size=X_nan.shape, p=[p, 1-p])
    X_nan = X_nan.mask(mask)   
    return X_nan
