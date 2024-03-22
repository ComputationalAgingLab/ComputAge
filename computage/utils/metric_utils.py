from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

def print_metrics(y_true: pd.Series | np.ndarray, 
                  y_pred: pd.Series | np.ndarray, 
                  return_result: bool = False
                  ) -> None | tuple[float, ...]:
    """
        Shortcut for printing all necessary metrics for aging clocks validation.
        Currently computes the following metrics:
            - MAE - mean absolute error
            - R2 - coefficient of determination
            - r - Pearson correlation coefficient
            - pval_r - p-value for Pearson corr. coef.

        parameters:
            - y_true: array-like - true values (ages)

            - y_pred: array-like - predicted values (ages)

            - return_result: bool - if `True` returns all calculated metrics as a tuple
                of four elements.

        returns:
            None or (MAE, R2, r, pval_r)
    """

    r2 = r2_score(y_true, y_pred)
    r, pval_r = pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"MAE = {mae:.3f}")
    print(f"R2 = {r2:.3f}")
    print(f"r = {r:.3f}")
    if return_result:
        return (mae, r2, r, pval_r)
