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

def impute_from_average(dnam, cpgs_to_impute=None):
    """
    Impute all missing values in a DNA methylation dataset using the average from the dataset itself.

    Args:
        dnam (pd.DataFrame): DataFrame with samples as columns and CpG sites as rows.
        cpgs_to_impute (list of str, optional): List of CpG sites to impute.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    if cpgs_to_impute is None: 
        X_filled = dnam.where(dnam.notna(), dnam.mean(axis=1), axis=0)
    else:
        X_filled = dnam.copy()
        for cpg in cpgs_to_impute:
            for null_a in dnam.loc[cpg].isnull():
                if null_a:
                    X_filled.loc[cpg] =  dnam.loc[cpg].mean()    

    return X_filled

### IMPUTATION implementation from biolearn: https://github.com/bio-learn/biolearn/blob/master/ ###
### TODO: rewrite this function
'''
def hybrid_impute(dnam, cpg_source, required_cpgs, threshold=0.8):
    """
    Imputes missing values in a DNA methylation dataset based on a threshold. 
    Sites with data below the threshold are replaced from an external source, 
    while others are imputed using the average of existing values.

    Args:
        dnam (pd.DataFrame): DataFrame with samples as columns and CpG sites as rows.
        cpg_source (pd.Series): Series containing reference values for CpG sites.
        required_cpgs (list of str): List of CpG sites that need to be in the final dataset.
        threshold (float, optional): Threshold for determining imputation strategy. Default is 0.8.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.

    Raises:
        ValueError: If certain required CpG sites are missing from both the dataset and the cpg_source.
    """
    # Drop rows below the threshold, these will be replaced entirely from cpg_source
    cpgs_below_threshold = dnam.notna().mean(axis=1) < threshold
    dnam = dnam.drop(dnam[cpgs_below_threshold].index)

    # Impute remaining rows using impute_from_average
    df_filled = impute_from_average(dnam)

    missing_cpgs_from_dataset = set(required_cpgs) - set(df_filled.index)
    missing_cpgs_from_source = [
        cpg for cpg in missing_cpgs_from_dataset if cpg not in cpg_source
    ]

    if missing_cpgs_from_source:
        raise ValueError(
            f"Tried to fill the following cpgs but they were missing from cpg_source: {missing_cpgs_from_source}"
        )

    for cpg in missing_cpgs_from_dataset:
        df_filled.loc[cpg] = cpg_source.loc[cpg]

    return df_filled.sort_index()
'''