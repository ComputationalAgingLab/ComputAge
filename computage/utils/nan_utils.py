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

### IMPUTATION implementation from biolearn: https://github.com/bio-learn/biolearn/blob/master/ ###
def impute_from_standard(X, cpg_averages, cpgs_to_impute=None):
    """
    Impute all missing values in a DNA methylation dataset using the averages from an external dataset.

    Args:
        dnam (pd.DataFrame): DataFrame with samples as columns and CpG sites as rows.
        cpg_averages (pd.Series): Series containing reference averages for CpG sites.
        cpgs_to_impute (list of str, optional): List of CpG sites to impute. Missing cpgs will only be imputed if in this list.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    if cpgs_to_impute:
        impute_rows = X.loc[cpgs_to_impute]
        impute_rows = impute_rows.apply(lambda col: col.fillna(cpg_averages))
        df_filled = X.combine_first(impute_rows)
    else:
        df_filled = X.apply(lambda col: col.fillna(cpg_averages))
    return df_filled


def impute_from_average(dnam, cpgs_to_impute=None):
    """
    Impute all missing values in a DNA methylation dataset using the average from the dataset itself.

    Args:
        dnam (pd.DataFrame): DataFrame with samples as columns and CpG sites as rows.
        cpgs_to_impute (list of str, optional): List of CpG sites to impute.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    # Protect input from mutation
    dnam_copy = dnam.copy()
    means = dnam_copy.mean(axis=1)

    if cpgs_to_impute:
        # Filter out non-existent CpG sites
        existing_cpgs = [
            cpg for cpg in cpgs_to_impute if cpg in dnam_copy.index
        ]

        # Apply imputation only to existing CpG sites
        mask = dnam_copy.loc[existing_cpgs].isna()
        dnam_copy.loc[existing_cpgs] = dnam_copy.loc[existing_cpgs].where(
            ~mask, means[existing_cpgs], axis=0
        )
    else:
        dnam_copy = dnam_copy.where(dnam_copy.notna(), means, axis=0)

    return dnam_copy


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