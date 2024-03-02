import numpy as np

def EN_nan_row_predict(x, model):
    nonanmask = ~np.isnan(x)
    return (model.coef_[nonanmask] * x[nonanmask]).sum() + model.intercept_

def introduce_nans(X, p: float = 0.1):
    X_nan = X.copy()
    mask = np.random.choice([True, False], size=X_nan.shape, p=[p, 1-p])
    X_nan = X_nan.mask(mask)   
    return X_nan