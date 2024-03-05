from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import pandas as pd
from typing import Union, List

from .base import DeAgeBaseEstimator

class DeAge(DeAgeBaseEstimator):
    """
    The Great DeAge Estimator!
    Implements PLS dimensionality reduction with different regression heads for 
    age prediction.
    
    n_components: {int, str} - number of PLS components. 'auto' - use auto for 
        seaching the best number of components based on the MAE of prediction.

    head_estimator: str - ['linear', 'gpr', 'kdm', 'rf']

    TODO docstring
    """

    def __init__(self, 
                 n_components: Union[int, str] = 2, 
                 head_estimator: str = 'linear',
                 n_splits: int = 5,
                 rel_criterion: float = 1e-2,
                 random_state: Union[int, None] = None
                 ):
        super().__init__()
        
        self.head_estimator = head_estimator
        self.n_components = n_components
        self.n_splits = n_splits
        self.rel_criterion = rel_criterion
        self.random_state = random_state

    def fit(self, 
            X: pd.DataFrame, 
            y: Union[pd.DataFrame, list, np.ndarray, pd.Series]) -> None:
        #store feature names
        self.train_features = X.columns

        if self.n_components == 'auto':
            print('Start searching for the best number of PLS components.')
            self.chosen_n_components = self._search_best_n_components(X, y)
            print(f'The best number of components is {self.chosen_n_components}')
        else:
            self.chosen_n_components = self.n_components
        
        # Step1: fit projector
        self.pls = PLSRegression(n_components=self.chosen_n_components, scale=True)
        self.pls.fit(X, y)

        self.V = self.pls.x_weights_  #eigenvectors matrix
        self.L = self.pls.x_loadings_ #loadings matrix

        # Step2: fit regressor, in case of `linear`, the embedded to PLS linear regressor is used
        if self.head_estimator == 'linear':
            self.head = LinearRegression()
            self.head.fit(self.pls.transform(X), y)
        elif self.head_estimator == 'gpr':
            pass #TODO add fit of GPR
        else:
            raise NotImplementedError

    def predict(self, 
                X: pd.DataFrame,) -> pd.Series:
        input_indices = X.index.copy()
        #check if dataset contains NaN
        if ~np.isnan(X).any(axis=0).any(): 
            X_ = self.pls.transform(X)
            return self.head.predict(X_)
        else:
            #an attempt to vectorize a prediction of NaN-enriched dataset
            X_scaled = (X - self.pls._x_mean) / self.pls._x_std
            mask = np.isfinite(X_scaled).to_numpy()[:, :, None]

            #saving and masking transforming matrices
            Vnew = np.repeat(self.V[None, :, :], X_scaled.shape[0], axis=0) * mask #[n x p x k]
            Lnew = np.repeat(self.L[None, :, :], X_scaled.shape[0], axis=0) * mask

            LVprod = np.matmul(np.transpose(Lnew, (0, 2, 1)), Vnew)
            LVinv = map(lambda n: np.linalg.pinv(n), LVprod) #<- TODO: check if it is bad, performance bottleneck?
            LVinv = np.asarray(list(LVinv))
            Rnew = np.matmul(Vnew, LVinv) #construct a new rotation matrix

            X_scaled = X_scaled.fillna(0.).to_numpy()[:, :, np.newaxis]

            # resulting in a [n x k] matrix of transformed data
            Xp_scaled = np.sum(X_scaled * Rnew, axis=1)
            y_pred = self.head.predict(Xp_scaled)
            return pd.Series(data=y_pred, index=input_indices)
    
    def _search_best_n_components(self,
                                  X: pd.DataFrame, 
                                  y: Union[pd.DataFrame, list, np.ndarray, pd.Series]) -> int:
        best_score = 0
        best_n_components = 1
        cv = ShuffleSplit(n_splits=self.n_splits, random_state=self.random_state)
        for n_components in range(1, X.shape[1]):
            _model = PLSRegression(n_components)
            cv_scores = cross_val_score(_model, X, y, scoring='neg_mean_absolute_error', cv=cv)
            mean_score = np.mean(cv_scores)
            new_criterion = np.abs((mean_score - best_score) / best_score)
            if new_criterion < self.rel_criterion:
                break
            else:
                best_n_components = n_components
                best_score = mean_score
        return best_n_components
        