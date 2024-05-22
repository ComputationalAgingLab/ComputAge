from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import pandas as pd

from .kdm import KlemeraDoubalEstimator
from .base import DeAgeBaseEstimator

class PLS1(DeAgeBaseEstimator):
    """
    The Great PLS1 Estimator!
    Implements PLS dimensionality reduction with different regression heads for 
    age prediction.
    
    parameters:
        head_estimator: str - ['linear', 'gpr', 'kdm', 'rf']

        n_components: {int, str} - number of PLS components. 'auto' - use auto for 
            seaching the best number of components based on the MAE of prediction.
        
        n_splits : int - number of cross validation splits for best components selection

        rel_criterion: float - relative error upon achieving which the search of 
            best number of components stops.

        random_state {int, None} - random state for cross validation shuffle split.

        TODO docstring for other parameters when they will be added
    """

    def __init__(self, 
                 n_components: int | str = 2, 
                 head_estimator: str = 'linear',
                 n_splits: int = 5,
                 rel_criterion: float = 1e-2,
                 random_state: int | None = None
                 ):
        super().__init__()
        
        self.head_estimator = head_estimator
        self.n_components = n_components
        self.n_splits = n_splits
        self.rel_criterion = rel_criterion
        self.random_state = random_state

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.DataFrame | list | np.ndarray | pd.Series) -> None:
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
        #explained variance - it is not expected to be large.
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = self.pls.x_scores_.var(ddof=1, axis=0) / total_var

        self.V = self.pls.x_weights_  #eigenvectors matrix
        self.L = self.pls.x_loadings_ #loadings matrix

        # Step2: fit regressor, in case of `linear`, the embedded to PLS linear regressor is used
        if self.head_estimator == 'linear':
            self.head = LinearRegression()
            self.head.fit(self.pls.transform(X), y)
        elif self.head_estimator == 'gpr':
            pass #TODO add fit of GPR
        elif self.head_estimator == 'kdm':
            self.head = KlemeraDoubalEstimator(
                                feature_selection_method='all',
                                feature_pval_threshold=0.05,
                                lasso_preselection=False,
                                weighing='rse')
            X_df = pd.DataFrame(self.pls.transform(X), 
                                columns=['pls' + str(i) for i in range(self.chosen_n_components)],
                                index=X.index
                                )
            self.head.fit(X_df, y)
        else:
            raise NotImplementedError

    def predict(self, 
                X: pd.DataFrame,) -> pd.Series:
        input_indices = X.index.copy()
        #check if dataset contains NaN
        if ~np.isnan(X).any(axis=0).any(): 
            X_df = pd.DataFrame(self.pls.transform(X), 
                    columns=['pls' + str(i) for i in range(self.chosen_n_components)],
                    index=X.index
                    )
            return self.head.predict(X_df)
        
        #an attempt to vectorize a prediction of NaN-enriched dataset
        X_scaled = (X - self.pls._x_mean) / self.pls._x_std
        mask = np.isfinite(X_scaled).to_numpy()[:, :, None]

        #saving and masking transforming matrices
        Vnew = np.repeat(self.V[None, :, :], X_scaled.shape[0], axis=0) * mask #[n x p x k]
        Lnew = np.repeat(self.L[None, :, :], X_scaled.shape[0], axis=0) * mask

        LVprod = np.matmul(np.transpose(Lnew, (0, 2, 1)), Vnew)
        LVinv = map(np.linalg.pinv, LVprod) #<- TODO: check if it is bad, performance bottleneck?
        LVinv = np.asarray(list(LVinv))
        Rnew = np.matmul(Vnew, LVinv) #construct a new rotation matrix

        X_scaled = X_scaled.fillna(0.).to_numpy()[:, :, np.newaxis]

        # resulting in a [n x k] matrix of transformed data
        Xp_scaled = np.sum(X_scaled * Rnew, axis=1)
        X_df = pd.DataFrame(Xp_scaled, 
                columns=['pls' + str(i) for i in range(self.chosen_n_components)],
                index=X.index
                )
        y_pred = self.head.predict(X_df)
        return pd.Series(data=y_pred, index=input_indices)
    
    def _search_best_n_components(self,
                                  X: pd.DataFrame, 
                                  y: pd.DataFrame | list | np.ndarray | pd.Series) -> int:
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
            best_n_components = n_components
            best_score = mean_score
        return best_n_components
