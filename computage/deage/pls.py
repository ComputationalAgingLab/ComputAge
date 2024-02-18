from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import numpy as np

from .base import DeAgeBaseEstimator

class DeAge(DeAgeBaseEstimator):
    """
    The Great DeAge Estimator!
    Implements PLS dimensionality reduction with different regression heads for 
    age prediction.
    
    n_components: {int, str} - number of PLS components. 'auto' - use auto for 
        seaching the best number of components based on the MAE of prediction.

    head_estimator: str - ['linear', 'gpr', 'kdm']
    """

    def __init__(self, n_components=2, head_estimator:str='linear'):
        super().__init__()
        
        self.head_estimator = head_estimator
        self.n_components = n_components

    def fit(self, X, y):
        #store feature names
        self.train_features = X.columns

        if self.n_components == 'auto':
            self.chosen_n_components = self.search_best_n_components(self, X, y)
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

    def predict(self, X):
        #check if dataset contains NaN
        if ~np.isnan(X).any(axis=0).any(): 
            X_ = self.pls.transform(X)
            return self.head.predict(X_)
        else:
            #an attempt to vectorize a prediction of NaN-enriched dataset
            X_scaled = (X - self.pls._x_mean) / self.pls._x_std
            mask = np.isfinite(X_scaled).to_numpy()[:, :, None]

            Vnew = np.repeat(self.V[None, :, :], X_scaled.shape[0], axis=0) * mask #[n x p x k]
            Lnew = np.repeat(self.L[None, :, :], X_scaled.shape[0], axis=0) * mask

            LVprod = np.matmul(np.transpose(Lnew, (0, 2, 1)), Vnew)
            LVinv = map(lambda n: np.linalg.pinv(n), LVprod) #<- is it bad, performance bottleneck?
            LVinv = np.asarray(list(LVinv))
            Rnew = np.matmul(Vnew, LVinv) #construct a new rotation matrix

            X_scaled = X_scaled.fillna(0.).to_numpy()[:, :, np.newaxis]

            # resulting in a [n x k] matrix
            Xp_scaled = np.sum(X_scaled * Rnew, axis=1)
            return self.head.predict(Xp_scaled)
    
    def search_best_n_components(self, X, y):
        # TODO
        return 1