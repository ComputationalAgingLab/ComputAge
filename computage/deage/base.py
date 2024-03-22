from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
#from util import get_model_file

class PublishedClocksBaseEstimator(BaseEstimator, ABC):
    """
    Basic class providing better interface for working with published aging clocks.
    The key features are:
    - better textual and HTML representation displayed in terminals and IDEs
      including number of parameters and reported accuracies according to papers;
    - interpretation of features;
    - possibility of mapping features to another conventional annotation, e.g. cgXXX.. -> chrN:coord1-coord2;
    """    

    def features_interpretation(self):
        pass

    def map_features(self, assembly='all'):
        pass

    @abstractmethod
    def fit(self, X, y):
        ...

    @abstractmethod
    def predict(self, X):
        ...





class DeAgeBaseEstimator(PublishedClocksBaseEstimator, ABC):    
    """
    Basic class of DeAge
    The key difference of DeAge-based model is their capability of working
    with NaN-enriched data
    """
    # https://docs.python.org/3/library/abc.html
    # @abstractmethod
    # def fit(self, X, y):
    #     pass

    def _validate_data(self, X, y):
        # always need y to be without NaN values
        assert ~np.any(np.isnan(y)), "Input y contains NaN"

        # X is allowed to contain NaN values, but each column should contain at least one non-Nan value
        assert isinstance(X, pd.DataFrame), "Input X should be a pandas.DataFrame"
        assert np.all(~np.all(np.isnan(X), 0)), "Input X contains columns with NaN only"
        assert np.all(~np.all(np.isnan(X), 1)), "Input X contains rows with NaN only"

        return None
    
    @abstractmethod
    def predict(self, X, y):
        ...
    
class LinearMethylationModel(PublishedClocksBaseEstimator):
    def __init__(
        self, model_file_path, transform, preprocess=None) -> None:
        self.transform = transform
        self.model_file_path = model_file_path
        self.model_data = pd.read_csv(self.model_file_path)
        self.features = self.model_data[['Feature_ID']]
        self.coefficients = self.model_data[['Coef']]
        self.preprocess = preprocess

    def fit(self,X,y):
            self.is_fitted_ = True

    def predict(self, X):
        """
        X - датасет для распознавания, принимает в себя pandas 
        table with Feature1, F2, F3 для всех строк (на пересечениях циферки)
        """    

        pass

