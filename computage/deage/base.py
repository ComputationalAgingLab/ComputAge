from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import warnings
import numpy as np
import pandas as pd
import pickle

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

    def map_features(self, 
                     assembly: str = 'all'):
        pass

    @abstractmethod
    def predict(self, 
                X: pd.DataFrame | np.ndarray):
        ...


class DeAgeBaseEstimator(PublishedClocksBaseEstimator, ABC):    
    """
    Basic class of DeAge
    The key difference of DeAge-based model is their capability of working
    with NaN-enriched data
    """

    def _validate_data(self, 
                       X: pd.DataFrame | np.ndarray, 
                       y: pd.Series | np.ndarray
                       ):
        # always need y to be without NaN values
        assert ~np.any(np.isnan(y)), "Input y contains NaN"

        # X is allowed to contain NaN values, but each column should contain at least one non-Nan value
        if not isinstance(X, pd.DataFrame):
            warnings.warn('X is not pandas.DataFrame type. Arbitrary feature names will be assigned!')
        assert np.all(~np.all(np.isnan(X), 0)), "Input X contains columns with NaN only"
        assert np.all(~np.all(np.isnan(X), 1)), "Input X contains rows with NaN only"

        return None
    
    @abstractmethod
    def fit(self,
            X: pd.DataFrame | np.ndarray, 
            y: pd.Series | np.ndarray
            ):
        ...

    def save_model(self, path):
        pickle.dump(self, open(path, 'wb'))