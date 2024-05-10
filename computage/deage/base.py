from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import os
import sys


#from util import get_model_file
scripts_working_directory = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(scripts_working_directory,'../models_library/raw_models/')
model_files = os.listdir(models_path)
model_mames = list(map(lambda a: a.replace('.csv','').lower(), model_files))

dict_model_names_paths = dict([(key, value)
          for i, (key, value) in enumerate(zip(model_mames, model_files))])
'''
{'epitoc2': 'EpiTOC2.csv', 'dunedinpoam': 'DunedinPoAm.csv', 
'mccartneyblood_2018_alcohol': 'McCartneyBlood_2018_Alcohol.csv', 
'mccartneyblood_2018_education': 'McCartneyBlood_2018_Education.csv', 
'linblood99cpg_2016': 'LinBlood99CpG_2016.csv', 'linblood3cpg_2016': 'LinBlood3CpG_2016.csv', 
'hannumlung_2013': 'HannumLung_2013.csv', 
'horvathmultishrunken_2013': 'HorvathMultiShrunken_2013.csv', 'hannumblood_2013': 'HannumBlood_2013.csv', 
'horvathmulti_2013': 'HorvathMulti_2013.csv', 'mccartneyblood_2018_bmi': 'McCartneyBlood_2018_BMI.csv', 
'mccartneyblood_2018_hdl': 'McCartneyBlood_2018_HDL.csv', 'mccartneyblood_2018_ldl': 'McCartneyBlood_2018_LDL.csv', 
'hannumbreast_2013': 'HannumBreast_2013.csv', 'knightblood_2016': 'KnightBlood_2016.csv', 'yingdamage': 'YingDamAge.csv', 
'mccartneyblood_2018_waisttipratio': 'McCartneyBlood_2018_WaistTipRatio.csv', 'hannumkidney_2013': 'HannumKidney_2013.csv',
 'mccartneyblood_2018_totalfat': 'McCartneyBlood_2018_TotalFat.csv', 'yingcausage': 'YingCausAge.csv', 
 'mccartneyblood_2018_tc': 'McCartneyBlood_2018_TC.csv', 'mccartneyblood_2018_smoking': 'McCartneyBlood_2018_Smoking.csv', 
 'yingadaptage': 'YingAdaptAge.csv', 'phenoage': 'PhenoAge.csv'}
'''


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
        self, name = 'phenoage', transform=None, preprocess=None) -> None:
        self.transform = transform
        self.name = name
        self.model_file_path = os.path.join(models_path,dict_model_names_paths[self.name])
        self.model_data = pd.read_csv(self.model_file_path)
        self.features = self.model_data[['Feature_ID']][1:]
        self.coefficients = np.array(self.model_data[['Coef']][1:])
        self.preprocess = preprocess

    def fit(self,X,y):
            self.is_fitted_ = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        X - датасет для распознавания, принимает в себя pandas 
        table with Feature1, F2, F3 для всех строк (на пересечениях циферки)
        """
        samples = list(X.columns)[1:]    
        X[samples] = X[samples].apply(pd.to_numeric)
        X_merged = X.merge(self.model_data.iloc[1:], left_on='ID_REF', right_on='Feature_ID', how='right')
        vectors = np.array(X_merged[samples].to_numpy())
        prediction = np.matmul( self.coefficients.transpose(), vectors)
        prediction += self.model_data.iloc[0,1]
        pd_prediction = pd.DataFrame()
        pd_prediction['sample'] = samples
        pd_prediction['prediction'] = prediction.transpose()

        return(pd_prediction)

class pickleModel(PublishedClocksBaseEstimator):
    def __init__(
        self, name = 'phenoage', transform=None, preprocess=None) -> None:
        self.transform = transform
        self.name = name
        self.model_file_path = os.path.join(models_path,dict_model_names_paths[self.name])
        self.model_data = pd.read_csv(self.model_file_path)
        self.features = self.model_data[['Feature_ID']][1:]
        self.coefficients = np.array(self.model_data[['Coef']][1:])
        self.preprocess = preprocess


