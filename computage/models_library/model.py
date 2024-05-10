import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from computage.deage.base import PublishedClocksBaseEstimator
from computage.models_library.definitions import *

class LinearMethylationModel(PublishedClocksBaseEstimator):
    def __init__(self, 
                 name, 
                 imputation=None, 
                 transform=None, 
                 preprocess=None, 
                ) -> None:
        self.name = name
        self.imputation = imputation
        self.transform = transform
        self.preprocess = preprocess

        #load the model
        self.model_file_path = get_clock_file(self.file)
        self.model_data = pd.read_csv(self.model_file_path)
        self.features = self.model_data[['Feature_ID']][1:]
        self.coefficients = np.array(self.model_data[['Coef']][1:])
        self.preprocess = preprocess

        #self.coefficients = pd.read_csv(get_clock_file(self.model_params['file']), index_col=0)

        #use params defined in model metadata if not given in the init
        if self.imputation is None:
            self.imputation = self.model_params.get('default_imputation', "sesame_450k")
        if self.imputation == 'sesame_450k':
            ivalues = pd.read_csv(get_clock_file("sesame_450k_median.csv"), index_col=0)
            self.imputation_values = ivalues.loc[self.coefficients.index]

        if self.transform is None:
            self.transform =  self.model_params.get("transform", identity)
            
        if self.preprocess is None:
            self.preprocess = self.model_params.get("preprocess", identity)

    def introduce_nans():
        pass

    def predict(self, 
                X: pd.DataFrame
                ) -> pd.Series:
        if self.imputation == 'none':
            X_ = X.reindex(columns=self.coefficients.index, fill_value=0.)
        elif self.imputation == 'sesame_450k':
            X_ = X.reindex(columns=self.coefficients.index).fillna(self.imputation_values['median'])
        elif self.imputation == 'average':
            X_ = X.reindex(columns=self.coefficients.index)
            averages = X_.mean(axis=0).fillna(0.) #fill with 0 if no values in a column
            X_ = X_.fillna(averages)
        
        #preprocess block (currently we don't have preprocess functions)
        X_ = self.preprocess(X_)
        
        # Vectorized multiplication: multiply CoefficientTraining with all columns of dnam_data
        wsum = X_.multiply(self.coefficients['CoefficientTraining']).sum(axis=1)

        # Return as a DataFrame
        return wsum.apply(self.transform)

    def get_methylation_sites(self):
        return list(self.coefficients.index)