import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from computage.settings import ROOTDIR
from computage.deage.base import PublishedClocksBaseEstimator
from computage.models_library.definitions import *

class LinearMethylationModel(PublishedClocksBaseEstimator):
    def __init__(self, 
                 name, 
                 imputation=None, 
                 transform=None, 
                 preprocess=None, 
                ) -> None:
        self.imputation = imputation
        self.name = name
        #self.transform = transform
        self.preprocess = preprocess
        

        #load the model
        self.model_file_path = get_clock_file(self.name)
        self.model_data = pd.read_csv(self.model_file_path)
        features = self.model_data[['Feature_ID']][1:]
        self.features = features['Feature_ID'].values
        self.coefficients = np.array(self.model_data[['Coef']][1:])
        self.preprocess = preprocess

        self.intercept = self.model_data.iloc[0,1]

        #if self.imputation == 'sesame_450k':
        ivalues = pd.read_csv(os.path.join(ROOTDIR, "models_library/raw_models/sesame_450k_median.csv"), index_col=0)
        self.imputation_values = ivalues.loc[self.features]

        #self.coefficients = pd.read_csv(get_clock_file(self.model_params['file']), index_col=0)

        #use params defined in model metadata if not given in the init
        # if self.imputation is None:
        #     self.imputation = self.model_params.get('default_imputation', "sesame_450k")
        

        # if self.transform is None:
        #     self.transform =  self.model_params.get("transform", identity)
            
        # if self.preprocess is None:
        #     self.preprocess = self.model_params.get("preprocess", identity)

    def introduce_nans():
        pass

    def fit():
        is_fitted = True
        return(is_fitted)

    def predict(self, 
                X: pd.DataFrame, imputation = None
                ) -> pd.Series:
        if self.imputation == 'none' or self.imputation is None:
            X_ = X.reindex(columns=self.features).fillna(0)
        elif self.imputation == 'sesame_450k':
            X_ = X.reindex(columns=self.features).fillna(self.imputation_values['median'])
        elif self.imputation == 'average':
            X_ = X.reindex(columns=self.features)
            averages = X_.mean(axis=0).fillna(0.) #fill with 0 if no values in a column
            X_ = X_.fillna(averages)
        
        #preprocess block (currently we don't have preprocess functions)
        #X_ = self.preprocess(X_)
        
        # Vectorized multiplication: multiply CoefficientTraining with all columns of dnam_data
        #wsum = X_.multiply(self.coefficients).sum(axis=1)
        if isinstance(X_, np.ndarray):
            wsum = np.matmul( X_, self.coefficients)
        else:
            #print(type(X_.values))
            #print(type(X_.values))
            #print(X_.shape)
            #print(self.coefficients.shape)
            wsum = np.matmul( np.array(X_.values), self.coefficients.astype(float))     
        
        wsum += self.intercept

        # Return as a DataFrame
        return wsum

    def get_methylation_sites(self):
        return self.features