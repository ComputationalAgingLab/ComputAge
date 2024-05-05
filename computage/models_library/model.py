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
        self.model_meta = model_definitions[name]
        self.model_params = model_definitions[name]['model']
        assert self.model_params['type'] == 'LinearMethylationModel', "Model is not Linear Methylation Model!"
        self.coefficients = pd.read_csv(get_clock_file(self.model_params['file']), index_col=0)

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




class GrimAgeModel(PublishedClocksBaseEstimator):
    def __init__(self, 
                 name, 
                 imputation=None, 
                 transform=None, 
                 preprocess=None,
                 meta_imputation=None, 
                ) -> None:
        self.name = name
        self.imputation = imputation
        self.transform = transform
        self.preprocess = preprocess
        self.meta_imputation = meta_imputation

        #load the model
        self.model_meta = model_definitions[name]
        self.model_params = model_definitions[name]['model']
        assert self.model_params['type'] == 'GrimAgeModel', "Model is not GrimAge Model!"
        self.coefficients = pd.read_csv(get_clock_file(self.model_params['file']), index_col=0)    
        self.cpgs = self.get_methylation_sites()

        #use params defined in model metadata if not given in the init
        if self.imputation is None:
            self.imputation = self.model_params.get('default_imputation', "sesame_450k")
        if self.imputation == 'sesame_450k':
            ivalues = pd.read_csv(get_clock_file("sesame_450k_median.csv"), index_col=0)
            self.imputation_values = ivalues.loc[self.cpgs]

        if self.transform is None:
            self.transform =  self.model_params.get("transform", identity)
            
        if self.preprocess is None:
            self.preprocess = self.model_params.get("preprocess", identity)


    def predict(self, dnam, meta):
        if self.imputation == 'none':
            X_ = dnam.reindex(columns=self.cpgs, fill_value=0.)
        elif self.imputation == 'sesame_450k':
            X_ = dnam.reindex(columns=self.cpgs).fillna(self.imputation_values['median'])
        elif self.imputation == 'average':
            X_ = dnam.reindex(columns=self.cpgs)
            averages = X_.mean(axis=0).fillna(0.) #fill with 0 if no values in a column
            X_ = X_.fillna(averages)

        # Add metadata rows to dnam DataFrame
        meta_ = meta.copy()
        if 'Gender' not in meta_.columns:
            meta_['Gender'] = 'U'
        df = X_.copy()
        df["Age"] = meta_["Age"]
        df["Intercept"] = 1.

        grouped = self.coefficients.groupby("Y.pred")
        all_data = pd.DataFrame()
        for name, group in grouped:
            if name == "COX":
                cox_coefficients = group.set_index("var")["beta"]
            elif name == "transform":
                transform = group.set_index("var")["beta"]
                m_age = transform["m_age"]
                sd_age = transform["sd_age"]
                m_cox = transform["m_cox"]
                sd_cox = transform["sd_cox"]
            else:
                sub_clock_result = self.calculate_sub_clock(df, group)
                all_data[name] = sub_clock_result

        all_data["Age"] = meta_["Age"]
        all_data["Female"] = meta_["Gender"].map({'M':0., 'F':1., 'U':0.5})
        all_data["COX"] = all_data.mul(cox_coefficients, axis=1).sum(axis=1)
        
        age_key = "DNAmGrimAge"
        accel_key = "AgeAccelGrim"
        # Calculate DNAmGrimAge
        Y = (all_data["COX"] - m_cox) / sd_cox
        result = (Y * sd_age) + m_age
        
        # Calculate AgeAccelGrim - this is a total antistatistical bullshit
        # lm = LinearRegression().fit(
        #     all_data[["Age"]].values, all_data[age_key].values
        # )
        # predictions = lm.predict(all_data[["Age"]].values)
        # all_data['Predictions'] = predictions
        # all_data[accel_key] = all_data[age_key] - predictions

        return result
    
    def calculate_sub_clock(self, X, coefficients):
        # Filter coefficients for only those present in df
        relevant_coefficients = coefficients[coefficients["var"].isin(X.columns)]

        # Create a Series from the relevant coefficients, indexed by 'var'
        coefficients_series = relevant_coefficients.set_index("var")["beta"]

        # Align coefficients with df's rows and multiply, then sum across CpG sites for each sample
        result = X[coefficients_series.index].\
                multiply(coefficients_series, axis=1).\
                sum(axis=1)

        return result


    def get_methylation_sites(self):
        filtered_df = self.coefficients[
            ~self.coefficients.index.isin(["COX", "transform"])
        ]
        unique_vars = set(filtered_df["var"]) - {"Intercept", "Age", "Female"}
        return list(unique_vars)
