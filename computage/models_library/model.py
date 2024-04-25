import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# from biolearn.dunedin_pace import dunedin_pace_normalization
from computage.deage.base import PublishedClocksBaseEstimator
from computage.settings import ROOTDIR
import os

#function for clock file retrieval
def get_clock_file(filename):
    clock_file_path = os.path.join(ROOTDIR, "models_library/clocks", filename)  
    return clock_file_path

#Horvath-specific ELU-like transform
def anti_trafo(x, adult_age=20):
    y = np.where(
        x < 0, (1 + adult_age) * np.exp(x) - 1, (1 + adult_age) * x + adult_age
    )
    return y

#identity transform if not given other
def identity(x):
    return x

### model definitions are taken from biolearn: https://github.com/bio-learn/biolearn/blob/master/ ###

model_definitions = {
    "Horvathv1": {
        "year": 2013,
        "species": "Human",
        "tissue": "Multi-tissue",
        "source": "https://genomebiology.biomedcentral.com/articles/10.1186/gb-2013-14-10-r115",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Horvath1.csv",
            "transform": lambda sum: anti_trafo(sum + 0.696),
        },
    },
    "Hannum": {
        "year": 2013,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.sciencedirect.com/science/article/pii/S1097276512008933",
        "output": "Age (Years)",
        "model": {"type": "LinearMethylationModel", "file": "Hannum.csv"},
    },
    "Lin": {
        "year": 2016,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.aging-us.com/article/100908/text",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Lin.csv",
            "transform": lambda sum: sum + 12.2169841,
        },
    },
    "PhenoAge": {
        "year": 2018,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.aging-us.com/article/101414/text",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "PhenoAge.csv",
            "transform": lambda sum: sum + 60.664,
        },
    },
    "YingCausAge": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.biorxiv.org/content/10.1101/2022.10.07.511382v2",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "YingCausAge.csv",
            "transform": lambda sum: sum + 86.80816381,
        },
    },
    "YingDamAge": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.biorxiv.org/content/10.1101/2022.10.07.511382v2",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "YingDamAge.csv",
            "transform": lambda sum: sum + 543.4315887,
        },
    },
    "YingAdaptAge": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.biorxiv.org/content/10.1101/2022.10.07.511382v2",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "YingAdaptAge.csv",
            "transform": lambda sum: sum - 511.9742762,
        },
    },
    "Horvathv2": {
        "year": 2018,
        "species": "Human",
        "tissue": "Skin + blood",
        "source": "https://www.aging-us.com/article/101508/text",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Horvath2.csv",
            "transform": lambda sum: anti_trafo(sum - 0.447119319),
        },
    },
    "PEDBE": {
        "year": 2019,
        "species": "Human",
        "tissue": "Buccal",
        "source": "https://www.pnas.org/doi/10.1073/pnas.1820843116",
        "output": "Age (Years)",
        "model": {
            "type": "LinearMethylationModel",
            "file": "PEDBE.csv",
            "transform": lambda sum: anti_trafo(sum - 2.1),
        },
    },
    "Zhang_10": {
        "year": 2019,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.nature.com/articles/ncomms14617",
        "output": "Mortality Risk",
        "model": {"type": "LinearMethylationModel", "file": "Zhang_10.csv"},
    },
    # "DunedinPoAm38": {
    #     "year": 2020,
    #     "species": "Human",
    #     "tissue": "Blood",
    #     "source": "https://elifesciences.org/articles/54870#s2",
    #     "output": "Aging Rate (Years/Year)",
    #     "model": {
    #         "type": "LinearMethylationModel",
    #         "file": "DunedinPoAm38.csv",
    #         "transform": lambda sum: sum - 0.06929805,
    #     },
    # },
    # "DunedinPACE": {
    #     "year": 2022,
    #     "species": "Human",
    #     "tissue": "Blood",
    #     "source": "https://www.proquest.com/docview/2634411178",
    #     "output": "Aging Rate (Years/Year)",
    #     "model": {
    #         "type": "LinearMethylationModel",
    #         "file": "DunedinPACE.csv",
    #         "transform": lambda sum: sum - 1.949859,
    #         "preprocess": dunedin_pace_normalization,
    #         "default_imputation": "none",
    #     },
    # },
    "GrimAgeV1": {
        "year": 2019,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6366976/",
        "output": "Mortality Adjusted Age (Years)",
        "model": {"type": "GrimageModel", "file": "GrimAgeV1.csv"},
    },
    "GrimAgeV2": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "source": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9792204/",
        "output": "Mortality Adjusted Age (Years)",
        "model": {"type": "GrimageModel", "file": "GrimAgeV2.csv"},
    },
    "DNAmTL": {
        "year": 2019,
        "species": "Human",
        "tissue": "Blood, Adipose",
        "source": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6738410/",
        "output": "Telomere Length",
        "model": {
            "type": "LinearMethylationModel",
            "file": "DNAmTL.csv",
            "transform": lambda sum: sum - 7.924780053,
        },
    },
    "HRSInCHPhenoAge": {
        "year": 2022,
        "species": "Human",
        "tissue": "Blood",
        "output": "Age (Years)",
        "source": "https://www.nature.com/articles/s43587-022-00248-2",
        "model": {
            "type": "LinearMethylationModel",
            "file": "HRSInCHPhenoAge.csv",
            "transform": lambda sum: sum + 52.8334080,
        },
    },
    "Knight": {
        "year": 2016,
        "species": "Human",
        "tissue": "Cord Blood",
        "source": "https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1068-z",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "Knight.csv",
            "transform": lambda sum: sum + 41.7,
        },
    },
    "LeeControl": {
        "year": 2019,
        "species": "Human",
        "tissue": "Placenta",
        "source": "https://www.aging-us.com/article/102049/text",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "LeeControl.csv",
            "transform": lambda sum: sum + 13.06182,
        },
    },
    "LeeRefinedRobust": {
        "year": 2019,
        "species": "Human",
        "tissue": "Placenta",
        "source": "https://www.aging-us.com/article/102049/text",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "LeeRefinedRobust.csv",
            "transform": lambda sum: sum + 30.74966,
        },
    },
    "LeeRobust": {
        "year": 2019,
        "species": "Human",
        "tissue": "Placenta",
        "source": "https://www.aging-us.com/article/102049/text",
        "output": "Gestational Age",
        "model": {
            "type": "LinearMethylationModel",
            "file": "LeeRobust.csv",
            "transform": lambda sum: sum + 24.99772,
        },
    },
}

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




### TODO: rewrite this class
# class GrimageModel:
#     def __init__(self, coef_file, **metadata):
#         self.coefficients = pd.read_csv(
#             get_clock_file(coef_file), index_col=0
#         )
#         self.metadata = metadata

#     @classmethod
#     def from_definition(cls, clock_definition):
#         model_def = clock_definition["model"]
#         return cls(
#             model_def["file"],
#             **{k: v for k, v in clock_definition.items() if k != "model"},
#         )

#     def predict(self, geo_data):
#         if "sex" not in geo_data.metadata or "age" not in geo_data.metadata:
#             raise ValueError("Metadata must contain 'sex' and 'age' columns")

#         df = geo_data.dnam

#         # Transposing metadata so that its structure aligns with dnam (columns as samples)
#         transposed_metadata = geo_data.metadata.transpose()

#         # Add metadata rows to dnam DataFrame
#         df.loc["Age"] = transposed_metadata.loc["age"]
#         df.loc["Female"] = transposed_metadata.loc["sex"].apply(
#             lambda x: 1 if x == 1 else 0
#         )
#         df.loc["Intercept"] = 1

#         grouped = self.coefficients.groupby("Y.pred")
#         all_data = pd.DataFrame()

#         for name, group in grouped:
#             if name == "COX":
#                 cox_coefficients = group.set_index("var")["beta"]
#             elif name == "transform":
#                 transform = group.set_index("var")["beta"]
#                 m_age = transform["m_age"]
#                 sd_age = transform["sd_age"]
#                 m_cox = transform["m_cox"]
#                 sd_cox = transform["sd_cox"]
#             else:
#                 sub_clock_result = self.calculate_sub_clock(df, group)
#                 all_data[name] = sub_clock_result

#         all_data["Age"] = geo_data.metadata["age"]
#         all_data["Female"] = geo_data.metadata["sex"].apply(
#             lambda x: 1 if x == 1 else 0
#         )

#         all_data["COX"] = all_data.mul(cox_coefficients).sum(axis=1)
#         age_key = "DNAmGrimAge"
#         accel_key = "AgeAccelGrim"
#         # Calculate DNAmGrimAge
#         Y = (all_data["COX"] - m_cox) / sd_cox
#         all_data[age_key] = (Y * sd_age) + m_age

#         # Calculate AgeAccelGrim
#         lm = LinearRegression().fit(
#             all_data[["Age"]].values, all_data[age_key].values
#         )
#         predictions = lm.predict(all_data[["Age"]].values)
#         all_data[accel_key] = all_data[age_key] - predictions

#         # Drop COX column after computations
#         all_data.drop("COX", axis=1, inplace=True)

#         return all_data

#     def calculate_sub_clock(self, df, coefficients):
#         # Filter coefficients for only those present in df
#         relevant_coefficients = coefficients[
#             coefficients["var"].isin(df.index)
#         ]

#         # Create a Series from the relevant coefficients, indexed by 'var'
#         coefficients_series = relevant_coefficients.set_index("var")["beta"]

#         # Align coefficients with df's rows and multiply, then sum across CpG sites for each sample
#         result = (
#             df.loc[coefficients_series.index]
#             .multiply(coefficients_series, axis=0)
#             .sum()
#         )

#         return result

#     def rename_columns(self, data, old_names, new_names):
#         for old_name, new_name in zip(old_names, new_names):
#             data.rename(columns={old_name: new_name}, inplace=True)

#     def methylation_sites(self):
#         filtered_df = self.coefficients[
#             ~self.coefficients.index.isin(["COX", "transform"])
#         ]
#         unique_vars = set(filtered_df["var"]) - {"Intercept", "Age", "Female"}
#         return list(unique_vars)
