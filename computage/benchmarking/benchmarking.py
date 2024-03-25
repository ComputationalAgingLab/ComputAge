import os
from os.path import exists
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

class EpiClocksBenchmarking:
    def __init__(self, 
                 models_config: dict,
                 datasets_config: dict,
                 experiment_prefix: str = 'test', 
                 correction_threshold: float = 0.05,
                 save_results: bool = True,
                 output_folder: str ='./bench_results', 
                 low_memory: bool = True,
                 verbose: int = 1,
                 ) -> None:
        """
        Docstring ...
        """
        self.models_config = models_config
        self.datasets_config = datasets_config
        self.n_datasets = len(datasets_config)
        self.n_models = len(models_config['in_library']) + len(models_config['new_models'])
        self.low_memory = low_memory
        self.save_results = save_results
        self.output_folder = output_folder
        self.experiment_prefix = experiment_prefix
        self.correction_threshold = correction_threshold
        self.verbose = verbose
        print(f'{self.n_models} models will be tested on {self.n_datasets} datasets.')
        
    
    @staticmethod
    def check_folder(path):
        if not exists(path):
            os.mkdir(path)

    def check_existence(self):
        """Check existense of all datasets in config."""
        for gse, conf in self.datasets_config.items():
            path, _, _ = conf.values()
            assert exists(path), f'A dataset file {gse} does not exist at the path!'

    def download_datasets(self):
        pass

    def run(self) -> None:
        #preparation
        self.check_folder(self.output_folder)
        self.check_existence()

        #go!
        self.bench_results_AAP = pd.DataFrame()
        self.bench_results_AA0 = pd.DataFrame()
        self.datasets_predictions = {}

        ###Predict datasets and gather predictions
        for gse, conf in tqdm(self.datasets_config.items(), 
                              total=self.n_datasets,
                              desc='Datasets'):
            #import data
            path, cond, test = conf.values()
            dnam, meta = pd.read_pickle(path, compression='gzip').values()
            
            #################################
            ###Should be modified in future## 
            #################################
            predictions = {}
            with tqdm(total=self.n_models, desc='Models', leave=False) as pbar:
                # two options here: our prediction and biolearn prediction, our is currently developing
                in_library_keys = list(self.models_config['in_library'].keys())
                for key in in_library_keys:
                    predictions[key] = self.biolearn_predict(dnam, meta, key, imputation_method='none') #tmp
                    pbar.update(1)

                #de novo clocks prediction                     
                new_models_keys = list(self.models_config['new_models'].keys())
                for key in new_models_keys:
                    path = self.models_config['new_models'][key]['path']
                    model = pickle.load(open(path, 'rb'))
                    try:
                        dnam_ = dnam.reindex(columns = list(model.pls.feature_names_in_)).copy()
                    except:
                        dnam_ = dnam.reindex(columns = list(model.feature_names_in_)).copy()
                        dnam_ = dnam_.fillna(0.)
                    
                    preds_ = model.predict(dnam_)
                    predictions[key] = self.check_predictions(preds_, dnam_)
                    pbar.update(1)

            #######################################
            #######################################

            pred = pd.DataFrame(predictions)
            self.datasets_predictions[gse] = pred.copy()
            #meta filtering
            no_age_na_indices = meta[~meta['Age'].isna()].index
            meta = meta.loc[no_age_na_indices]

            if test == 'AAP':
                pvals = self.AAP_test(pred, meta, gse, cond)
                self.bench_results_AAP[f'{cond}:{gse}:AAP'] = pd.Series(pvals)
            elif test == 'AA0': 
                pvals = self.AA0_test(pred, meta, gse, cond)
                self.bench_results_AA0[f'{cond}:{gse}:AA0'] = pd.Series(pvals)
            else:
                NotImplementedError("Following tests are currently available: ['AAP', 'AA0'].")
        
        #multiple testing correction of results. Note each model's pvalues are corrected individually.
        self.bench_results = pd.concat([self.bench_results_AAP, self.bench_results_AA0], axis=1)
        corrected_results_AAP = self.bench_results_AAP.T.apply(self.correction, axis=0).T 
        corrected_results_AA0 = self.bench_results_AA0.T.apply(self.correction, axis=0).T 
        self.corrected_results = pd.concat([corrected_results_AAP, corrected_results_AA0], axis=1)
        self.corrected_results_bool = self.corrected_results < self.correction_threshold

        if self.save_results:
            self.bench_results.to_csv(os.path.join(self.output_folder, 
                                                   f'{self.experiment_prefix}_bench_pvals.csv'))
            self.corrected_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_adj_pvals.csv'))
            self.corrected_results_bool.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_bools.csv'))
            

    def biolearn_predict(self, dnam, meta, model_key, imputation_method='none'):
        from biolearn.data_library import GeoData
        from biolearn.model_gallery import ModelGallery
        gallery = ModelGallery()
        data = GeoData(meta, dnam.T) 
        #published clocks prediction
        results = gallery.get(model_key, 
                              imputation_method=imputation_method).predict(data)
        return results['Predicted']

    def AAP_test(self, pred, meta, gse, cond):
        #calculating mann-whitney test for difference in age acceleration between disease and healthy cohorts
        disease_idx = meta.index[meta['Condition'] == cond]
        healthy_idx = meta.index[meta['Condition'] == 'HC']
        if self.verbose > 0:
            print(f'{cond}:{gse} - AAP testing {len(disease_idx)} disease versus {len(healthy_idx)} healthy samples')
        pvals = {}
        for col in pred.columns:
            disease_true = meta.loc[disease_idx, 'Age'].values
            healthy_true = meta.loc[healthy_idx, 'Age'].values
            disease_pred = pred.loc[disease_idx, col].values
            healthy_pred = pred.loc[healthy_idx, col].values
            disease_delta = disease_pred - disease_true
            healthy_delta = healthy_pred - healthy_true
            _, pval = mannwhitneyu(disease_delta, healthy_delta, alternative='greater')
            pvals[col] = pval
        return pvals

    def AA0_test(self, pred, meta, gse, cond):
        #calculating wilcoxon test for positive age (>0) acceleration in disease cohort
        disease_idx = meta.index[meta['Condition'] == cond]
        if self.verbose > 0:
            print(f'{cond}:{gse} - AA0 testing {len(disease_idx)} disease samples')
        pvals = {}
        for col in pred.columns:
            disease_true = meta.loc[disease_idx, 'Age'].values
            disease_pred = pred.loc[disease_idx, col].values
            disease_delta = disease_pred - disease_true
            _, pval = wilcoxon(disease_delta, alternative='greater')
            pvals[col] = pval
        return pvals        
    
    @staticmethod
    def check_predictions(preds: np.ndarray | pd.Series, 
                          dnam: pd.DataFrame
                          ):
        if type(preds) == np.ndarray:
            return pd.Series(preds, index=dnam.index)
        else:
            return pd.Series(preds.values, index=dnam.index)

    @staticmethod
    def correction(x):
        return multipletests(x, method='fdr_bh')[1] #returns adjusted p-values only