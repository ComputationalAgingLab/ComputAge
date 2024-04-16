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
                 tissue_types: str | list[str] = 'BSB',
                 experiment_prefix: str = 'test', 
                 test_age_prediction: bool = True,
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
        self.tissue_types = ['Blood', 'Saliva', 'Buccal'] if tissue_types=='BSB' else tissue_types
        self.tissue_types = [self.tissue_types] if type(self.tissue_types) == str else self.tissue_types
        self.n_datasets = len(datasets_config)
        self.n_models = len(models_config['in_library']) + len(models_config['new_models'])
        self.low_memory = low_memory
        self.save_results = save_results
        self.output_folder = output_folder
        self.experiment_prefix = experiment_prefix
        self.test_age_prediction = test_age_prediction
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

    def download_datasets(self): #TODO: add downloader for files
        pass

    def run(self) -> None:
        """
        Run epigenetic clocks benchmarking!
        """
        #preparation
        self.check_folder(self.output_folder)
        self.check_existence()

        #go!
        self.bench_results_AA2 = pd.DataFrame()
        self.bench_results_AA1 = pd.DataFrame()
        self.datasets_predictions = {}
        self.datasets_metadata = {}

        ###Predict datasets and gather predictions
        for gse, conf in tqdm(self.datasets_config.items(), 
                              total=self.n_datasets,
                              desc='Datasets'):
            #import data
            path, conditions, test = conf.values()
            dnam, meta = pd.read_pickle(path, compression='gzip').values()
            #initial tissue filtering
            tissue_indices = meta[meta['Tissue'].isin(self.tissue_types)].index
            meta = meta.loc[tissue_indices]
            dnam = dnam.loc[tissue_indices]
            
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

            for cond in conditions:
                pred = pd.DataFrame(predictions)
                self.datasets_predictions[gse] = pred.copy()
                meta['GSE'] = gse
                self.datasets_metadata[gse] = meta.copy()
                #meta filtering
                no_age_na_indices = meta[~meta['Age'].isna()].index
                meta = meta.loc[no_age_na_indices]

                if test == 'AA2':
                    pvals = self.AA2_test(pred, meta, gse, cond)
                    if pvals is None:
                        continue
                    self.bench_results_AA2[f'{gse}:{cond}:AA2'] = pd.Series(pvals)
                elif test == 'AA1': 
                    pvals = self.AA1_test(pred, meta, gse, cond)
                    if pvals is None:
                        continue
                    self.bench_results_AA1[f'{gse}:{cond}:AA1'] = pd.Series(pvals)
                else:
                    NotImplementedError("Following tests are currently available: ['AA2', 'AA1'].")
            
        
        #multiple testing correction of results. Note each model's pvalues are corrected individually.
        self.bench_results = pd.concat([self.bench_results_AA2, self.bench_results_AA1], axis=1)
        corrected_results_AA2 = self.bench_results_AA2.T.apply(self.correction, axis=0).T 
        corrected_results_AA1 = self.bench_results_AA1.T.apply(self.correction, axis=0).T 
        self.corrected_results = pd.concat([corrected_results_AA2, corrected_results_AA1], axis=1)
        self.corrected_results_bool = self.corrected_results < self.correction_threshold

        #optional test of chronological age prediction accuracy
        if self.test_age_prediction:
            self.CA_prediction_results = self.CA_prediction_test()

        if self.save_results:
            self.bench_results.to_csv(os.path.join(self.output_folder, 
                                                   f'{self.experiment_prefix}_bench_pvals.csv'))
            self.corrected_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_adj_pvals.csv'))
            self.corrected_results_bool.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_bools.csv'))
            self.CA_prediction_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_CA_pred_MAE.csv'))
            

    def biolearn_predict(self, dnam, meta, model_key, imputation_method='none'):
        from biolearn.data_library import GeoData
        from biolearn.model_gallery import ModelGallery
        gallery = ModelGallery()
        data = GeoData(meta, dnam.T) 
        #published clocks prediction
        results = gallery.get(model_key, 
                              imputation_method=imputation_method).predict(data)
        return results['Predicted']

    def AA2_test(self, pred, meta, gse, cond):
        #calculating mann-whitney test for difference in age acceleration between disease and healthy cohorts
        disease_idx = meta.index[meta['Condition'] == cond]
        healthy_idx = meta.index[meta['Condition'] == 'HC']
        if (len(disease_idx) == 0) or (len(healthy_idx) == 0):
            print(f'{gse}:{cond} - {len(disease_idx)} disease and {len(healthy_idx)} healthy samples found - AA2 test is impossible. Skip!')
            return None
        if self.verbose > 0:
            print(f'{gse}:{cond} - AA2 testing {len(disease_idx)} disease versus {len(healthy_idx)} healthy samples')
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

    def AA1_test(self, pred, meta, gse, cond):
        #calculating wilcoxon test for positive age (>0) acceleration in disease cohort
        disease_idx = meta.index[meta['Condition'] == cond]
        if self.verbose > 0:
            print(f'{gse}:{cond} - AA1 testing {len(disease_idx)} disease samples')
        pvals = {}
        for col in pred.columns:
            disease_true = meta.loc[disease_idx, 'Age'].values
            disease_pred = pred.loc[disease_idx, col].values
            disease_delta = disease_pred - disease_true
            _, pval = wilcoxon(disease_delta, alternative='greater')
            pvals[col] = pval
        return pvals    

    def CA_prediction_test(self) -> pd.DataFrame:
        full_meta = pd.concat(self.datasets_metadata.values(), axis=0)
        full_pred = pd.concat(self.datasets_predictions.values(), axis=0)
        hc_index = full_meta[~full_meta['Age'].isna() & (full_meta['Condition'] == 'HC')].index
        hc_age = full_meta.loc[hc_index]['Age']
        hc_pred = full_pred.loc[hc_index]
        absdiff = np.abs(hc_pred.subtract(hc_age.values, axis=0))
        if self.verbose > 0:
            print(f'Compute MAE metric based on {absdiff.shape[0]} healthy control samples.')
        result = pd.concat([absdiff.mean(axis=0), absdiff.sem(axis=0, ddof=1)], 
                            keys=['MAE', 'MAE_SE'], axis=1).reset_index()
        result = result.sort_values('MAE')
        return result
    
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