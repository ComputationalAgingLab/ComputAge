import os
from os.path import exists
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests
from computage.utils.data_utils import cond2class

class EpiClocksBenchmarking:
    def __init__(self, 
                 models_config: dict,
                 datasets_config: dict,
                 tissue_types: str | list[str] = 'BSB',
                 age_limits: list | None = [18, 90],
                 age_limits_class_exclusions: list[str] | None = ['PGS'],
                 experiment_prefix: str = 'test',
                 delta_assumption: str = 'normal',
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
        self.age_limits = age_limits
        self.age_limits_class_exclusions = age_limits_class_exclusions
        self.n_datasets = len(datasets_config)
        self.n_models = len(models_config['in_library']) + len(models_config['new_models'])
        self.low_memory = low_memory
        self.save_results = save_results
        self.output_folder = output_folder
        self.experiment_prefix = experiment_prefix
        self.delta_assumption = delta_assumption
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
        self.bench_deltas_AA2 = pd.DataFrame()
        self.bench_deltas_AA1 = pd.DataFrame()
        self.datasets_predictions = {}
        self.datasets_metadata = {}

        ###Predict datasets and gather predictions
        for gse, conf in tqdm(self.datasets_config.items(), 
                              total=self.n_datasets,
                              desc='Datasets'):
            #import data
            path, conditions, test = conf.values()
            dnam, meta = pd.read_pickle(path, compression='gzip').values()
            meta['GSE'] = gse
            #initial tissue filtering
            meta['Age'] = meta['Age'].astype(float)
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
            pred = pd.DataFrame(predictions)
            self.datasets_predictions[gse] = pred.copy()
            self.datasets_metadata[gse] = meta.copy()
            #meta filtering
            no_na_indices = meta[~meta['Age'].isna()].index
            meta = meta.loc[no_na_indices]
            meta['Age'] = meta['Age'].astype(float) #TODO: add function checking this behaviour
            
            for cond in conditions:
                cl = cond2class([cond])[0]
                if cl not in self.age_limits_class_exclusions:
                    meta_ = meta[(self.age_limits[0] <= meta['Age']) & 
                                 (meta['Age'] <= self.age_limits[1])].copy()
                else:
                    meta_ = meta.copy()
                if test == 'AA2':
                    pvals, deltas = self.AA2_test(pred, meta_, gse, cond)
                    if pvals is None:
                        continue
                    dname = f'{gse}:{cond}:AA2'
                    deltas['Dataset'] = dname
                    self.bench_results_AA2[dname] = pd.Series(pvals)
                    self.bench_deltas_AA2 = pd.concat([self.bench_deltas_AA2, deltas], axis=0)
                elif test == 'AA1': 
                    pvals, deltas = self.AA1_test(pred, meta_, gse, cond)
                    if pvals is None:
                        continue
                    dname = f'{gse}:{cond}:AA1'
                    deltas['Dataset'] = dname
                    self.bench_results_AA1[f'{gse}:{cond}:AA1'] = pd.Series(pvals)
                    self.bench_deltas_AA1 = pd.concat([self.bench_deltas_AA1, deltas], axis=0)
                else:
                    NotImplementedError("Following tests are currently available: ['AA2', 'AA1'].")
            
        
        #multiple testing correction of results. Note each model's pvalues are corrected individually.
        self.bench_results = pd.concat([self.bench_results_AA2, self.bench_results_AA1], axis=1)
        corrected_results_AA2 = self.bench_results_AA2.T.apply(self.correction, axis=0).T 
        corrected_results_AA1 = self.bench_results_AA1.T.apply(self.correction, axis=0).T 
        self.corrected_results = pd.concat([corrected_results_AA2, corrected_results_AA1], axis=1)
        self.corrected_results_bool = self.corrected_results < self.correction_threshold

        #additional test of chronological age prediction accuracy
        #and age acceleration prediction bias
        self.CA_prediction_results = self.CA_prediction_test()
        self.CA_bias_results = self.CA_bias_test()

        if self.save_results:
            self.bench_results.to_csv(os.path.join(self.output_folder, 
                                                   f'{self.experiment_prefix}_bench_pvals.csv'))
            self.corrected_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_adj_pvals.csv'))
            self.corrected_results_bool.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_bools.csv'))
            self.CA_prediction_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_CA_pred_MAE.csv'))
            self.CA_bias_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_CA_pred_bias.csv'))
            

    def biolearn_predict(self, dnam, meta, model_key, imputation_method='none'):
        from biolearn.data_library import GeoData
        from biolearn.model_gallery import ModelGallery
        gallery = ModelGallery()
        ###TMP###
        # if 'Gender' not in meta.columns:
        #     meta['Gender'] = np.nan
        # meta = meta.rename(columns={'Age':'age', 'Gender':'sex'})
        # meta['age'] = meta['age'].astype(float)
        # meta['sex'] = meta['sex'].map({'M':2, 'F':1})
        #########
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
        deltas = pd.DataFrame()
        for col in pred.columns:
            disease_true = meta.loc[disease_idx, 'Age'].values
            healthy_true = meta.loc[healthy_idx, 'Age'].values
            disease_pred = pred.loc[disease_idx, col].values
            healthy_pred = pred.loc[healthy_idx, col].values
            disease_delta = disease_pred - disease_true
            healthy_delta = healthy_pred - healthy_true
            if self.delta_assumption == 'normal':
                _, pval = ttest_ind(disease_delta, healthy_delta, equal_var=False, alternative='greater')
            elif self.delta_assumption == 'none':
                _, pval = mannwhitneyu(disease_delta, healthy_delta, alternative='greater')
            else:
                raise NotImplementedError()
            pvals[col] = pval
            deltas = pd.concat([deltas, 
                                pd.DataFrame({'Condition':['AAC']*len(disease_delta) + ['HC']*len(healthy_delta), 
                                              'Delta':np.concatenate([disease_delta, healthy_delta]),
                                              'Model':col})
                                ])
        return pvals, deltas

    def AA1_test(self, pred, meta, gse, cond):
        #calculating wilcoxon test for positive age (>0) acceleration in disease cohort
        disease_idx = meta.index[meta['Condition'] == cond]
        if (len(disease_idx) == 0):
            print(f'{gse}:{cond} - {len(disease_idx)} disease samples found - AA1 test is impossible. Skip!')
            return None
        if self.verbose > 0:
            print(f'{gse}:{cond} - AA1 testing {len(disease_idx)} disease samples')
        pvals = {}
        deltas = pd.DataFrame()
        for col in pred.columns:
            disease_true = meta.loc[disease_idx, 'Age'].values
            disease_pred = pred.loc[disease_idx, col].values
            disease_delta = disease_pred - disease_true
            if self.delta_assumption == 'normal':
                _, pval = ttest_1samp(disease_delta, popmean=0., alternative='greater')
            elif self.delta_assumption == 'none':
                _, pval = wilcoxon(disease_delta, alternative='greater')
            else:
                raise NotImplementedError()
            pvals[col] = pval
            deltas = pd.concat([deltas, 
                                pd.DataFrame({'Condition':['AAC']*len(disease_delta), 
                                              'Delta':disease_delta,
                                              'Model':col})
                                ])
        return pvals, deltas
    
    def CA_prediction_test(self) -> pd.DataFrame:
        full_meta = pd.concat(self.datasets_metadata.values(), axis=0)
        full_meta['Age'] = full_meta['Age'].astype(float)
        full_pred = pd.concat(self.datasets_predictions.values(), axis=0)
        hc_index = full_meta[~full_meta['Age'].isna() & (full_meta['Condition'] == 'HC')].index
        hc_age = full_meta.loc[hc_index]['Age']
        hc_pred = full_pred.loc[hc_index]
        absdiff = np.abs(hc_pred.subtract(hc_age.values, axis=0))
        if self.verbose > 0:
            print(f'Compute MedAE metric based on {absdiff.shape[0]} healthy control samples.')
        result = pd.concat([absdiff.median(axis=0)], 
                            keys=['MAE'], axis=1).reset_index()
        result = result.sort_values('MAE')
        return result    
    
    def CA_bias_test(self) -> pd.DataFrame:
        full_meta = pd.concat(self.datasets_metadata.values(), axis=0)
        full_meta['Age'] = full_meta['Age'].astype(float)
        full_pred = pd.concat(self.datasets_predictions.values(), axis=0)
        hc_index = full_meta[~full_meta['Age'].isna() & (full_meta['Condition'] == 'HC')].index
        hc_age = full_meta.loc[hc_index]['Age']
        hc_pred = full_pred.loc[hc_index]
        diff = hc_pred.subtract(hc_age.values, axis=0)
        result = pd.concat([diff.median(axis=0)], keys=['MedE'], axis=1).reset_index()
        result['absMedE'] = np.abs(result['MedE'])
        result = result.sort_values('absMedE', ascending=True)
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