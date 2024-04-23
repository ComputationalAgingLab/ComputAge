import os
from os.path import exists
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt

from computage.utils.data_utils import cond2class, download_meta, download_dataset
from computage.plots.benchplots import plot_class_bench, plot_medae, plot_bias

class EpiClocksBenchmarking:
    def __init__(self, 
                 models_config: dict,
                 datasets_config: dict,
                 tissue_types: str | list[str] = 'BSB',
                 age_limits: list | None = [18, 90],
                 age_limits_class_exclusions: list[str] | None = ['PGS'],
                 experiment_prefix: str = 'test',
                 delta_assumption: str = 'normal',
                 pvalue_threshold: float = 0.05,
                 save_results: bool = True,
                 plot_results: bool = True,
                 save_data: bool = False,
                 output_folder: str ='./bench_results', 
                 verbose: int = 1,
                 ) -> None:
        """
        Initialize the benchmarking of epigenetic aging clocks.

        Parameters
        ----------
        models_config : dict
            Configuration dictionary where keys are model names and values contain model-specific parameters.
        datasets_config : dict
            Configuration dictionary where keys are dataset names and values are dataset-specific parameters.
        tissue_types : str or list of str, default 'BSB'
            Tissue types to include in the benchmarking. Single string or list of strings representing tissue types.
            'BSB' is a special key aliasing of ['Blood', 'Saliva', 'Buccal'].
        age_limits : list, default [18, 90]
            Age range to include in the analysis. List containing minimum and maximum age.
        age_limits_class_exclusions : list of str, default ['PGS']
            List of classes to be excluded from application age limits filtering.
            For example, 'PGS' in the list means that people with progeria are tested
            versus healthy control without exclusion of people younger than 18 years 
            (lower limit of age_limits).
        experiment_prefix : str, default 'test'
            Prefix for the experiment name, used in saving results.
        delta_assumption : str, default 'normal'
            Assumption for distribution of age acceleration (\Delta). 
            If 'normal', one-sided Welch's test is applied for AA2 hypothesis testing and
            one-sided t-test for AA1 hypothesis testing. If 'none' one-sided nonparametric 
            Mann-Whitney test is applied for AA2 and Wilcoxon test is used for AA1.
        pvalue_threshold : float, default 0.05
            P-value threshold for statistical significance in the AA2 and AA1 tasks.
            This P-value threshold is applied after multiple testing correction procedure.
        save_results : bool, default True
            If True, save the benchmarking results to files.
        plot_results : bool, default True
            If True, generate and save plots of the results.
        save_data : bool, default False
            If True, save the methylation data and metadata used in the benchmarking.
            Note these data are not filtered with respect to config and class parameters.
        output_folder : str, default './bench_results'
            Path to the folder where results and plots will be saved.
        verbose : int, default 1, indicating the verbosity level.
        """
        
        self.models_config = models_config
        self.datasets_config = datasets_config
        self.tissue_types = ['Blood', 'Saliva', 'Buccal'] if tissue_types=='BSB' else tissue_types
        self.tissue_types = [self.tissue_types] if type(self.tissue_types) == str else self.tissue_types
        self.age_limits = age_limits
        self.age_limits_class_exclusions = age_limits_class_exclusions
        self.n_datasets = len(datasets_config)
        self.n_models = len(models_config['in_library']) + len(models_config['new_models'])
        self.save_results = save_results
        self.plot_results = plot_results
        self.save_data = save_data
        self.meta_table_path = os.path.join(output_folder, 'meta_table.xlsx')
        self.output_folder = os.path.join(output_folder, experiment_prefix)
        self.figure_folder = os.path.join(self.output_folder, 'figures')
        self.data_folder = os.path.join(self.output_folder, 'benchdata')
        self.experiment_prefix = experiment_prefix
        self.delta_assumption = delta_assumption
        self.pvalue_threshold = pvalue_threshold
        self.verbose = verbose
        print(f'{self.n_models} models will be tested on {self.n_datasets} datasets.')
        
    
    @staticmethod
    def check_folder(path):
        if not exists(path):
            os.mkdir(path)

    def check_existence(self):
        """Check existense of all datasets in config."""
        toremove = []
        for gse, conf in self.datasets_config.items():
            path, _, _ = conf.values()
            if not exists(path):
                print(f'A dataset file {gse} does not exist at the path! Start download...')
                if not exists(self.meta_table_path):
                    self.meta_table = download_meta(self.meta_table_path, open_file=True)
                else:
                    self.meta_table = pd.read_excel(self.meta_table_path)
                try:
                    download_dataset(self.meta_table, gse, self.data_folder)
                    self.datasets_config[gse]['path'] = os.path.join(self.data_folder, f'{gse}.pkl.gz')
                    print(f'Saved to {os.path.join(self.data_folder, f"{gse}.pkl.gz")}.')
                except:
                    print('Oops! We currently do not have this dataset.')
                    toremove.append(gse)
        for gse in toremove:        
            del self.datasets_config[gse]
                

    def run(self) -> None:
        """
        Run epigenetic clocks benchmarking!
        """
        #preparation
        self.check_folder(self.output_folder)
        self.check_folder(self.figure_folder)
        self.check_folder(self.data_folder)
        self.check_existence()

        #go!
        self.bench_results_AA2 = pd.DataFrame()
        self.bench_results_AA1 = pd.DataFrame()
        self.bench_deltas_AA2 = pd.DataFrame()
        self.bench_deltas_AA1 = pd.DataFrame()
        self.datasets_predictions = {}
        self.datasets_metadata = {}
        self.datasets_data = {} if self.save_data else None

        ###Predict datasets and gather predictions
        for gse, conf in tqdm(self.datasets_config.items(), 
                              total=self.n_datasets,
                              desc='Datasets'):
            #import data
            path, conditions, test = conf.values()
            dnam, meta = pd.read_pickle(path, compression='gzip').values()
            meta['GSE'] = gse
            if self.save_data:
                self.datasets_data[gse] = {}
                self.datasets_data[gse]['data'] = dnam.copy()
                self.datasets_data[gse]['meta'] = meta.copy()
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
            meta['Age'] = meta['Age'].astype(float) 
            
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
        self.corrected_results_AA2 = self.bench_results_AA2.T.apply(self.correction, axis=0).T 
        self.corrected_results_AA1 = self.bench_results_AA1.T.apply(self.correction, axis=0).T 
        self.corrected_results_AA2_bool = self.corrected_results_AA2 < self.pvalue_threshold
        self.corrected_results_AA1_bool = self.corrected_results_AA1 < self.pvalue_threshold

        #auxilary tests of chronological age prediction accuracy
        #and age acceleration prediction bias
        self.CA_prediction_results = self.CA_prediction_test()
        self.CA_bias_results = self.CA_bias_test()

        if self.save_results:
            self.bench_results_AA2.to_csv(os.path.join(self.output_folder, 
                                                   f'{self.experiment_prefix}_bench_AA2_pvals.csv'))
            self.bench_results_AA1.to_csv(os.path.join(self.output_folder, 
                                                   f'{self.experiment_prefix}_bench_AA1_pvals.csv'))            
            self.corrected_results_AA2_bool.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_AA2_corrected_bools.csv'))
            self.corrected_results_AA1_bool.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_AA1_corrected_bools.csv'))
            self.CA_prediction_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_CA_pred_MAE.csv'))
            self.CA_bias_results.to_csv(os.path.join(self.output_folder, 
                                                       f'{self.experiment_prefix}_bench_CA_pred_bias.csv'))
            #save the whole bench class
            pd.to_pickle(self, os.path.join(self.output_folder, 
                                            f'{self.experiment_prefix}_bench.pkl'))
            
        if self.plot_results:
            self.plot_bench_results()

        # Warning: extremely memory consuming operation!
        if self.save_data:
            pass
            

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
    

    def plot_bench_results(self):
        #plot AA2
        if len(self.corrected_results_AA2_bool) > 0:
            plot_class_bench(self.corrected_results_AA2_bool, 
                                figsize=(11.0, 0.5 * (len(self.corrected_results_AA2_bool) + 1)))
            plt.savefig(os.path.join(self.figure_folder, 'AA2_main.pdf'), format='pdf', dpi=180)
            plt.close()

        #plot AA1
        if len(self.corrected_results_AA1_bool) > 0:
            plot_class_bench(self.corrected_results_AA1_bool, 
                                figsize=(11.0, 0.5 * (len(self.corrected_results_AA1_bool) + 1)))
            plt.savefig(os.path.join(self.figure_folder, 'AA1_main.pdf'), format='pdf', dpi=180)
            plt.close()
        
        #plot chronological age prediction accuracy results
        plot_medae(self.CA_prediction_results, figsize=(11., 4.6), upper_bound=18) 
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder, 'CA_MedAE_main.pdf'), format='pdf', dpi=180)
        plt.close()

        #plot chronological age prediction bias results
        plot_bias(self.CA_bias_results, figsize=(11, 4.6), ylims=[-10, 10])
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder, 'AA_bias_main.pdf'), format='pdf', dpi=180)
        plt.close()