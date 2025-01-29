import os
from os.path import exists
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from scipy.stats import mannwhitneyu, wilcoxon, ttest_1samp, ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt

from computage.models_library.model import LinearMethylationModel, GrimAgeModel, model_definitions
from computage.utils.data_utils import cond2class, download_meta, download_dataset, construct_config
from computage.plots.benchplots import plot_class_bench, plot_medae, plot_bias
from computage.models_library.definitions import get_clock_file
from huggingface_hub import snapshot_download

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
                 data_repository: str = 'huggingface',
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
        self.n_datasets = sum([len(v['conditions']) for k,v in datasets_config.items()])
        self.n_models = len(models_config['in_library']) + len(models_config['new_models'])
        self.save_results = save_results
        self.plot_results = plot_results
        self.save_data = save_data
        self.root_folder = output_folder
        self.meta_table_path = os.path.join(self.root_folder, 'meta_table.xlsx')
        self.data_folder = os.path.join(self.root_folder, 'benchdata')
        self.output_folder = os.path.join(self.root_folder, experiment_prefix)
        self.figure_folder = os.path.join(self.output_folder, 'figures')
        self.data_repository = data_repository
        self.experiment_prefix = experiment_prefix
        self.delta_assumption = delta_assumption
        self.pvalue_threshold = pvalue_threshold
        self.verbose = verbose
        self.imputation_values = pd.read_csv(get_clock_file("sesame_450k_median.csv"), index_col=0)
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

    def download_benchmark_from_huggingface(self):        
        snapshot_download(
                        repo_id='computage/computage_bench', 
                        repo_type="dataset",
                        local_dir=self.data_folder,
                        allow_patterns=[
                                        'computage_bench_meta.tsv', 
                                        'data/benchmark/*.parquet'
                                        ]
                        )

    def run(self) -> None:
        """
        Run epigenetic clocks benchmarking!
        """
        if self.verbose > 0:
            print(f'Run benchmarking!')
        #output and files preparation
        self.check_folder(self.root_folder)
        self.check_folder(self.output_folder)
        self.check_folder(self.figure_folder)
        self.check_folder(self.data_folder)
        
        if self.verbose > 0: 
            print(f'Check data.')
        if self.data_repository == 'huggingface':
            self.download_benchmark_from_huggingface()
            samples_meta = pd.read_csv(os.path.join(self.data_folder, 'computage_bench_meta.tsv'), 
                           sep='\t', index_col=0)
        else:
            self.check_existence()

        #prepare dataconfig
        if self.data_repository == 'huggingface':
            self.datasets_config = construct_config(self.data_folder, self.datasets_config, split='benchmark')
        
        #initialize models
        self.models = self.prepare_models()
        
        if len(self.models_config['new_models']) > 0:
            self.newmodels = self.prepare_new_models()

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
            if self.data_repository == 'huggingface':
                dnam = pd.read_parquet(path).T
                meta = samples_meta[samples_meta['DatasetID'] == gse].copy()
                meta = meta.loc[dnam.index]
            else:
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
            ##############RUN################ 
            #################################
            predictions = {}
            with tqdm(total=self.n_models, desc='Models', leave=False) as pbar:
                # two options here: our prediction and biolearn prediction, our is currently developing
                in_library_keys = list(self.models_config['in_library'].keys())
                for key in in_library_keys:
                    # predictions[key] = self.biolearn_predict(dnam, meta, key, imputation_method='none') #tmp
                    if model_definitions[key]["model"]["type"] == "LinearMethylationModel":
                        predictions[key] = self.models[key].predict(dnam)
                    else:
                        predictions[key] = self.models[key].predict(dnam, meta)
                    pbar.update(1)

                #de novo clocks prediction                     
                new_models_keys = list(self.models_config['new_models'].keys())
                for key in new_models_keys:
                    imputation = self.models_config['new_models'][key].get('imputation', 'none')
                    dnam_ = self.impute_nans(
                                        dnam, 
                                        features=self.newmodels[key].feature_names_in_,
                                        method=imputation,
                                        ivalues=self.newmodels[key].ivalues
                                        )
                    preds_ = self.newmodels[key].predict(dnam_)
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
        self.total_score = self.compute_total_score()

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
            #remove redundant elements before pickling
            del self.models
            #save the whole bench class
            pd.to_pickle(self, os.path.join(self.output_folder, 
                                            f'{self.experiment_prefix}_bench.pkl'))
            
        if self.plot_results:
            self.plot_bench_results()

        # Warning: extremely memory consuming operation!
        if self.save_data:
            pass
            
    def prepare_models(self) -> dict:
        models = {}
        for name, params in self.models_config['in_library'].items():
            if model_definitions[name]["model"]["type"] == "LinearMethylationModel":
                m = LinearMethylationModel(name, **params)
            else:
                m = GrimAgeModel(name, **params)
            models[name] = m
        return models

    def prepare_new_models(self) -> dict:
        models = {}
        for name, params in self.models_config['new_models'].items():
            m = pickle.load(open(params['path'], 'rb'))
            m.ivalues = self.imputation_values.loc[m.feature_names_in_]
            models[name] = m
        return models   
    
    def impute_nans(self, 
                    X: pd.DataFrame, 
                    features: pd.Index | list | np.ndarray, 
                    method: str = 'none',
                    ivalues: pd.Series|None = None,
                    ) -> pd.DataFrame:
        if method == 'none':
            X_ = X.reindex(columns=features).fillna(0.)
        elif method == 'sesame_450k':
            X_ = X.reindex(columns=features).fillna(ivalues['median'])
        elif method == 'average':
            X_ = X.reindex(columns=features)
            averages = X_.mean(axis=0).fillna(0.) #fill with 0 if no values in a column
            X_ = X_.fillna(averages)
        return X_

    def AA2_test(self, pred, meta, gse, cond):
        #calculating mann-whitney test for difference in age acceleration between disease and healthy cohorts
        disease_idx = meta.index[meta['Condition'] == cond]
        healthy_idx = meta.index[meta['Condition'] == 'HC']
        if (len(disease_idx) == 0) or (len(healthy_idx) == 0):
            print(f'{gse}:{cond} - {len(disease_idx)} disease and {len(healthy_idx)} healthy samples found - AA2 test is impossible. Skip!')
            return None, None
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
            return None, None
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
    
    def compute_total_score(self) -> pd.Series:
        aa2_score = self.corrected_results_AA2_bool.sum(axis=1) 
        aa1_score = self.corrected_results_AA1_bool.sum(axis=1)
        mad = self.CA_prediction_results.set_index('index')['MAE']
        relu_md = np.maximum(0, self.CA_bias_results.set_index('index')['MedE'])
        total_score = aa2_score + aa1_score * (1 - relu_md / mad)
        total_score = total_score.sort_values(ascending=False)
        total_score = round(total_score, 1)
        return total_score
    
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
        return multipletests(x.fillna(1.0), method='fdr_bh')[1] #returns adjusted p-values only
    

    def plot_bench_results(self):
        #plot AA2
        if len(self.corrected_results_AA2_bool) > 0:
            plot_class_bench(self.corrected_results_AA2_bool, 
                                figsize=(10.5, 0.48 * (len(self.corrected_results_AA2_bool) + 1)),
                                firstcolwidth = 1.9,
                                )
            plt.savefig(os.path.join(self.figure_folder, 'AA2_main.pdf'), format='pdf', dpi=180)
            plt.show()

        #plot AA1
        if len(self.corrected_results_AA1_bool) > 0:
            plot_class_bench(self.corrected_results_AA1_bool, 
                                figsize=(9, 0.48 * (len(self.corrected_results_AA1_bool) + 1)),
                                classcolwidth=0.64,
                                totalcolwidth=0.7,
                                firstcolwidth = 1.2
                                )
            plt.savefig(os.path.join(self.figure_folder, 'AA1_main.pdf'), format='pdf', dpi=180)
            plt.show()
        
        #plot chronological age prediction accuracy results
        _, colordict = plot_medae(self.CA_prediction_results, figsize=(2.8, 5), upper_bound=18) 
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder, 'CA_MedAE_main.pdf'), format='pdf', dpi=180)
        plt.show()

        #plot chronological age prediction bias results
        plot_bias(self.CA_bias_results, colordict, figsize=(2.8, 5), xlims=[-20, 20])
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_folder, 'AA_bias_main.pdf'), format='pdf', dpi=180)
        plt.show()


def run_benchmark(models_config: dict, 
				  datasets_config: dict = None,
				  experiment_prefix: str = 'my_model_test',
				  output_folder:str = './benchmark'):
	"""
    Runs a benchmarking experiment on a set of epigenetic clock models against 
    all datasets in the benchmark.

    Parameters:
        models_config (dict): 
            A dictionary containing configurations for the epigenetic 
            clock models to be tested. 
        datasets_config (dict, optional): 
            A dictionary specifying the datasets to be used for benchmarking.
            If not provided, the default `datasets_config_main` from `computage.configs.datasets_bench_config` is used.                           
        experiment_prefix (str): 
            A string prefix used to identify the experiment in output files. Default is 'my_model_test'.
        output_folder (str): 
            The folder path where the benchmarking results and data will be saved. Default is './benchmark'.

    Returns:
        EpiClocksBenchmarking: An instance of the EpiClocksBenchmarking class containing the results 
        of the benchmarking experiment.
    """
	if datasets_config is None:
		from computage.utils.data_utils import get_bsb_config
		datasets_config = get_bsb_config()

	bench = EpiClocksBenchmarking(
		models_config=models_config,
		datasets_config=datasets_config,
		tissue_types='BSB',
		age_limits = [18, 90],
		age_limits_class_exclusions= ['PGS'],
		experiment_prefix=experiment_prefix,
		delta_assumption = 'normal',
		pvalue_threshold=0.05,
		save_results=True,
		save_data=False,
		output_folder=output_folder,
		data_repository='huggingface',
		verbose=1
	)   
	bench.run()
	return bench