#todo: 
# assign types of parameters
# import mapply
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import linregress
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LassoCV
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.multitest import multipletests
import pickle
from .base import DeAgeBaseEstimator
from computage.analyzer.timecourse import linear_time_analysis


class KlemeraDoubalEstimator(DeAgeBaseEstimator):    
    def __init__(self, 
                 cv: int = 10,
                 cv_val_size: float = 0.2,
                 cv_stratify: np.ndarray | pd.Series | None = None,
                 feature_selection_method: str = 'forward',
                 feature_selection_criterion: str = 'mse',
                 feature_pval_threshold: float = 0.5,
                 feature_stability_test: float = 0.2, #tmp
                 lasso_preselection: bool = True,
                 lasso_n_alphas: int = 50,
                 weighing: str = 'rse',
                 max_features: int = 10000,
                 nan_train_threshold: float = 0.3, #tmp
                 orthogonal_features: bool = False, #tmp
                 n_jobs: int = 8,
                 verbose: bool | int = 0,
                 ):
        """
        Klemera-Doubal Estimator - a model for estimation of a biological age.
        It works with aggregation of multiple univariate linear regression estimators,
        we call them `small models` of biomarkers. Aggregation is 
        
        Parameters
        ----------
        cv : int, default=10
            A number of cross-validation folds for splitting train data during fit.
            Note that small models are retrained for each cv fold.
        
        cv_val_size : float, default=0.2
            Should be between 0.0 and 1.0 and represent the proportion of the 
            dataset to include in the val split.
        
        cv_stratify : array-like, default=None
            If not None, data is split in a stratified fashion, using this as the 
            class labels.
        
        feature_selection_method : {'all', 'forward', 'backward', 'top_n', top_pct'}, default='forward'
            If all :
                No feature selection, all features given in `X` are used for prediction.
            If forward :
                Performs sequential forward feature selection (as in sklearn) selecting feature one-by-one. 
                Features are selected by the descending order of their absolute Pearson correlation with `y`.
                A new feature is accepted if it improves `feature_selection_criterion`.
            If backward :
                Performs sequential backward feature selection (as in sklearn) removing feature one-by-one. 
                Features are removed by the ascending order of their absolute Pearson correlation with `y`.
                A feature is removed if its removal improves `feature_selection_criterion`.
            If top_n :
                Searches for the best value of top `n` by absolute Pearson correlation features 
                optimizing `feature_selection_criterion`.
            If top_pct :
                Searches for the best percent (%) by absolute Pearson correlation features 
                optimizing `feature_selection_criterion`.
        
        feature_selection_criterion : {'mse', 'Bvar'}, default='mse'
            If mse :
                Use mean-squared error minimization as the feature selection criterion.
            If Bvar :
                Use variance of biological age estimate as the feature selection criterion.
        
        feature_pval_threshold : float, default=0.5
            Filter out biomarkers with slope coefficient p-value (a small model) large than the threshold.
            Note that by default the threshold is weak.
            
        lasso_preselection : bool, default=True
            Run LassoCV from sklearn to select features best optimizing Lasso objective. 
            These features are, however, suboptimal for Klemera-Doubal, so, `feature_selection_method` 
            is applied to the Lasso pre-selected features subsequently.
            Note: we recommend to do standard-scaling of your dataset with if lasso pre-selection is True.

        lasso_n_alphas : int, default=50
            Number of LassoCV steps of `alpha` parameter (see sklearn documentation for details
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html).
            This parameter strongly affects time of feature pre-selection. 

        weighing : {'rse', 'r2', 'r2|rse', None}, default='rse'
            Weights of small models in Klemera-Doubal objective. Note that 1/rse^2 is used in the original paper that
            corresponds the default 'rse' paremeter. However, other weighings can be useful in some situations such as
            'r2', 'r2/rse' or even no weights ('None' corresponds to all weights equal to 1).
        
        max_features : int, default=10000
            Maximum number of features from which a method will select (features are ordered by absolute Pearson 
            correlation with `y`). This parameter affects only 'forward', 'all', 'top_n', and 'top_pct' feature selection
            methods. May be especially useful in case you work with methylation data :)

        n_jobs : int, default=None
            Number of CPUs to use during the small models training. This parameter is passed to `mapply` - a package 
            providing a parallel call of method `.apply()` for pandas.DataFrame.

        verbose : bool or int, default=0
            Amount of verbosity.

        """
        self.cv = cv
        self.cv_val_size = cv_val_size
        self.cv_stratify = cv_stratify
        self.feature_selection_method = feature_selection_method
        self.feature_selection_criterion = feature_selection_criterion
        self.max_features = max_features
        self.feature_pval_threshold = feature_pval_threshold
        self.feature_stability_test = feature_stability_test
        self.lasso_preselection = lasso_preselection
        self.lasso_n_alphas = lasso_n_alphas
        self.weighing = weighing
        self.nan_train_threshold = nan_train_threshold if nan_train_threshold is not None else 0.
        self.orthogonal_features = orthogonal_features
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.features = []
        self.best_n_features = None
        self.best_pct_features = None
        self.train_mode = True
        self.metrics = None
    
    def sample_features(self, l, p):
        size = int(len(l) * p) + 1
        return np.random.choice(l, size, replace=False).tolist()

    def _forward_feature_selection_strategy(self):
        if self.lasso_preselection:
            feat_order = self.lasso_selected_features
        else:
            feat_order = pd.concat([m['r_abs'] for m in self.cv_models], axis=1).dropna(axis=0).mean(1).sort_values(ascending=False).index
        self.metrics = []
        current_criterion = np.inf
        for i in tqdm(feat_order[:self.max_features]):
            tmp_mae_train, tmp_mae_test = [], []
            tmp_crit_train, tmp_crit_test = [], []
            tmp_r2s_train, tmp_r2s_test = [], []
            for m, f in zip(self.cv_models, self.folds):
                X_train, X_test, y_train, y_test = f
                new_feature = [i]
                if self.feature_stability_test:
                    if (len(self.features) > int(1 / self.feature_stability_test)):
                        cur_features = self.sample_features(self.features, 1 - self.feature_stability_test)
                    else:
                        cur_features = self.features[:]
                else:
                    cur_features = self.features[:]
                X_train_top = X_train[cur_features + new_feature]
                X_test_top = X_test[cur_features + new_feature]
                self.model = m.loc[cur_features + new_feature]
                y_pred_train = self.predict(X_train_top)
                y_pred_test = self.predict(X_test_top)
                
                tmp_mae_train.append(median_absolute_error(y_train, y_pred_train))
                tmp_mae_test.append(median_absolute_error(y_test, y_pred_test))
                tmp_r2s_train.append(r2_score(y_train, y_pred_train))
                tmp_r2s_test.append(r2_score(y_test, y_pred_test))

                if self.feature_selection_criterion == 'mse':
                    tmp_crit_train.append(mean_squared_error(y_train, y_pred_train))
                    tmp_crit_test.append(mean_squared_error(y_test, y_pred_test))
                elif self.feature_selection_criterion == 'Bvar':
                    r = self.model['rvalue']
                    tmp_crit_train.append(self._compute_Bvar(y_pred_train, y_train, r)[0])
                    tmp_crit_test.append(self._compute_Bvar(y_pred_test, y_test, r)[0])
                else:
                    raise NotImplementedError('Wrong feature selection criterion! Choose one of ["mse", "Bvar"].')
                
            crit = np.mean(tmp_crit_test) 
            if crit <= current_criterion: 
                current_criterion = crit
                self.features.append(new_feature[0])
            else:
                continue
            
            self.metrics += [[np.mean(tmp_crit_test), np.mean(tmp_crit_test),
                              np.mean(tmp_mae_train), np.mean(tmp_mae_test),
                              np.mean(tmp_r2s_train), np.mean(tmp_r2s_test)
                             ]]
    
    def _top_n_feature_selection_strategy(self):
        if self.lasso_preselection:
            feat_order = self.lasso_selected_features
        else:
            feat_order = pd.concat([m['r_abs'] for m in self.cv_models], axis=1).dropna(axis=0).mean(1).sort_values(ascending=False).index
        max_features = min(self.max_features, len(feat_order))
        self.metrics = []
        for i in tqdm(range(1, max_features)):
            tmp_mae_train, tmp_mae_test = [], []
            tmp_crit_train, tmp_crit_test = [], []
            tmp_r2s_train, tmp_r2s_test = [], []
            for m, f in zip(self.cv_models, self.folds):
                X_train, X_test, y_train, y_test = f
                X_train_top = X_train[feat_order[:i]]
                X_test_top = X_test[feat_order[:i]]
                self.model = m.loc[feat_order[:i]]
                y_pred_train = self.predict(X_train_top)
                y_pred_test = self.predict(X_test_top)
                
                tmp_mae_train.append(median_absolute_error(y_train, y_pred_train))
                tmp_mae_test.append(median_absolute_error(y_test, y_pred_test))
                tmp_r2s_train.append(r2_score(y_train, y_pred_train))
                tmp_r2s_test.append(r2_score(y_test, y_pred_test))

                tmp_mae_train.append(median_absolute_error(y_train, y_pred_train))
                tmp_mae_test.append(median_absolute_error(y_test, y_pred_test))
                tmp_r2s_train.append(r2_score(y_train, y_pred_train))
                tmp_r2s_test.append(r2_score(y_test, y_pred_test))

                if self.feature_selection_criterion == 'mse':
                    tmp_crit_train.append(mean_squared_error(y_train, y_pred_train))
                    tmp_crit_test.append(mean_squared_error(y_test, y_pred_test))
                elif self.feature_selection_criterion == 'Bvar':
                    r = self.model['rvalue']
                    tmp_crit_train.append(self._compute_Bvar(y_pred_train, y_train, r)[0])
                    tmp_crit_test.append(self._compute_Bvar(y_pred_test, y_test, r)[0])
                else:
                    raise NotImplementedError('Wrong feature selection criterion! Choose one of ["mse", "Bvar"].')
            
            self.metrics += [[np.mean(tmp_crit_train), np.mean(tmp_crit_test),
                              np.mean(tmp_mae_train), np.mean(tmp_mae_test),
                              np.mean(tmp_r2s_train), np.mean(tmp_r2s_test)
                             ]]
        met = np.asarray(self.metrics)
        self.best_n_features = np.argmin(met[:, 1]) + 1
        self.best_pct_features = self.best_n_features / feat_order.shape[0] * 100
        self.features = self._model.index

    def _backward_feature_selection_strategy(self):
        if self.lasso_preselection:
            feat_order = self.lasso_selected_features[::-1] #inverse order
        else:
            feat_order = pd.concat([m['r_abs'] for m in self.cv_models], axis=1).dropna(axis=0).mean(1).sort_values(ascending=False).index[::-1]
        self.features = feat_order.copy().tolist()
        passed_features = []
        self.metrics = []
        current_criterion = np.inf
        flag = 'all'

        def _find_first_unique_element(list1, list2):
            unique_elements = set(list2)
            for element in list1:
                if element not in unique_elements:
                    return element
            return None  # If no unique element is found

        while len(passed_features) != len(feat_order):
            tmp_features = self.features[:]
            if flag != 'all':
                feat = _find_first_unique_element(tmp_features, passed_features)
                tmp_features.remove(feat)
                
            tmp_mae_train, tmp_mae_test = [], []
            tmp_crit_train, tmp_crit_test = [], []
            tmp_r2s_train, tmp_r2s_test = [], []
            for m, f in zip(self.cv_models, self.folds):
                X_train, X_test, y_train, y_test = f
                X_train_top = X_train[tmp_features]
                X_test_top = X_test[tmp_features]
                self.model = m.loc[tmp_features]
                y_pred_train = self.predict(X_train_top)
                y_pred_test = self.predict(X_test_top)
                
                if self.feature_selection_criterion == 'mse':
                    tmp_crit_train.append(mean_squared_error(y_train, y_pred_train))
                    tmp_crit_test.append(mean_squared_error(y_test, y_pred_test))
                elif self.feature_selection_criterion == 'Bvar':
                    r = self.model['rvalue']
                    tmp_crit_train.append(self._compute_Bvar(y_pred_train, y_train, r)[0])
                    tmp_crit_test.append(self._compute_Bvar(y_pred_test, y_test, r)[0])
                else:
                    raise NotImplementedError('Wrong feature selection criterion! Choose one of ["mse", "Bvar"].')
            
            crit = np.mean(tmp_crit_test) 
            if crit <= current_criterion: 
                current_criterion = crit
                if flag != 'all': 
                    self.features.remove(feat)

            if flag != 'all': 
                passed_features.append(feat)
                if len(passed_features) % 100 == 0:
                    print(f'{len(passed_features)} features passed')
            
            flag = 'one'
            self.metrics += [[np.mean(tmp_crit_train), np.mean(tmp_crit_test),
                              np.mean(tmp_mae_train), np.mean(tmp_mae_test),
                              np.mean(tmp_r2s_train), np.mean(tmp_r2s_test)
                             ]]    

    def fit(self, X, y):
        self._validate_data(X, y)
        self.y_avg = np.mean(y)
        self.y_max = np.max(y)
        self.y_min = np.min(y)
        
        #for compatibility with sklearn
        if type(X) == np.ndarray:
            X = pd.DataFrame(X, 
                             columns=['X' + str(i) for i in range(X.shape[1])])
        #drop columns with more than `nan_train_threshold` of nans or infs
        X = X.loc[:, np.isfinite(X).sum(axis=0) / X.shape[0] >= self.nan_train_threshold].copy()

        def _fit_feature(y, x):
            idx = np.isfinite(x) #y should always be finite
            y = y[idx] 
            x = x[idx]
            s, i, r, p, serr = linregress(y, x)
            x_cap = y * s + i
            r2 = r2_score(x, x_cap)
            rss = (np.square(x_cap - x)).sum()
            rse = np.sqrt(rss / (x.shape[0] - 2))
            return s, i, r, p, serr, rse, r2

        #train all estimators
        if self.verbose > 0: print('Training estimators on full data.')
        self._model = X.apply(lambda x: _fit_feature(y, x), result_type='expand').reset_index(drop=True).rename(index={
                                                                                                 0: 'slope', 
                                                                                                 1: 'intercept', 
                                                                                                 2: 'rvalue', 
                                                                                                 3: 'p-value', 
                                                                                                 4: 'stderr', 
                                                                                                 5: 'rse',
                                                                                                 6: 'r2'}).T
        self._model['r_abs'] = np.abs(self._model['rvalue'])
        self._model = self._model.sort_values('r_abs', ascending=False).dropna(axis=0)
        self._model['fi'] = self._model['r_abs'] / self._model['rse']**2
        self._model['adj.p-value'] = multipletests(self._model['p-value'], method='fdr_bh')[1]

        #weights of individual parts of biological age estimate, treat them as feature importances
        if self.weighing == 'rse':
            self._model['weight'] = self._model['rse']
        elif self.weighing == 'r2':
            self._model['weight'] = 1 / np.sqrt(self._model['r2'])
        elif self.weighing == 'r2|rse':
            self._model['weight'] = self._model['rse'] / np.sqrt(self._model['r2'])
        else:
            self._model['weight'] = 1.0

        if (self.feature_selection_method == 'all') or (self.feature_selection_method is None):
            self.model = self._model.copy()
            self.features = self.model.index
            self.train_mode = False

        elif self.feature_selection_method in ['forward', 'backward', 'top_n', 'top_pct']:
            #p-value filtering to prevent overfit on cv-folds
            if self.feature_pval_threshold is not None:
                pval_passed_features = self._model[self._model['adj.p-value'] < self.feature_pval_threshold].index
                X = X[pval_passed_features]
                if self.verbose > 0: print(f'{len(pval_passed_features)} features remained after applying p-value threshold.')

            
            if self.lasso_preselection:
                if self.verbose > 0: print('Run Lasso.')
                Xtmp = X.fillna(0.)
                lasso = LassoCV(verbose=self.verbose, 
                                n_alphas=self.lasso_n_alphas, 
                                n_jobs=self.n_jobs).fit(Xtmp, y)
                passed = ~(lasso.coef_ == 0)
                self.lasso_selected_features = Xtmp.columns[passed]
                if self.verbose > 0: print(f'{len(self.lasso_selected_features)} features remained after Lasso pre-selection.')

            if (self.cv == 0) or (self.cv is None):
                print('Number of cross-validation folds is 0! Overfitting to train set is expected.')
                self.folds = [X.copy(), X.copy(), y.copy(), y.copy()]
            elif self.cv == 1:
                print('Number of cross-validation folds is 1! Overfitting to validation set is expected.')
                self.folds = [tts(X, y, test_size=self.cv_val_size, 
                                   random_state=10, stratify=self.cv_stratify,)]
            else:
                self.folds = [tts(X, y, test_size=self.cv_val_size, 
                                   random_state=i, stratify=self.cv_stratify,) for i in range(self.cv)]
            
            if self.verbose > 0: print('Training estimators on different cv-folds.')
            # mapply.init(n_workers=self.n_jobs, chunk_size=100, max_chunks_per_worker=10, progressbar=False)
            self.cv_models = []
            for f in self.folds:
                X_train, _, y_train, _ = f
                model = X_train.apply(lambda x: _fit_feature(y_train, x), result_type='expand').reset_index(drop=True).rename(index={0: 'slope', 
                                                                                                                1: 'intercept', 
                                                                                                                2: 'rvalue', 
                                                                                                                3: 'p-value', 
                                                                                                                4: 'stderr', 
                                                                                                                5: 'rse',
                                                                                                                6: 'r2'}).T
                model['r_abs'] = np.abs(model['rvalue'])
                #weights of individual parts of biological age estimate, treat them as feature importances
                if self.weighing == 'rse':
                    model['weight'] = model['rse']
                elif self.weighing == 'r2':
                    model['weight'] = 1 / np.sqrt(model['r2'])
                elif self.weighing == 'r2|rse':
                    model['weight'] = model['rse'] / np.sqrt(model['r2'])
                else:
                    model['weight'] = 1.0

                self.cv_models.append(model)

            if self.feature_selection_method == 'forward':
                self._forward_feature_selection_strategy()
            elif self.feature_selection_method == 'backward': #top_n, top_pct
                self._backward_feature_selection_strategy()
            else:
                self._top_n_feature_selection_strategy()
            
            if len(self.features) != 0:
                if self.verbose > 0: print('Remove redundant features.')
                selected_features = self._model.index.intersection(self.features)
                if self.verbose > 0: print(f'{len(selected_features)} features remained after selection.')
                self.model = self._model.loc[selected_features].sort_values('r_abs', ascending=False)
            else:
                self.model = self._model
            self.train_mode = False
            
        else:
            raise NotImplementedError('Wrong feature selection method! Choose one of ["forward", "backward", "top_n", "top_pct", "all"].')
        
        #compute Biological age variance estimate
        B = self.predict(X, feature_names=self.features)
        r = self._model.loc[self.features, 'rvalue']
        self.Bvar_, self.BECvar_, self.BEBvar_, self.BECBvar_, self.rchar_  = self._compute_Bvar(B, y, r) #Bvar_ is a Var(B - C)
        self.Bvar_dict = {'Var(B-C)':self.Bvar_, 
                          'Var(BE-C)':self.BECvar_,
                          'Var(BE-B)':self.BEBvar_,
                          'Var(BEC-B)':self.BECBvar_,
                          'rchar':self.rchar_
                          }
        if self.verbose > 0: print(f'Bvar={round(self.Bvar_, 3)}')
        if self.verbose > 0: print('Finished!')
           
    def predict(self, X, feature_names=None):
        #make compatible with sklearn
        if type(X) == np.ndarray:
            X = pd.DataFrame(X, columns=['X' + str(i) for i in range(X.shape[1])])
        if self.train_mode:
            return self._predict_vectorized(self.model, X, feature_names=None)  
        else:
            X = pd.DataFrame(X).T if type(X) == pd.Series else X   
            if type(X) == pd.DataFrame:
                #manually selected features are always predicted by feature_selection_method = all;
                #self._model preserving all features is used for prediction
                if feature_names is not None:
                    return self._predict_vectorized(self._model, X, feature_names=feature_names)
                
                #feature_names = None leads to the prediction based on the chosen selection method in train.
                if self.feature_selection_method in ["forward", "backward", 'all']:
                    intersection = self.model.index.intersection(X.columns)
                    X_ = X[intersection]
                    return self._predict_vectorized(self.model, X_, feature_names=intersection)
                elif self.feature_selection_method == 'top_n':
                    intersection = self.model.index.intersection(X.columns)
                    X_ = X[intersection]
                    #the following mask preserves top n features for each sample
                    mask_nonan = np.isfinite(X_)
                    mask_top = np.cumsum(mask_nonan, axis=1) <= self.best_n_features
                    mask = mask_nonan * mask_top
                    X_ = X_ * mask.replace(False, np.nan)
                    return self._predict_vectorized(self.model, X_, feature_names=intersection)
                elif self.feature_selection_method == 'top_pct':
                    intersection = self.model.index.intersection(X.columns)
                    X_ = X[intersection]
                    #the following mask preserves top percent of features for each sample
                    mask_nonan = np.isfinite(X_)
                    mask_top = np.cumsum(mask_nonan, axis=1).le(np.round(mask_nonan.sum(1) * self.best_pct_features / 100), axis=0)
                    mask = mask_nonan * mask_top
                    X_ = X_ * mask.replace(False, np.nan)
                    return self._predict_vectorized(self.model, X_, feature_names=intersection)
                else:
                    raise NotImplementedError('Wrong feature selection method! Choose one of ["forward", "backward", "top_n", "top_pct", "all"].')
            else:
                raise "Please use pd.DataFrame or pd.Series input types with known feature names."


    def predict_BAC(self, X_target, y_target, X_base=None, y_base=None, feature_names=None):
        feature_names = self.features if feature_names is None else feature_names
        if (X_base is not None) and (y_base is not None):
            B = self.predict(X_base, feature_names=feature_names)
            r = self._model.loc[feature_names, 'rvalue']
            self.Bvar_, self.BECvar_, self.BEBvar_, self.BECBvar_, self.rchar_ = self._compute_Bvar(B, y_base, r)
            self.y_avg = np.mean(y_base)

        if type(X_target) == pd.DataFrame:
            return self._predict_vectorized(self._model, X_target, feature_names, y=y_target, Bvar=self.Bvar_)
        elif type(X_target) == pd.Series:
            return self._predict_vectorized(self._model, pd.DataFrame(X_target).T, feature_names, y=y_target, Bvar=self.Bvar_)   
        else:
            raise "Please use pd.DataFrame or pd.Series input types with known feature names." 

    def _predict_vectorized(self, model, X, feature_names, y=None, Bvar=None):
        indx = X.index
        if (y is not None) and (Bvar is not None):
            Bvarterm = 1 / Bvar
            Cterm = y.values / Bvar
        else:
            Cterm, Bvarterm = (0., 0.)
        if (feature_names is None) or (len(feature_names) == 0):
            feature_names = ...
        X_ = X.loc[:, feature_names].copy()
        mask = np.isfinite(X_) 
        w = np.repeat(model.loc[feature_names, 'slope'].values[None, :], X_.shape[0], axis=0) * mask
        b = np.repeat(model.loc[feature_names, 'intercept'].values[None, :], X_.shape[0], axis=0) * mask
        s = np.repeat(model.loc[feature_names, 'weight'].values[None, :], X_.shape[0], axis=0) * mask
        denominator = np.square(w / s).sum(axis=1) + Bvarterm
        nominator = ((X_ - b) * w / np.square(s)).sum(1) + Cterm
        #if sum(denominator) == 0, then it is better to use average as an estimate
        return pd.Series(np.where(denominator==0., self.y_avg, nominator / denominator), index=indx)    

    def _compute_Bvar(self, B, y, r):
        rchar = np.sum(r**2 / np.sqrt(1 - r**2)) / np.sum(np.abs(r) / np.sqrt(1 - r**2))
        #Bvar can me negative for small m = len(r), it is better to use MSE as an estimator
        dif = (B - y)
        BECvar = np.var(dif, ddof=1)
        BEBvar = ((1 - rchar**2) / rchar**2) * (y.max() - y.min())**2 / (12 * len(r))
        if self.orthogonal_features:
            if BECvar <= BEBvar:
                BCvar = BECvar
            else:
                BCvar = BECvar - BEBvar
        else:
            # TMP, correlated features create paradoxes in variance estimation, 
            #so more rude estimate will be used
            BCvar = BECvar 
        BECBvar = BEBvar / (1 + BEBvar / BCvar)
        return (BCvar, BECvar, BEBvar, BECBvar, rchar) 
        
    
    # def plot_metrics(self):
    #     met = np.asarray(self.metrics)
    #     met = pd.DataFrame(met, columns=["Loss Train", "Loss Val", "MedAE Train", "MedAE Val", "R2 Train", "R2 Val"])

    #     fig, axs = plt.subplots(1, 3, layout='constrained', figsize=(13, 5))
    #     metric_names = ["Loss", "MedAE", "R2"]
    #     for i, metric in enumerate(metric_names):
    #         ax = axs[i]
    #         sns.lineplot(met[[f"{metric} Train", f"{metric} Val"]].reset_index(), dashes=False, ax=ax)
    #         ax.set_title(metric)
    #         ax.set_xlabel('Number of features')
    #         if i != 1:
    #             ax.legend().remove()
    #         else:
    #             handles, labels = ax.get_legend_handles_labels()
    #             ax.legend(handles, [l.split(" ")[1] for l in labels], loc="upper center")
    #         if metric == "R2":
    #             ax.set_ylim([-1, 1.05])
    #     sns.despine(fig=fig)
    #     plt.show()
        
        
### TODO from Dmitrii

## add top n% custom feature selection by top-n/top-%
## testing all
## add NaN interpolation filling with linear models

### TODO from Evgeniy
## change weights in KD calculation from s to a and place them to the nominator
