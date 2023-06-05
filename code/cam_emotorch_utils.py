#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_emotorch_utils.py
"""


class LogitTransform():

    def __init__(self, affine_intercept=None):
        self.affine_intercept = affine_intercept
        self.bypass = True if affine_intercept is None else False

    def affine_compress(self, x):
        return x if self.bypass else (1 - 2 * self.affine_intercept) * x + self.affine_intercept

    def affine_stretch(self, x):
        return x if self.bypass else (1 + 2 * self.affine_intercept) * x - self.affine_intercept

    def logit_transform(self, x):
        import numpy as np
        import warnings
        if x <= 0:
            warnings.warn("customWarning: Divide by Zero in logit (x<=1) :: ", x)
            return np.nan
        return x if self.bypass else np.log(x / (1 - x))

    def logistic_transform(self, x):
        import numpy as np
        return x if self.bypass else (1 + np.exp(-x))**-1

    def y_transform(self, x):
        return x if self.bypass else self.logit_transform(self.affine_compress(x))

    def yhat_transform(self, x):
        return x if self.bypass else self.affine_stretch(self.logistic_transform(x))


class ProspectTransform():

    def __init__(self, alpha_=1.0, beta_=1.0, lambda_=1.0, intercept=0.0, noise_scale=0.0, log1p=False):
        self.alpha_ = alpha_
        self.beta_ = beta_
        self.lambda_ = lambda_
        self.sigma_ = noise_scale
        self.b0_ = intercept
        self.log1p = log1p

    def _kahneman(self, x):
        if x < 0:
            y = -1 * self.lambda_ * ((-1 * x) ** self.beta_)
        else:
            y = x ** self.alpha_
        return y

    def _power_fn_(self, X_):
        import numpy as np
        power_fn = np.vectorize(self._kahneman)
        return power_fn(X_)

    def _log1p_fn_(self, X_):
        import numpy as np

        def _nu_(x): return np.sign(x) * np.log1p(np.abs(x))

        log_fn = np.vectorize(_nu_)

        return log_fn(X_)

    def transform(self, df):
        import numpy as np

        if isinstance(df, list):
            shape = len(df)
        elif isinstance(df, (int, float)):
            shape = 1
        else:
            shape = df.shape

        if self.sigma_ > 0:
            noise = np.zeros(shape)
        else:
            noise = np.random.normal(loc=0.0, scale=self.sigma_, size=shape)

        X_ = np.add(np.subtract(df, self.b0_), noise)
        if self.log1p:
            Xtranformed = self._log1p_fn_(X_)
        else:
            Xtranformed = self._power_fn_(X_)

        if isinstance(df, (int, float)):
            Xtranformed = Xtranformed[0]

        return Xtranformed


class DataTransform():
    """
    For training, return X,Y pair with a index map that relates the observations
    For testing, take an X containing any stim identities and transform it independent of 
    its relationship to the training data.

    To score Y data, just need to know the new X:Y map (not the stim IDs used to train the model)
    """

    def __init__(self, index_pad=0, _thin_=1, affine_intercept_=None, prospect_transform_param_=None, scale_param_=None, verbose=False):
        import random

        self.index_pad = index_pad
        self._thin_ = _thin_

        self.affine_intercept = affine_intercept_
        self.yt = LogitTransform(affine_intercept_)

        self._prospect_transform_param_fn_ = prospect_transform_param_['fn']
        self._prospect_transform_param_ = prospect_transform_param_['param']
        self.prospect_transform_kwargs_dict = dict()
        self.xt_dict = dict()

        self._scale_param_fn_ = scale_param_['fn']
        self._scale_param_ = scale_param_['param']
        self.scale_kwargs_dict = dict()
        self.xscaler_dict = dict()

        self.total_stim_train_ = None

        self.X_cols_in = None
        self.X_cols = None

        self.desc = dict()

        self.desc_hash = None
        self.persistance_seed = random.randint(0, 1e22)

        self.verbose = verbose

    def _thin_df_(self, X_in_, thin_):
        """
        thinned according to its own stim identity -- all stim included
        """
        import numpy as np
        import pandas as pd

        ### prune model samples ###

        if thin_ != self._thin_:
            from warnings import warn
            warn(f"WARNING: Thin Change from {self._thin_} to {thin_}")

        thisX_stim_counts = X_in_.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        indexer_counts = list()
        thinned_X_array = list()
        for i_stim, stim in thisX_stim_counts.iterrows():
            indexer = (X_in_['pot'] == stim['pot']) & (X_in_['outcome'] == stim['outcome'])
            indexer_counts.append(np.sum(indexer))
            if i_stim == 0:
                print(f'Thinning X by {thin_}, samples included: {len(np.arange(0,np.sum(indexer),thin_))}')
            thinned_X_array.append(X_in_.loc[indexer, :].iloc[np.arange(0, np.sum(indexer), thin_), :].copy())

        ### assert that there's an even number of observations for each stimulus in this X
        # (only important for the current pytorch implementation that reshapes the data to be cuboidal)

        xthin = pd.concat(thinned_X_array).reset_index(drop=True)

        assert np.unique(indexer_counts).size == 1, f"np.unique(indexer_counts):\n{np.unique(indexer_counts)}, \n\n np.unique(indexer_counts).size: {np.unique(indexer_counts).size}"  # data is made cuboidal

        return xthin, thisX_stim_counts

    def _get_id_index_(self, M, with_respect_to_stim_id=None):
        import numpy as np

        stim_counts = M.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        if with_respect_to_stim_id is None:
            stim_id_table = stim_counts.drop(['count'], axis=1)
        else:
            stim_id_table = with_respect_to_stim_id

        uniform = True if len(stim_counts['count'].unique()) == 1 else False

        data_binary_table = np.full((M.shape[0], stim_id_table.shape[0]), 0, dtype=bool)

        J = np.zeros((M.shape[0]), dtype=int)

        J_sample = np.zeros((M.shape[0]), dtype=int) if uniform else None

        ### assign an ID number to each row of observed M
        for i_stim, stim in stim_id_table.iterrows():
            data_binary_table[:, i_stim] = (M['pot'] == stim['pot']) & (M['outcome'] == stim['outcome'])

            J[data_binary_table[:, i_stim]] = stim.name

            if uniform:
                J_sample[data_binary_table[:, i_stim]] = np.arange(0, data_binary_table[:, i_stim].sum())

        return J, J_sample

    def apply_prospect_transform(self, X_in_):

        X_transformed = X_in_.copy()
        for col, fn in self.xt_dict.items():
            X_transformed.loc[:, col] = fn.transform(X_transformed.loc[:, col].values.reshape(-1, 1))

        return X_transformed

    def apply_scaling(self, X_in_):

        X_scaled = X_in_.copy()
        for col, fn in self.xscaler_dict.items():
            X_scaled.loc[:, col] = fn.transform(X_in_.loc[:, col].values.reshape(-1, 1))

        return X_scaled

    def fit_X_transform(self, X_list_):
        import numpy as np
        import pandas as pd
        from copy import deepcopy
        from sklearn.preprocessing import StandardScaler
        from pprint import pprint

        expected_length = 0
        thinnedXlist = list()
        for X_in_ in X_list_:
            np.testing.assert_array_equal(X_in_.columns, X_list_[0].columns)
            X_thinned, _ = self._thin_df_(X_in_, self._thin_)
            thinnedXlist.append(X_thinned)
            expected_length += X_thinned.shape[0]

        allX = pd.concat(thinnedXlist)

        assert allX.shape[0] == expected_length

        self.X_cols_in = allX.columns.to_list()

        self.X_cols = allX.columns.drop(['pot', 'outcome']).to_list()

        self.total_stim_train_ = allX.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        self.prospect_transform_kwargs_dict = self._prospect_transform_param_fn_(allX, self._prospect_transform_param_)
        self.scale_kwargs_dict = self._scale_param_fn_(allX, self._scale_param_)

        for col in self.X_cols_in:
            self.desc[col] = list()

            if self.prospect_transform_kwargs_dict[col] is not None:
                self.xt_dict[col] = ProspectTransform(**self.prospect_transform_kwargs_dict[col])
            self.desc[col].append(('prospect_transform', deepcopy(self.prospect_transform_kwargs_dict[col])))

            if self.scale_kwargs_dict[col] is not None:
                self.xscaler_dict[col] = StandardScaler(copy=True, **self.scale_kwargs_dict[col])

        X_transformed = self.apply_prospect_transform(allX)

        for col in self.xscaler_dict:
            self.xscaler_dict[col].fit(X_transformed.loc[:, col].values.reshape(-1, 1))

        for col in self.xscaler_dict:
            self.desc[col].append(('StandardScaler', {
                'kwargs': self.xscaler_dict[col].get_params(deep=True),
                'scale_': self.xscaler_dict[col].scale_,
                'mean_': self.xscaler_dict[col].mean_,
                'var_': self.xscaler_dict[col].var_,
                'n_samples_': self.xscaler_dict[col].n_samples_seen_
            }))

        print('----self.desc-------vvvv')
        pprint(self.desc)
        print('----self.desc-------^^^^')

    def X_apply(self, X_in_, thin_=None):
        import numpy as np

        if thin_ is None:
            thin_ = self._thin_

        assert X_in_.columns.to_list() == self.X_cols_in

        stim_id0 = X_in_.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        X_thinned, X_stim_counts = self._thin_df_(X_in_, thin_)
        temp_len = X_thinned.shape[0]

        Jx, Jx_sample = self._get_id_index_(X_thinned, with_respect_to_stim_id=None)

        X_transformed = self.apply_prospect_transform(X_thinned)

        X_transformed_scaled = self.apply_scaling(X_transformed)
        X_transformed_scaled.drop(['pot', 'outcome'], axis=1, inplace=True)

        stim_id1 = X_thinned.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        if not np.array_equal(stim_id0.loc[:, ('pot', 'outcome')].to_numpy(), stim_id1.loc[:, ('pot', 'outcome')].to_numpy()):
            from warnings import warn
            warn(f"ALERT: Stim ID Changed::")
            print('------Original:')
            print(stim_id0)
            print('------Final:')
            print(stim_id1)

        assert X_transformed_scaled.shape[0] == temp_len
        assert not X_transformed_scaled.isnull().any().any(), f"{X_transformed_scaled.isnull().any()}"
        assert not np.isnan(X_transformed_scaled.to_numpy()).any()

        return {'X': X_transformed_scaled.to_numpy(), 'Jx': Jx, 'Jx_sample': Jx_sample, 'Kx': X_transformed_scaled.shape[1], 'n_stimuli': X_stim_counts.shape[0], 'X_stim_counts': X_stim_counts, 'pot_col': X_thinned['pot'], 'outcome_col': X_thinned['outcome']}

    def gen_x_y_pair(self, X_in_, Y_in_):

        import numpy as np
        import pandas as pd

        assert X_in_.columns.to_list() == self.X_cols_in

        Y_cols = Y_in_.columns  # .drop(['pot', 'outcome'])

        X_cols = X_in_.columns.drop(['pot', 'outcome'])

        ### Don't filter data at all (with_respect_to_stim_id=None)
        X_dict = self.X_apply(X_in_, thin_=self._thin_)

        ### self.stim_id_table defined here
        stim_id_table = X_dict['X_stim_counts'].drop(['count'], axis=1)

        X_values = X_dict['X']

        Xdf = pd.DataFrame(data=X_values, columns=X_cols)
        Xdf['pot'] = X_dict['pot_col'].to_numpy()
        Xdf['outcome'] = X_dict['outcome_col'].to_numpy()

        assert X_values.shape[0] == Xdf.shape[0]

        ######## Y

        stimuli_counts_y = Y_in_.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})
        Ytransformed = Y_in_.drop(['pot', 'outcome'], axis=1).applymap(self.yt.y_transform)

        Y_values = Ytransformed.to_numpy()

        Ydf = Ytransformed.copy(deep=True)
        Ydf['pot'] = Y_in_['pot']
        Ydf['outcome'] = Y_in_['outcome']

        assert np.all([np.all(stimuli_counts_y.index == stim_id_table.index), np.all(stimuli_counts_y['pot'] == stim_id_table['pot']), np.all(stimuli_counts_y['outcome'] == stim_id_table['outcome'])])
        assert np.all(stimuli_counts_y.loc[:, ('pot', 'outcome')] == stim_id_table.loc[:, ('pot', 'outcome')])

        Jx, Jx_sample = self._get_id_index_(Xdf, with_respect_to_stim_id=stim_id_table)

        np.testing.assert_array_equal(Jx, X_dict['Jx'])
        np.testing.assert_array_equal(Jx_sample, X_dict['Jx_sample'])

        Jy, _ = self._get_id_index_(Ydf, with_respect_to_stim_id=stim_id_table)

        max0 = np.max(X_in_.drop(['pot', 'outcome'], axis=1).to_numpy())
        max1 = np.max(X_values)

        data_dfs = {
            'X': X_values,
            'Y': Y_values,
            'x_pot_col': X_dict['pot_col'],
            'x_outcome_col': X_dict['outcome_col'],
            'Xdf': Xdf,
            'Ydf': Ydf,
            'stim_id_map': stim_id_table,
            'Jy': Jy,
            'Jx': Jx,
            'Jx_sample': Jx_sample}

        data = {
            'Nx': X_values.shape[0],  # number of predictor observations (samples from webppl)
            'Kx': X_values.shape[1],  # number of predictors (computed appraisal features)

            'Ny': Y_values.shape[0],  # number of empirical observations (people's emotion response vectors)
            'Ky': Y_values.shape[1],  # number of emotions (20)

            'X': X_values,  # column matrix of precitor values

            'Y': Y_values,  # column matrix of empirical responses

            'n_stimuli': stim_id_table.shape[0],  # number of (pot,outcome) combinations
            'n_samples': np.max(Jx_sample) + 1,  # number of webppl samples for each (pot,outcome) combination

            'Jy': Jy + self.index_pad,  # stimulus identity of each row of Y <Nx x 1>
            'Jx': Jx + self.index_pad,  # stimulus identity of each row of X <Nx x 1>
            'Jx_sample': Jx_sample + self.index_pad,  # index of webppl sample
        }

        return data, data_dfs

    def get_params(self):
        return {
            '_thin_': self._thin_,
            'index_pad': self.index_pad,
            'affine_intercept_': self.affine_intercept,
            'prospect_transform_kwargs_dict_': self.prospect_transform_kwargs_dict,
            'scale_kwargs_dict_': self.scale_kwargs_dict,
            'X_cols_in': self.X_cols_in,
            'X_cols': self.X_cols,
            'total_stim_train_': self.total_stim_train_,
            'desc': self.desc,
            'desc_hash': 'none',  # not implemented
            'persistance_seed': self.persistance_seed,
        }


class FilterAppraisalDict():

    def __init__(self, feature_selector, return_ev=False):
        self.filter = feature_selector
        self.return_ev = return_ev

    def _filter_df_columns_(self, df, column_list):
        X = df.reindex(columns=column_list, copy=True)
        return X.reset_index(drop=True, inplace=False)

    def apply(self, emodict_iaf):
        import re
        from copy import deepcopy

        dfwide_iaf_allfeatures = get_wide_df_(deepcopy(emodict_iaf), return_ev=self.return_ev)

        ### filter columns ###
        regex = re.compile(self.filter)
        iaf_list = list(filter(regex.search, dfwide_iaf_allfeatures.columns))

        dfwide_iaf = self._filter_df_columns_(dfwide_iaf_allfeatures, iaf_list)
        dfwide_iaf['pot'] = dfwide_iaf_allfeatures['pot'].values
        dfwide_iaf['outcome'] = dfwide_iaf_allfeatures['outcome'].values

        return dfwide_iaf


def get_wide_df_(emodict, return_ev=True):
    '''
    Takes a dict with keys {*outcomes, nobs}
    '''
    import numpy as np
    import pandas as pd
    from cam_webppl_utils import unweightProbabilities

    nobsdf = emodict['nobs']
    outcomes = list(nobsdf.columns)
    df_array = list()
    for outcome in outcomes:
        pots_by_outcome_temp = nobsdf.index[nobsdf[outcome] > 0]
        for i_pot, pot in enumerate(pots_by_outcome_temp):
            data_slice = emodict[outcome].loc[pot, slice('emotionIntensities', 'prob')]
            nobs = nobsdf[outcome].loc[pot]

            data_slice_unweighted = unweightProbabilities(data_slice, nobs=nobs)
            np.testing.assert_almost_equal(data_slice_unweighted[('prob', 'prob')].sum(), 1.0)

            if return_ev:
                ev_vector = np.matmul(data_slice_unweighted[('prob', 'prob')].values.T, data_slice_unweighted['emotionIntensities'].values)
                df_out = pd.DataFrame(np.atleast_2d(ev_vector), columns=data_slice_unweighted['emotionIntensities'].columns)
            else:
                data_slice_unweighted.columns = data_slice_unweighted.columns.droplevel(0)
                df_out = data_slice_unweighted
            df_out['pot'] = pot
            df_out['outcome'] = outcome
            df_array.append(df_out)

    return pd.concat(df_array)


def prep_generic_data_pair_(ppldata):
    import numpy as np
    import pandas as pd
    from cam_webppl_utils import unweightProbabilities

    composite_training_emodict = dict()
    composite_training_emodict['nobs'] = ppldata['empiricalEmotionJudgments']['nobs'].copy(deep=True)
    composite_training_emodict['nobs'].loc[:, :] = int(0)

    composite_training_iafdict = dict()
    composite_training_iafdict['nobs'] = ppldata['level4IAF']['nobs'].copy(deep=True)
    composite_training_iafdict['nobs'].loc[:, :] = int(0)

    for outcome in ppldata['labels']['outcomes']:
        new_emp_data_list = list()
        new_iaf_data_list = list()

        for pot in ppldata['empiricalEmotionJudgments']['nobs'].index.get_level_values(0):
            df_temp_generic = unweightProbabilities(ppldata['empiricalEmotionJudgments'][outcome].loc[pot, :], ppldata['empiricalEmotionJudgments']['nobs'].loc[pot, outcome])
            df_temp_generic['pots'] = pot
            new_emp_data_list.append(df_temp_generic.set_index('pots'))

            df_temp_genericiaf = unweightProbabilities(ppldata['level4IAF'][outcome].loc[pot, :], ppldata['level4IAF']['nobs'].loc[pot, outcome])
            df_temp_genericiaf['pots'] = pot
            new_iaf_data_list.append(df_temp_genericiaf.set_index('pots'))

        composite_training_emodict[outcome] = pd.concat(new_emp_data_list)
        composite_training_iafdict[outcome] = pd.concat(new_iaf_data_list)

        for pot in np.unique(ppldata['empiricalEmotionJudgments']['nobs'].index.get_level_values(0)):
            nobs = composite_training_emodict[outcome].loc[pot, :].shape[0]
            composite_training_emodict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
            composite_training_emodict['nobs'].loc[pot, outcome] = nobs

            nobs = composite_training_iafdict[outcome].loc[pot, :].shape[0]
            composite_training_iafdict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
            composite_training_iafdict['nobs'].loc[pot, outcome] = nobs

    return composite_training_emodict, composite_training_iafdict


def prep_specific_data_pair_(distal_ppldata, nobsdf_template):
    import numpy as np
    import pandas as pd
    from cam_webppl_utils import unweightProbabilities

    composite_training_emodict = dict()
    composite_training_emodict['nobs'] = nobsdf_template.copy(deep=True)
    composite_training_emodict['nobs'].loc[:, :] = int(0)

    composite_training_iafdict = dict()
    composite_training_iafdict['nobs'] = nobsdf_template.copy(deep=True)
    composite_training_iafdict['nobs'].loc[:, :] = int(0)

    for a1 in ['C', 'D']:
        for a2 in ['C', 'D']:
            outcome = f'{a1}{a2}'
            new_emp_data_list = list()
            new_iaf_data_list = list()

            for obspot in distal_ppldata[a1]['empiricalEmotionJudgments']['nobs'].index:
                assert obspot in composite_training_emodict['nobs'].index.get_level_values(0)

            for pot in composite_training_emodict['nobs'].index.get_level_values(0):

                ### add distal prior empirical ratings
                if pot in distal_ppldata[a1]['empiricalEmotionJudgments']['nobs'].index:
                    df_temp_distal = unweightProbabilities(distal_ppldata[a1]['empiricalEmotionJudgments'][outcome].loc[pot, :], distal_ppldata[a1]['empiricalEmotionJudgments']['nobs'].loc[pot, outcome])
                    df_temp_distal['pots'] = pot
                    new_emp_data_list.append(df_temp_distal.set_index('pots'))

                    df_temp_iaf = unweightProbabilities(distal_ppldata[a1]['level4IAF'][outcome].loc[pot, :], distal_ppldata[a1]['level4IAF']['nobs'].loc[pot, outcome])
                    df_temp_iaf['pots'] = pot
                    new_iaf_data_list.append(df_temp_iaf.set_index('pots'))

            composite_training_emodict[outcome] = pd.concat(new_emp_data_list)
            composite_training_iafdict[outcome] = pd.concat(new_iaf_data_list)

            for pot in np.unique(composite_training_emodict[outcome].index.get_level_values(0)):
                nobs = composite_training_emodict[outcome].loc[pot, :].shape[0]
                composite_training_emodict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
                composite_training_emodict['nobs'].loc[pot, outcome] = nobs

                nobs = composite_training_iafdict[outcome].loc[pot, :].shape[0]
                composite_training_iafdict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
                composite_training_iafdict['nobs'].loc[pot, outcome] = nobs

    return composite_training_emodict, composite_training_iafdict


def getEmpiricalModel_pair_(emodict_emp, emodict_iaf, return_ev=True, feature_selector=''):
    import numpy as np
    from copy import deepcopy

    ### empirical ###
    dfwide_emp = get_wide_df_(emodict_emp, return_ev=return_ev).reset_index(drop=True, inplace=False)

    if not return_ev:
        dfwide_emp.drop('prob', axis=1, inplace=True)

    ### model features ###
    emodict_iaf_temp = deepcopy(emodict_iaf)
    emodict_iaf_temp['nobs'][emodict_emp['nobs'] == 0] = 0

    forward_data_prep = FilterAppraisalDict(feature_selector, return_ev=return_ev)
    dfwide_iaf = forward_data_prep.apply(emodict_iaf_temp)

    if return_ev:
        np.testing.assert_equal(dfwide_emp.shape[0], dfwide_iaf.shape[0])
        np.testing.assert_array_equal(dfwide_emp['pot'], dfwide_iaf['pot'])

    return dfwide_emp, dfwide_iaf, forward_data_prep


##### data prep func

def prospect_transform_appraisal_variables_func(X_, prospect_transform_param):
    """
    apply prospect_transform_param['base_kwargs'] to base features
    apply prospect_transform_param['repu_kwargs'] to reputation features
    do not apply any transform to PEa2 vars that are log1p transformed
    """
    import re
    prospect_transform_kwargs_dict_ = dict()
    for col in X_.columns.to_list():
        if bool(re.match(r"\S+\[base\S+\]", col)):  # base
            prospect_transform_kwargs_dict_[col] = prospect_transform_param['base_kwargs']
        elif bool(re.match(r"\S+\[repu\S+\]", col)):  # repu
            prospect_transform_kwargs_dict_[col] = prospect_transform_param['repu_kwargs']
        elif col in ['PEa2pot']:
            prospect_transform_kwargs_dict_[col] = prospect_transform_param['base_kwargs']
        elif col in ['PEa2lnpotunval', 'PEa2lnpot', 'PEa2raw', 'PEa2unval', 'absPEa2lnpot']:
            prospect_transform_kwargs_dict_[col] = None
        elif bool(re.match(r"^(pot|outcome)$", col)):  # skip
            prospect_transform_kwargs_dict_[col] = None
        else:
            raise ValueError(f"prospect_transform_fn_(): {col} not found in {X_.columns.to_list()}")
    return prospect_transform_kwargs_dict_


def scale_appraisal_variables_func(X_, scale_param):
    """
    scale base, reputation, and \pi_{a_2} features by scale_param['all'] (e.g. unit variance, mean kept)
    """
    import re

    scale_kwargs_dict_ = dict()
    for col in X_.columns.to_list():
        if bool(re.match(r"\S+\[base\S+\]", col)):  # base
            scale_kwargs_dict_[col] = scale_param['all']
        elif bool(re.match(r"\S+\[repu\S+\]", col)):  # repu
            scale_kwargs_dict_[col] = scale_param['all']
        elif col in ['PEa2lnpotunval', 'PEa2lnpot', 'PEa2pot', 'PEa2raw', 'PEa2unval', 'absPEa2lnpot']:
            scale_kwargs_dict_[col] = scale_param['all']
        elif bool(re.match(r"^(pot|outcome)$", col)):  # skip
            scale_kwargs_dict_[col] = None
        else:
            assert False, f"scale_features_1(): {col} mismatch in {X_.columns.to_list()}"
    return scale_kwargs_dict_


def reformat_ppldata_allplayers(cpar_path=None, feature_selector=None, feature_selector_label=None):
    from cam_collect_torch_results import get_ppldata_from_cpar

    ppldata, distal_prior_ppldata = get_ppldata_from_cpar(cpar_path=cpar_path)

    nobsdf_template = ppldata['empiricalEmotionJudgments']['nobs'].copy()

    ppldatasets = dict()

    composite_emodict_, composite_cafdict_ = prep_generic_data_pair_(ppldata)
    Y_filtered, X_filtered, _ = getEmpiricalModel_pair_(composite_emodict_, composite_cafdict_, feature_selector=feature_selector, return_ev=False)

    ppldatasets['generic'] = dict(
        X=X_filtered,
        Y=Y_filtered,
    )

    for stimid, ppldatadistal_ in distal_prior_ppldata.items():
        composite_emodict_, composite_cafdict_ = prep_specific_data_pair_(ppldatadistal_, nobsdf_template)
        Y_filtered, X_filtered, _ = getEmpiricalModel_pair_(composite_emodict_, composite_cafdict_, feature_selector=feature_selector, return_ev=False)

        ppldatasets[stimid] = dict(
            X=X_filtered,
            Y=Y_filtered,
        )

    return ppldatasets


def format_data_for_torch(dfs_):

    import numpy as np
    import pandas as pd

    outcomes = ['CC', 'CD', 'DC', 'DD']
    pots_ = sorted(dfs_['x_pot_col'].unique().tolist())

    Yshort_ = dict()
    X_ = np.full([len(outcomes), len(pots_), np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < 4 outcome, n pot, m features >
    for i_outcome, outcome in enumerate(outcomes):
        Yshort_[outcome] = list()
        for i_pot, pot in enumerate(pots_):
            idxer = ((dfs_['x_outcome_col'] == outcome) & (dfs_['x_pot_col'] == pot)).to_numpy()
            X_[i_outcome, i_pot, :, :] = dfs_['X'][idxer, :]

            idyer = ((dfs_['Ydf']['outcome'] == outcome) & (dfs_['Ydf']['pot'] == pot)).to_numpy()
            Yshort_[outcome].append(dfs_['Y'][idyer, :])

    Xdims = dict(
        outcome=outcomes,
        pot=pots_,
        sample=np.unique(dfs_['Jx_sample']),
        iaf=dfs_['Xdf'].columns.drop(['pot', 'outcome']).tolist(),
    )
    Ydims = dict(
        outcome=outcomes,
        pot=pots_,
        response='variable',
        emotion=dfs_['Ydf'].columns.drop(['pot', 'outcome']).tolist(),
    )

    jx_map_list = list()
    jy_map_list = list()
    Xlong_ = np.full([np.unique(dfs_['Jx']).size, np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < n pot-outcome, n appraisal features >
    for jx in range(len(np.unique(dfs_['Jx']))):

        Xlong_[jx, :, :] = dfs_['X'][dfs_['Jx'] == jx, :]

        jx_map_list.append([np.array([jx]), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'pot'].unique(), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'outcome'].unique()])

        jy_map_list.append([np.array([jx]), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'pot'].unique(), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'outcome'].unique()])

    assert not np.any(np.isnan(X_))
    assert not np.any(np.isnan(Xlong_))

    torchdata = dict(
        Xshort=X_,
        Xshortdims=Xdims,
        Yshort=Yshort_,
        Yshortdims=Ydims,
        Xlong=Xlong_,
        Ylong=dfs_['Y'],
        Jy=dfs_['Jy'],
        Jxmap=pd.DataFrame(np.concatenate(jx_map_list, axis=1).T, columns=['J', 'pot', 'outcome']),
        Jymap=pd.DataFrame(np.concatenate(jy_map_list, axis=1).T, columns=['J', 'pot', 'outcome']),
    )

    return torchdata


def sbatch_torch_array(job_array, behavior=None, codedir_path=None, dataoutbase_path=None, job_name=None, mem_per_job=None, cpus_per_task=None, partition=None, time=None, exclude=None, dependency=None):

    import numpy as np
    import dill
    import subprocess
    import re
    from cam_utils import random_string_alphanumeric

    ### hardcoded path ###
    script_path = codedir_path / "launch_emotorch.sbatch"

    data_cache_dir = dataoutbase_path / "temp_pickles_torch"
    data_cache_dir.mkdir(parents=True, exist_ok=True)

    assert len(job_array) > 0, "job_array is empty"
    assert script_path.parent.is_dir(), f"script_path.parent is not a directory: {script_path.parent}"
    assert script_path.is_file(), f"script_path is not a file: {script_path}"
    assert data_cache_dir.parent.is_dir(), f"data_cache_dir.parent is not a directory: {data_cache_dir.parent}"
    assert not (behavior is None), "behavior is required"

    if not data_cache_dir.is_dir():
        data_cache_dir.mkdir(parents=True, exist_ok=True)

    if job_name is None:
        job_name = "camTorchDefault"

    if mem_per_job is None:
        mem_per_job = 8

    if cpus_per_task is None:
        cpus_per_task = 1

    if time is None:
        time = "0-04:00:00"
    elif isinstance(time, int):
        time = f"0-{time}:00:00"

    pickle_dir_path = None
    file_unique = False
    while not file_unique:
        pickle_dir_path = data_cache_dir / f'{job_name}_{random_string_alphanumeric()}'
        if not pickle_dir_path.exists():
            file_unique = True

    pickle_dir_path.mkdir(parents=True, exist_ok=True)
    pickle_path = pickle_dir_path / 'all_data.dill'

    log_dir_path = pickle_dir_path / 'logs'
    log_dir_path.mkdir(parents=True, exist_ok=True)

    output_pattern = log_dir_path / f"{job_name}_%A_%a.txt"

    with open(pickle_path, 'wb') as f:
        dill.dump(job_array, f, protocol=-4)

    cmd_list = [
        "sbatch",
        f"--array=1-{len(job_array)}",
        f"--job-name={job_name}",
        f"--output={str(output_pattern)}",
        f"--mem={mem_per_job}GB",
        f"--cpus-per-task={cpus_per_task}",
        f"--time={time}",
    ]

    if not (partition is False or partition is None or partition == ''):
        cmd_list.append(f"--partition={partition}")  # use-everything

    if exclude is not None:
        cmd_list.append(f"--exclude={exclude}")

    if dependency is not None:
        dep_array = np.array(dependency)
        if dep_array.ndim == 0:
            # If the input is a scalar, convert it to a 1D array
            dep_array = np.array([dep_array.item()])
        dependency_str = ':'.join(np.array(dep_array.flatten(), dtype='str').tolist())
        cmd_list.append(f"--dependency=afterok:{dependency_str}")

    cmd_list.extend([
        str(script_path),
        str(script_path),
        str(pickle_path),
        behavior,
    ])

    clout = subprocess.run(cmd_list, capture_output=True, encoding='utf-8')

    error = False
    try:
        depend = re.search(r'([0-9]+)', clout.stdout.strip()).group(0)
    except AttributeError:
        error = True
    finally:
        if error or clout.returncode != 0:
            print(f"ERROR: sbatch returned: {clout.returncode}")
        else:
            print(f"sbatch returned: {clout.returncode}")
        print(f">> {' '.join(clout.args)}")
        if clout.stdout.strip():
            print(f"  {clout.stdout}")
        if clout.stderr.strip():
            print(f"  {clout.stderr}")
    if error:
        raise ValueError("sbatch_torch_array(): failed to get job id")

    return dict(dependency=depend, clout=clout, cmd=' '.join(cmd_list))
