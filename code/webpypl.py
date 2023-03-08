#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""webpypl.py
Webppl play goldenballs
"""


class Timer:

    def __init__(self):
        self._start_time = None
        self._timepoints = list()
        self.__last_timepoint = None

    def _format_time(self, time_delta):
        return f"{time_delta:0.4f}"

    def start(self):
        # """Start"""
        import time
        if self._start_time is not None:
            raise Exception(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()
        self.__last_timepoint = self._start_time

    def lap(self, label=None):
        import time
        t = time.perf_counter()
        if label is None:
            label = len(self._timepoints)
        self._timepoints.append((f'{label}', t - self._start_time, t - self.__last_timepoint))
        self.__last_timepoint = t

    def report(self):
        print('\n-=-=-=-=-=-=-=-=-=-=-=-=-= TIMER')
        for tp in self._timepoints:
            print(f"  {self._format_time(tp[1])} :: {self._format_time(tp[2])} >> {tp[0]} <<")
        print('-=-=-=-=-=-=-=-=-=-=-=-=-= TIMER')


class gameData:

    def __init__(self, emoDict):
        # from webpypl import get_wide_df_
        self.emoDict = emoDict
        self.dfwide_ev = get_wide_df_(emoDict, return_ev=True)
        self.dfwide = get_wide_df_(emoDict, return_ev=False)

    def df(self, ev=False, column_filter=None, outcome_filter=None, pot_filter=None):
        if ev:
            df = self.dfwide_ev
        else:
            df = self.dfwide

        return df


class logit_logistic_transform():

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

    def get_functions(self):
        def y_transform(x): return x if self.bypass else self.logit_transform(self.affine_compress(x))

        def yhat_transform(x): return x if self.bypass else self.affine_stretch(self.logistic_transform(x))

        return y_transform, yhat_transform


class prospect_transform():

    def __init__(self, alpha_=1.0, beta_=1.0, lambda_=1.0, intercept=0.0, noise_scale=0.0, log1p=False):
        self.alpha_ = alpha_
        self.beta_ = beta_
        self.lambda_ = lambda_
        self.sigma_ = noise_scale
        self.b0_ = intercept
        self.log1p = log1p

    def kahneman(self, x):
        if x < 0:
            y = -1 * self.lambda_ * ((-1 * x) ** self.beta_)
        else:
            y = x ** self.alpha_
        return y

    def vectorized(self):
        import numpy as np
        return np.vectorize(self.kahneman)

    def _power_fn_(self, X_):
        import numpy as np

        power_fn = self.vectorized()

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


class filter_iaf_dict():

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

        ### filter columns
        regex = re.compile(self.filter)
        iaf_list = list(filter(regex.search, dfwide_iaf_allfeatures.columns))

        dfwide_iaf = self._filter_df_columns_(dfwide_iaf_allfeatures, iaf_list)
        dfwide_iaf['pot'] = dfwide_iaf_allfeatures['pot'].values
        dfwide_iaf['outcome'] = dfwide_iaf_allfeatures['outcome'].values

        return dfwide_iaf


class transform_data():
    """
    For training, return X,Y pair with a index map that relates the observations
    For testing, take an X containing any stim identities and transform it independant of 
    it's relationship to the training data.

    To score Y data, just need to know the new X:Y map (not the stim IDs used to train the model)
    """
    import numpy as np
    import pandas as pd

    def __init__(self, _thin_=1, index_pad=0, affine_intercept_=None, prospect_transform_kwargs_=None, vars_to_exclude_from_prospect_scaling=None, scale_kwargs_=None, verbose=False):
        from sklearn.preprocessing import StandardScaler

        self._thin_ = _thin_
        self.index_pad = index_pad

        self.scale_kwargs = scale_kwargs_
        self.xscaler = StandardScaler(copy=True, **scale_kwargs_)

        self.prospect_transform_kwargs = prospect_transform_kwargs_
        self.xt = prospect_transform(**prospect_transform_kwargs_)
        self.vars_to_exclude_from_prospect_scaling = vars_to_exclude_from_prospect_scaling

        self.affine_intercept = affine_intercept_
        self.yt = logit_logistic_transform(affine_intercept_)

        self.X_cols_in = None
        self.X_cols = None

        self.verbose = verbose

    def _thin_df_(self, X_in_, thin_):
        """
        thinned according to it's own stim identity -- all stim included
        """
        import numpy as np
        import pandas as pd
        ### prune model samples

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
        # (only important for the current pytorch implementation that reshapes the data to be cuboidal

        xthin = pd.concat(thinned_X_array).reset_index(drop=True)

        assert np.unique(indexer_counts).size == 1  # data is made cuboidal

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

        ### assign an ID numer to each row of observed M
        for i_stim, stim in stim_id_table.iterrows():
            data_binary_table[:, i_stim] = (M['pot'] == stim['pot']) & (M['outcome'] == stim['outcome'])

            J[data_binary_table[:, i_stim]] = stim.name

            if uniform:
                J_sample[data_binary_table[:, i_stim]] = np.arange(0, data_binary_table[:, i_stim].sum())

        return J, J_sample

    def apply_prospect_transform(self, X_in_):
        import numpy as np

        #### deal with variables that should not be prospect scaled
        if self.vars_to_exclude_from_prospect_scaling:
            X_thinned_original = X_in_.copy()

            print('self.vars_to_exclude_from_prospect_scaling')
            print(self.vars_to_exclude_from_prospect_scaling)

            X_thinned_cols_heldout = X_in_.loc[:, self.vars_to_exclude_from_prospect_scaling].copy()
            X_in_.loc[:, self.vars_to_exclude_from_prospect_scaling].copy()

            ### get column location
            vars_to_exclude_from_prospect_scaling_loc = list()
            for varid in self.vars_to_exclude_from_prospect_scaling:
                assert np.sum(X_in_.columns.to_numpy() == varid) == 1
                temploc = np.argwhere(X_in_.columns.to_numpy() == varid)[0]
                assert len(temploc) == 1
                vars_to_exclude_from_prospect_scaling_loc.append(temploc[0])

            X_thinned = X_in_.drop(columns=self.vars_to_exclude_from_prospect_scaling)

        else:
            X_thinned = X_in_

        X_unscaled = self.xt.transform(X_thinned.drop(['pot', 'outcome'], axis=1))

        print('X_unscaled.shape 1 ')
        print(X_unscaled.shape)

        ### add columns left out of prospect scaling back
        if self.vars_to_exclude_from_prospect_scaling:
            print(f"Excluding from prospect transform: {vars_to_exclude_from_prospect_scaling_loc}")

            for iloc, hoc in zip(vars_to_exclude_from_prospect_scaling_loc, X_thinned_cols_heldout.columns.to_list()):
                X_unscaled = np.insert(X_unscaled, iloc, X_thinned_cols_heldout.loc[:, hoc].to_numpy(), axis=1)

            for iloc, hoc in zip(vars_to_exclude_from_prospect_scaling_loc, X_thinned_cols_heldout.columns.to_list()):
                np.testing.assert_array_equal(X_thinned_original.loc[:, hoc].to_numpy(), np.squeeze(X_unscaled[:, iloc]))

            assert X_unscaled.shape[0] == X_thinned_original.shape[0]
            assert X_unscaled.shape[1] == len(X_in_.columns.drop(['pot', 'outcome']))

        return X_unscaled

    def fit_X_transform(self, X_list_):
        import numpy as np
        import pandas as pd

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

        X_unscaled = self.apply_prospect_transform(allX)

        self.xscaler.fit(X_unscaled)

        # if self.verbose:
        #     print('>>>>>>>>>>StandardScalerFit<<<<<<<<')
        #     print(f"Using StandardScaler with scale: {self.xscaler.scale_}, mean: {self.xscaler.mean_} var: {self.xscaler.var_}")

    def X_apply(self, X_in_, thin_=None):
        import numpy as np

        if thin_ is None:
            thin_ = self._thin_

        assert X_in_.columns.to_list() == self.X_cols_in

        stim_id0 = X_in_.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        X_thinned, X_stim_counts = self._thin_df_(X_in_, thin_)

        ### Don't filter data at all
        Jx, Jx_sample = self._get_id_index_(X_thinned, with_respect_to_stim_id=None)

        X_unscaled = self.apply_prospect_transform(X_thinned)

        X_scaled = self.xscaler.transform(X_unscaled)

        stim_id1 = X_thinned.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        if not np.array_equal(stim_id0.loc[:, ('pot', 'outcome')].to_numpy(), stim_id1.loc[:, ('pot', 'outcome')].to_numpy()):
            from warnings import warn
            warn(f"ALERT: Stim ID Changed::")
            print('------Original:')
            print(stim_id0)
            print('------Final:')
            print(stim_id1)

        # if self.verbose:
        #     print(f"Using StandardScaler with scale: {self.xscaler.scale_}, mean: {self.xscaler.mean_} var: {self.xscaler.var_}")

        return {'X': X_scaled, 'Jx': Jx, 'Jx_sample': Jx_sample, 'Kx': X_scaled.shape[1], 'n_stimuli': X_stim_counts.shape[0], 'X_stim_counts': X_stim_counts, 'pot_col': X_thinned['pot'], 'outcome_col': X_thinned['outcome']}

    def gen_x_y_pair(self, X_in_, Y_in_):
        """

        """
        import numpy as np
        import pandas as pd

        assert X_in_.columns.to_list() == self.X_cols_in

        Y_cols = Y_in_.columns.drop(['pot', 'outcome'])

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

        print([np.all(stimuli_counts_y.index == stim_id_table.index), np.all(stimuli_counts_y['pot'] == stim_id_table['pot']), np.all(stimuli_counts_y['outcome'] == stim_id_table['outcome'])])
        assert np.all([np.all(stimuli_counts_y.index == stim_id_table.index), np.all(stimuli_counts_y['pot'] == stim_id_table['pot']), np.all(stimuli_counts_y['outcome'] == stim_id_table['outcome'])])
        assert np.all(stimuli_counts_y.loc[:, ('pot', 'outcome')] == stim_id_table.loc[:, ('pot', 'outcome')])

        Jx, Jx_sample = self._get_id_index_(Xdf, with_respect_to_stim_id=stim_id_table)

        np.testing.assert_array_equal(Jx, X_dict['Jx'])
        np.testing.assert_array_equal(Jx_sample, X_dict['Jx_sample'])

        Jy, _ = self._get_id_index_(Ydf, with_respect_to_stim_id=stim_id_table)

        # print(f'\ndataformater: \n\t X range ({X_values.min():0.2}, {X_values.max():0.2}), \n\t Y range ({Y_values.min():0.2}, {Y_values.max():0.2}) \n\t Xcols: [{",".join(X_cols)}]')

        maxx0 = np.max(X_in_.drop(['pot', 'outcome'], axis=1).to_numpy())
        maxx1 = np.max(X_values)
        print(f"DataFormater::\n\tMax {maxx0:.2}  -->  {maxx1:.2}")

        ### verification check for data
        # print('-----NOTINGNPRINGS--------')
        # tempdf = self.xscaler.fit_transform(self.xt.transform(X_in_.loc[:,X_cols]))
        # for ix,rx in enumerate(X_values):
        #     row_matches = (np.abs(tempdf-rx) < 1e-11).all(axis=1)
        #     if row_matches.sum() == 0:
        #         for ii in list(np.arange(-15,0,1)):
        #             print(f"X failed: For 10^{ii}, mismatched {(np.abs(tempdf-rx) > 10.0**float(ii)).any(axis=1).sum()}")
        #     # assert row_matches.sum() > 0
        #     if np.sum( (X_in_.loc[row_matches,('pot','outcome')] == stim_id_table.iloc[ Jx[ix], : ]).all(axis=1) ) == 0:
        #         print('so broken')
        #     assert 0 < np.sum( (X_in_.loc[row_matches,('pot','outcome')] == stim_id_table.iloc[ Jx[ix], : ]).all(axis=1) )
        #     np.testing.assert_array_equal(Xdf.loc[:,('pot','outcome')].iloc[ix], stim_id_table.iloc[ Jx[ix], : ])

        # tempdf = Y_in_.loc[:,Y_cols].applymap( self.yt.y_transform )
        # for ix,rx in enumerate(Y_values):
        #     print('BROKENY')
        #     row_matches = (np.abs(tempdf-rx) < 1e-11).all(axis=1)
        #     if row_matches.sum() < 1:
        #         for ii in np.arange(-15,0,1):
        #             print(f"Y failed: For 10^{ii}, mismatched {(np.abs(tempdf-rx) > 10.0**float(ii)).any(axis=1).sum()}")
        #     assert row_matches.sum() > 0
        #     assert 0 < np.sum( (Y_in_.loc[row_matches,('pot','outcome')] == stim_id_table.iloc[ Jy[ix], : ]).all(axis=1) )
        #     np.testing.assert_array_equal(Ydf.loc[:,('pot','outcome')].iloc[ix], stim_id_table.iloc[ Jy[ix], : ])

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
            'Kx': X_values.shape[1],  # number of predictors (22 inverse appraisal features)

            'Ny': Y_values.shape[0],  # number of empirical observations (people's emotion response vectors)
            'Ky': Y_values.shape[1],  # number of emotions (20)

            'X': X_values,  # column matrix of precitor values

            'Y': Y_values,  # column matrix of empirical responses

            'n_stimuli': stim_id_table.shape[0],  # number of (pot,outcome) combinations (96)
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
            'prospect_transform_kwargs_': self.prospect_transform_kwargs,
            'scale_kwargs_': self.scale_kwargs,
            'X_cols_in': self.X_cols_in,
            'X_cols': self.X_cols,
            'total_stim_train_': self.total_stim_train_,
            'x_scale_fit': {'kwargs': self.xscaler.get_params(deep=True), 'scale_': self.xscaler.scale_, 'mean_': self.xscaler.mean_, 'var': self.xscaler.var_, 'n_samples_': self.xscaler.n_samples_seen_},
        }


class transform_data_multifunction():
    """
    For training, return X,Y pair with a index map that relates the observations
    For testing, take an X containing any stim identities and transform it independant of 
    it's relationship to the training data.

    To score Y data, just need to know the new X:Y map (not the stim IDs used to train the model)
    """
    import numpy as np
    import pandas as pd

    def __init__(self, index_pad=0, _thin_=1, affine_intercept_=None, prospect_transform_param_=None, scale_param_=None, verbose=False):
        import random
        # from sklearn.preprocessing import StandardScaler

        self.index_pad = index_pad
        self._thin_ = _thin_

        self.affine_intercept = affine_intercept_
        self.yt = logit_logistic_transform(affine_intercept_)

        self._prospect_transform_param_fn_ = prospect_transform_param_['fn']
        self._prospect_transform_param_ = prospect_transform_param_['param']
        self.prospect_transform_kwargs_dict = dict()
        self.xt_dict = dict()
        # self.xt = prospect_transform(**prospect_transform_kwargs_)

        self._scale_param_fn_ = scale_param_['fn']
        self._scale_param_ = scale_param_['param']
        self.scale_kwargs_dict = dict()
        self.xscaler_dict = dict()
        # self.xscaler = StandardScaler(copy=True, **scale_kwargs_)

        self.total_stim_train_ = None

        self.X_cols_in = None
        self.X_cols = None

        self.desc = dict()

        self.desc_hash = None
        self.persistance_seed = random.randint(0, 1e22)

        self.verbose = verbose

    def _thin_df_(self, X_in_, thin_):
        """
        thinned according to it's own stim identity -- all stim included
        """
        import numpy as np
        import pandas as pd
        ### prune model samples

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
        # (only important for the current pytorch implementation that reshapes the data to be cuboidal

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

        ### assign an ID numer to each row of observed M
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
                self.xt_dict[col] = prospect_transform(**self.prospect_transform_kwargs_dict[col])
            self.desc[col].append(('prospect_transform', deepcopy(self.prospect_transform_kwargs_dict[col])))

            if self.scale_kwargs_dict[col] is not None:
                self.xscaler_dict[col] = StandardScaler(copy=True, **self.scale_kwargs_dict[col])
            self.desc[col].append(('StandardScaler', deepcopy(self.scale_kwargs_dict[col])))

        X_transformed = self.apply_prospect_transform(allX)

        for col in self.xscaler_dict:
            if self.xscaler_dict[col] is not None:
                self.xscaler_dict[col].fit(X_transformed.loc[:, col].values.reshape(-1, 1))
                self.desc[col].append({'kwargs': self.xscaler_dict[col].get_params(deep=True), 'scale_': self.xscaler_dict[col].scale_, 'mean_': self.xscaler_dict[col].mean_, 'var_': self.xscaler_dict[col].var_, 'n_samples_': self.xscaler_dict[col].n_samples_seen_})

        print('----self.desc-------vvvv')
        from pprint import pprint
        pprint(self.desc)
        print('----self.desc-------^^^^')
        # if self.verbose:
        #     print('>>>>>>>>>>StandardScalerFit<<<<<<<<')
        #     print(f"Using StandardScaler with scale: {self.xscaler.scale_}, mean: {self.xscaler.mean_} var: {self.xscaler.var_}")

    def X_apply(self, X_in_, thin_=None):
        import numpy as np

        if thin_ is None:
            thin_ = self._thin_

        assert X_in_.columns.to_list() == self.X_cols_in

        stim_id0 = X_in_.groupby(['outcome', 'pot']).size().reset_index().rename(columns={0: 'count'})

        X_thinned, X_stim_counts = self._thin_df_(X_in_, thin_)
        temp_len = X_thinned.shape[0]

        ### Don't filter data at all
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

        # if self.verbose:
        #     print(f"Using StandardScaler with scale: {self.xscaler.scale_}, mean: {self.xscaler.mean_} var: {self.xscaler.var_}")

        assert X_transformed_scaled.shape[0] == temp_len
        assert not X_transformed_scaled.isnull().any().any(), f"{X_transformed_scaled.isnull().any()}"
        assert not np.isnan(X_transformed_scaled.to_numpy()).any()

        return {'X': X_transformed_scaled.to_numpy(), 'Jx': Jx, 'Jx_sample': Jx_sample, 'Kx': X_transformed_scaled.shape[1], 'n_stimuli': X_stim_counts.shape[0], 'X_stim_counts': X_stim_counts, 'pot_col': X_thinned['pot'], 'outcome_col': X_thinned['outcome']}

    def gen_x_y_pair(self, X_in_, Y_in_):
        """

        """
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

        print([np.all(stimuli_counts_y.index == stim_id_table.index), np.all(stimuli_counts_y['pot'] == stim_id_table['pot']), np.all(stimuli_counts_y['outcome'] == stim_id_table['outcome'])])
        assert np.all([np.all(stimuli_counts_y.index == stim_id_table.index), np.all(stimuli_counts_y['pot'] == stim_id_table['pot']), np.all(stimuli_counts_y['outcome'] == stim_id_table['outcome'])])
        assert np.all(stimuli_counts_y.loc[:, ('pot', 'outcome')] == stim_id_table.loc[:, ('pot', 'outcome')])

        Jx, Jx_sample = self._get_id_index_(Xdf, with_respect_to_stim_id=stim_id_table)

        np.testing.assert_array_equal(Jx, X_dict['Jx'])
        np.testing.assert_array_equal(Jx_sample, X_dict['Jx_sample'])

        Jy, _ = self._get_id_index_(Ydf, with_respect_to_stim_id=stim_id_table)

        # print(f'\ndataformater: \n\t X range ({X_values.min():0.2}, {X_values.max():0.2}), \n\t Y range ({Y_values.min():0.2}, {Y_values.max():0.2}) \n\t Xcols: [{",".join(X_cols)}]')

        maxx0 = np.max(X_in_.drop(['pot', 'outcome'], axis=1).to_numpy())
        maxx1 = np.max(X_values)
        print(f"DataFormater::\n\tMax {maxx0:.2}  -->  {maxx1:.2}")

        ### verification check for data
        # print('-----NOTINGNPRINGS--------')
        # tempdf = self.xscaler.fit_transform(self.xt.transform(X_in_.loc[:,X_cols]))
        # for ix,rx in enumerate(X_values):
        #     row_matches = (np.abs(tempdf-rx) < 1e-11).all(axis=1)
        #     if row_matches.sum() == 0:
        #         for ii in list(np.arange(-15,0,1)):
        #             print(f"X failed: For 10^{ii}, mismatched {(np.abs(tempdf-rx) > 10.0**float(ii)).any(axis=1).sum()}")
        #     # assert row_matches.sum() > 0
        #     if np.sum( (X_in_.loc[row_matches,('pot','outcome')] == stim_id_table.iloc[ Jx[ix], : ]).all(axis=1) ) == 0:
        #         print('so broken')
        #     assert 0 < np.sum( (X_in_.loc[row_matches,('pot','outcome')] == stim_id_table.iloc[ Jx[ix], : ]).all(axis=1) )
        #     np.testing.assert_array_equal(Xdf.loc[:,('pot','outcome')].iloc[ix], stim_id_table.iloc[ Jx[ix], : ])

        # tempdf = Y_in_.loc[:,Y_cols].applymap( self.yt.y_transform )
        # for ix,rx in enumerate(Y_values):
        #     print('BROKENY')
        #     row_matches = (np.abs(tempdf-rx) < 1e-11).all(axis=1)
        #     if row_matches.sum() < 1:
        #         for ii in np.arange(-15,0,1):
        #             print(f"Y failed: For 10^{ii}, mismatched {(np.abs(tempdf-rx) > 10.0**float(ii)).any(axis=1).sum()}")
        #     assert row_matches.sum() > 0
        #     assert 0 < np.sum( (Y_in_.loc[row_matches,('pot','outcome')] == stim_id_table.iloc[ Jy[ix], : ]).all(axis=1) )
        #     np.testing.assert_array_equal(Ydf.loc[:,('pot','outcome')].iloc[ix], stim_id_table.iloc[ Jy[ix], : ])

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
            'Kx': X_values.shape[1],  # number of predictors (22 inverse appraisal features)

            'Ny': Y_values.shape[0],  # number of empirical observations (people's emotion response vectors)
            'Ky': Y_values.shape[1],  # number of emotions (20)

            'X': X_values,  # column matrix of precitor values

            'Y': Y_values,  # column matrix of empirical responses

            'n_stimuli': stim_id_table.shape[0],  # number of (pot,outcome) combinations (96)
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
            'desc_hash': 'none',  # TODO
            'persistance_seed': self.persistance_seed,
            # 'x_scale_fit': {'kwargs': self.xscaler.get_params(deep=True), 'scale_': self.xscaler.scale_, 'mean_': self.xscaler.mean_, 'var': self.xscaler.var_, 'n_samples_': self.xscaler.n_samples_seen_},
        }


def encode_number(n, desired_prefix=None):
    import numpy as np
    import decimal

    done = False
    prefixes = ['d', 'c', 'm', 'n', 'o', 'p', 'q', 'r', 's']
    if desired_prefix is not None and desired_prefix in prefixes:
        p = prefixes.index(desired_prefix) + 1

        nout_test = n * 10 ** p
        if nout_test - np.floor(nout_test) <= 0:
            nout = f"{prefixes[p - 1]}{int(round(nout_test))}"
            done = True

    if not done:

        n_str = f"{n:f}"
        if '.' in n_str and not np.all(np.array(list(n_str.split('.')[1])) == '0'):
            rp = len(n_str.strip('0').split('.')[1])
            nout = f"{prefixes[rp - 1]}{int(np.floor(n * 10 ** rp))}"

        else:
            nout = f"{int(round(n))}"
    return nout


def cache_data(data, outdir=None, fname_base='datacache', pickler='pickle', overwrite_existing=True, verbose=True, hashlen=16):
    from pathlib import Path
    import pickle
    import dill
    from hashlib import blake2b

    def randomStringDigits(stringLength=8):
        """Generate a random string of letters and digits """
        import string
        import random
        lettersAndDigits = string.ascii_lowercase + string.digits
        return ''.join(random.choice(lettersAndDigits) for _ in range(stringLength))

    if pickler == 'pickle':
        pickler_ = pickle
    elif pickler == 'dill':
        pickler_ = dill

    assert isinstance(fname_base, str), type(fname_base)

    """
    demohash = hashlib.md5('somestring'.encode('ascii')).hexdigest()
    """

    ### save temporary file ###

    # fpath_temp0 = outdir.resolve() / fname_base
    fpath_temp0 = outdir / fname_base
    assert fpath_temp0.suffix in ['', '.pkl']
    fpath_temp1 = fpath_temp0.with_suffix('')
    # assert not fpath_temp1.exists()

    fpath_unique_ = False
    while not fpath_unique_:
        fpath_temp = fpath_temp1.parent / f"_tempcache_{fpath_temp1.name}_{randomStringDigits()}.pkl"
        fpath_unique_ = not fpath_temp.exists()

    assert fpath_temp.parent == fpath_temp0.parent

    fpath_temp.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath_temp, 'wb') as f:
        pickler_.dump(data, f, protocol=-1)

    with open(fpath_temp, 'rb') as f:
        hasher = blake2b(digest_size=hashlen)
        while chunk := f.read(8192):  # 8 kiB
            hasher.update(chunk)
    file_hash_ = hasher.hexdigest()

    fpath_final = fpath_temp.parent / f"{fpath_temp1}_{file_hash_}.pkl"

    if fpath_final.is_file():
        if overwrite_existing:
            try:
                fpath_final.unlink()
            except Exception as e:
                print(f"WARNING failed to delete >>{e}<< at: {fpath_final}")
            data_cache_path = fpath_temp.rename(fpath_final)
            if verbose:
                print(f"data cache replaced at {data_cache_path}")
                if fpath_temp.is_file():
                    print(f"ERROR -- temp file not deleted at {fpath_temp}")
        else:
            fpath_temp.unlink()
            data_cache_path = fpath_final
            if verbose:
                print(f"data cache already exists, leaving original at {data_cache_path}")
    else:
        data_cache_path = fpath_temp.rename(fpath_final)
        if verbose:
            print(f"data cache saved at {data_cache_path}")

    if fpath_temp.is_file():
        print(f"ERROR2 -- temp file not deleted at {fpath_temp}")
    return data_cache_path


def saveScript(paths):
    import subprocess
    from pathlib import PurePath
    import shutil
    import time

    codedumpsdir = paths['expDir'] / '_codedumps_gitignore' / 'simultaneoussymmetric'
    codedumpsdir.mkdir(exist_ok=True, parents=True)
    dataout_codedumpsdir = paths['dataOut'] / 'code_dump'
    dataout_codedumpsdir.mkdir(exist_ok=True, parents=True)
    timestr = time.strftime("%Y_%m_%d-%H%M%S")

    for ext in ['py', 'wppl', 'sbatch', 'stan']:
        for file in list(paths['code'].glob(f'[A-Za-z]*.{ext}')):
            shutil.copy(paths['code'] / PurePath(file), codedumpsdir / PurePath(f'{file.name}-{timestr}').with_suffix(file.suffix))
            shutil.copy(paths['code'] / PurePath(file), dataout_codedumpsdir / PurePath(f'{file.name}').with_suffix(file.suffix))

    freezeout_conda = subprocess.run('conda list --export > ' + str(dataout_codedumpsdir / 'environment-package-list.txt'), check=True, capture_output=True, shell=True)
    freezeout_pip = subprocess.run('pip freeze > ' + str(dataout_codedumpsdir / 'environment-package-list-pip.txt'), check=True, capture_output=True, shell=True)


def compileModel(paths):
    import subprocess
    import shutil

    webppl_model_copy = paths['dataOut'] / 'webppl_exe' / paths['model'].name
    webppl_model_copy.parent.mkdir(exist_ok=True, parents=True)

    ### copy webppl model to executable directory ###
    assert paths['model'].is_file()
    shutil.copy(paths['model'], webppl_model_copy)

    executable = webppl_model_copy.with_suffix('.js')

    cmd = [
        "webppl", str(webppl_model_copy),
        "--require", "webppl-json",
        "--compile",
        "--out", str(executable)]
    print(' '.join(cmd))
    return executable, subprocess.run(cmd, check=True, capture_output=True)


def playGame(model=None, param=None, pot=None, seed=None):
    import subprocess
    import json

    assert isinstance(pot, float) or isinstance(pot, int)
    param['pot'] = pot

    cmd = [
        "node", str(model),
        "--param", "{}".format(json.dumps(param))]

    if seed is not None:
        assert isinstance(seed, int)
        cmd.extend(["--random-seed", str(seed)])

    # print(cmd)

    return subprocess.run(cmd, check=True, capture_output=True)


# def playGoldenBalls(model, param, path):
#     import subprocess
#     import os
#     import json
#     from pprint import pprint
#
#     cmd = [
#     "webppl", model,
#     "--require", "webppl-viz",
#     "--require", "webppl-json",
#     "--",
#     "--param", "{}".format(json.dumps(param))]
#     print(cmd) ###debug
#     return subprocess.run(cmd, stdout=subprocess.PIPE)


def importPPLdata(jsondata, supports):
    '''
    for importing data with a finite number of known supports (e.g. probability of a1)
    '''
    import numpy as np
    import pandas as pd
    probs = np.zeros(len(supports), dtype='float')
    for idx, obs in enumerate(supports):
        if obs in jsondata["support"]:
            probs[idx] = jsondata["probs"][jsondata["support"].index(obs)]
    df = pd.DataFrame(dict(support=supports, prob=probs))
    assert df.isnull().sum().sum() == 0
    df.set_index('support', inplace=True)
    np.testing.assert_almost_equal(df['prob'].sum(), 1.0, decimal=9, err_msg='probabilities do not sum to 1', verbose=True)
    return df


def importPPLdataContinuous(jsondata):
    ### NOT USED
    import numpy as np
    import pandas as pd
    xs = np.array(jsondata['support'])
    probs = np.array([jsondata['probs']], dtype='float')
    d = np.concatenate((xs, probs.T), axis=1)
    df = pd.DataFrame(d)
    cols = list(df.columns)
    cols[-1] = 'prob'
    df.columns = cols
    np.testing.assert_almost_equal(df.iloc[:, -1].sum(), 1.0, decimal=9, err_msg='probabilities do not sum to 1', verbose=True)
    return df


def importPPLdataContinuous2(jsondata):
    ### NOT USED
    import numpy as np
    import pandas as pd
    xs = np.array(jsondata['support'])
    probs = np.array(jsondata['probs'], dtype='float')
    a = np.vstack((xs, probs)).T
    # print(a.shape)
    return pd.DataFrame(data=a, columns=('support', 'prob'))


def importPPLdataDict(datain):
    import numpy as np
    import pandas as pd
    import itertools
    import sys

    # if len(featureclasses) < 1: list(range()
    error_ = False
    overflow_ = list()
    support = []
    probs = []
    featureList = list(datain['support'][0].keys())
    subfeatureList = []
    for feature in featureList:
        subfeatureList.append(list(range(len(datain['support'][0][feature]))))
    for obs in range(len(datain['probs'])):
        supportRow = []
        for feature in featureList:
            ### Test datatype
            for val_ in datain['support'][obs][feature]:
                if not (isinstance(val_, float) or isinstance(val_, int)) or val_ is None:
                    error_ = True
                    print('featureList')
                    print(featureList)
                    print('subfeatureList')
                    print(subfeatureList)
                assert not error_, f"val >>{val_}<< in feature >>{feature}<< in obs >>{obs}<< is not float/int but >>{type(val_)}<<"

                if np.abs(val_) > sys.float_info.max / 10.0 or (np.abs(val_) > 0 and np.abs(val_) < sys.float_info.min * 10.0):
                    overflow_.append(val_)
                    if np.abs(val_) > sys.float_info.max:
                        print(f"Numerical Warning, val >>{val_}<< in feature >>{feature}<< in obs >>{obs}<< exceeds system thresholds of ({sys.float_info.min}, {sys.float_info.max})")
                        val_ = np.sign(val_) * sys.float_info.max

            ###
            supportRow.append(datain['support'][obs][feature])
        support.append(list(itertools.chain(*supportRow)))
        probs.append(datain['probs'][obs])
    df = pd.DataFrame(support, columns=makeLabelHierarchy([featureList, subfeatureList]), dtype=float)
    se = pd.Series(probs, dtype=float)
    df[('prob', 'prob')] = se.values
    assert df.isnull().sum().sum() == 0
    assert not np.any(np.isnan(df.to_numpy()))

    return df, overflow_


def importPPLdataWithLinkerFn(jsondata, labels, repackageFn):
    import numpy as np
    import pandas as pd

    probs = np.array(jsondata['probs'], dtype='float')
    xs = np.full((len(jsondata['support']), len(labels)), np.nan, dtype=float)
    for i_obs, obs in enumerate(jsondata['support']):
        xs[i_obs, :] = repackageFn(obs)

    df = pd.DataFrame(data=np.insert(xs, xs.shape[1], probs, axis=1), columns=(np.append(labels, 'prob')))
    assert df.isnull().sum().sum() == 0
    return df


def makeLabelHierarchy(labels):
    import pandas as pd
    import itertools
    # def depthCount(x):
    #     return int(isinstance(x, list)) and len(x) and 1 + max(map(depthCount, x))

    # for llevel in range(depthCount(labels)-2, -1, -1):
    featureClassLabels = []
    for idx, label in enumerate(labels[0]):
        featureClassLabels.append(list(itertools.repeat(label, len(labels[1][idx]))))
    arrays = [list(itertools.chain(*featureClassLabels)), list(itertools.chain(*labels[1]))]
    tuples = list(zip(*arrays))
    cols = pd.MultiIndex.from_tuples(tuples)
    return cols

#################


def getMarginals(data, features):
    ### NOT USED
    import numpy as np
    import pandas as pd
    marginals = np.empty((len(features), 2))
    marginals[:] = np.nan
    for idx, feature in enumerate(features):
        support = np.array(data.index.tolist())
        supportMask = support[:, idx].astype(bool)

        s1 = pd.Series(supportMask)
        s0 = np.invert(s1)

        d1 = data[s1.values]
        d0 = data[s0.values]

        marginals[idx, 1] = d1.sum()
        marginals[idx, 0] = d0.sum()

    df = pd.DataFrame(dict(feature=features, prob0=marginals[:, 0], prob1=marginals[:, 1]))
    df.set_index('feature', inplace=True)
    ##### check to make sure all rows sum to 1
    np.testing.assert_almost_equal(df['prob'].sum(), 1.0, decimal=9, err_msg='probabilities do not sum to 1', verbose=True)
    return df


def marginalizeContinuous(dfin, featureList, bypassCheck=False):

    ### uses column labels
    df = dfin.copy(deep=True)
    ### drop the first column heiarchy if column multiindex
    if (isinstance(df.columns[0], list) or isinstance(df.columns[0], tuple)) and len(list(df.columns)[0]) > 1:
        df.columns = df.columns.droplevel(0)
    # avilableGroupings = list(df.columns[:-1])
    return condensePPLdata(df, featureList, bypassCheck)  # df.groupby(featureList)['prob'].sum()


# def marginalizeContinuous2(df, icol):
#     ### uses column numbers
#     df2 = df.iloc[:,[icol,-1]]
#     df2.set_index(df2.columns[[0]].tolist(), inplace=True)
#     return df2.groupby(df2.index).sum()


def get_expected_vector_from_multiplesets(df, col_labels, nobs, set_prior=None):
    import numpy as np
    import pandas as pd

    if set_prior is None:
        '''Use a uniform prior over sets (pots) if none is provided'''
        uniform_set_prior_prob = nobs.shape[0]**-1
        set_prior_templist = np.full((nobs.shape[0]), np.nan)
        for i_key, key in enumerate(nobs.index.to_list()):
            set_prior_templist[i_key] = uniform_set_prior_prob
        set_prior = pd.Series(set_prior_templist, index=nobs.index.to_list())

    e_vec = df.loc[:, col_labels].to_numpy()  # n_observations x emotions matrix
    obs_prob = df.loc[:, 'prob'].to_numpy()  # column vector of P(\vec{e}|outcome,pot)
    pots_col = df.index.get_level_values(0).to_numpy(dtype=np.float64)  # column vector of set identity (which pot)

    set_prior_by_obs = np.full_like(pots_col, np.nan)
    for i_obspot, obspot in enumerate(pots_col):
        set_prior_by_obs[i_obspot] = set_prior.loc[obspot]  # column vector of P(set), i.e. P(pot)

    overall_vec_prob = set_prior_by_obs * obs_prob  # column vector of P(\vec{e}|outcome) = P(\vec{e}|outcome,pot) * P(pot)

    grand_prob = np.tile(overall_vec_prob.reshape(-1, 1), (1, e_vec.shape[1]))  # n_observations x n_emotions matrix

    assert np.allclose(grand_prob.sum(axis=0), 1)

    ### dot product of each column of intensity values e_i with P(e_i|outcome) yielding expected vector across pots, E[(\vec{e}|outcome)]
    expected_vector = pd.Series(np.sum(e_vec * grand_prob, axis=0), index=col_labels)
    variances = np.sum(np.square(e_vec) * grand_prob, axis=0) - np.square(np.sum(e_vec * grand_prob, axis=0))
    return expected_vector, variances


def get_ev_emodf(dfin0, col_labels, nobs, sets_prior=None):
    """
    We'll presume the nobs df has the relevant conditions (pots)
    the EV[ \vec{e} ] is, \sum \vec{e_{i,j}} \cdot P( \vec{e_{i,j}} | pot=j ) \cdot P( pot=j )

    df['prob'] gives P( \vec{e_{i,j}} | pot=j ) (for empirical data, simply 1/n_observations for that pot)
    set_prior gives P( pot=j )
    """

    import numpy as np
    import warnings

    warnings.warn('nobs currently only being used for index labels')

    dfin = dfin0.copy(deep=True)

    dfin.columns = dfin.columns.droplevel(level=0)

    ### make sure index and nobs are the same

    topkeys = np.unique(dfin.index.get_level_values(0).to_numpy(dtype=np.float64))

    np.testing.assert_array_equal(np.unique(nobs.index.to_numpy(dtype=np.float64)), topkeys)

    return get_expected_vector_from_multiplesets(dfin, col_labels, nobs, set_prior=sets_prior)


def marginalizeContinuousAcrossMultilevel(df, col_labels, nobs=None, set_prior=None, bypassCheck=False):
    '''
    concatonates MultiIndex column df according to first column level
    '''
    import numpy as np
    import pandas as pd

    mdf = df.loc[:, (*col_labels, 'prob')]
    topkeys = np.unique(mdf.index.get_level_values(0))
    tempdflist = [None] * len(topkeys)
    for i_key, key in enumerate(topkeys):
        tempdflist[i_key] = mdf.loc[key, ].copy(deep=True)

    # marginal.prob = marginal.prob.divide(marginal.prob.sum())
    # elif not bypassCheck: np.testing.assert_approx_equal(marginal.prob.sum(), len(topkeys))

    if nobs is None:
        ### normalize assuming pot dfs represent the same number of observations (normalization for simulation, not empirical)
        nobs = np.array([1.0] * len(topkeys), dtype='float')

    ### if there are unequal observations per set, adjust probabilities accordingly

    nobs_type = {pd.core.series.Series: 'series', np.ndarray: 'ndarray', list: 'list'}[type(nobs)]

    performDEBUGcheck = False
    if set_prior is None:
        set_prior = np.full(len(nobs), 1 / len(nobs), dtype=float)
        performDEBUGcheck = True
    elif isinstance(set_prior, list):
        set_prior = np.array(set_prior)

    np.testing.assert_almost_equal(set_prior.sum(), np.round(set_prior.sum()))  # make sure it's an integer value
    if np.round(set_prior.sum()) > 1.0:
        set_prior = set_prior / np.round(set_prior.sum())  # normalize if sum is not already 1

    set_prob = set_prior * nobs / np.sum(nobs) * np.inner(set_prior, nobs / np.sum(nobs))**-1

    for i_key, key in enumerate(topkeys):
        if nobs_type == 'series':
            this_prob = set_prob.loc[key]
        elif nobs_type == 'ndarray':
            this_prob = set_prob[i_key]
        elif nobs_type == 'list':
            this_prob = set_prob[i_key]
        else:
            this_prob = set_prob[i_key]

        if performDEBUGcheck:
            if nobs_type == 'series':
                this_nobs = nobs.loc[key]
            elif nobs_type == 'ndarray':
                this_nobs = nobs[i_key]
            elif nobs_type == 'list':
                this_nobs = nobs[i_key]
            else:
                this_nobs = nobs[i_key]

            np.testing.assert_almost_equal(tempdflist[i_key].prob.multiply(this_prob).values, tempdflist[i_key].prob.multiply(this_nobs).divide(np.sum(nobs)).values)

        tempdflist[i_key].prob = tempdflist[i_key].prob.multiply(this_prob)

    final_expanded = pd.concat(tempdflist)
    assert final_expanded.shape[0] == mdf.shape[0]

    return condensePPLdata(final_expanded, col_labels, bypassCheck)


#################


def condensePPLdata(marginaldf, col_labels, bypassCheck=False):
    import numpy as np

    if not bypassCheck:
        np.testing.assert_approx_equal(marginaldf.prob.sum(), 1.0)

    return marginaldf.groupby(col_labels)['prob'].sum()


def unweightProbabilities(dfin, nobs=None):
    import numpy as np
    from copy import deepcopy

    # convert series with index (e.g. from marginalizeContinuous) into df
    if len(dfin.shape) == 1:
        dfincopy = dfin.reset_index()
    else:
        dfincopy = deepcopy(dfin)

    if dfincopy.shape[0] == 0:
        np.testing.assert_equal(np.any(np.equal([nobs, nobs], [0, None])), True)
        dfincopy.loc[0, :] = np.full((1, dfincopy.shape[1]), np.nan, dtype=float)
        dfincopy.loc[0, 'prob'] = 0.0
        nobs = 1
        df = dfincopy
    else:
        # get index of non-zero probability Observations
        nzidx = np.squeeze(dfincopy.loc[:, 'prob'].values > 0)
        # drop zeros
        df = dfincopy.iloc[nzidx, :]  # return copy

    newprobs = np.squeeze(df.loc[:, 'prob'].values)

    # assert that df is normalized
    np.testing.assert_almost_equal(newprobs.sum(), 1.0)

    ## get lowest common denominator
    if nobs is None:
        ### this is bypassed for empty df since nobs is set to 1
        lcd = newprobs.min()
    else:
        lcd = 1 / nobs
    if not np.isclose(lcd, newprobs.max()):
        repfactor = newprobs / lcd
        repfactor_int = np.round(repfactor).astype(int)
        np.testing.assert_almost_equal(repfactor, np.round(repfactor), decimal=6)
        df = df.iloc[df.index.repeat(repfactor_int), :]
        df.loc[:, 'prob'] = lcd
        df.reset_index(drop=True, inplace=True)

    np.testing.assert_equal(df.shape[0], int(np.round(1 / lcd)))

    return df


def emoDict_to_numpy(emoDict, outcomes=None, pots=None, emotions=None):
    import numpy as np

    if outcomes is None:
        outcomes = emoDict['nobs'].columns.to_list()
    if pots is None:
        pots = emoDict['nobs'].index.to_list()
    if emotions is None:
        emotions = emoDict[outcomes[0]].loc[:, 'emotionIntensities'].columns.to_list()

    emoNpy_dict = dict()

    for i_outcome, outcome in enumerate(outcomes):
        emoNpy_dict[outcome] = dict()
        for i_pot, pot in enumerate(pots):

            idx_pot_empir = emoDict[outcome].index.get_level_values('pots') == pot
            assert idx_pot_empir.sum() == emoDict['nobs'].loc[pot, outcome]  # ignore empirical probs if all equal

            emoNpy_dict[outcome][pot] = emoDict[outcome].loc[idx_pot_empir, ('emotionIntensities', emotions)].to_numpy()

    return emoNpy_dict


def getEV(df, bypassCheck=False):
    import numpy as np
    if df.size:
        x = df.index
        p = df.values
        EV = np.inner(x, p)
        Var = np.inner(p, np.square(x)) - np.inner(x, p)**2
    else:
        EV, Var = np.nan, np.nan
    if not bypassCheck:
        np.testing.assert_almost_equal(p.sum(), 1.0)
    return EV, Var


def get_expected_vector_(X, p):
    import numpy as np

    np.testing.assert_almost_equal(np.sum(p), 1.0)

    if len(p.shape) == 1:
        p = p.reshape(-1, 1)
        assert len(p.shape) == 2
        assert p.shape[1] == 1

    prob = np.tile(p, (1, X.shape[1]))

    return np.sum(X * prob, axis=0)


def get_expected_emo_vector_from_df_(dfin):
    import numpy as np
    import pandas as pd

    if isinstance(dfin, pd.core.series.Series):
        e_vecs = np.expand_dims(dfin.drop(labels='prob').to_numpy(), axis=0)
        p = dfin['prob'].to_numpy().reshape(-1, 1)
        labels = dfin['emotionIntensities'].index

        assert e_vecs.shape[1] == len(labels)
        assert len(e_vecs.shape) == 2
        assert len(p.shape) == 2

    else:
        idx = pd.IndexSlice
        e_vecs = dfin.loc[:, idx['emotionIntensities', :]].to_numpy()
        p = dfin.loc[:, idx['prob', :]].to_numpy()
        labels = dfin['emotionIntensities'].columns

    expected_vector = get_expected_vector_(e_vecs, p)

    return pd.Series(expected_vector, index=labels)


def get_expected_vector_by_pot_outcome(emodict, outcomes=None, pots=None):
    import numpy as np
    import pandas as pd

    if outcomes is None:
        outcomes = emodict['nobs'].columns.to_list()
    if pots is None:
        pots = emodict['nobs'].index.to_list()

    expected_vectors = list()
    for outcome in outcomes:
        for pot in pots:
            expected_vector = get_expected_emo_vector_from_df_(emodict[outcome].loc[pot, :])
            expected_vector['outcome'] = outcome
            expected_vector['pot'] = pot
            expected_vectors.append(expected_vector)
    return pd.concat(expected_vectors, axis=1).T


def get_expected_vector_by_outcome(emodict, emotions=None, outcomes=None, sets_prior=None):
    import numpy as np
    import pandas as pd

    if emotions is None:
        print("FIXTHIS")
    if outcomes is None:
        outcomes = emodict['nobs'].columns.to_list()

    expected_vectors = list()
    for i_outcome, outcome in enumerate(outcomes):
        expected_vector, _ = get_ev_emodf(emodict[outcome], emotions, nobs=emodict['nobs'][outcome], sets_prior=sets_prior)
        expected_vector['outcome'] = outcome
        expected_vectors.append(expected_vector)

    return pd.concat(expected_vectors, axis=1).T


def calc_ev_and_delta_from_emodicts(emodict_player=None, emodict_generic=None, emotion_labels=None, outcomes=None):
    """
    Returns E[player emotions | outcome], E[generic emotions | outcome] and difference, filtering by (outcome,pot) combinations in player emodict.
    Given:
        emodict_player = model_data['results']['test']['model_predictions'][stimid]
        emodict_generic = model_data['results']['train']['model_predictions']['generic']
    Call as:
        calc_ev_and_delta_from_emodicts(emodict_player=emodict_player, emodict_generic=emodict_generic, emotion_labels=ppldata['labels']['emotions'], outcomes=ppldata['labels']['outcomes'])

        or 

        player_ev_empir, generic_ev_empir, ev_deltas_empir = calc_ev_and_delta_from_emodicts(emodict_player=model_data['results']['test']['empiricalEmotionJudgments'][stimid], emodict_generic=model_data['results']['train']['empiricalEmotionJudgments']['generic'], emotion_labels=emotion_labels, outcomes=outcomes)
    """
    # from webpypl import get_ev_emodf, filter_emodict_by_nobsdf, get_expected_vector_by_outcome, get_expected_vector_by_pot_outcome

    emodict_generic_filtered = filter_emodict_by_nobsdf(emodict_generic, emodict_player['nobs'])

    generic_model_ev_byoutcome = get_expected_vector_by_outcome(emodict_generic_filtered, emotions=emotion_labels, outcomes=outcomes).set_index('outcome')

    specific_model_ev_byoutcome = get_expected_vector_by_outcome(emodict_player, emotions=emotion_labels, outcomes=outcomes).set_index('outcome')

    return specific_model_ev_byoutcome, generic_model_ev_byoutcome, specific_model_ev_byoutcome - generic_model_ev_byoutcome


def getFitBOEmotionParam(path, samples='*', optcycles='*'):
    import os
    import os.path
    import json
    from pprint import pprint
    import glob

    def optprogress(x):
        import re
        return int(re.findall(r'\d+', re.findall(r'optcycles\d+', x)[0])[0])
    wd1 = os.getcwd()
    os.chdir(path)
    files = glob.glob('bayesOptimizationResults_samples{}_optcycles{}.json'.format(samples, optcycles))
    if optcycles == '*':
        files_sorted = sorted(files, key=optprogress)
        target = files_sorted[-1]
    else:
        assert len(files) == 1
        target = files[-1]
    pprint(target)
    with open(target, mode='rt') as data_file:
        jsonin = json.load(data_file)
    os.chdir(wd1)
    return jsonin['webpplparam']


def getFitEmotionParam(path, samples='*', burn='*'):
    import os
    import os.path
    import json
    import glob

    # if samples is None: samples = '*'
    # if burn is None: burn = '*'
    wd1 = os.getcwd()
    os.chdir(path)
    param = {}
    for fname in glob.glob('optimizationResults_*_samples{}_burn{}.json'.format(samples, burn)):
        with open(fname, mode='rt') as data_file:
            param.update(json.load(data_file))

    os.chdir(wd1)
    return param


def modelStats_(data=None, model_predictions=None):
    import numpy as np
    import pandas as pd
    import scipy.stats

    np.testing.assert_equal(data.shape, model_predictions.shape)

    if isinstance(model_predictions, pd.DataFrame) or isinstance(model_predictions, pd.Series):
        y = data.values.flatten()
        f = model_predictions.values.flatten()
    else:
        y = data.flatten()
        f = model_predictions.flatten()

    y_mean = np.mean(y)  # mean of observed data
    SStot = np.sum(np.power(np.subtract(y, y_mean), 2))  # total sum of squares (proportional to the variance of the data)
    sqrerr = np.power(np.subtract(y, f), 2)
    SSres = np.sum(sqrerr)  # residual sum of squares
    R2 = 1 - SSres / SStot
    RMSE = np.sqrt(np.mean(sqrerr))
    r = scipy.stats.pearsonr(y, f)[0]

    return {'RMSE': RMSE, 'R2': R2, 'pearson_r': r}


def compareDMs(dms1, dms2, corr_method, fillDiag=None):
    import numpy as np
    import scipy
    dm_compared = np.full((len(dms1), len(dms2)), np.nan, dtype=float)
    for idx_dm1, dm1 in enumerate(dms1):
        for idx_dm2, dm2 in enumerate(dms2):

            dm_compared[idx_dm1, idx_dm2] = corr_method['corr'](dm1, dm2)

            # x = dm1
            # y = dm2
            # if np.unique(x).size < x.size or np.unique(y).size < y.size:
            #     import warnings
            #     warnings.warn('duplicate ranks')
            #     if np.unique(x).size < x.size: print(x)
            #     if np.unique(y).size < y.size: print(y)

            x = scipy.stats.rankdata(dm1)
            y = scipy.stats.rankdata(dm2)
            # if np.unique(x).size < x.size or np.unique(y).size < y.size:
            #     import warnings
            #     warnings.warn('duplicate ranks')
            #     if np.unique(x).size < x.size: print(x)
            #     if np.unique(y).size < y.size: print(y)

    if fillDiag is not None:
        np.fill_diagonal(dm_compared, fillDiag)

    ranked_scores = []
    for idx_dm1, dm1 in enumerate(dms1):
        x = dm_compared[idx_dm1, :]
        ranked_scores.append(scipy.stats.rankdata(-x))  # x -> rank1=lowest, -x -> rank1=highest, np.nan always given lowest order (regardless of +/-)

    return dm_compared, np.array(ranked_scores)


def rankDMs(dm_compared, dm_tocompare_labels, dm_tocompare_labels_ideal, method='rank1_is_max', verbose=True):
    import numpy as np
    import scipy

    if verbose:
        print('\nDM \tTop Match \t Score'.format())

    ideal_score = [None] * len(dm_tocompare_labels)
    ranked_labels = [None] * len(dm_tocompare_labels)
    ranked_scores = [None] * len(dm_tocompare_labels)
    top_scoring = [None] * len(dm_tocompare_labels)
    for idx_dm, dm_label in enumerate(dm_tocompare_labels):
        x = dm_compared[idx_dm, :]
        orderingConstant = {'rank1_is_max': -1, 'rank1_is_min': 1}[method]
        x[np.isnan(x)] = orderingConstant * np.inf
        ranks = scipy.stats.rankdata(orderingConstant * x)  # x -> 1=lowest, -x -> 1=highest, np.nan always given lowest order (regardless of +/-)
        sortidx = np.argsort(ranks)
        ranked_scores[idx_dm] = x[sortidx]

        ### top scoring dm
        labels_sorted = np.array(dm_tocompare_labels)[sortidx]
        ranked_labels[idx_dm] = labels_sorted
        top_scoring[idx_dm] = labels_sorted[0]

        ### score of ideal dm
        ideal_score[idx_dm] = list(labels_sorted).index(dm_tocompare_labels_ideal[idx_dm])

        if verbose:
            print('{}\t{}\t{}'.format(dm_label, top_scoring[idx_dm], ideal_score[idx_dm]))

    if verbose:
        print('loss: {} - {} = {}\n'.format((np.array(ideal_score) + 1).sum(), len(ideal_score), sum(ideal_score)))

    return np.array(ideal_score) + 1  # , ranked_labels, ranked_scores, top_scoring


def getCI(data, alpha):
    import scipy.stats
    # import pymc3

    ### TODO pick ci
    mean, var, std = scipy.stats.bayes_mvs(data, alpha=alpha)  # bayesian esitimates of confidence intervals

    # ci[i_feature,i_outcome] = pymc3.stats.hpd(expandedObservations.loc[:,feature].values)

    return mean


def calculate_expectedValue_perFeature(ppldataLabels_features, ppldataLabels_outcomes, emotionData, alpha=0.95):
    import numpy as np

    ev = np.zeros((len(ppldataLabels_features), len(ppldataLabels_outcomes)), dtype=float)
    variance = np.zeros_like(ev, dtype=float)
    ci = np.full_like(ev, None, dtype=object)
    # nsamples = np.zeros((len(ppldataLabels_features),len(ppldataLabels_outcomes)))

    for i_feature, feature in enumerate(ppldataLabels_features):
        for i_outcome, outcome in enumerate(ppldataLabels_outcomes):

            dfin = emotionData[outcome].copy()
            if dfin.columns.nlevels == 2:
                dfin.columns = dfin.columns.droplevel(level=0)

            nobs = emotionData['nobs'][outcome]

            dfout = marginalizeContinuousAcrossMultilevel(dfin, [feature], nobs=nobs)

            ev[i_feature, i_outcome], variance[i_feature, i_outcome] = getEV(dfout)
            expandedObservations = unweightProbabilities(dfout, nobs=nobs.sum())

            ci[i_feature, i_outcome] = getCI(expandedObservations.loc[:, feature].values, alpha=alpha)[1]

    return ev, variance, ci


def get_wide_df_(emodict, return_ev=True):
    '''
    Takes a dict with keys {[outcomes], nobs}
    '''
    import numpy as np
    import pandas as pd
    # from webpypl import unweightProbabilities

    nobsdf = emodict['nobs']
    outcomes = list(nobsdf.columns)
    df_array = list()
    for outcome in outcomes:
        pots_by_outcome_temp = nobsdf.index[nobsdf[outcome] > 0]
        for i_pot, pot in enumerate(pots_by_outcome_temp):
            data_slice = emodict[outcome].loc[pot, slice('emotionIntensities', 'prob')]
            nobs = nobsdf[outcome].loc[pot]
            # emodict_temp1['prob']['prob'][0] = emodict_temp1['prob']['prob'][0]**-1
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


def widedf_to_emoDict_(dfin, outcomes, emotions):
    import numpy as np
    import pandas as pd

    emoDict = {'nobs': pd.DataFrame(data=np.zeros((len(np.unique(dfin['pot'])), len(outcomes))), index=np.unique(dfin['pot']), columns=outcomes, dtype=np.int64)}
    for outcome in outcomes:
        df_list = list()
        df_outcome = dfin.loc[dfin['outcome'] == outcome, :]
        pots_by_outcome = np.unique(df_outcome['pot'])
        for pot in pots_by_outcome:
            df_pot = df_outcome.loc[df_outcome['pot'] == pot, :].drop(['pot', 'outcome'], axis=1)

            df_pot.columns = pd.MultiIndex.from_tuples([('emotionIntensities', feature) for feature in dfin.columns if feature in emotions])

            df_pot[('prob', 'prob')] = df_pot.shape[0]**-1

            df_pot.index = range(df_pot.shape[0])

            df_list.append(df_pot)

            emoDict['nobs'].loc[pot, outcome] = df_pot.shape[0]

        emoDict[outcome] = pd.concat(df_list, keys=pots_by_outcome)
        emoDict[outcome].rename_axis(index=('pots', None), inplace=True)

    return emoDict


def filter_emodict_by_nobsdf(dictin, nobs_filter):
    import pandas as pd
    #import copy

    nobsdf = nobs_filter.copy()
    nobsdf.loc[:, :] = int(0)
    filtered_dict = dict()
    for outcome in nobs_filter.columns.to_list():
        pots_to_include = nobs_filter.index.to_numpy()[nobs_filter[outcome] > 0]
        filtered_df_list = list()

        for pot in pots_to_include:
            # df_temp = copy.deepcopy(dictin[outcome].loc[pot,:])
            df_temp = dictin[outcome].loc[pot, :].copy()
            df_temp['pots'] = pot
            nobsdf.loc[pot, outcome] = df_temp.shape[0]
            filtered_df_list.append(df_temp.set_index('pots'))

        if filtered_df_list:
            filtered_dict[outcome] = pd.concat(filtered_df_list)
        else:
            import warnings
            warnings.warn(f'No observations for outcome {outcome}')

    filtered_dict['nobs'] = nobsdf

    return filtered_dict


def getEmpiricalModel_pair_(emodict_emp, emodict_iaf, return_ev=True, feature_selector=''):
    import numpy as np
    from copy import deepcopy

    ### empirical
    dfwide_emp = get_wide_df_(emodict_emp, return_ev=return_ev).reset_index(drop=True, inplace=False)
    if not return_ev:
        dfwide_emp.drop('prob', axis=1, inplace=True)

    ### model features
    emodict_iaf_temp = deepcopy(emodict_iaf)
    emodict_iaf_temp['nobs'][emodict_emp['nobs'] == 0] = 0

    forward_data_prep = filter_iaf_dict(feature_selector, return_ev=return_ev)
    dfwide_iaf = forward_data_prep.apply(emodict_iaf_temp)

    if return_ev:
        np.testing.assert_equal(dfwide_emp.shape[0], dfwide_iaf.shape[0])
        np.testing.assert_array_equal(dfwide_emp['pot'], dfwide_iaf['pot'])

    return dfwide_emp, dfwide_iaf, forward_data_prep


def getEmpiricalModel_pair_unbalanced_(emodict_emp, emodict_iaf, return_ev=True, feature_selector=r''):  # wip
    import numpy as np
    import re
    from copy import deepcopy

    def filter_df_columns_(dfin, column_list):
        X = dfin.reindex(columns=column_list, copy=True)
        return X.reset_index(drop=True, inplace=False)

    dfwide_emp = get_wide_df_(emodict_emp, return_ev=return_ev).reset_index(drop=True, inplace=False)
    if not return_ev:
        dfwide_emp.drop('prob', axis=1, inplace=True)

    emodict_iaf_temp = deepcopy(emodict_iaf)
    emodict_iaf_temp['nobs'][emodict_emp['nobs'] == 0] = 0

    dfwide_iaf_allfeatures = get_wide_df_(emodict_iaf_temp, return_ev=return_ev)

    ### filter columns
    regex = re.compile(feature_selector)
    iaf_list = list(filter(regex.search, dfwide_iaf_allfeatures.columns))

    dfwide_iaf = filter_df_columns_(dfwide_iaf_allfeatures, iaf_list)
    dfwide_iaf['pot'] = dfwide_iaf_allfeatures['pot'].values
    dfwide_iaf['outcome'] = dfwide_iaf_allfeatures['outcome'].values

    if return_ev:
        np.testing.assert_equal(dfwide_emp.shape[0], dfwide_iaf.shape[0])
        np.testing.assert_array_equal(dfwide_emp['pot'], dfwide_iaf['pot'])

    return dfwide_emp, dfwide_iaf


def simulateEmpiricalModel_pair_(Y_train_full, outcomes, corr_mean_method=None, affine_intercept=0.02, prospect_fn=None, n_samples='match_empirical'):
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    emotions = Y_train_full.columns.drop(['pot', 'outcome']).to_list()
    stimuli = Y_train_full.loc[:, ('pot', 'outcome')].drop_duplicates().values

    ########
    ### Transform Y into linear space
    ########
    yt = logit_logistic_transform(affine_intercept)

    Y_train_full_transformed = yt.y_transform(Y_train_full.drop(['pot', 'outcome'], axis=1))
    # Y_train_full_transformed = Y_train_full.drop(['pot','outcome'], axis=1).applymap(y_transform)
    Y_train_full_transformed['pot'], Y_train_full_transformed['outcome'] = Y_train_full['pot'], Y_train_full['outcome']

    ########
    ### Calculate EV, SD and CorrMat for each stimulus
    ########

    corr_mats_dict = dict()
    sd_vecs = list()
    ev_vecs = list()

    for i_stim, stim in enumerate(stimuli):
        pot, outcome = stim
        stim_indexer = np.all(Y_train_full_transformed.loc[:, ('pot', 'outcome')] == stimuli[i_stim], axis=1)

        emotion_values_transformed = Y_train_full_transformed.loc[stim_indexer, emotions]

        ev = emotion_values_transformed.mean(axis=0)
        ev['pot'], ev['outcome'] = pot, outcome
        ev_vecs.append(ev)

        corr_mat = emotion_values_transformed.corr()
        if outcome not in corr_mats_dict:
            corr_mats_dict[outcome] = list()
        corr_mats_dict[outcome].append(corr_mat)

        sd_vec = emotion_values_transformed.std()
        sd_vecs.append(sd_vec)

    ev_df = pd.concat(ev_vecs, axis=1).T
    ev_df_linear_approx = ev_df.copy()

    ########
    ### linear models of each emotion,outcome vs potsize :: (|emotion,outcome)
    ### ### get mean vector :: (|pot,outcome)
    ########

    for i_emotion, emotion in enumerate(emotions):
        for i_outcome, outcome in enumerate(outcomes):
            x = np.array(ev_df.loc[ev_df['outcome'] == outcome, 'pot'].values, dtype=float)

            # x_transformed = ppldata['potspacing'](x)
            # x_log_transformed = np.log1p(x)
            x_transformed = prospect_fn(x)

            y_ev_by_outcome = ev_df.loc[ev_df['outcome'] == outcome, emotion]

            model = LinearRegression(fit_intercept=True).fit(x_transformed.reshape(-1, 1), y_ev_by_outcome.values)
            y_ev_pred = model.predict(x_transformed.reshape(-1, 1))

            ev_df_linear_approx.loc[ev_df['outcome'] == outcome, emotion] = y_ev_pred

            # # plt.plot(x_transformed, y_ev_by_outcome)
            # plt.plot(x_log_transformed, y_ev_by_outcome)
            # plt.plot(x_log_transformed, y_pred)

    ########
    ### mean corr matrix across all pots within each outcome :: (|outcome)
    ########

    corr_mat_mean_dict = dict()
    for outcome in corr_mats_dict:
        mat_list = list()
        for mat in corr_mats_dict[outcome]:
            mat_list.append(mat.values)
        corrs_mat_3d = np.stack(mat_list)

        corr_mat_mean = np.full((len(emotions), len(emotions)), np.nan, dtype=float)
        for m in range(len(emotions)):
            for n in range(len(emotions)):
                if m == n:
                    corr_mat_mean[m, n] = 1.0
                else:
                    corr_mat_mean[m, n] = corr_mean_method(corrs_mat_3d[:, m, n])

        corr_mat_mean_dict[outcome] = corr_mat_mean

    ########
    ### calc cov_mat based on mean corr matrix and within (pot,outcome) SDs :: (|pot,outcome)
    ########
    # cov_mat_mean_dict = dict()
    # for outcome in corr_mat_mean_dict:
    #     cov_mat_mean_dict[outcome] = np.diag()

    cov_mat_list = list()
    for i_stim, stim in enumerate(stimuli):

        pot = stim[0]
        outcome = stim[1]

        sd_vec = sd_vecs[i_stim]

        cov_mat = np.diag(sd_vec) @ corr_mat_mean_dict[outcome] @ np.diag(sd_vec)
        cov_mat_list.append(cov_mat)

    ########
    ### gen data vectors (balanced) based on EV from linear model and CovMat :: (|pot,outcome)
    ########

    simulated_y_list = list()
    for i_stim, stim in enumerate(stimuli):
        pot, outcome = stim

        cov_mat = cov_mat_list[i_stim]
        mu_vec_temp = ev_df_linear_approx.loc[(ev_df_linear_approx['pot'] == pot) & (ev_df_linear_approx['outcome'] == outcome), emotions].values
        mu_vec = np.array(mu_vec_temp.flatten(), dtype=float)

        if n_samples == 'match_empirical':
            nobs = np.sum(np.all(Y_train_full_transformed.loc[:, ('pot', 'outcome')] == stimuli[i_stim], axis=1))
        else:
            nobs = n_samples

        y_sim = np.random.multivariate_normal(mu_vec, cov_mat, size=(nobs))

        y_sim_df = pd.DataFrame(data=y_sim, columns=emotions)
        y_sim_df['pot'], y_sim_df['outcome'] = pot, outcome

        simulated_y_list.append(y_sim_df)

    return pd.concat(simulated_y_list)


#################################


def rankdata(x):
    import numpy as np
    import scipy.stats

    ranks = scipy.stats.rankdata(x, method='average')
    ranks[np.isnan(x)] = np.nan

    return ranks


def FTrz(r_array):
    import numpy as np

    fisher_transform = np.vectorize(lambda r: 0.5 * np.log((1 + r) / (1 - r)))  # = arctanh(r_i)

    return fisher_transform(r_array)


def FTzr(z_array):
    import numpy as np

    arctanh = np.vectorize(lambda z: (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1))  # = arctanh(z_i)

    return arctanh(z_array)


def bootstrap(data, n=1000, func=None):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    import numpy as np

    if func is None:
        func = np.mean

    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()

    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1 + p) / 2.
        l_pval = (1 - u_pval)
        l_indx = int(np.floor(n * l_pval))
        u_indx = int(np.floor(n * u_pval))
        return (simulations[l_indx], simulations[u_indx])
    return (ci)


def get_intersection(list_1, list_2):
    """Remove duplicates from list while preserving order.
     Parameters
    ----------
    list_in: Iterable
     Returns
    -------
    list
        List of first occurences in order
    """
    _list = []
    for item in list_1:
        if item in list_2 and item not in _list:
            _list.append(item)
    return _list


def notify(title, text, annouce):
    import os
    print('\nwrapper finished\n')
    if annouce:
        os.system("""
                  osascript -e 'display notification "{}" with title "{}"'
                  """.format(text, title))
