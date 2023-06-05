#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_webppl_gendata_helpers.py
"""


class Timer:

    def __init__(self):
        self._start_time = None
        self._timepoints = list()
        self.__last_timepoint = None

    def _format_time(self, time_delta):
        return f"{time_delta:0.4f}"

    def start(self):
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
        print('-=-=-=-=-=-=-=-=-=-=-=-=-= TIMER\n')


def cache_all_code(code_dir=None, dump_dir=None):
    import subprocess
    from pathlib import PurePath
    import shutil

    dataout_codedumpsdir = dump_dir / 'code_dump'
    dataout_codedumpsdir.mkdir(exist_ok=True, parents=True)

    for ext in ['py', 'wppl', 'sbatch', 'stan']:
        for file in list(code_dir.glob(f'[A-Za-z]*.{ext}')):
            shutil.copy(code_dir / PurePath(file), dataout_codedumpsdir / PurePath(f'{file.name}').with_suffix(file.suffix))

    freezeout_conda = subprocess.run('conda list --export > ' + str(dataout_codedumpsdir / 'environment-package-list.txt'), check=True, capture_output=True, shell=True)
    freezeout_pip = subprocess.run('pip freeze > ' + str(dataout_codedumpsdir / 'environment-package-list-pip.txt'), check=True, capture_output=True, shell=True)


def _make_feature_list_(df_wide, feature_list, mix_ratio, rescale_intensities_, rescale_intensities_pia2_):
    import numpy as np

    temp_df = df_wide.loc[:, df_wide.columns.isin(feature_list + ["pi_a2", "a_1"])].copy()
    temp_df.loc[:, 'n_'] = int(1)

    temp_df['a1_temp'] = (temp_df['a_1'] == 'C').to_numpy().astype(int)

    check_total = True
    if mix_ratio['C'] == 0:
        temp_df1 = temp_df.loc[temp_df['a_1'] == 'D', :].copy()
        check_total = False
    elif mix_ratio['D'] == 0:
        temp_df1 = temp_df.loc[temp_df['a_1'] == 'C', :].copy()
        check_total = False
    else:
        temp_df1 = temp_df.copy()

    temp_df1.drop(columns=['a_1'], inplace=True)

    temp_df0 = temp_df1.groupby(feature_list + ["pi_a2", "a1_temp"]).sum().reset_index()

    if check_total:
        assert temp_df0['n_'].sum() == df_wide.shape[0]

    idx_C_ = (temp_df0['a1_temp'] == 1).to_numpy()
    idx_D_ = (temp_df0['a1_temp'] == 0).to_numpy()
    assert np.sum(idx_C_) + np.sum(idx_D_) == temp_df0.shape[0]

    n_C_ = temp_df0.loc[idx_C_, 'n_'].to_numpy()
    n_D_ = temp_df0.loc[idx_D_, 'n_'].to_numpy()

    if check_total:
        assert np.sum(temp_df0['n_']) == df_wide.shape[0]

    prob_C_ = (temp_df0.loc[idx_C_, 'n_'].to_numpy() / np.sum(temp_df0.loc[idx_C_, 'n_'])) * mix_ratio['C']
    prob_D_ = (temp_df0.loc[idx_D_, 'n_'].to_numpy() / np.sum(temp_df0.loc[idx_D_, 'n_'])) * mix_ratio['D']
    assert np.isclose([np.sum(prob_C_) + np.sum(prob_D_)], [1]), f"sum prob C: {np.sum(prob_C_):0.4f}, sum prob D: {np.sum(prob_D_):0.4f}"

    n_C_mix_ = temp_df0.loc[idx_C_, 'n_'].to_numpy() * mix_ratio['C']
    n_D_mix_ = temp_df0.loc[idx_D_, 'n_'].to_numpy() * mix_ratio['D']

    if n_C_mix_.size > 0:
        n_C_mix_normed_ = n_C_mix_ / np.min(np.concatenate((n_C_mix_, n_D_mix_)))
    else:
        n_C_mix_normed_ = n_C_mix_

    if n_D_mix_.size > 0:
        n_D_mix_normed_ = n_D_mix_ / np.min(np.concatenate((n_C_mix_, n_D_mix_)))
    else:
        n_D_mix_normed_ = n_D_mix_

    emp_data_dict = dict()
    for feature in feature_list:
        values_C_ = rescale_intensities_(temp_df0.loc[idx_C_, feature].to_numpy())
        values_D_ = rescale_intensities_(temp_df0.loc[idx_D_, feature].to_numpy())
        emp_data_dict[feature] = np.concatenate((values_C_, values_D_), axis=0).tolist()

    values_C_pia2_ = rescale_intensities_pia2_(temp_df0.loc[idx_C_, "pi_a2"].to_numpy())
    values_D_pia2_ = rescale_intensities_pia2_(temp_df0.loc[idx_D_, "pi_a2"].to_numpy())
    emp_data_dict["pi_a2_C"] = np.concatenate((values_C_pia2_, values_D_pia2_), axis=0).tolist()
    emp_data_dict['n'] = np.concatenate((n_C_mix_normed_.round(14), n_D_mix_normed_.round(14)), axis=0).tolist()

    emp_data_dict['debug'] = {
        'mix_ratio_C': mix_ratio['C'],
        'mix_ratio_D': mix_ratio['D'],
        'counts_C': temp_df0.loc[idx_C_, 'n_'].to_numpy().tolist(),
        'counts_D': temp_df0.loc[idx_D_, 'n_'].to_numpy().tolist(),
        'n_C_mix_': n_C_mix_normed_.tolist(),
        'n_D_mix_': n_D_mix_normed_.tolist(),
    }

    ### reshape into response vectors ###
    data_reshaped_labels = list()
    data_reshaped_ = np.full((len(emp_data_dict['n']), len(feature_list) + 1), np.nan, dtype=float)
    for i_feature, feature in enumerate(feature_list):
        data_reshaped_[:, i_feature] = emp_data_dict[feature]
        data_reshaped_labels.append(feature)
    data_reshaped_[:, len(feature_list)] = emp_data_dict['pi_a2_C']
    data_reshaped_labels.append('pi_a2_C')
    assert not np.any(np.isnan(data_reshaped_))

    data_reshaped_list = list()
    for i_resp in range(data_reshaped_.shape[0]):
        data_reshaped_list.append(data_reshaped_[i_resp, :].tolist())

    return emp_data_dict, data_reshaped_list, data_reshaped_labels


def make_kde_lists_mulivar_(df_wide92=None, df_wide62=None, prior_a2C=0.5, space='unit', **kwargs):

    import numpy as np

    mix_ratio = {'C': prior_a2C, 'D': 1.0 - prior_a2C}

    def flip_pia2_axis_to_C(x): return np.abs(x - 5)

    if space == 'unit':
        n = 49
        rescaled_min = 0.0
        rescaled_max = 1.0
        def rescale_intensities_(x): return x

        n_pia2 = 6
        rescaled_min_pia2 = 0
        rescaled_max_pia2 = 5
        def rescale_intensities_pia2_(x): return flip_pia2_axis_to_C(x) / rescaled_max_pia2

    if space == 'raw':
        n = 49
        rescaled_min = int(0)
        rescaled_max = int(n - 1)
        def rescale_intensities_(x): return np.rint((n - 1) * x).astype(int)

        n_pia2 = 6
        rescaled_min_pia2 = 0
        rescaled_max_pia2 = 5
        def rescale_intensities_pia2_(x): return flip_pia2_axis_to_C(x)

    elif space == 'squeeze':
        n = 49
        rescaled_min = 1 / (2 * n)
        rescaled_max = 1 - 1 / (2 * n)
        def rescale_intensities_(x): return (2 * (n - 1) * x + 1) / (2 * n)

        n_pia2 = 6
        rescaled_min_pia2 = 0
        rescaled_max_pia2 = 5
        def rescale_intensities_pia2_(x): return ((flip_pia2_axis_to_C(x) + 1) * 2 - 1) / 12

    elif space == 'logit':
        n = 49
        affine_b = kwargs['affine_b']
        logit_k = kwargs['logit_k']  # 1.0
        rescaled_min = affine_b
        rescaled_max = 1.0 - affine_b
        def affine_compress_(x): return x * (1.0 - 2.0 * affine_b) + affine_b
        def logit_(x): return np.log(x / (1.0 - x)) / logit_k
        def rescale_intensities_(x): return logit_(affine_compress_(x))

        n_pia2 = 6
        rescaled_min_pia2 = 0
        rescaled_max_pia2 = 5
        def rescale_intensities_pia2_(x): return ((flip_pia2_axis_to_C(x) + 1) * 2 - 1) / 12

    if df_wide62 is not None:
        emp_data_anongame, emp_data_anongame_vectors, emp_data_anongame_vector_labels = _make_feature_list_(df_wide62, ["bMoney", "bAIA", "bDIA"], mix_ratio, rescale_intensities_, rescale_intensities_pia2_)
    else:
        emp_data_anongame, emp_data_anongame_vectors, emp_data_anongame_vector_labels = _make_feature_list_(df_wide92, ["bMoney", "bAIA", "bDIA"], mix_ratio, rescale_intensities_, rescale_intensities_pia2_)

    emp_data_publicgame, emp_data_publicgame_vectors, emp_data_publicgame_vector_labels = _make_feature_list_(df_wide92, ["bMoney", "bAIA", "bDIA", "rMoney", "rAIA", "rDIA"], mix_ratio, rescale_intensities_, rescale_intensities_pia2_)

    if df_wide62 is not None:
        emp_data_anongame_temp_C, _, _ = _make_feature_list_(df_wide62, ["bMoney", "bAIA", "bDIA"], {"C": 1, "D": 0}, rescale_intensities_, rescale_intensities_pia2_)
        emp_data_anongame_temp_D, _, _ = _make_feature_list_(df_wide62, ["bMoney", "bAIA", "bDIA"], {"C": 0, "D": 1}, rescale_intensities_, rescale_intensities_pia2_)
        agg_emp_data6 = {
            'bMoneyC': np.array(emp_data_anongame_temp_C['bMoney']),
            'bAIAC': np.array(emp_data_anongame_temp_C['bAIA']),
            'bDIAC': np.array(emp_data_anongame_temp_C['bDIA']),
            'bMoneyD': np.array(emp_data_anongame_temp_D['bMoney']),
            'bAIAD': np.array(emp_data_anongame_temp_D['bAIA']),
            'bDIAD': np.array(emp_data_anongame_temp_D['bDIA']),
        }
    else:
        agg_emp_data6 = None

    agg_emp_data9 = dict()
    if mix_ratio['C'] > 0:
        emp_data_pubgame_temp_C, _, _ = _make_feature_list_(df_wide92, ["bMoney", "bAIA", "bDIA", "rMoney", "rAIA", "rDIA"], {"C": 1, "D": 0}, rescale_intensities_, rescale_intensities_pia2_)
        agg_emp_data9['bMoneyC'] = np.array(emp_data_pubgame_temp_C['bMoney'])
        agg_emp_data9['bAIAC'] = np.array(emp_data_pubgame_temp_C['bAIA'])
        agg_emp_data9['bDIAC'] = np.array(emp_data_pubgame_temp_C['bDIA'])
    else:
        emp_data_pubgame_temp_C = {'bMoney': list(), 'bAIA': list(), 'bDIA': list()}
        agg_emp_data9['bMoneyC'] = np.array([])
        agg_emp_data9['bAIAC'] = np.array([])
        agg_emp_data9['bDIAC'] = np.array([])

    if mix_ratio['D'] > 0:
        emp_data_pubgame_temp_D, _, _ = _make_feature_list_(df_wide92, ["bMoney", "bAIA", "bDIA", "rMoney", "rAIA", "rDIA"], {"C": 0, "D": 1}, rescale_intensities_, rescale_intensities_pia2_)
        agg_emp_data9['bMoneyD'] = np.array(emp_data_pubgame_temp_D['bMoney'])
        agg_emp_data9['bAIAD'] = np.array(emp_data_pubgame_temp_D['bAIA'])
        agg_emp_data9['bDIAD'] = np.array(emp_data_pubgame_temp_D['bDIA'])
    else:
        emp_data_pubgame_temp_D = {'bMoney': list(), 'bAIA': list(), 'bDIA': list()}
        agg_emp_data9['bMoneyD'] = np.array([])
        agg_emp_data9['bAIAD'] = np.array([])
        agg_emp_data9['bDIAD'] = np.array([])

    if df_wide62 is not None:
        agg_emp_data69 = {
            'bMoneyC': np.concatenate((emp_data_anongame_temp_C['bMoney'], emp_data_pubgame_temp_C['bMoney']), axis=0),
            'bAIAC': np.concatenate((emp_data_anongame_temp_C['bAIA'], emp_data_pubgame_temp_C['bAIA']), axis=0),
            'bDIAC': np.concatenate((emp_data_anongame_temp_C['bDIA'], emp_data_pubgame_temp_C['bDIA']), axis=0),
            'bMoneyD': np.concatenate((emp_data_anongame_temp_D['bMoney'], emp_data_pubgame_temp_D['bMoney']), axis=0),
            'bAIAD': np.concatenate((emp_data_anongame_temp_D['bAIA'], emp_data_pubgame_temp_D['bAIA']), axis=0),
            'bDIAD': np.concatenate((emp_data_anongame_temp_D['bDIA'], emp_data_pubgame_temp_D['bDIA']), axis=0),
        }
    else:
        agg_emp_data69 = None

    json_resp_vectors = {
        'anongame': emp_data_anongame_vectors,
        'publicgame': emp_data_publicgame_vectors,
        'anongame_n': emp_data_anongame['n'],
        'publicgame_n': emp_data_publicgame['n'],
        'anongame_labels': emp_data_anongame_vector_labels,
        'publicgame_labels': emp_data_publicgame_vector_labels,
        # 'debug_anon': emp_data_anongame['debug'],
        'debug_pub': emp_data_publicgame['debug'],
    }

    json_feature_vectors = {
        'anongame': emp_data_anongame,
        'publicgame': emp_data_publicgame,
    }

    return {'json_resp_vectors': json_resp_vectors, 'json_feature_vectors': json_feature_vectors}, {'agg_emp_data6': agg_emp_data6, 'agg_emp_data9': agg_emp_data9, 'agg_emp_data69': agg_emp_data69}


def get_inferred_repu_values(agg_emp_data, repu_values_from):

    jsonout = dict()

    if repu_values_from == 'internal':
        jsonout['inferred_reputation_values_source'] = 'internal'
    else:
        jsonout['inferred_reputation_values_source'] = 'empirical'

    if repu_values_from == 'empiricalKDE':
        inferred_reputation_values = {
            'C': {
                'Money': agg_emp_data['bMoneyC'].tolist(),
                'AIA': agg_emp_data['bAIAC'].tolist(),
                'DIA': agg_emp_data['bDIAC'].tolist(),
            },
            'D': {
                'Money': agg_emp_data['bMoneyD'].tolist(),
                'AIA': agg_emp_data['bAIAD'].tolist(),
                'DIA': agg_emp_data['bDIAD'].tolist(),
            },
        }
        jsonout['inferred_reputation_values'] = inferred_reputation_values

    elif repu_values_from == 'empiricalExpectation':
        inferred_reputation_values_expectation = {
            'C': {
                'Money': agg_emp_data['bMoneyC'].mean(),
                'AIA': agg_emp_data['bAIAC'].mean(),
                'DIA': agg_emp_data['bDIAC'].mean(),
            },
            'D': {
                'Money': agg_emp_data['bMoneyD'].mean(),
                'AIA': agg_emp_data['bAIAD'].mean(),
                'DIA': agg_emp_data['bDIAD'].mean(),
            },
        }
        jsonout['inferred_reputation_values_expectation'] = inferred_reputation_values_expectation

    elif repu_values_from == 'binary':
        inferred_reputation_values_expectation = {
            'C': {
                'Money': 0.0,
                'AIA': 1.0,
                'DIA': 0.0,
            },
            'D': {
                'Money': 1.0,
                'AIA': 0.0,
                'DIA': 1.0,
            },
        }
        jsonout['inferred_reputation_values_expectation'] = inferred_reputation_values_expectation

    return jsonout


def gen_empir_kde_genericplayers_multivarkdemixture(df_wide9=None, df_wide6=None, prior_a2C=0.5, repu_values_from=None):

    jsonout_temp, agg_emp_data_temp = make_kde_lists_mulivar_(df_wide92=df_wide9, df_wide62=df_wide6, prior_a2C=prior_a2C, space='squeeze')

    kdedata = jsonout_temp['json_resp_vectors']

    agg_emp_data = agg_emp_data_temp['agg_emp_data6']

    kdedata.update(get_inferred_repu_values(agg_emp_data, repu_values_from))

    return kdedata, f'generic_pa2c-{prior_a2C:0.3f}_repu-{repu_values_from}',


def gen_empir_kde_specificplayers_multivarkdemixture(df_wide9=None, stimid=None, a1=None, repu_values_from=None):

    df_wide9_thisplayer_ = df_wide9.loc[(df_wide9['face'] == stimid), :]
    df_wide9_thisplayer_a1 = df_wide9.loc[(df_wide9['face'] == stimid) & (df_wide9['a_1'] == a1), :]

    if a1 == 'C':
        prior_a2C = 1
    elif a1 == 'D':
        prior_a2C = 0
    else:
        raise ValueError(f'a1={a1} not understood')

    jsonout_temp, _ = make_kde_lists_mulivar_(df_wide92=df_wide9_thisplayer_a1, df_wide62=None, prior_a2C=prior_a2C, space='squeeze')

    kdedata = jsonout_temp['json_resp_vectors']

    ### get data for inferred reputation values -- \pi( \omega_{base} | a_1 ) ###
    _, agg_emp_data_temp = make_kde_lists_mulivar_(df_wide92=df_wide9_thisplayer_, df_wide62=None, prior_a2C=0.5, space='squeeze')

    kdedata.update(get_inferred_repu_values(agg_emp_data_temp['agg_emp_data9'], repu_values_from))

    return kdedata, f'{stimid}-{a1}_repu-{repu_values_from}',
