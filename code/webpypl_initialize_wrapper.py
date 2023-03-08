

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
    # emp_data_dict['prob'] = np.concatenate((prob_C_, prob_D_), axis=0)
    emp_data_dict['n'] = np.concatenate((n_C_mix_normed_.round(14), n_D_mix_normed_.round(14)), axis=0).tolist()

    emp_data_dict['debug'] = {
        'mix_ratio_C': mix_ratio['C'],
        'mix_ratio_D': mix_ratio['D'],
        'counts_C': temp_df0.loc[idx_C_, 'n_'].to_numpy().tolist(),
        'counts_D': temp_df0.loc[idx_D_, 'n_'].to_numpy().tolist(),
        'n_C_mix_': n_C_mix_normed_.tolist(),
        'n_D_mix_': n_D_mix_normed_.tolist(),
    }  # DEBUG

    ### reshape into response vectors
    rrr_labels = list()
    rrr = np.full((len(emp_data_dict['n']), len(feature_list) + 1), np.nan, dtype=float)
    for i_feature, feature in enumerate(feature_list):
        rrr[:, i_feature] = emp_data_dict[feature]
        rrr_labels.append(feature)
    rrr[:, len(feature_list)] = emp_data_dict['pi_a2_C']
    rrr_labels.append('pi_a2_C')
    assert not np.any(np.isnan(rrr))

    rrr_list = list()
    for i_resp in range(rrr.shape[0]):
        rrr_list.append(rrr[i_resp, :].tolist())

    return emp_data_dict, rrr_list, rrr_labels


def make_kde_lists_mulivar_(df_wide92, df_wide62=None, prior_a2C=0.5, space='unit', **kwargs):
    ########
    # ******
    ########
    """
    'raw' space = logistic

    """
    import numpy as np
    import scipy.stats

    """ multivariate KDE
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_density.KDEMultivariate.html
    """

    """
    For the time being, using the empirical values as the samples rather than learning a KDE
    Unlike prior KDE methods, not using the exp6 data for the generic public game
    """

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
        def rescale_intensities_(x): return np.rint((n - 1) * x).astype(np.int)

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

    # emp_data_publicgame_C = _make_feature_list_(df_wide92, ["bMoney","bAIA","bDIA","rMoney","rAIA","rDIA"], {'C':1.0, 'D':0.0}, rescale_intensities_, rescale_intensities_pia2_)
    # emp_data_publicgame_D = _make_feature_list_(df_wide92, ["bMoney","bAIA","bDIA","rMoney","rAIA","rDIA"], {'C':0.0, 'D':1.0}, rescale_intensities_, rescale_intensities_pia2_)

    ### CRITICAL
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

    """
    yielded:

    agg_prior...
    {'bMoney': (0.9972696493990538, 0.48953967929958275),
    'rMoney': (0.5304798703314113, 0.6288301519135389),
    'bAIA': (0.5107255246132126, 0.8815618998641543),
    'rAIA': (0.5654584958193075, 0.5355275737517022),
    'bDIA': (1.1231988675402091, 0.5730567570635731),
    'rDIA': (0.8748128043211957, 0.5813190338839319),
    'pi_a2': array([0.2410221 , 0.28660221, 0.18370166, 0.08218232, 0.11671271, 0.08977901])}
    """

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

    print('agg_emp_data.keys()')
    print(agg_emp_data.keys())

    if repu_values_from == 'internal':
        print(f'repu_values_from :: {repu_values_from}')

    elif repu_values_from == 'empiricalKDE':
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

# if stim == '276_1':
#     assert np.sum(wpplparam_distalprior_temp['prior_pi_a2'][0:3]) < np.sum(wpplparam_distalprior_temp['prior_pi_a2'][3:6]) ###TEMP


def package_generic_webppl_param_data(modelParam, prior_form, repu_values_from, df_wide9, df_wide6, json_dir, loadpickle, datacache, prior_a2C=0.5):
    import json

    if prior_form in ['multivarkdemixture', 'multivarkdemixtureLogit']:  # , 'kde', 'kdemixture', 'broadkdemixture']:

        if prior_form == 'multivarkdemixture':
            print('\n\n :::: PRIOR_FORM: multivarkdemixture MIX :::: \n\n')

            if prior_a2C in ['split', 'splitinv']:  # TEMP (200519)
                prior_a2C = 0.5

            jsonout_temp, agg_emp_data_temp = make_kde_lists_mulivar_(df_wide9, df_wide6, prior_a2C=prior_a2C, space='squeeze')

            jsonout = jsonout_temp['json_resp_vectors']  # json_resp_vectors, json_feature_vectors

            ### CRITICAL
            ########
            ### which data to use for inferred reputation values ###
            ########
            agg_emp_data = agg_emp_data_temp['agg_emp_data6']  # agg_emp_data6, agg_emp_data9, agg_emp_data69

        elif prior_form == 'multivarkdemixtureLogit':
            print('\n\n :::: PRIOR_FORM: multivarkdemixtureLogit MIX :::: \n\n')

            if prior_a2C in ['split', 'splitinv']:  # TEMP (200519)
                prior_a2C = 0.5

            jsonout_temp, agg_emp_data_temp = make_kde_lists_mulivar_(df_wide9, df_wide6, prior_a2C=prior_a2C, space='logit', affine_b=modelParam['affine_b'], logit_k=modelParam['logit_k'])

            jsonout = jsonout_temp['json_resp_vectors']  # json_resp_vectors, json_feature_vectors

            ### CRITICAL
            ########
            ### which data to use for inferred reputation values ###
            ########
            agg_emp_data = agg_emp_data_temp['agg_emp_data6']

        elif prior_form == 'kdemixture':
            print('\n\nKDE MIX --- BROKEN\n\n')
            # _, agg_emp_data = make_kde_lists_bya1_(df_wide9, df_wide6)
            # modelParam['prior_pi_a2C'] = agg_emp_data['pi_a2C'][::-1]
            # modelParam['prior_pi_a2D'] = agg_emp_data['pi_a2D'][::-1]
            # jsonout = {
            #     'bMoneyC': agg_emp_data['bMoneyC'].tolist(),
            #     'bMoneyD': agg_emp_data['bMoneyD'].tolist(),
            #     'bAIAC': agg_emp_data['bAIAC'].tolist(),
            #     'bAIAD': agg_emp_data['bAIAD'].tolist(),
            #     'bDIAC': agg_emp_data['bDIAC'].tolist(),
            #     'bDIAD': agg_emp_data['bDIAD'].tolist(),
            #     ###
            #     'rMoneyC': agg_emp_data['rMoneyC'].tolist(),
            #     'rMoneyD': agg_emp_data['rMoneyD'].tolist(),
            #     'rAIAC': agg_emp_data['rAIAC'].tolist(),
            #     'rAIAD': agg_emp_data['rAIAD'].tolist(),
            #     'rDIAC': agg_emp_data['rDIAC'].tolist(),
            #     'rDIAD': agg_emp_data['rDIAD'].tolist(),
            # }

        elif prior_form == 'kde':
            print('\n\nKDE --- BROKEN\n\n')
            # jsonout_temp, agg_emp_data_temp = make_kde_lists_mulivar_(df_wide9, df_wide6, prior_a2C=prior_a2C, space='squeeze')
            # jsonout = jsonout_temp['json_resp_vectors'] ### json_resp_vectors, json_feature_vectors
            # agg_emp_data = agg_emp_data_temp['agg_emp_data6'] ### agg_emp_data6, agg_emp_data9, agg_emp_data69 ### which data to use for inferred reputation values
            # # modelParam['prior_pi_a2'] = agg_prior['pi_a2'][::-1] ###DEBUG make_kde_lists_mulivar_ return list of responses, not histogram
            # if not (loadpickle and datacache.is_file()):
            #     json_out_path = json_dir / 'generic_kde.json'
            #     with open(json_out_path, 'w+') as f:
            #         json.dump(jsonout, f)
            #     print(f'json dump to {json_dir}')
            #     modelParam['path_to_kde'] = str(json_out_path)

        ######################

        print(f'prior_form :: {prior_form}')
        jsonout.update(get_inferred_repu_values(agg_emp_data, repu_values_from))

        if not (loadpickle and datacache.is_file()):
            json_out_path = json_dir / f'generic_kde-{prior_a2C:0.3f}.json'
            with open(json_out_path, 'w+') as f:
                json.dump(jsonout, f)

            print(f'json dump to {json_dir}')

            modelParam['path_to_kde'] = str(json_out_path)

    else:
        print(f"prior_form {prior_form} not understood")
        raise f"prior_form {prior_form} not understood"

    return modelParam, jsonout


def package_distal_webppl_param_data(modelParam, prior_form, repu_values_from, df_wide9, json_dir, loadpickle, datacache, stimid=None, a1=None):
    import json

    df_wide9_thisplayer_ = df_wide9.loc[(df_wide9['face'] == stimid), :]
    df_wide9_thisplayer_a1 = df_wide9.loc[(df_wide9['face'] == stimid) & (df_wide9['a_1'] == a1), :]

    if a1 == 'C':
        prior_a2C = 1
    elif a1 == 'D':
        prior_a2C = 0
    else:
        raise "some error"

    if prior_form in ['multivarkdemixture', 'multivarkdemixtureLogit']:  # , 'kde', 'kdemixture', 'broadkdemixture']:

        if prior_form == 'multivarkdemixture':
            print('\n\n :::: PRIOR_FORM: multivarkdemixture MIX :::: \n\n')

            jsonout_temp, _ = make_kde_lists_mulivar_(df_wide9_thisplayer_a1, df_wide62=None, prior_a2C=prior_a2C, space='squeeze')

            jsonout = jsonout_temp['json_resp_vectors']  # json_resp_vectors, json_feature_vectors
            # agg_emp_data ### agg_emp_data6, agg_emp_data9, agg_emp_data69 ### which data to use for inferred reputation values

            print(f"prior_form :: {prior_form}")
            ### get data for inferred reputation values -- \pi( \omega_{base} | a_1 )
            _, agg_emp_data_temp = make_kde_lists_mulivar_(df_wide9_thisplayer_, df_wide62=None, prior_a2C=0.5, space='squeeze')
            bbb = get_inferred_repu_values(agg_emp_data_temp['agg_emp_data9'], repu_values_from)
            print(bbb.keys())
            jsonout.update(bbb)

        elif prior_form == 'multivarkdemixtureLogit':
            print('\n\n :::: PRIOR_FORM: multivarkdemixtureLogit MIX :::: \n\n')

            jsonout_temp, _ = make_kde_lists_mulivar_(df_wide9_thisplayer_a1, df_wide62=None, prior_a2C=prior_a2C, space='logit', affine_b=modelParam['affine_b'], logit_k=modelParam['logit_k'])

            jsonout = jsonout_temp['json_resp_vectors']  # json_resp_vectors, json_feature_vectors

            print(f"prior_form :: {prior_form}")
            ### get data for inferred reputation values -- \pi( \omega_{base} | a_1 )
            _, agg_emp_data_temp = make_kde_lists_mulivar_(df_wide9_thisplayer_, df_wide62=None, prior_a2C=0.5, space='logit', affine_b=modelParam['affine_b'], logit_k=modelParam['logit_k'])
            bbb = get_inferred_repu_values(agg_emp_data_temp['agg_emp_data9'], repu_values_from)
            print(bbb.keys())
            jsonout.update(bbb)

        if not (loadpickle and datacache.is_file()):
            json_out_path = json_dir / f'specific_kde_{stimid}-{a1}.json'
            with open(json_out_path, 'w+') as f:
                json.dump(jsonout, f)

            print(f'json dump to {json_dir}')

            modelParam['path_to_kde'] = str(json_out_path)

    else:
        print(f"prior_form {prior_form} not understood")
        raise f"prior_form {prior_form} not understood"

    return modelParam, jsonout


def importPPLdata_parallel(datain, cpar):
    from webpypl_importjson import importPPLmodel_
    from iaa_import_empirical_data_wrapper import importEmpirical_exp10_
    stimid, a_1, data_path_, wpplparam_, pots_, verbose_ = datain
    stim_a1_ppldata, _ = importPPLmodel_(data_path_, wpplparam_, pots_, verbose_)

    # Import distal prior emotion attributions
    # load empirical exp10
    subject_stats_ = importEmpirical_exp10_(stim_a1_ppldata, cpar, stimulus=stimid, condition=a_1, update_ppldata=True, bypass_plotting=True)

    return (stimid, a_1, stim_a1_ppldata, subject_stats_)


def initialize_wrapper(cpar):
    import numpy as np
    import pickle
    import dill
    from copy import deepcopy
    import time

    from iaa_import_empirical_data_wrapper import importEmpirical_exp3_7_11_, importEmpirical_exp10_
    from webpypl_importjson import Game, importPPLmodel_
    from iaa_import_empirical_data_wrapper import importEmpirical_InversePlanning_exp6_exp9

    environment = cpar.environment
    paths = cpar.paths
    runModel = cpar.cache['webppl']['runModel']
    loadpickle = cpar.cache['webppl']['loadpickle']
    hotwire_precached_data = cpar.cache['webppl'].get('hotwire_precached_data', False)
    removeOldData = cpar.cache['webppl']['removeOldData']
    saveOnExec = cpar.cache['webppl']['saveOnExec']
    empir_load_param = cpar.empir_load_param
    wppl_model_spec = cpar.wppl_model_spec
    verbose = cpar.plot_param['verbose']
    seed = getattr(cpar, 'seed', None)

    prior_form = wppl_model_spec['prior_form']
    repu_values_from = wppl_model_spec['repu_values_from']
    inf_param = wppl_model_spec['inf_param']

    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
    rng = np.random.default_rng(seed)

    a1_labels = ['C', 'D']

    datacache = paths['wpplDataCache']

    if hotwire_precached_data and datacache.is_file():
        runModel = False

    # %%

    t = Timer()
    t.start()

    inversePlanningDict = dict()
    inversePlanningDict_subject_stats = dict()
    empdf, data_stats = importEmpirical_InversePlanning_exp6_exp9(cpar, 'base', bypass_plotting=True)
    inversePlanningDict.update(empdf)
    inversePlanningDict_subject_stats['exp6'] = data_stats
    empdf, data_stats = importEmpirical_InversePlanning_exp6_exp9(cpar, 'repu', bypass_plotting=True)
    inversePlanningDict.update(empdf)
    inversePlanningDict_subject_stats['exp9'] = data_stats

    df_wide9 = inversePlanningDict['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide'].copy()
    df_wide6 = inversePlanningDict['empiricalInverseJudgmentsExtras_BaseGeneric']['df_wide'].copy()

    t.lap('invplan1')
    # %%

    if inf_param == 'rejection':
        inferenceParam = {
            'm0': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # lvl0: base agent makes decision
            'm1': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # lvl2: reputation agent makes decision
            'm3': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "rejection", "samples": 3000, "maxScore": 0, "incremental": "true"}  # infer values of inverse appraisal features
        }
    elif inf_param == 'MCMC1':
        inferenceParam = {
            'm0': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # lvl0: base agent makes decision
            'm1': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # lvl2: reputation agent makes decision
            'm3': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "MCMC", "samples": 3000, "lag": 3, "burn": 3000, "kernel": "MH"}  # infer values of inverse appraisal features
        }
    elif inf_param == 'incrementalMH1':
        inferenceParam = {
            'm0': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "incrementalMH"},  # lvl0: base agent makes decision
            'm1': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "incrementalMH"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "incrementalMH"},  # lvl2: reputation agent makes decision
            'm3': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "incrementalMH"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "MCMC", "samples": 3000, "lag": 3, "burn": 2000, "kernel": "incrementalMH"}  # infer values of inverse appraisal features
        }
    elif inf_param == 'rapidtest':
        inferenceParam = {
            'm0': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # lvl0: base agent makes decision
            'm1': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # lvl2: reputation agent makes decision
            'm3': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"}  # infer values of inverse appraisal features
        }
    elif inf_param == 'meddebug':
        inferenceParam = {
            'm0': {"method": "MCMC", "samples": 500, "lag": 3, "burn": 1000, "kernel": "MH"},  # lvl0: base agent makes decision
            'm1': {"method": "MCMC", "samples": 500, "lag": 3, "burn": 1000, "kernel": "MH"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "MCMC", "samples": 500, "lag": 3, "burn": 1000, "kernel": "MH"},  # lvl2: reputation agent makes decision
            'm3': {"method": "MCMC", "samples": 500, "lag": 3, "burn": 1000, "kernel": "MH"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"}  # infer values of inverse appraisal features
        }

    # prior on this player's estimation of the probability of a_2 = C (the other player's action) D->C
    # prior_pi_a2_C ### must appear here as P(a2=C), which is what the webppl model takes

    payoffMatrices = {"weakPD": {"C": {"otherC": 0.5, "otherD": 0}, "D": {"otherC": 1, "otherD": 0}},
                      "strongPD": {"C": {"otherC": 0.5, "otherD": 0}, "D": {"otherC": 1, "otherD": 0.25}},
                      "hawkDove": {"C": {"otherC": 0.5, "otherD": 0.25}, "D": {"otherC": 0.75, "otherD": 0.25}}
                      }

    # %%

    modelParam = {
        'pot': 0.0,
        'a1': ['C', 'D'],
        'lambda': [2.0, 2.0],  # softmax optimality parameter [base, reputation]
        'affine_b': 2.0 / 49.0,  # TEMP cpar.cache['webppl']['runModel']
        'logit_k': 0.9,  # TEMP
        'kde_width': wppl_model_spec['kde_width'],
        'prior_a2C': wppl_model_spec['prior_a2C'],
        'payoffMatrix': payoffMatrices['weakPD'],
    }
    if wppl_model_spec['refpoint_type'] == 'Power':
        modelParam['baseRefPointDist'] = wppl_model_spec['refpoint_type']
        modelParam['baseRefPoint'] = {'Money': {'scale': 5.0, 'a': 3.0, 'b': 2.0}}
    elif wppl_model_spec['refpoint_type'] == 'Norm':
        modelParam['baseRefPointDist'] = wppl_model_spec['refpoint_type']
        modelParam['baseRefPoint'] = {'Money': {'mu': 1000, 'sigma': 250}}
    elif wppl_model_spec['refpoint_type'] == 'Norm350':
        modelParam['baseRefPointDist'] = 'Norm'
        modelParam['baseRefPoint'] = {'Money': {'mu': 1000, 'sigma': 350}}
    elif wppl_model_spec['refpoint_type'] == 'Gamma':
        modelParam['baseRefPointDist'] = wppl_model_spec['refpoint_type']
        modelParam['baseRefPoint'] = {'Money': {'shape': 3, 'scale': 500}}
    elif wppl_model_spec['refpoint_type'] == 'None':
        modelParam['baseRefPointDist'] = wppl_model_spec['refpoint_type']
        modelParam['baseRefPoint'] = {'Money': 'None'}

    t.lap('prior form start')

    # ### import data
    json_dir = None
    if not (loadpickle and datacache.is_file()):
        import random
        not_unique_ = True
        while not_unique_:
            json_dir = paths['dataOutBase'] / 'temp_json_for_webppl' / f'json_for_webppl_{random.randint(0,10000)}'
            if not json_dir.is_dir():
                not_unique_ = False
        json_dir.mkdir(exist_ok=True, parents=True)

    modelParam_generic, jsonout_generic_ = package_generic_webppl_param_data(deepcopy(modelParam), prior_form, repu_values_from, df_wide9, df_wide6, json_dir, loadpickle, datacache, prior_a2C=wppl_model_spec['prior_a2C'])

    t.lap('end prior form')
    # %%
    ### initialize run
    wpplparam = {**modelParam_generic, **inferenceParam}

    ### if webppl will be run, save cpar ###
    if runModel or hotwire_precached_data:

        ### save cpar with cache settings ###
        cache_setting_ = deepcopy(cpar.cache['webppl'])
        cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

        pickle_path = paths['dataOut'] / f'cpar.dill'
        pickle_path.parent.mkdir(exist_ok=True, parents=True)
        if pickle_path.is_file():
            pickle_path.unlink()
        with open(pickle_path, 'wb') as f:
            dill.dump(cpar, f, protocol=-4)

        ### restore run settings ###
        cpar.cache['webppl'] = cache_setting_

    modelseed = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
    game_full = Game('full', wpplparam, paths['dataOut'] / 'wppl_samples', [2.0, 11, 25, 46, 77, 124, 194, 299, 457, 694, 1049, 1582, 2381, 3580, 5378, 8075, 12121, 18190, 27293, 40948, 61430, 92153, 138238, 207365])
    if runModel:
        game_full.play(paths, environment=environment, removeOldData=removeOldData, saveOnExec=saveOnExec, seed=modelseed)
    del modelseed

    modelseed = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
    game_exp3 = Game('exp3', wpplparam, paths['dataOut'] / 'wppl_samples', [3.5, 30, 139, 269.5, 310, 822.5, 1030.5, 1116.5, 1283.5, 1562.5, 1588, 1598.5, 1803, 2300.5, 2318, 2835.5, 2843.5, 2983, 3331, 3532.5, 5322.5, 5958, 6559, 7726.5, 9675, 15744.5, 19025.5, 21719.5, 24145, 28802, 30304.5, 31403, 36159, 40606, 50221, 56488, 57518, 61381.5, 65673.5, 80954.5, 81899, 94819, 130884, 130944])
    if runModel:
        game_exp3.play(paths, executable_path=game_full.executable, environment=environment, removeOldData=removeOldData, saveOnExec=saveOnExec, seed=modelseed)
    del modelseed

    t.lap('game play')
    # %% #########################

    distal_prior_game = dict()
    stimids = np.unique(df_wide9['face']).tolist()
    for i_stim, stim in enumerate(stimids):
        distal_prior_game[stim] = dict()
        for a1 in a1_labels:
            modelParam_distalPlayer, jsonout_distalPlayer_ = package_distal_webppl_param_data(deepcopy(modelParam), prior_form, repu_values_from, df_wide9, json_dir, loadpickle, datacache, stimid=stim, a1=a1)
            wpplparam_distalPlayer = {**modelParam_distalPlayer, **inferenceParam}

            modelseed = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
            distal_prior_game[stim][a1] = Game(f'distal-{stim}-{a1}', wpplparam_distalPlayer, paths['dataOut'] / 'wppl_samples', [124.0, 694.0, 1582.0, 5378.0, 12121.0, 27293.0, 61430.0, 138238.0])
            if runModel:
                print(f'running distal prior {stim} - {a1}  ({i_stim+1}/{len(stimids)+1})')
                distal_prior_game[stim][a1].play(paths, executable_path=game_full.executable, environment=environment, removeOldData=removeOldData, saveOnExec=saveOnExec, seed=modelseed)
            del modelseed

    if wppl_model_spec['prior_a2C'] in ['split', 'splitinv']:
        game_full_split = dict()
        for a1 in a1_labels:
            modelParam_generic_split, jsonout_generic_split_ = package_generic_webppl_param_data(deepcopy(modelParam), prior_form, repu_values_from, df_wide9, df_wide6, json_dir, loadpickle, datacache, prior_a2C={'C': 1.0, 'D': 0.0}[a1])
            wpplparam_split = {**modelParam_generic_split, **inferenceParam}

            modelseed = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
            game_full_split[a1] = Game(f'generic-{a1}', wpplparam_split, paths['dataOut'] / 'wppl_samples', [2.0, 11, 25, 46, 77, 124, 194, 299, 457, 694, 1049, 1582, 2381, 3580, 5378, 8075, 12121, 18190, 27293, 40948, 61430, 92153, 138238, 207365])
            if runModel:
                game_full_split[a1].play(paths, executable_path=game_full.executable, environment=environment, removeOldData=removeOldData, saveOnExec=saveOnExec, seed=modelseed)
            del modelseed

    t.lap('json')
    # %% ########################

    # ### import data
    print(f'##KASH## Starting')
    if loadpickle and datacache.is_file():
        print(f'##KASH## Loading cached data from {datacache}')
        with open(datacache, 'rb') as f:
            ppldata, ppldata_exp3, distal_prior_ppldata, wpplparam = pickle.load(f)
    else:
        print(f'##KASH## Starting to import json data. Will be cached at: {datacache}')
        t_import = Timer()
        t_import.start()
        ### load empirical data for exp3, exp7, exp11
        ppldata, ppldata_exp3, wpplparam = importEmpirical_exp3_7_11_(cpar, game_full, game_exp3, verbose=verbose)

        t_import.lap('import 3 7 11')

        if cpar.data_spec['exp10']['data_load_param']['print_responses']:
            bypass_plotting = False
        else:
            bypass_plotting = True

        subject_stats_all_ = importEmpirical_exp10_(ppldata, cpar, stimulus='all', condition=None, update_ppldata=False, bypass_plotting=bypass_plotting)
        ppldata['subject_stats']['exp10all'] = subject_stats_all_

        t_import.lap('importEmpirical_exp10_')

        # start parallel load data

        from joblib import Parallel, delayed, cpu_count

        import_ppl_list = list()
        for stim in stimids:
            for a1 in a1_labels:
                import_ppl_list.append((stim, a1, distal_prior_game[stim][a1].data_path_, distal_prior_game[stim][a1].wpplparam, distal_prior_game[stim][a1].pots, verbose))

        t_import.lap('import_ppl_list')

        with Parallel(n_jobs=min(len(import_ppl_list), cpu_count())) as pool:
            ppldata_list_loaded = pool(delayed(importPPLdata_parallel)(ppl_data_, cpar) for ppl_data_ in import_ppl_list)

        t_import.lap('importPPLdata_parallel')

        distal_prior_ppldata = dict()
        for stim in stimids:
            distal_prior_ppldata[stim] = dict()
        for stim, a1, ppl_data_loaded, subject_stats in ppldata_list_loaded:
            distal_prior_ppldata[stim][a1] = ppl_data_loaded
            distal_prior_ppldata[stim][a1]['subject_stats_'] = subject_stats

        t_import.lap('distal_prior_ppldata')

        #############

        if wppl_model_spec['prior_a2C'] == 'split':

            aaaa = dict()
            for a1 in ['C', 'D']:
                aaaa[a1], _ = importPPLmodel_(game_full_split[a1].data_path_, game_full_split[a1].wpplparam, game_full_split[a1].pots, verbose)

            ppldata['level4IAF_mergedPrior'] = ppldata.pop('level4IAF')
            ppldata['level4IAF'] = {'CC': aaaa['C']['level4IAF']['CC'].copy(), 'CD': aaaa['C']['level4IAF']['CD'].copy(), 'DC': aaaa['D']['level4IAF']['DC'].copy(), 'DD': aaaa['D']['level4IAF']['DD'].copy(), 'nobs': aaaa['C']['level4IAF']['nobs'].copy()}
            ppldata['ppldata_splitPrior'] = aaaa

        elif wppl_model_spec['prior_a2C'] == 'splitinv':

            aaaa = dict()
            for a1 in ['C', 'D']:
                aaaa[a1], _ = importPPLmodel_(game_full_split[a1].data_path_, game_full_split[a1].wpplparam, game_full_split[a1].pots, verbose)

            ppldata['level4IAF_mergedPrior'] = ppldata.pop('level4IAF')
            ppldata['level4IAF'] = {'CC': aaaa['D']['level4IAF']['CC'].copy(), 'CD': aaaa['D']['level4IAF']['CD'].copy(), 'DC': aaaa['C']['level4IAF']['DC'].copy(), 'DD': aaaa['C']['level4IAF']['DD'].copy(), 'nobs': aaaa['C']['level4IAF']['nobs'].copy()}
            ppldata['ppldata_splitPrior'] = aaaa

        # end parallel load data

        # add inverse planning empirical ratings
        ppldata.update(inversePlanningDict)
        ppldata['subject_stats'].update(inversePlanningDict_subject_stats)
        if datacache.is_file():
            print(f'##KASH## Removing cached data at {datacache}')
            datacache.unlink()
        with open(datacache, 'wb') as f:
            pickle.dump((ppldata, ppldata_exp3, distal_prior_ppldata, wpplparam), f, pickle.HIGHEST_PROTOCOL)
        print(f'##KASH## Cacheing data to {datacache} \n\t')

    t.lap('import')
    t.report()

    # %%

    ### TODO is setting wpplparam here necessary?
    setattr(cpar, 'wpplparam', wpplparam)

    return ppldata, ppldata_exp3, distal_prior_ppldata
