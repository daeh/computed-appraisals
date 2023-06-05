#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_import_empirical_data.py
"""


def import_empirical_data_wrapper(path_data, path_subjecttracker, data_load_param, import_responses_fn=None, import_participants_fn=None, calc_filter_criteria_fn=None, package_fn=None, followup_fn=None, plot_param=None, bypass_plotting=False, debug=False):
    import numpy as np
    import pandas as pd
    import warnings

    data_stats = {
        'label': data_load_param.get('label', 'none'),
        'nsub_loaded': None,
        'nsub_retained': None,
        'nresp_loaded': None,
        'nresp_retained': None,
        'nresp_per_sub_retained': None,
        '_nobs_unfiltered': None,
    }

    if debug:
        data_stats['debug'] = dict()

    ### fetch response data, filter out practice question
    datasheet_temp = import_responses_fn(path_data, data_stats, data_load_param=data_load_param)

    ### fetch participants data
    datasheet_participants = import_participants_fn(path_subjecttracker, data_load_param=data_load_param)
    data_stats['nsub_loaded'] = datasheet_participants.shape[0]

    ### calculate exclusions criteria
    filter_criteria = calc_filter_criteria_fn(datasheet_temp, datasheet_participants, data_stats, data_load_param=data_load_param)

    assert filter_criteria.loc[:, 'subjectId'].unique().shape[0] == filter_criteria.shape[0]
    filter_fn_dict = data_load_param['filter_fn']
    filter_values_dict = {'subjectId': filter_criteria['subjectId'].copy()}

    for criteria, fn in filter_fn_dict.items():
        assert criteria in filter_criteria.columns
        filter_values_dict[criteria] = filter_criteria[criteria].apply(fn)
        assert filter_values_dict[criteria].dtype == bool

    filter_ = pd.DataFrame(filter_values_dict)
    filter_.set_index('subjectId', inplace=True)

    include_series = filter_.all(axis=1)
    include_series_idx = include_series.values
    subjects_included = include_series[include_series].index
    subjects_excluded = include_series[~include_series].index

    datasheet_participants['subjectIncluded'] = include_series_idx

    if data_load_param['ncond'] is None:
        data_load_param['ncond'] = datasheet_participants['randCondNum'].abs().max() + 1

    if datasheet_participants['randCondNum'].abs().max() + 1 != data_load_param['ncond'] or datasheet_participants['randCondNum'].abs().unique().size != data_load_param['ncond']:

        if debug:
            msg = f"expected conditions not equal to found conditions ({data_load_param['ncond']} expected from load param, {datasheet_participants['randCondNum'].abs().max()+1} expected from empirical data, {datasheet_participants['randCondNum'].abs().unique().size} found in empirical data)"
            warnings.warn(msg)

    assert datasheet_participants['randCondNum'].abs().max() + 1 <= data_load_param['ncond']
    assert len(datasheet_participants['randCondNum'].abs().unique()) <= data_load_param['ncond']

    response_selector = datasheet_temp['subjectId'].isin(subjects_included)

    if response_selector.sum() == 0:
        warnings.warn(f'No responses found')

    data_included = datasheet_temp.loc[response_selector, :].copy()
    data_excluded = datasheet_temp.loc[~response_selector, :].copy()

    ### filter response data by included participants
    dataout = package_fn(data_included, subjects_included, data_stats, data_load_param=data_load_param)

    data_stats['nsub_retained'] = len(subjects_included)
    data_stats['nresp_retained'] = data_included.shape[0]
    data_stats['final_nobs'] = dataout['nobs'].copy()
    if not data_included.shape[0] / len(subjects_included) == data_stats['nresp_per_sub_retained']:
        data_stats['potential_problem'] = f" nresp/nsub = {data_included.shape[0]/len(subjects_included)}, expect {data_stats['nresp_per_sub_retained']}, grand_selector vs nobs: {data_included.shape[0]} vs {dataout['nobs'].sum().sum()}"

    phptable = np.zeros((data_load_param['ncond'], 4))
    for i_cond in range(data_load_param['ncond']):
        total_count = np.sum(np.abs(datasheet_participants['randCondNum']) == i_cond)
        valid_count = np.sum(np.abs(datasheet_participants.loc[datasheet_participants['subjectIncluded'], 'randCondNum']) == i_cond)
        phptable[i_cond, :] = np.array([i_cond, 0, total_count, valid_count])

    mmin = np.unique(np.abs(datasheet_participants.loc[datasheet_participants['subjectIncluded'], 'randCondNum']), return_counts=True)[1].min()
    mmax = np.unique(np.abs(datasheet_participants.loc[datasheet_participants['subjectIncluded'], 'randCondNum']), return_counts=True)[1].max()

    data_stats['php_table'] = phptable
    data_stats['min_resp_per_cond'] = mmin
    data_stats['max_resp_per_cond'] = mmax
    data_stats['filter_'] = filter_

    # print responses, etc.
    if not followup_fn is None and not bypass_plotting:
        if isinstance(followup_fn, list):
            for fn_ in followup_fn:
                fn_(data_included, data_excluded, datasheet_participants, data_load_param=data_load_param, plot_param=plot_param, data_stats=data_stats)
        else:
            fn_(data_included, data_excluded, datasheet_participants, data_load_param=data_load_param, plot_param=plot_param, data_stats=data_stats)

    if debug:
        data_stats['debug']['filter_criteria'] = filter_criteria
        data_stats['debug']['data_included'] = data_included
        data_stats['debug']['data_excluded'] = data_excluded
        data_stats['debug']['datasheet_temp'] = datasheet_temp
        data_stats['debug']['subjects_excluded'] = subjects_excluded
        data_stats['debug']['datasheet_participants'] = datasheet_participants
        data_stats['debug']['filter_criteria'] = filter_criteria

    return dataout, datasheet_participants, data_stats


def make_nobs_df(datasheet_in, outcomes=None, pots=None):
    import numpy as np
    import pandas as pd

    if outcomes is None:
        outcomes = np.unique(datasheet_in['outcome'].values)
    if pots is None:
        pots = np.unique(datasheet_in['pot'].values)

    index_outcome = np.full((len(outcomes), datasheet_in.shape[0]), False, dtype=bool)
    index_pot = np.full((len(pots), datasheet_in.shape[0]), False, dtype=bool)
    for i_outcome, outcome in enumerate(outcomes):
        index_outcome[i_outcome, :] = datasheet_in['outcome'] == outcome
    for i_pot, pot in enumerate(pots):
        index_pot[i_pot, :] = datasheet_in['pot'] == pot
    nobsarray = np.full((len(pots), len(outcomes)), 0, dtype=int)
    for i_outcome, outcome in enumerate(outcomes):
        for i_pot, pot in enumerate(pots):
            nobsarray[i_pot, i_outcome] = (index_outcome[i_outcome, :] & index_pot[i_pot, :]).sum()

    nobsdf = pd.DataFrame(data=nobsarray, index=pots, columns=outcomes, dtype=int)
    nobsdf.index.set_names(['pots'], inplace=True)

    assert nobsdf.sum().sum() == datasheet_in.shape[0]

    return nobsdf


def make_invplan_nobs_df(datasheet_in, a1_list=None, pots=None):
    import numpy as np
    import pandas as pd

    if a1_list is None:
        a1_list = np.unique(datasheet_in['a_1'].values)
    if pots is None:
        pots = np.unique(datasheet_in['pot'].values)

    index_a1 = np.full((len(a1_list), datasheet_in.shape[0]), False, dtype=bool)
    index_pot = np.full((len(pots), datasheet_in.shape[0]), False, dtype=bool)
    for i_a1, a1 in enumerate(a1_list):
        index_a1[i_a1, :] = datasheet_in['a_1'] == a1
    for i_pot, pot in enumerate(pots):
        index_pot[i_pot, :] = datasheet_in['pot'] == pot
    nobsarray = np.full((len(pots), len(a1_list)), 0, dtype=int)
    for i_a1, a1 in enumerate(a1_list):
        for i_pot, pot in enumerate(pots):
            nobsarray[i_pot, i_a1] = (index_a1[i_a1, :] & index_pot[i_pot, :]).sum()

    nobsdf = pd.DataFrame(data=nobsarray, index=pots, columns=a1_list, dtype=int)
    nobsdf.index.set_names(['pots'], inplace=True)

    assert nobsdf.sum().sum() == datasheet_in.shape[0]

    return nobsdf


def import_responses_exp10(path_data, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    if data_load_param is None:
        data_load_param = dict()
    drop_practice_trial = data_load_param.get('drop_practice_trial', True)

    stimulus = data_load_param['stimulus']
    condition = data_load_param['condition']
    emoLabels = data_load_param['emoLabels']

    #######
    ### Read in response data
    #######

    stim_label = stimulus
    condition_label = condition

    if stimulus in ['', 'all']:
        print(f"{stimulus} -- {condition}")
        assert condition is None
        stim_label = 'all'
        condition_label = ''

    data_stats['label'] += stim_label + condition_label

    def unitscale(x): return int(x) / 48
    def restore_quotes(x): return str(x).replace('non-profit', 'nonprofit').replace('-', '\"').replace('  ', ', ')

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={
                                     "stimulus": str,
                                     "pronoun": str,
                                     # "desc": str,
                                     "decisionThis": str,
                                     "decisionOther": str,
                                     "pot": float,
                                     "respTimer": float,
                                     "gender": str,
                                     "subjectId": str,
                                     "Data_Set": float,
                                     "HITID": str},
                                 converters={
                                     "desc": restore_quotes,
                                     "e_amusement": unitscale,
                                     "e_annoyance": unitscale,
                                     "e_confusion": unitscale,
                                     "e_contempt": unitscale,
                                     "e_devastation": unitscale,
                                     "e_disappointment": unitscale,
                                     "e_disgust": unitscale,
                                     "e_embarrassment": unitscale,
                                     "e_envy": unitscale,
                                     "e_excitement": unitscale,
                                     "e_fury": unitscale,
                                     "e_gratitude": unitscale,
                                     "e_guilt": unitscale,
                                     "e_joy": unitscale,
                                     "e_pride": unitscale,
                                     "e_regret": unitscale,
                                     "e_relief": unitscale,
                                     "e_respect": unitscale,
                                     "e_surprise": unitscale,
                                     "e_sympathy": unitscale,
                                 },
                                 )

    rename_dict_alphabetical = {
        "e_amusement": "Amusement",
        "e_annoyance": "Annoyance",
        "e_confusion": "Confusion",
        "e_contempt": "Contempt",
        "e_devastation": "Devastation",
        "e_disappointment": "Disappointment",
        "e_disgust": "Disgust",
        "e_embarrassment": "Embarrassment",
        "e_envy": "Envy",
        "e_excitement": "Excitement",
        "e_fury": "Fury",
        "e_gratitude": "Gratitude",
        "e_guilt": "Guilt",
        "e_joy": "Joy",
        "e_pride": "Pride",
        "e_regret": "Regret",
        "e_relief": "Relief",
        "e_respect": "Respect",
        "e_surprise": "Surprise",
        "e_sympathy": "Sympathy",
    }

    if emoLabels is None:
        emoLabels = [rename_dict_alphabetical[key] for key in rename_dict_alphabetical]

    ### reorder to match ppldata['lables']['emotions']
    rename_dict = dict()
    for emotion in emoLabels:
        for jsl, pyl in rename_dict_alphabetical.items():
            if pyl == emotion:
                rename_dict[jsl] = pyl
    assert len(rename_dict) == len(rename_dict_alphabetical)

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis", "decisionOther"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    outcomekey = {'Split': {'Split': 'CC', 'Stole': 'CD'}, 'Stole': {'Split': 'DC', 'Stole': 'DD'}}
    outcome = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        outcome[idx] = outcomekey[datasheet_temp['decisionThis'][idx]][datasheet_temp['decisionOther'][idx]]
    datasheet_temp['outcome'] = pd.Series(outcome).astype(CategoricalDtype(ordered=False, categories=['CC', 'CD', 'DC', 'DD']))

    data_stats['nresp_loaded'] = datasheet_temp.shape[0]

    if drop_practice_trial:
        a0 = datasheet_temp['stimulus'] != '244_2'
    else:
        a0 = np.full_like(datasheet_temp['stimulus'] != '000_0', True, dtype=bool)

    data_stats['_nobs_unfiltered'] = make_nobs_df(datasheet_temp)

    if stimulus in ['all', '']:
        a1 = np.full_like(a0, True, dtype=bool)
        a2 = np.full_like(a0, True, dtype=bool)
    else:
        assert stimulus != '244_2', "This is the practice trial"
        a1 = datasheet_temp['stimulus'] == stimulus
        a2 = datasheet_temp['decisionThis'] == {'C': 'Split', 'D': 'Stole'}[condition]

    include_idx = (a0 & a1 & a2)

    return datasheet_temp.loc[include_idx, :]


def import_responses_exp11(path_data, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    if data_load_param is None:
        data_load_param = dict()
    emoLabels = data_load_param.get('emoLabels', None)

    drop_practice_trial = data_load_param.get('drop_practice_trial', True)

    #######
    ### Read in response data
    #######

    def unitscale(x): return int(x) / 48
    def restore_quotes(x): return str(x).replace('non-profit', 'nonprofit').replace('-', '\"').replace('  ', ', ')

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={
                                     "stimulus": str,
                                     "pronoun": str,
                                     # "desc": str,
                                     "decisionThis": str,
                                     "decisionOther": str,
                                     "pot": float,
                                     "respTimer": float,
                                     "gender": str,
                                     "subjectId": str,
                                     "Data_Set": float,
                                     "HITID": str},
                                 converters={
                                     "e_amusement": unitscale,
                                     "e_annoyance": unitscale,
                                     "e_confusion": unitscale,
                                     "e_contempt": unitscale,
                                     "e_devastation": unitscale,
                                     "e_disappointment": unitscale,
                                     "e_disgust": unitscale,
                                     "e_embarrassment": unitscale,
                                     "e_envy": unitscale,
                                     "e_excitement": unitscale,
                                     "e_fury": unitscale,
                                     "e_gratitude": unitscale,
                                     "e_guilt": unitscale,
                                     "e_joy": unitscale,
                                     "e_pride": unitscale,
                                     "e_regret": unitscale,
                                     "e_relief": unitscale,
                                     "e_respect": unitscale,
                                     "e_surprise": unitscale,
                                     "e_sympathy": unitscale,
                                     "desc": restore_quotes,
                                 },
                                 )

    rename_dict_alphabetical = {
        "e_amusement": "Amusement",
        "e_annoyance": "Annoyance",
        "e_confusion": "Confusion",
        "e_contempt": "Contempt",
        "e_devastation": "Devastation",
        "e_disappointment": "Disappointment",
        "e_disgust": "Disgust",
        "e_embarrassment": "Embarrassment",
        "e_envy": "Envy",
        "e_excitement": "Excitement",
        "e_fury": "Fury",
        "e_gratitude": "Gratitude",
        "e_guilt": "Guilt",
        "e_joy": "Joy",
        "e_pride": "Pride",
        "e_regret": "Regret",
        "e_relief": "Relief",
        "e_respect": "Respect",
        "e_surprise": "Surprise",
        "e_sympathy": "Sympathy",
    }

    if emoLabels is None:
        emoLabels = [rename_dict_alphabetical[key] for key in rename_dict_alphabetical]

    ### reorder to match ppldata['lables']['emotions']
    rename_dict = dict()
    for emotion in emoLabels:
        for jsl, pyl in rename_dict_alphabetical.items():
            if pyl == emotion:
                rename_dict[jsl] = pyl
    assert len(rename_dict) == len(rename_dict_alphabetical)

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis", "decisionOther"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    outcomekey = {'Split': {'Split': 'CC', 'Stole': 'CD'}, 'Stole': {'Split': 'DC', 'Stole': 'DD'}}
    outcome = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        outcome[idx] = outcomekey[datasheet_temp['decisionThis'][idx]][datasheet_temp['decisionOther'][idx]]
    datasheet_temp['outcome'] = pd.Series(outcome).astype(CategoricalDtype(ordered=False, categories=['CC', 'CD', 'DC', 'DD']))

    data_stats['nresp_loaded'] = datasheet_temp.shape[0]

    if drop_practice_trial:
        include_idx = datasheet_temp['stimulus'] != '244_2'

    data_stats['_nobs_unfiltered'] = make_nobs_df(datasheet_temp)

    return datasheet_temp.loc[include_idx, :]


def import_responses_exp7(path_data, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    if data_load_param is None:
        data_load_param = dict()
    emoLabels = data_load_param.get('emoLabels', None)

    #######
    ### Read in response data
    #######

    def unitscale(x): return int(x) / 48

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={"condition": str,
                                        "Version": str,
                                        "stimID": str,
                                        "stimulus": str,
                                        "decisionThis": str,
                                        "decisionOther": str,
                                        "pot": float,
                                        "randStimulusFace": str,
                                        "gender": str,
                                        "subjectId": str,
                                        "Data_Set": float,
                                        "HITID": str},
                                 converters={
                                     "q1responseArray": unitscale,
                                     "q2responseArray": unitscale,
                                     "q3responseArray": unitscale,
                                     "q4responseArray": unitscale,
                                     "q5responseArray": unitscale,
                                     "q6responseArray": unitscale,
                                     "q7responseArray": unitscale,
                                     "q8responseArray": unitscale,
                                     "q9responseArray": unitscale,
                                     "q10responseArray": unitscale,
                                     "q11responseArray": unitscale,
                                     "q12responseArray": unitscale,
                                     "q13responseArray": unitscale,
                                     "q14responseArray": unitscale,
                                     "q15responseArray": unitscale,
                                     "q16responseArray": unitscale,
                                     "q17responseArray": unitscale,
                                     "q18responseArray": unitscale,
                                     "q19responseArray": unitscale,
                                     "q20responseArray": unitscale,
                                 },
                                 )

    rename_dict_alphabetical = {
        "q1responseArray": "Amusement",
        "q2responseArray": "Annoyance",
        "q3responseArray": "Confusion",
        "q4responseArray": "Contempt",
        "q5responseArray": "Devastation",
        "q6responseArray": "Disappointment",
        "q7responseArray": "Disgust",
        "q8responseArray": "Embarrassment",
        "q9responseArray": "Envy",
        "q10responseArray": "Excitement",
        "q11responseArray": "Fury",
        "q12responseArray": "Gratitude",
        "q13responseArray": "Guilt",
        "q14responseArray": "Joy",
        "q15responseArray": "Pride",
        "q16responseArray": "Regret",
        "q17responseArray": "Relief",
        "q18responseArray": "Respect",
        "q19responseArray": "Surprise",
        "q20responseArray": "Sympathy",
    }

    if emoLabels is None:
        emoLabels = [rename_dict_alphabetical[key] for key in rename_dict_alphabetical]

    ### reorder to match ppldata['lables']['emotions']
    rename_dict = dict()
    for emotion in emoLabels:
        for jsl, pyl in rename_dict_alphabetical.items():
            if pyl == emotion:
                rename_dict[jsl] = pyl
    assert len(rename_dict) == len(rename_dict_alphabetical)

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis", "decisionOther"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    outcomekey = {'Split': {'Split': 'CC', 'Stole': 'CD'}, 'Stole': {'Split': 'DC', 'Stole': 'DD'}}
    outcome = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        outcome[idx] = outcomekey[datasheet_temp['decisionThis'][idx]][datasheet_temp['decisionOther'][idx]]
    datasheet_temp['outcome'] = pd.Series(outcome).astype(CategoricalDtype(ordered=False, categories=['CC', 'CD', 'DC', 'DD']))

    data_stats['nresp_loaded'] = datasheet_temp.shape[0]
    data_stats['_nobs_unfiltered'] = make_nobs_df(datasheet_temp)

    return datasheet_temp


def import_responses_InversePlanning_Base_widedf_exp6(path_data, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    # a1_labels, path_data, path_subjecttracker

    #######
    ### Read in exp 6.2 response data
    #######

    def unitscale(x): return int(x) / 48

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={
                                     "condition": str,
                                     "Version": str,
                                     "stimID": str,
                                     "stimulus": str,
                                     "decisionThis": str,
                                     "decisionOther": str,
                                     "pot": float,
                                     "randStimulusFace": str,
                                     "BTS_actual_otherDecisionConfidence": int,
                                     "subjectId": str,
                                     "gender": str,
                                     "Data_Set": float,
                                     "HITID": str},
                                 converters={
                                     "q1responseArray": unitscale,
                                     "q2responseArray": unitscale,
                                     "q3responseArray": unitscale,
                                 },
                                 )

    rename_dict = {
        "q1responseArray": "bMoney",
        "q2responseArray": "bAIA",
        "q3responseArray": "bDIA",
        "BTS_actual_otherDecisionConfidence": "pi_a2",
    }

    assert max(datasheet_temp['q1responseArray']) == 1
    assert min(datasheet_temp['q1responseArray']) == 0

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    decision_this_key = {'Split': 'C', 'Stole': 'D'}
    decision_this = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        decision_this[idx] = decision_this_key[datasheet_temp['decisionThis'][idx]]
    datasheet_temp['a_1'] = pd.Series(decision_this).astype(CategoricalDtype(ordered=False, categories=['C', 'D']))

    data_stats['nresp_loaded'] = datasheet_temp.shape[0]

    data_stats['_nobs_unfiltered'] = make_invplan_nobs_df(datasheet_temp, a1_list=['C', 'D'])

    return datasheet_temp


def import_responses_InversePlanning_Repu_widedf_exp9(path_data, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    from pprint import pprint

    drop_practice_trial = data_load_param.get('drop_practice_trial', True)

    #######
    ### Read in exp 9 data
    #######

    def unitscale(x): return int(x) / 48
    def restore_quotes(x): return str(x).replace('non-profit', 'nonprofit').replace('-', '\"').replace('  ', ', ')
    # #### subject data

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={
                                     "stimulus": str,
                                     "pronoun": str,
                                     # "desc": str,
                                     "decisionThis": str,
                                     "pot": float,
                                     "respTimer": float,
                                     "BTS_actual_otherDecisionConfidence": int,
                                     "gender": str,
                                     "subjectId": str,
                                     "Data_Set": float,
                                     "HITID": str,
                                 },
                                 converters={
                                     "q_bMoney_Array": unitscale,
                                     "q_rMoney_Array": unitscale,
                                     "q_bAIA_Array": unitscale,
                                     "q_rAIA_Array": unitscale,
                                     "q_bDIA_Array": unitscale,
                                     "q_rDIA_Array": unitscale,
                                     "desc": restore_quotes,
                                 },
                                 )

    rename_dict = {
        "q_bMoney_Array": "bMoney",
        "q_bAIA_Array": "bAIA",
        "q_bDIA_Array": "bDIA",
        "q_rMoney_Array": "rMoney",
        "q_rAIA_Array": "rAIA",
        "q_rDIA_Array": "rDIA",
        "BTS_actual_otherDecisionConfidence": "pi_a2",
    }

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    decision_this_key = {'Split': 'C', 'Stole': 'D'}
    decision_this = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        decision_this[idx] = decision_this_key[datasheet_temp['decisionThis'][idx]]
    datasheet_temp['a_1'] = pd.Series(decision_this).astype(CategoricalDtype(ordered=False, categories=['C', 'D']))

    data_stats['nresp_loaded'] = datasheet_temp.shape[0]

    if drop_practice_trial:
        include_idx = datasheet_temp['stimulus'] != '244_2'

    data_stats['_nobs_unfiltered'] = make_invplan_nobs_df(datasheet_temp, a1_list=['C', 'D'])

    return datasheet_temp.loc[include_idx, :]


def import_participants_exp10_exp11(path_subjecttracker, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    datasheet_participants = pd.read_csv(path_subjecttracker,
                                         header=0, index_col=None,
                                         dtype={
                                             "subjectId": str,
                                             "randCondNum": int,
                                             "validationRadio": str,
                                             "subjectValidation1": bool,
                                             "expTime_min": float,
                                             "minRespTime_sec": float,
                                             "iwould_large": int,
                                             "iwould_small": int,
                                             "iexpectOther_large": int,
                                             "iexpectOther_small": int,
                                             "dem_gender": str,
                                             "dem_language": str,
                                             "browser_version": str,
                                             "browser": str,
                                             "visible_area": str,
                                             "val_recognized": str,
                                             "val_familiar": str,
                                             "val_feedback": str,
                                             "Data_Set": float,
                                             "HITID": str,
                                             "Excluded": bool,
                                             "val0(7510)": str,
                                             "val1(disdain)": str,
                                             "val2(jealousy)": str,
                                             "val3(AF25HAS)": str,
                                             "val4(steal)": str,
                                             "val5(pia2_D_a2_C)": str,
                                             "val6(pia2_D_a2_C)": str,
                                         }
                                         )

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    return datasheet_participants


def import_participants_exp7(path_subjecttracker, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    datasheet_participants = pd.read_csv(path_subjecttracker,
                                         header=0, index_col=None,
                                         dtype={
                                             "subjectId": str,
                                             "randCondNum": int,
                                             "validationRadio": str,
                                             "subjectValidation1": bool,
                                             "dem_gender": str,
                                             "dem_language": str,
                                             "val_recognized": str,
                                             "val_feedback": str,
                                             "Data_Set": float,
                                             "HITID": str,
                                             "Excluded": bool,
                                         },
                                         )

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    return datasheet_participants


def import_participants_InversePlanning_Base_widedf_exp6(path_subjecttracker, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    def randcond(xin):
        if xin == 'NaN':
            x = np.nan
        else:
            if isinstance(xin, str):
                assert f"{int(xin)}" == xin
                xin = int(xin)
            assert isinstance(xin, (int, float))
            x = int(xin)
        return x

    datasheet_participants = pd.read_csv(path_subjecttracker,
                                         header=0, index_col=None,
                                         dtype={
                                             "subjectId": str,
                                             "validationRadio": str,
                                             "subjectValidation1": bool,
                                             "dem_gender": str,
                                             "dem_language": str,
                                             "val_recognized": str,
                                             "val_feedback": str,
                                             "Data_Set": float,
                                             "HITID": str,
                                             "Excluded": bool,
                                         },
                                         converters={
                                             "randCondNum": randcond,
                                         }
                                         )

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    return datasheet_participants


def import_participants_InversePlanning_Repu_widedf_exp9(path_subjecttracker, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    datasheet_participants = pd.read_csv(
        path_subjecttracker,
        header=0,
        index_col=None,
        dtype={
            "subjectId": str,
            "randCondNum": int,
            "validationRadio": str,
            "subjectValidation1": bool,
            "expTime_min": float,
            "minRespTime_sec": float,
            "dem_gender": str,
            "dem_language": str,
            "browser_version": str,
            "browser": str,
            "visible_area": str,
            "val_recognized": str,
            "val_feedback": str,
            "Data_Set": float,
            "HITID": str,
            "Excluded": bool,
            "val0(7510)": str,
            "val1(disdainful)": str,
            "val2(split)": str,
            "val3(three/rAIA)": str,
            "val4(AF25HAS)": str,
        }
    )

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    return datasheet_participants


def response_filter(datasheet_in, subject_ids, emoLabels):
    import numpy as np

    neg_emos = ['Devastation', 'Disappointment', 'Fury', 'Annoyance']
    pos_emos = ['Relief', 'Excitement', 'Joy']
    joint_emos = [*neg_emos, *pos_emos]
    validation_valencecorr = np.zeros(datasheet_in.shape[0], dtype=bool)
    validation_valencecorr_value = [''] * datasheet_in.shape[0]
    participant_notation = np.full(subject_ids.shape[0], '', dtype=object)

    for iresp in range(datasheet_in.shape[0]):

        resp = datasheet_in.iloc[iresp, :]

        if resp.loc['outcome'] in ['CC', 'DC']:
            if np.mean(resp.loc[neg_emos]) >= np.mean(resp.loc[pos_emos]) and resp.loc['pot'] > 2000:
                validation_valencecorr[iresp] = True
                validation_valencecorr_value[iresp] += f"[neg >= pos]"
        if resp.loc['outcome'] in ['CD', 'DD']:
            if np.mean(resp.loc[pos_emos]) >= np.mean(resp.loc[neg_emos]):
                validation_valencecorr[iresp] = True
                validation_valencecorr_value[iresp] += f"[pos >= neg]"

        if np.max(resp.loc[emoLabels]) - np.min(resp.loc[emoLabels]) < 0.2:
            validation_valencecorr[iresp] = True
            validation_valencecorr_value[iresp] += f"[range {np.max(resp.loc[emoLabels]) - np.min(resp.loc[emoLabels]):0.2} < 0.2]"

        if resp.loc['outcome'] in ['CC', 'DC']:
            if np.mean(resp.loc[neg_emos]) > 0.3 and resp.loc['pot'] > 5000:
                validation_valencecorr[iresp] = True
                validation_valencecorr_value[iresp] += f"[neg {np.mean(resp.loc[neg_emos]):0.2} > 0.3]"

        if resp.loc['outcome'] in ['CD', 'DD']:
            if np.mean(resp.loc[pos_emos]) > 0.3:
                validation_valencecorr[iresp] = True
                validation_valencecorr_value[iresp] += f"[pos {np.mean(resp.loc[pos_emos]):0.2} > 0.3]"

    datasheet_in['validation_valencecorr'] = validation_valencecorr
    datasheet_in['validation_valencecorr_value'] = validation_valencecorr_value

    for isub, subid in enumerate(subject_ids):
        participant_notation[isub] = datasheet_in.loc[datasheet_in['subjectId'] == subid, 'validation_valencecorr'].sum()

    return participant_notation


def calc_filter_criteria_exp10(datasheet_temp, datasheet_participants, data_stats, data_load_param=None):
    import numpy as np

    stimulus = data_load_param['stimulus']
    emoLabels = data_load_param.get('emoLabels', None)

    if stimulus in ['', 'all']:
        unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
        unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

        ########
        # Test that all responses are associated with a batch_2_ subject
        ########
        np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

        ########
        # Test that all subjects have same number of responses
        ########
        nresponses = list()
        for subject in unique_sub_batch2:
            nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
        assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"

    ##########
    ### Subject Filter
    ##########

    for val_id in ["val0(7510)", "val1(disdain)", "val2(jealousy)", "val3(AF25HAS)", "val4(steal)", "val5(pia2_D_a2_C)", "val6(pia2_D_a2_C)"]:
        datasheet_participants[val_id].fillna('correct_response', inplace=True)

    datasheet_participants['response_filter'] = response_filter(datasheet_temp, datasheet_participants.loc[:, 'subjectId'], emoLabels)

    validation_df = datasheet_participants.loc[:, ('subjectId', 'Data_Set', 'Excluded')].copy()
    for val_id in ["val0(7510)", "val1(disdain)", "val2(jealousy)", "val3(AF25HAS)", "val4(steal)", "val5(pia2_D_a2_C)", "val6(pia2_D_a2_C)", "response_filter"]:
        validation_df[val_id] = datasheet_participants.loc[:, val_id].copy()

    return validation_df


def package_empdata_exp10(data_included, subjects_included, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    import warnings

    stimulus = data_load_param['stimulus']
    condition = data_load_param['condition']
    emoLabels = data_load_param['emoLabels']
    outcomes = data_load_param['outcomes']

    if stimulus in ['', 'all']:
        print(f"{stimulus} -- {condition}")
        assert condition is None

    ##############
    ## pull data by subject, repackage minimal variables
    ##############

    ### make categorical after selecting which data to include
    for catfield in ["subjectId", "stimulus", "pot", "HITID", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    empemodata_bysubject_list = list()
    for subID in subjects_included:
        subjectData = data_included.loc[data_included['subjectId'] == subID, :]

    ### if not selecting by stimuli, make by-subject dicts
    if stimulus in ['all', '']:
        nresp_list = list()
        empemodata_bysubject_list = list()
        for iSubject, subID in enumerate(subjects_included):
            subjectData = data_included.loc[(data_included['subjectId'] == subID), :]

            tempdict = dict()
            nTrials = subjectData.shape[0]
            prob = 1 / nTrials
            nresp_list.append(nTrials)

            for feature in emoLabels:
                tempdict[('emotionIntensities', feature)] = subjectData[feature]

            tempdict[('stimulus', 'outcome')] = subjectData['outcome']
            tempdict[('stimulus', 'pot')] = subjectData['pot']
            tempdict[('subjectId', 'subjectId')] = subjectData['subjectId']
            tempdict[('prob', 'prob')] = [prob] * nTrials

            empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

        alldata = pd.concat(empemodata_bysubject_list)

        assert len(np.unique(nresp_list)) == 1
        data_stats['nresp_per_sub_retained'] = nresp_list[0]

    else:
        emp_emo_data_bypot_list = list()
        for i_pot, pot in enumerate(data_included['pot'].cat.categories):
            potData = data_included.loc[(data_included['pot'] == pot), :]

            tempdict = dict()
            nTrials = potData.shape[0]
            if nTrials > 0:
                prob = 1 / nTrials

                for feature in emoLabels:
                    tempdict[('emotionIntensities', feature)] = potData[feature]

                tempdict[('stimulus', 'outcome')] = potData['outcome']
                tempdict[('stimulus', 'pot')] = potData['pot']
                tempdict[('subjectId', 'subjectId')] = potData['subjectId']
                tempdict[('prob', 'prob')] = [prob] * nTrials

                emp_emo_data_bypot_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))
            else:
                columns_temp = [('emotionIntensities', feature) for feature in emoLabels] + [('stimulus', 'outcome'), ('stimulus', 'pot'), ('subjectId', 'subjectId'), ('prob', 'prob')]
                emp_emo_data_bypot_list.append(pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns_temp)))

        alldata = pd.concat(emp_emo_data_bypot_list)
        data_stats['nresp_per_sub_retained'] = np.nan

    pots = np.unique(alldata[('stimulus', 'pot')])
    assert np.all(pots == data_included['pot'].cat.categories)

    ###
    tempdfdict = dict()
    for outcome in outcomes:
        tempdfdict[outcome] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(outcomes)), 0, dtype=int), index=pots, columns=outcomes, dtype=int)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalEmotionJudgments = dict()
    if stimulus in ['all', '']:
        outcome_loop = outcomes
    else:
        outcome_loop = {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[condition]
    for outcome in outcome_loop:
        for i_pot, pot in enumerate(pots):
            df = alldata.loc[((alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot)), ['emotionIntensities', 'prob']]
            nobsdf.loc[pot, outcome] = df.shape[0]
            if df.shape[0] > 0:
                df[('prob', 'prob')] = 1 / df.shape[0]
                tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

        if len(tempdfdict[outcome]) == 0:
            import warnings
            warnings.warn(f'Stim :: {stimulus} - {outcome} !!!! no data')
            empiricalEmotionJudgments[outcome] = 'no data'
        all_none = True
        for item in tempdfdict[outcome]:
            if item is not None:
                all_none = False
            break
        if not all_none:
            empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
        else:
            empiricalEmotionJudgments[outcome] = 'no data yet'
    empiricalEmotionJudgments['nobs'] = nobsdf

    return empiricalEmotionJudgments


def calc_filter_criteria_exp11(datasheet_temp, datasheet_participants, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    import warnings

    emoLabels = data_load_param.get('emoLabels', None)

    ########
    # Test that all subjects have same number of responses
    ########
    unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
    unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

    nresponses = list()
    for subject in unique_sub_batch2:
        nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
    assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"

    ########
    # Test that all responses are associated with a batch_2_ subject
    ########
    np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

    ##########
    ### Subject Filter
    ##########

    for val_id in ["val0(7510)", "val1(disdain)", "val2(jealousy)", "val3(AF25HAS)", "val4(steal)", "val5(pia2_D_a2_C)", "val6(pia2_D_a2_C)"]:
        datasheet_participants[val_id].fillna('correct_response', inplace=True)

    datasheet_participants['response_filter'] = response_filter(datasheet_temp, datasheet_participants.loc[:, 'subjectId'], emoLabels)

    validation_df = datasheet_participants.loc[:, ('subjectId', 'Data_Set', 'Excluded')].copy()
    for val_id in ["val0(7510)", "val1(disdain)", "val2(jealousy)", "val3(AF25HAS)", "val4(steal)", "val5(pia2_D_a2_C)", "val6(pia2_D_a2_C)", "response_filter"]:
        validation_df[val_id] = datasheet_participants.loc[:, val_id].copy()

    return validation_df


def package_empdata_exp11(data_included, subjects_included, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    import warnings

    emoLabels = data_load_param.get('emoLabels', None)
    outcomes = data_load_param['outcomes']

    ##############
    ## pull data by subject, repackage minimal variables
    ##############

    ### make categorical after selecting which data to include
    for catfield in ["subjectId", "stimulus", "pot", "HITID", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    nresp_list = list()
    empemodata_bysubject_list = list()
    for subID in subjects_included:
        subjectData = data_included.loc[data_included['subjectId'] == subID, :]

        tempdict = dict()
        nTrials = subjectData.shape[0]
        prob = 1 / nTrials
        nresp_list.append(nTrials)

        for feature in emoLabels:
            tempdict[('emotionIntensities', feature)] = subjectData[feature]

        tempdict[('stimulus', 'outcome')] = subjectData['outcome']
        tempdict[('stimulus', 'pot')] = subjectData['pot']
        tempdict[('subjectId', 'subjectId')] = subjectData['subjectId']
        tempdict[('prob', 'prob')] = [prob] * nTrials

        empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

    assert len(np.unique(nresp_list)) == 1
    data_stats['nresp_per_sub_retained'] = nresp_list[0]
    alldata = pd.concat(empemodata_bysubject_list)

    pots = np.unique(alldata[('stimulus', 'pot')])
    assert np.all(pots == data_included['pot'].cat.categories)

    ###
    tempdfdict = dict()
    for outcome in outcomes:
        tempdfdict[outcome] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(outcomes)), 0, dtype=int), index=pots, columns=outcomes, dtype=int)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalEmotionJudgments = dict()
    for outcome in outcomes:
        for i_pot, pot in enumerate(pots):
            df = alldata.loc[((alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot)), ['emotionIntensities', 'prob']]
            nobsdf.loc[pot, outcome] = df.shape[0]
            if df.shape[0] > 0:
                df[('prob', 'prob')] = 1 / df.shape[0]
                tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

        if len(tempdfdict[outcome]) == 0:
            import warnings
            warnings.warn(f'!!!! no data')
            empiricalEmotionJudgments[outcome] = 'no data'
        all_none = True
        for item in tempdfdict[outcome]:
            if item is not None:
                all_none = False
            break
        if not all_none:
            empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
        else:
            empiricalEmotionJudgments[outcome] = 'no data'
    empiricalEmotionJudgments['nobs'] = nobsdf

    return empiricalEmotionJudgments


def calc_filter_criteria_exp7(datasheet_temp, datasheet_participants, data_stats, data_load_param=None):
    import numpy as np

    emoLabels = data_load_param.get('emoLabels', None)

    ########
    # Test that all subjects have same number of responses
    ########
    unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
    unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

    nresponses = list()
    for subject in unique_sub_batch2:
        nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
    assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"

    ########
    # Test that all responses are associated with a batch_2_ subject
    ########
    np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

    ##########
    ### Subject Filter
    ##########

    datasheet_participants['response_filter'] = response_filter(datasheet_temp, datasheet_participants.loc[:, 'subjectId'], emoLabels)

    validation_df = datasheet_participants.loc[:, ('subjectId', 'Data_Set', 'Excluded')].copy()
    for val_id in ["subjectValidation1", "response_filter"]:
        validation_df[val_id] = datasheet_participants.loc[:, val_id].copy()

    return validation_df


def package_empdata_exp7(data_included, subjects_included, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    import warnings

    emoLabels = data_load_param.get('emoLabels', None)
    outcomes = data_load_param['outcomes']

    ##############
    ## pull data by subject, repackage minimal variables
    ##############

    ### make categorical after selecting which data to include
    for catfield in ["subjectId", "stimulus", "pot", "HITID", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    nresp_list = list()
    empemodata_bysubject_list = list()
    for subID in subjects_included:
        subjectData = data_included.loc[data_included['subjectId'] == subID, :]

        tempdict = dict()
        nTrials = subjectData.shape[0]
        prob = 1 / nTrials
        nresp_list.append(nTrials)

        for feature in emoLabels:
            tempdict[('emotionIntensities', feature)] = subjectData[feature]

        tempdict[('stimulus', 'outcome')] = subjectData['outcome']
        tempdict[('stimulus', 'pot')] = subjectData['pot']
        tempdict[('subjectId', 'subjectId')] = subjectData['subjectId']
        tempdict[('prob', 'prob')] = [prob] * nTrials

        empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

    assert len(np.unique(nresp_list)) == 1
    data_stats['nresp_per_sub_retained'] = nresp_list[0]
    alldata = pd.concat(empemodata_bysubject_list)

    pots = np.unique(alldata[('stimulus', 'pot')])
    assert np.all(pots == data_included['pot'].cat.categories)

    ###
    tempdfdict = dict()
    for outcome in outcomes:
        tempdfdict[outcome] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(outcomes)), 0, dtype=int), index=pots, columns=outcomes, dtype=int)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalEmotionJudgments = dict()
    for outcome in outcomes:
        for i_pot, pot in enumerate(pots):
            df = alldata.loc[((alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot)), ['emotionIntensities', 'prob']]
            nobsdf.loc[pot, outcome] = df.shape[0]
            if df.shape[0] > 0:
                df[('prob', 'prob')] = 1 / df.shape[0]
                tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

        empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
    empiricalEmotionJudgments['nobs'] = nobsdf

    return empiricalEmotionJudgments


def calc_filter_criteria_InversePlanning_Base_widedf_exp6(datasheet_temp, datasheet_participants, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    import warnings

    ########
    # Test that all subjects have same number of responses
    ########
    unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
    unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

    nresponses = list()
    for subject in unique_sub_batch2:
        nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
    assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"
    data_stats['nresp_per_sub_retained'] = nresponses[0]

    ########
    # Test that all responses are associated with a batch_2_ subject
    ########
    np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

    ##########
    ### Subject Filter
    ##########
    validation_df = datasheet_participants.loc[:, ('subjectId', 'Data_Set', 'Excluded')].copy()
    for val_id in ["subjectValidation1"]:
        validation_df[val_id] = datasheet_participants.loc[:, val_id].copy()

    return validation_df


def package_empdata_InversePlanning_Base_widedf_exp6(data_included, subjects_included, data_stats, data_load_param=None):
    from pandas.api.types import CategoricalDtype

    ##############
    ## pull data by subject, repackage minimal variables
    ##############

    ### make categorical after selecting which data to include

    data_included.rename(columns={'stimulus': 'face'}, inplace=True)

    for catfield in ["subjectId", "face", "pot", "HITID", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    features = ["bMoney", "bAIA", "bDIA", "pi_a2"]  # NB ordering

    shorthand = {
        "bMoney": "getting money",
        "bAIA": "not getting too much",
        "bDIA": "not getting too little",
        "pi_a2": "p_1's expectation of a_2"}

    shorthand_list = [
        "getting money\n(bMoney)",
        "not getting too much\n(bAIA)",
        "not getting too little\n(bDIA)",
        "belief about $a_2$\n($\pi_{a_2=D}$)"]

    data_reduced_wide = data_included[list(shorthand.keys()) + ['a_1', 'pot', 'face', 'subjectId']]

    data_reduced_wide_sorted = data_reduced_wide.sort_values(['a_1', 'pot', 'face'], ascending=[1, 1, 1], inplace=False)

    return package_empdata_InversePlanning_exp6_exp9(data_reduced_wide_sorted.reset_index(inplace=False, drop=True), features, shorthand, shorthand_list, data_load_param['a1_labels'])


def calc_filter_criteria_InversePlanning_Repu_widedf_exp9(datasheet_temp, datasheet_participants, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    import warnings

    ########
    # Test that all subjects have same number of responses
    ########
    unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
    unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

    nresponses = list()
    for subject in unique_sub_batch2:
        nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
    assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"
    data_stats['nresp_per_sub_retained'] = nresponses[0]

    ########
    # Test that all responses are associated with a batch_2_ subject
    ########
    np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

    ##########
    ### Subject Filter
    ##########

    for val_id in ["val0(7510)", "val1(disdainful)", "val2(split)", "val3(three/rAIA)", "val4(AF25HAS)"]:
        datasheet_participants[val_id].fillna('correct_response', inplace=True)

    validation_df = datasheet_participants.loc[:, ('subjectId', 'Data_Set', 'Excluded')].copy()
    for val_id in ["subjectValidation1", "val0(7510)", "val1(disdainful)", "val2(split)", "val3(three/rAIA)", "val4(AF25HAS)"]:
        validation_df[val_id] = datasheet_participants.loc[:, val_id].copy()

    return validation_df


def package_empdata_InversePlanning_Repu_widedf_exp9(data_included, subjects_included, data_stats, data_load_param=None):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    import warnings

    ##############
    ## pull data by subject, repackage minimal variables
    ##############

    ### make categorical after selecting which data to include

    data_included.rename(columns={'stimulus': 'face'}, inplace=True)

    for catfield in ["subjectId", "face", "pot", "HITID", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    features = ["bMoney", "bAIA", "bDIA", "rMoney", "rAIA", "rDIA", "pi_a2"]  # NB ordering

    shorthand = {
        "bMoney": "getting money",
        "rMoney": "reputation for not prioritizing money",
        "bAIA": "not getting too much",
        "rAIA": "reputation for being considerate",
        "bDIA": "not getting too little",
        "rDIA": "reputation for being competitive",
        "pi_a2": "$p_1$'s expectation of $a_2=D$"}

    shorthand_list = [
        "getting money\n(bMoney)",
        "reputation for not prioritizing money\n(rMoney)",
        "not getting too much\n(bAIA)",
        "reputation for being considerate\n(rAIA)",
        "not getting too little\n(bDIA)",
        "reputation for being competitive\n(rDIA)",
        "belief about $a_2$\n($\pi_{a_2=D}$)"]

    data_reduced_wide = data_included[list(shorthand.keys()) + ['a_1', 'pot', 'face', 'desc', 'subjectId']]

    data_reduced_wide_sorted = data_reduced_wide.sort_values(['a_1', 'pot', 'face'], ascending=[1, 1, 1], inplace=False)

    return package_empdata_InversePlanning_exp6_exp9(data_reduced_wide_sorted.reset_index(inplace=False, drop=True), features, shorthand, shorthand_list, data_load_param['a1_labels'])


def package_empdata_InversePlanning_exp6_exp9(df_wide, feature_list, shorthand, shorthand_list, a1_labels):
    import numpy as np
    import pandas as pd

    pots = np.unique(df_wide['pot'])
    assert np.all(pots == df_wide['pot'].cat.categories)

    ###
    tempdfdict = dict()
    for a_1 in a1_labels:
        tempdfdict[a_1] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(a1_labels)), 0, dtype=int), index=pots, columns=a1_labels, dtype=int)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalPreferenceJudgments = dict()
    iterables = [['feature'], feature_list]
    idx_a1 = [(df_wide['a_1'] == a_1) for a_1 in a1_labels]
    idx_pot = [(df_wide['pot'] == pot) for pot in pots]
    for i_a1, a_1 in enumerate(a1_labels):
        for i_pot, pot in enumerate(pots):
            df = df_wide.loc[idx_a1[i_a1] & idx_pot[i_pot], feature_list]
            np.testing.assert_array_equal(df.columns.to_list(), feature_list)
            df.columns = pd.MultiIndex.from_product(iterables)
            nobsdf.loc[pot, a_1] = df.shape[0]
            if df.shape[0] > 0:
                df[('prob', 'prob')] = 1 / df.shape[0]
                tempdfdict[a_1][i_pot] = df.reset_index(inplace=False, drop=True)

        empiricalPreferenceJudgments[a_1] = pd.concat(tempdfdict[a_1], axis=0, keys=pots, names=['pots', None])
    empiricalPreferenceJudgments['nobs'] = nobsdf

    empiricalPreferenceJudgments['rider_hack'] = {'df_wide': df_wide, 'feature_list': feature_list, 'shorthand': shorthand, 'shorthand_list': shorthand_list}

    return empiricalPreferenceJudgments


def print_emp_responses_emotions(data_included, data_excluded, datasheet_participants, data_load_param=None, plot_param=None, data_stats=None):
    import numpy as np
    import pandas as pd
    import warnings

    dataset_label = data_load_param.get('label', 'none')

    if data_load_param['print_responses']:
        emoLabels = data_load_param.get('emoLabels', None)
        savepath = data_load_param['savepath']

        sub_inc = datasheet_participants.loc[datasheet_participants['subjectIncluded'], 'subjectId']
        sub_exc = datasheet_participants.loc[~datasheet_participants['subjectIncluded'], 'subjectId']

        print_subject_filter(data_excluded, emoLabels, plot_param=plot_param, dataset=f'excluded/{dataset_label}_{sub_inc.shape[0]}')
        print_subject_filter(data_included, emoLabels, plot_param=plot_param, dataset=f'included/{dataset_label}_{sub_exc.shape[0]}')


def print_rand_conds(data_included, data_excluded, datasheet_participants, data_load_param=None, plot_param=None, data_stats=None):
    import numpy as np
    from cam_plot_utils import printFigList

    plt = plot_param['plt']

    dataset_label = data_load_param.get('label', 'none')
    savepath = plot_param['figsPath'] / 'subjectResponses'

    plt.close('all')
    figsout = list()

    ncond = data_load_param['ncond']

    valid_conds = np.abs(datasheet_participants.loc[datasheet_participants['subjectIncluded'], 'randCondNum'])
    hist_ = np.histogram(valid_conds, bins=ncond, density=False)[0]
    mmin = hist_.min()
    n_mmin = np.sum(hist_ == mmin)

    all_conds = np.abs(datasheet_participants.loc[:, 'randCondNum'])
    hist_t = np.histogram(all_conds, bins=ncond, density=False)[0]
    mmin_t = hist_t.min()
    n_mmin_t = np.sum(hist_t == mmin_t)

    figout, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
    figout.subplots_adjust(wspace=0.5)

    ax = axs[0]
    ax.hist(valid_conds, bins=range(ncond), color='orange', zorder=10)
    ax.hist(all_conds, bins=range(ncond), color='blue', zorder=1)
    ax.set_title(f"{dataset_label}\nmin valid: {mmin}(x{n_mmin}), min total: {mmin_t}(x{n_mmin_t})")

    ax = axs[1]
    ax.hist(valid_conds, bins=range(ncond), color='orange', zorder=10)
    ax.hist(all_conds, bins=range(ncond), color='blue', zorder=1)
    ax.set_ylim((0, 4))

    figsout.append((savepath / f'randCondHist_{dataset_label}.pdf', figout))

    _ = printFigList(figsout, plot_param)

    plt.close(figout)
    plt.close('all')


def print_subject_filter(datasheet, emoLabels, plot_param=None, dataset='', excluded_elsewhere=None):
    from cam_plot_utils import printFigList

    plt = plot_param['plt']
    savepath = plot_param['figsPath'] / 'subjectResponses'

    if excluded_elsewhere is None:
        excluded_elsewhere = list()

    plt.close('all')
    excluded_participants_figs = list()

    neg_emos = ['Devastation', 'Disappointment', 'Fury', 'Annoyance']
    pos_emos = ['Relief', 'Excitement', 'Joy']
    joint_emos = [*neg_emos, *pos_emos]

    subject_list = datasheet.loc[:, 'subjectId'].unique()

    for i_subid, subid in enumerate(subject_list):
        tempdf = datasheet.loc[datasheet.loc[:, 'subjectId'] == subid, :]
        selected_emo_idx = list()
        for emol in joint_emos:
            selected_emo_idx.append(emoLabels.index(emol))
        figout, axes = plt.subplots(2, 2, figsize=(8, 4), constrained_layout=True, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [2, 2]}, sharex=True, sharey=True)
        axs = axes.flatten()
        for i_outcome, outcome in enumerate(['CC', 'CD', 'DC', 'DD']):
            tempdf_outcome = tempdf.loc[tempdf.loc[:, 'outcome'] == outcome, :]
            titletxt = f"{outcome}"
            for iresp in range(tempdf_outcome.shape[0]):
                if tempdf_outcome.iloc[iresp, :]['stimulus'] != '244_2':
                    titletxt += f", {tempdf_outcome.iloc[iresp,:]['pot']}"
            for iresp in range(tempdf_outcome.shape[0]):
                if tempdf_outcome.iloc[iresp, :]['stimulus'] != '244_2':
                    linestyle = '--'
                    if tempdf_outcome.iloc[iresp, :]['validation_valencecorr']:
                        titletxt = '*' + titletxt + '\n' + tempdf_outcome.iloc[iresp, :]['validation_valencecorr_value']
                        axs[i_outcome].scatter(selected_emo_idx, tempdf_outcome.iloc[iresp, :].loc[joint_emos], color=['green', 'blue', 'red', 'black'][i_outcome], alpha=0.5)
                        linestyle = '-'
                    axs[i_outcome].plot(range(len(emoLabels)), tempdf_outcome.iloc[iresp, :].loc[emoLabels], linestyle, color=['green', 'blue', 'red', 'black'][i_outcome], alpha=0.5)

            axs[i_outcome].axvline(x=emoLabels.index('Surprise'), color='k', linewidth=1, alpha=0.2)
            axs[i_outcome].axhline(y=0.3, color='k', linewidth=1, alpha=0.2)
            axs[i_outcome].axhline(y=0.7, color='k', linewidth=1, alpha=0.2)
            axs[i_outcome].set_title(titletxt, fontsize=9)
            axs[i_outcome].set_xticks(range(len(emoLabels)))
            axs[i_outcome].set_xticklabels(emoLabels, rotation=-35, ha='left', rotation_mode='anchor', fontsize=9)
            axs[i_outcome].set_ylim([-0.1, 1.1])

        subj_exluded_elsewhere = 'o'
        if subid in excluded_elsewhere:
            subj_exluded_elsewhere = 'x'

        plt.suptitle(f"{subj_exluded_elsewhere}  {subid} ({i_subid}/{len(subject_list)})", fontsize=9)
        excluded_participants_figs.append((savepath / f'{dataset}' / f'sub_{subj_exluded_elsewhere}_{subid}.pdf', figout))

        plt.close(figout)

    _ = printFigList(excluded_participants_figs, plot_param)


def importEmpirical_exp10_(ppldata, cpar, stimulus='all', condition=None, update_ppldata=False, bypass_plotting=False):

    suffix = ''

    empiricalEmotionJudgments, datasheet_participants, data_stats = import_empirical_data_wrapper(cpar.paths['exp10xlsx'], cpar.paths['subjectrackerexp10'], {**cpar.data_spec['exp10']['data_load_param'], **dict(stimulus=stimulus, condition=condition)}, **cpar.data_spec['exp10']['import_fn_dict'], plot_param={**cpar.plot_param, **dict(figsPath=cpar.paths['figsOut'])}, bypass_plotting=bypass_plotting)

    if update_ppldata:
        outcomes = cpar.data_spec['exp10']['data_load_param']['outcomes']

        pots = data_stats['_nobs_unfiltered'].index.values

        ppldata['labels']['emotions' + suffix] = cpar.data_spec['exp10']['data_load_param']['emoLabels']
        ppldata['empiricalEmotionJudgments' + suffix] = empiricalEmotionJudgments

        ppldata['pots' + suffix] = pots

        ppldata['pots_byoutcome' + suffix] = {'all': ppldata['pots' + suffix]}
        for outcome in outcomes:
            pots_temp = data_stats['_nobs_unfiltered'].index[data_stats['_nobs_unfiltered'][outcome] > 0]
            ppldata['pots_byoutcome' + suffix][outcome] = pots_temp

        if 'subject_stats' not in ppldata:
            ppldata['subject_stats'] = dict()
        ppldata['subject_stats'].update(data_stats)

    return data_stats


def importEmpirical_exp7_11_(ppldata, cpar):
    import numpy as np
    import pandas as pd

    empiricalEmotionJudgments7, datasheet_participants_7, data_stats_7 = import_empirical_data_wrapper(cpar.paths['exp7xlsx'], cpar.paths['subjectrackerexp7'], cpar.data_spec['exp7']['data_load_param'], **cpar.data_spec['exp7']['import_fn_dict'], plot_param={**cpar.plot_param, **dict(figsPath=cpar.paths['figsOut'])})
    empiricalEmotionJudgments11, datasheet_participants_11, data_stats_11 = import_empirical_data_wrapper(cpar.paths['exp11xlsx'], cpar.paths['subjectrackerexp11'], cpar.data_spec['exp11']['data_load_param'], **cpar.data_spec['exp11']['import_fn_dict'], plot_param={**cpar.plot_param, **dict(figsPath=cpar.paths['figsOut'])})

    empiricalEmotionJudgments_combined = dict()
    nobsdf = empiricalEmotionJudgments7['nobs'].copy()
    allpots = list()
    for outcome in ppldata['labels']['outcomes']:

        empiricalEmotionJudgments_combined_lists = list()
        for pot in np.unique(empiricalEmotionJudgments7[outcome].index.get_level_values('pots')):
            potidx7 = empiricalEmotionJudgments7[outcome].index.get_level_values('pots') == pot
            if pot in empiricalEmotionJudgments11[outcome].index.get_level_values('pots'):
                potidx11 = empiricalEmotionJudgments11[outcome].index.get_level_values('pots') == pot

                tempdf7 = empiricalEmotionJudgments7[outcome].loc[potidx7, :]
                tempdf11 = empiricalEmotionJudgments11[outcome].loc[potidx11, :]
                tempdf = pd.concat([tempdf7, tempdf11])
                tempdf.loc[:, ('prob', 'prob')] = tempdf.shape[0]**-1
                nobsdf.loc[pot, outcome] = tempdf.shape[0]
            else:
                tempdf = empiricalEmotionJudgments7[outcome].loc[potidx7, :]

            empiricalEmotionJudgments_combined_lists.append(tempdf)

        empiricalEmotionJudgments_combined[outcome] = pd.concat(empiricalEmotionJudgments_combined_lists)

    empiricalEmotionJudgments_combined['nobs'] = nobsdf

    data_stats = {
        'exp7': data_stats_7,
        'exp11': data_stats_11,
        'exp711cat': {'nobsdf': nobsdf.copy()},
    }

    ppldata['labels']['emotions'] = cpar.data_spec['exp11']['data_load_param']['emoLabels']
    ppldata['empiricalEmotionJudgments'] = empiricalEmotionJudgments_combined

    ppldata['pots'] = empiricalEmotionJudgments_combined['nobs'].index.to_list()

    ppldata['pots_byoutcome'] = {'all': ppldata['pots']}
    for outcome in ppldata['labels']['outcomes']:
        ppldata['pots_byoutcome'][outcome] = empiricalEmotionJudgments_combined['nobs'].index[empiricalEmotionJudgments_combined['nobs'][outcome] > 0].to_list()

    if not 'subject_stats' in ppldata:
        ppldata['subject_stats'] = dict()
    ppldata['subject_stats'].update(data_stats)

    return data_stats


def importEmpirical_InversePlanning_exp6_exp9(cpar, level, bypass_plotting=False):
    import numpy as np

    if level == 'base':
        suffix = 'BaseGeneric'
        empiricalPreferenceJudgments, datasheet_participants, data_stats = import_empirical_data_wrapper(cpar.paths['exp6xlsx'], cpar.paths['subjectrackerexp6'], cpar.data_spec['exp6']['data_load_param'], **cpar.data_spec['exp6']['import_fn_dict'], plot_param={**cpar.plot_param, **dict(figsPath=cpar.paths['figsOut'])}, bypass_plotting=bypass_plotting)
        a1_labels = cpar.data_spec['exp6']['data_load_param']['a1_labels']
    elif level == 'repu':
        suffix = 'RepuSpecific'
        empiricalPreferenceJudgments, datasheet_participants, data_stats = import_empirical_data_wrapper(cpar.paths['exp9xlsx'], cpar.paths['subjectrackerexp9'], cpar.data_spec['exp9']['data_load_param'], **cpar.data_spec['exp9']['import_fn_dict'], plot_param={**cpar.plot_param, **dict(figsPath=cpar.paths['figsOut'])}, bypass_plotting=bypass_plotting)
        a1_labels = cpar.data_spec['exp9']['data_load_param']['a1_labels']

    rider_hack = empiricalPreferenceJudgments.pop('rider_hack')  # <--- df_wide, feature_list, shorthand, shorthand_list

    ################
    ####### make dict
    ################

    pots = data_stats['_nobs_unfiltered'].index.values

    if suffix == 'RepuSpecific':
        data_stats['nobs_by_stim'] = dict()
        nobsdf_unfil = data_stats['_nobs_unfiltered']
        df_wide = rider_hack['df_wide']
        stim_ids = np.unique(df_wide['face'].to_numpy())

        stim_idx_array = np.full((stim_ids.size, df_wide.shape[0]), 0, dtype=int)
        for i_stimid, stimid in enumerate(nobsdf_unfil.columns.to_numpy()):
            stim_idx_array[i_stimid, :] = df_wide['face'] == stimid
        a1_idx_array = np.full((nobsdf_unfil.shape[1], df_wide.shape[0]), 0, dtype=int)
        for i_a1, a1 in enumerate(nobsdf_unfil.columns.to_numpy()):
            a1_idx_array[i_a1, :] = df_wide['a_1'] == a1
        pot_idx_array = np.full((nobsdf_unfil.shape[0], df_wide.shape[0]), 0, dtype=int)
        for i_pot, pot in enumerate(nobsdf_unfil.index.to_numpy()):
            pot_idx_array[i_pot, :] = df_wide['a_1'] == pot

        for i_stimid, stimid in enumerate(nobsdf_unfil.columns.to_numpy()):
            sub_nobsdf = nobsdf_unfil.copy()
            sub_nobsdf.loc[:, :] = 0
            for i_a1, a1 in enumerate(nobsdf_unfil.columns.to_numpy()):
                for i_pot, pot in enumerate(nobsdf_unfil.index.to_numpy()):
                    sub_nobsdf.loc[pot, a1] = np.sum(a1_idx_array[i_a1, :] & pot_idx_array[i_pot, :])
            data_stats['nobs_by_stim'][stimid] = sub_nobsdf

    dataout = {'empiricalInverseJudgments_' + suffix: empiricalPreferenceJudgments}

    dataout['pots_' + suffix] = pots
    dataout['pots_byoutcome_' + suffix] = {'all': pots}
    nobsdf = empiricalPreferenceJudgments['nobs']
    for i_a1, a_1 in enumerate(a1_labels):
        dataout['pots_byoutcome_' + suffix][a_1] = nobsdf.index[nobsdf[a_1] > 0].to_numpy()

    dataout['empiricalInverseJudgmentsExtras_' + suffix] = rider_hack

    return dataout, data_stats
