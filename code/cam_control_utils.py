#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_control_utils.py
"""


class ControlObject():

    def __init__(self, environment, isInteractive=False):
        self.environment = environment
        self.isInteractive = isInteractive
        self.paths = dict()
        self.plot_param = dict()
        self.plot_control = dict()
        self.empir_load_param = dict()
        self.data_spec = dict()
        self.wppl_model_spec = dict()
        self.wpplparam = dict()
        self.cache = dict()
        self.seed = None

    def serialize(self):
        return self.__dict__


def merge_dicts(dict_base, dict_update, add_keys=False):
    """
    dict1 = {'one': 'one', 'two': 'two', 'three': 'three', 'four':{'five':'five', 'six':'six'}}
    dict2 = {'one': 1, 'two': 2, 'four':{'five':5, 'seven':7}, 'eight': 8}
    merge_dicts(dict1, dict2, False)
    print(dict1)
    print(dict2)
    """

    def _merge_dicts(d, dnew, add_keys_):
        """e.g.
        __keys_list__ = list()
        dict1 = {'one': 'one', 'two': 'two', 'three': 'three', 'four':{'five':'five', 'six':'six'}}
        dict2 = {'one': 1, 'two': 2, 'four':{'five':5, 'seven':7}, 'eight': 8}
        list(merge_dicts_(dict1, dict2, False))
        print(dict1)
        print(dict2)
        print(__keys_list__)

        __keys_list__ = list()
        dict1 = {'one': 'one', 'two': 'two', 'three': 'three', 'four':{'five':'five', 'six':'six'}}
        dict2 = {'one': 1, 'two': 2, 'four':{'five':5, 'seven':7}, 'eight': 8}
        list(merge_dicts_(dict1, dict2, True))
        print(dict1)
        print(dict2)
        print(__keys_list__)
        """
        for key, val in dnew.items():
            if isinstance(val, dict):
                if key in d:
                    yield from _merge_dicts(d[key], val, add_keys_)
                else:
                    __keys_list__.append((key, val))
                    if add_keys_:
                        d[key] = val
                        dnew[key] = 'used_by_merge_dicts'
            else:
                if key in d:
                    d[key] = val
                    dnew[key] = 'used_by_merge_dicts'
                else:
                    __keys_list__.append((key, val))
                    if add_keys_:
                        d[key] = val
                        dnew[key] = 'used_by_merge_dicts'

    def _find_unused(gen_obj, d, dnew):
        """e.g.
        dict1 = {'one': 'one', 'two': 'two', 'three': 'three', 'four':{'five':'five', 'six':'six'}}
        dict2 = {'one': 1, 'two': 2, 'four':{'five':5, 'seven':7}}
        res = find_unused(merge_dicts_(dict1, dict2), dict1, dict2)
        print(dict1)
        print(dict2)
        print(res)
        """

        def _find(dnew_):
            for key, val in dnew_.items():
                if isinstance(val, dict):
                    yield from _find(val)
                else:
                    if not val == 'used_by_merge_dicts':
                        yield from (key, val)

        _ = list(gen_obj)

        return list(_find(dnew))

    __keys_list__ = list()
    _ = list(_merge_dicts(dict_base, dict_update, add_keys))
    if __keys_list__:
        if add_keys:
            print('---ADDED:: ')
            for key, val in __keys_list__:
                print(f"{key} :: {val}")
        else:
            print('---KEYS NOT FOUND, UNUSED:: ')
            for key, val in __keys_list__:
                print(f"{key} :: {val}")

    return __keys_list__


def find_unmodified(dmod, dorigin):
    """
    dict1 = {'a': 1, 'b': 2, 'c': {'d': 1, 'e': 4}}
    dict2 = {'a': 1, 'b': 2, 'c': {'d': None, 'e': 4}}
    find_unmodified(dict1, dict2)
    """
    def _find(d1, d0):
        for key, val in d0.items():
            if isinstance(val, dict):
                if key in d1:
                    yield from _find(d1[key], val)
            else:
                if d1[key] == val and val == val:
                    _keys_list_.append((key, val))

    _keys_list_ = list()
    _ = list(_find(dmod, dorigin))

    return _keys_list_


def parse_param(update_dict_in=None):

    import os
    import sys
    from pathlib import Path
    from copy import deepcopy
    from pprint import pprint
    import warnings
    import time
    import numpy as np
    from cam_plot_utils import init_plot_objects

    # ## plotting
    if update_dict_in is None:
        update_dict_in = dict()
    update_dict_original = deepcopy(update_dict_in)
    update_dict = deepcopy(update_dict_in)

    long_names = update_dict.pop('long_names', False)

    strict = update_dict.pop('strict', 'strict')
    _ = update_dict_original.pop('strict')

    seed = update_dict.get('seed', None)
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
    update_dict['seed'] = 'used'

    isInteractive = update_dict.get('isInteractive', False)

    if str(Path.home()) == '/Users/dae':
        if str(Path().cwd().expanduser()) == '/Users/dae/mp/om_iaa_code':
            raise ValueError('Remote kernel failed to activate')
        environment = 'local'
        print(f'environment: {environment}')
    elif str(Path.home()) == '/home/daeda':
        environment = 'remotekernel'
        print(f'environment: {environment}; {len(os.sched_getaffinity(0))} CPU')
    else:
        environment = 'undetermined'
        print(f'environment: {environment}; {len(os.sched_getaffinity(0))} CPU; home: {Path.home()}')

    ###############

    control_parameters = ControlObject(environment, isInteractive)
    setattr(control_parameters, 'seed', seed)

    unused_list = list()
    added_list = list()

    #########
    ## cache
    #########

    runModel_ = True  # CACHE
    loadpickle_ = False  # CACHE
    hotpatch_precached_data_ = False  # CACHE
    removeOldData_ = True  # CACHE
    saveOnExec_ = True  # CACHE

    cache_ = {
        'webppl': {
            'runModel': runModel_,
            'loadpickle': loadpickle_,
            'hotpatch_precached_data': hotpatch_precached_data_,
            'removeOldData': removeOldData_,
            'saveOnExec': saveOnExec_,
        },
    }
    setattr(control_parameters, 'cache', cache_)
    ### update only the keys that overlap
    unused_ = merge_dicts(control_parameters.cache, update_dict.get('cache', dict()))
    if unused_:
        unused_list.extend(unused_)
        warnings.warn('UNUSED UPDATE ITEMS:')
        print(unused_)

    #############################################################

    #########
    ## paths
    #########

    expdir = update_dict.pop('expdir', None)
    _ = update_dict_original.pop('expdir')

    dataOutBase = update_dict.pop('dataOutBase', None)
    _ = update_dict_original.pop('dataOutBase')

    if expdir is None:
        if control_parameters.environment in ['cluster', 'remotekernel']:
            expdir = Path('/home/daeda/itegb_cam/')
        elif control_parameters.environment == 'local':
            expdir = Path('/Users/dae/coding/-GitRepos/itegb_computed-appraisals/')

    if dataOutBase is None:
        if control_parameters.environment in ['cluster', 'remotekernel']:
            dataOutBase = Path(f'/om2/user/daeda/iaa_dataout/')
        elif control_parameters.environment == 'local':
            dataOutBase = expdir / f'dataOut/'

    codedir = expdir / 'code'
    assert codedir.is_dir(), f"project code path does not exist: {codedir}"
    assert dataOutBase.parent.is_dir(), f"dataOut path does not exist: {dataOutBase.parent}"
    sys.path.append(str(codedir))

    #############################################################

    #########
    ## default values
    #########

    qc_exclusion = ['original', 'none', 'allquestions'][0]
    subjRespFilter = 'unfiltered'

    wppl_model_spec_ = {
        'wppl_model_path': None,
        'prospect_fn': 'log',
        'refpoint_type': 'Norm',
        'inf_param': 'MCMC3k',
        'generic_repu_values_from': ['internal', 'empiricalExpectation'][0],
        'distal_repu_values_from': ['internal', 'empiricalExpectation'][1],
        'prior_a2C_cmd': ['uniform', 0.4][1],
        'kde_width': 0.01,
        'prior_form': 'multivarkdemixture',
        'priors_only': ['full', 'priorOnly'][0],
        'cfa1_form': '',
    }

    setattr(control_parameters, 'wppl_model_spec', wppl_model_spec_)
    unused_ = merge_dicts(control_parameters.wppl_model_spec, update_dict.get('wppl_model_spec', dict()))
    if unused_:
        unused_list.extend(unused_)
        warnings.warn('UNUSED UPDATE ITEMS:')
        print(unused_)

    empir_load_param_ = {
        'exp10load': qc_exclusion,
        'exp11load': qc_exclusion,
        'subjRespFilter': subjRespFilter,
        'print_responses': False
    }
    setattr(control_parameters, 'empir_load_param', empir_load_param_)
    unused_ = merge_dicts(control_parameters.empir_load_param, update_dict.get('empir_load_param', dict()))
    if unused_:
        unused_list.extend(unused_)
        warnings.warn('UNUSED UPDATE ITEMS:')
        print(unused_)

    #########
    ## naming variables
    #########

    prospect_fn_ = {'log': '', 'powerfn': 'powerfn_', 'nopsscale': 'nopsscale_'}[control_parameters.wppl_model_spec['prospect_fn']]

    inf_param_ = control_parameters.wppl_model_spec['inf_param']

    repu_values_from_str_ = f"REPU{control_parameters.wppl_model_spec['generic_repu_values_from']}-{control_parameters.wppl_model_spec['distal_repu_values_from']}"

    if control_parameters.wppl_model_spec['prior_a2C_cmd'] == 'uniform':
        prior_a2C = 'uniform'
    elif control_parameters.wppl_model_spec['prior_a2C_cmd'] == 'split':
        prior_a2C = 'split'
    elif control_parameters.wppl_model_spec['prior_a2C_cmd'] == 'splitinv':
        prior_a2C = 'splitinv'
    else:
        prior_a2C = float(control_parameters.wppl_model_spec['prior_a2C_cmd'])

    control_parameters.wppl_model_spec['prior_a2C'] = prior_a2C

    if control_parameters.wppl_model_spec['prior_form'] == 'kde':
        prior_form_ = 'kde'
    elif control_parameters.wppl_model_spec['prior_form'] == 'kdemixture':
        prior_form_ = 'kde-mixture'
    elif control_parameters.wppl_model_spec['prior_form'] == 'broadkdemixture':
        prior_form_ = 'broadkde-mixture'
    elif control_parameters.wppl_model_spec['prior_form'] == 'multivarkdemixture':
        prior_form_ = 'multivarkde-mixture'
    elif control_parameters.wppl_model_spec['prior_form'] == 'multivarkdemixtureLogit':
        prior_form_ = 'multivarkdeLogit-mixture'
    else:
        prior_form_ = ''

    if control_parameters.wppl_model_spec['cfa1_form'] == '':
        cfa1_form_ = ''
    else:
        cfa1_form_ = f"_{control_parameters.wppl_model_spec['cfa1_form']}"

    #########
    ## define names
    #########

    modelstr_list = ['agent-model']
    if prospect_fn_ != '':
        modelstr_list.append(prospect_fn_)
    if cfa1_form_ != '':
        modelstr_list.append(cfa1_form_)
    if prior_form_ != '':
        modelstr_list.append(prior_form_)
    if control_parameters.wppl_model_spec['priors_only'] == 'priorOnly':
        modelstr_list.append("priors")

    datastr_list = ['wppldata']
    datastr_list.append(inf_param_)
    if long_names:
        if prospect_fn_ != '':
            datastr_list.append(prospect_fn_)
        if cfa1_form_ != '':
            datastr_list.append(cfa1_form_)
        if prior_form_ != '':
            datastr_list.append(prior_form_)
        datastr_list.append(repu_values_from_str_)
        datastr_list.append(f"psref{control_parameters.wppl_model_spec['refpoint_type']}")
        datastr_list.append(f"priora1{str(control_parameters.wppl_model_spec['prior_a2C']).replace('0.','')}")
    if control_parameters.wppl_model_spec['priors_only'] == 'priorOnly':
        datastr_list.append('priorOnly')

    if control_parameters.wppl_model_spec['wppl_model_path'] is None:
        webppl_model_name_ = f"{'_'.join(modelstr_list)}.wppl"
    else:
        webppl_model_name_ = control_parameters.wppl_model_spec['wppl_model_path'].name

    datadir_name_ = '_'.join(datastr_list)

    #########
    ## paths
    #########

    exp_name_ = update_dict.get('exp_name', 'iaa_dataout')
    if 'exp_name' in update_dict:
        update_dict['exp_name'] = None

    paths = {
        'expDir': expdir,
        'code': expdir / 'code',
        'dataOutBase': dataOutBase / exp_name_,
        'dataOut': dataOutBase / exp_name_ / datadir_name_,
        'wpplDataCache': dataOutBase / exp_name_ / datadir_name_ / "webppl_cache.pkl"
    }

    if control_parameters.wppl_model_spec['wppl_model_path'] is None:
        paths['model'] = paths['code'] / webppl_model_name_
    else:
        paths['model'] = control_parameters.wppl_model_spec['wppl_model_path']

    paths['figsOut'] = paths['dataOut'] / 'figs'
    paths['byPot'] = paths['figsOut'] / 'byPotsize'

    # output directories
    paths['figsPub'] = paths['figsOut'] / 'pub'
    paths['txtPub'] = paths['figsOut'] / 'pub' / 'textvars'
    # input directories
    datain_dir = expdir / 'dataIn'
    paths['stimuli'] = datain_dir / 'statics'

    # empir data
    paths['exp6xlsx'] = datain_dir / 'exp6' / 'trial_data_composite.csv'
    paths['subjectrackerexp6'] = datain_dir / 'exp6' / 'participant_data_composite.csv'
    paths['exp7xlsx'] = datain_dir / 'exp7' / 'trial_data_composite.csv'
    paths['subjectrackerexp7'] = datain_dir / 'exp7' / 'participant_data_composite.csv'
    paths['exp9xlsx'] = datain_dir / 'exp9' / 'trial_data_composite.csv'
    paths['subjectrackerexp9'] = datain_dir / 'exp9' / 'participant_data_composite.csv'
    paths['exp10xlsx'] = datain_dir / 'exp10' / 'trial_data_composite.csv'
    paths['subjectrackerexp10'] = datain_dir / 'exp10' / 'participant_data_composite.csv'
    paths['exp11xlsx'] = datain_dir / 'exp11' / 'trial_data_composite.csv'
    paths['subjectrackerexp11'] = datain_dir / 'exp11' / 'participant_data_composite.csv'

    setattr(control_parameters, 'paths', paths)

    plot_param_ = init_plot_objects(paths['dataOut'], isInteractive=isInteractive)

    plot_control_ = {
        'set1aba': {
            'printFigsByPotsize': False,  # broken for exp3
            'printJointPosteriorHeatmaps': False,
        }
    }

    setattr(control_parameters, 'plot_param', plot_param_)
    unused_ = merge_dicts(control_parameters.plot_param, update_dict.get('plot_param', dict()))
    if unused_:
        unused_list.extend(unused_)
        warnings.warn('UNUSED UPDATE ITEMS:')
        print(unused_)

    setattr(control_parameters, 'plot_control', plot_control_)
    unused_ = merge_dicts(control_parameters.plot_control, update_dict.get('plot_control', dict()))
    if unused_:
        unused_list.extend(unused_)
        warnings.warn('UNUSED UPDATE ITEMS:')
        print(unused_)

    print(f"\n---Operating in {control_parameters.paths['dataOutBase']}\n")
    print('env:\t{}\nexp:\t{} \ncode:\t{} \nrunModel:\t{} \nremoveOldData: {}'.format(control_parameters.environment, str(control_parameters.paths['expDir']), str(control_parameters.paths['code']), control_parameters.cache['webppl']['runModel'], control_parameters.cache['webppl']['removeOldData']))

    emoLabelsOrdered = ["Devastation", "Disappointment", "Contempt", "Disgust", "Envy", "Fury", "Annoyance", "Embarrassment", "Regret", "Guilt", "Confusion", "Surprise", "Sympathy", "Amusement", "Relief", "Respect", "Gratitude", "Pride", "Excitement", "Joy"]
    outcomes = ['CC', 'CD', 'DC', 'DD']
    data_spec = dict()

    #### Exp 10

    from cam_import_empirical_data import print_rand_conds, print_emp_responses_emotions
    from cam_import_empirical_data import import_responses_exp10, import_participants_exp10_exp11, calc_filter_criteria_exp10, package_empdata_exp10
    data_spec_emp_exp10 = {
        'label': 'exp10',
        'description': '',
        'drop_practice_trial': True,
        'filter_fn': {  # acceptable values evaluate to True. Criteria not included here are ignored
            'Data_Set': lambda x: x >= 10 and x < 11,
            'Excluded': lambda x: not x,
            'val0(7510)': lambda x: x == 'correct_response',
            'val1(disdain)': lambda x: x == 'correct_response',
            'val2(jealousy)': lambda x: x == 'correct_response',
            'val3(AF25HAS)': lambda x: x == 'correct_response',
            'val4(steal)': lambda x: x == 'correct_response',
            # 'val5(pia2_D_a2_C)': lambda x: x == 'correct_response',
            # 'val6(pia2_D_a2_C)': lambda x: x == 'correct_response',
        },
        'pots': None,
        'outcomes': outcomes,
        'emoLabels': emoLabelsOrdered,
        'ncond': None,
        'print_responses': False,
        ### import_fn_dict
        'import_responses_fn': import_responses_exp10,
        'import_participants_fn': import_participants_exp10_exp11,
        'calc_filter_criteria_fn': calc_filter_criteria_exp10,
        'package_fn': package_empdata_exp10,
        'followup_fn': [print_rand_conds, print_emp_responses_emotions],
    }

    added_ = merge_dicts(data_spec_emp_exp10, update_dict.get('data_spec_emp_exp10', dict()), add_keys=True)
    if added_:
        added_list.extend(added_)
        warnings.warn('ADDED UPDATE ITEMS:')
        print(added_)

    data_spec['exp10'] = {'data_load_param': dict(), 'import_fn_dict': dict()}
    for key in ['import_responses_fn', 'import_participants_fn', 'calc_filter_criteria_fn', 'package_fn', 'followup_fn']:
        data_spec['exp10']['import_fn_dict'][key] = data_spec_emp_exp10.pop(key, None)
    data_spec['exp10']['data_load_param'] = data_spec_emp_exp10

    #### Exp 11

    from cam_import_empirical_data import import_responses_exp11, import_participants_exp10_exp11, calc_filter_criteria_exp11, package_empdata_exp11
    data_spec_emp_exp11 = {
        'label': 'exp11',
        'description': '',
        'drop_practice_trial': True,
        'filter_fn': {  # acceptable values evaluate to True. Criteria not included here are ignored
            'Data_Set': lambda x: x >= 11 and x < 12,
            'Excluded': lambda x: not x,
            'val0(7510)': lambda x: x == 'correct_response',
            'val1(disdain)': lambda x: x == 'correct_response',
            'val2(jealousy)': lambda x: x == 'correct_response',
            'val3(AF25HAS)': lambda x: x == 'correct_response',
            'val4(steal)': lambda x: x == 'correct_response',
            # 'val5(pia2_D_a2_C)': lambda x: x == 'correct_response',
            # 'val6(pia2_D_a2_C)': lambda x: x == 'correct_response',
        },
        'pots': None,
        'outcomes': outcomes,
        'emoLabels': emoLabelsOrdered,
        'ncond': 32,
        'print_responses': False,
        ### import_fn_dict
        'import_responses_fn': import_responses_exp11,
        'import_participants_fn': import_participants_exp10_exp11,
        'calc_filter_criteria_fn': calc_filter_criteria_exp11,
        'package_fn': package_empdata_exp11,
        'followup_fn': [print_rand_conds, print_emp_responses_emotions],
    }

    added_ = merge_dicts(data_spec_emp_exp11, update_dict.get('data_spec_emp_exp11', dict()), add_keys=True)
    if added_:
        added_list.extend(added_)
        warnings.warn('ADDED UPDATE ITEMS:')
        print(added_)

    data_spec['exp11'] = {'data_load_param': dict(), 'import_fn_dict': dict()}
    for key in ['import_responses_fn', 'import_participants_fn', 'calc_filter_criteria_fn', 'package_fn', 'followup_fn']:
        data_spec['exp11']['import_fn_dict'][key] = data_spec_emp_exp11.pop(key, None)
    data_spec['exp11']['data_load_param'] = data_spec_emp_exp11

    #### Exp 7

    from cam_import_empirical_data import import_responses_exp7, import_participants_exp7, calc_filter_criteria_exp7, package_empdata_exp7
    data_spec_emp_exp7 = {
        'label': 'exp7',
        'description': '',
        'filter_fn': {  # acceptable values evaluate to True. Criteria not included here are ignored
            'Data_Set': lambda x: x >= 7 and x < 8,
            'Excluded': lambda x: not x,
            'subjectValidation1': lambda x: x,
        },
        'pots': None,
        'outcomes': outcomes,
        'emoLabels': emoLabelsOrdered,
        'ncond': 132,
        'print_responses': False,
        ### import_fn_dict
        'import_responses_fn': import_responses_exp7,
        'import_participants_fn': import_participants_exp7,
        'calc_filter_criteria_fn': calc_filter_criteria_exp7,
        'package_fn': package_empdata_exp7,
        'followup_fn': [print_rand_conds, print_emp_responses_emotions],
    }

    added_ = merge_dicts(data_spec_emp_exp7, update_dict.get('data_spec_emp_exp7', dict()), add_keys=True)
    if added_:
        added_list.extend(added_)
        warnings.warn('ADDED UPDATE ITEMS:')
        print(added_)

    data_spec['exp7'] = {'data_load_param': dict(), 'import_fn_dict': dict()}
    for key in ['import_responses_fn', 'import_participants_fn', 'calc_filter_criteria_fn', 'package_fn', 'followup_fn']:
        data_spec['exp7']['import_fn_dict'][key] = data_spec_emp_exp7.pop(key, None)
    data_spec['exp7']['data_load_param'] = data_spec_emp_exp7

    #### Exp 6

    from cam_import_empirical_data import import_responses_InversePlanning_Base_widedf_exp6, import_participants_InversePlanning_Base_widedf_exp6, calc_filter_criteria_InversePlanning_Base_widedf_exp6, package_empdata_InversePlanning_Base_widedf_exp6
    data_spec_emp_exp6 = {
        'label': 'exp6',
        'description': '',
        'filter_fn': {  # acceptable values evaluate to True. Criteria not included here are ignored
            'Data_Set': lambda x: x >= 6.1 and x < 7,
            'Excluded': lambda x: not x,
            'subjectValidation1': lambda x: x,
        },
        'a1_labels': ['C', 'D'],
        'pots': None,
        'ncond': None,
        'print_responses': False,
        ### import_fn_dict
        'import_responses_fn': import_responses_InversePlanning_Base_widedf_exp6,
        'import_participants_fn': import_participants_InversePlanning_Base_widedf_exp6,
        'calc_filter_criteria_fn': calc_filter_criteria_InversePlanning_Base_widedf_exp6,
        'package_fn': package_empdata_InversePlanning_Base_widedf_exp6,
        'followup_fn': [print_rand_conds],
    }

    added_ = merge_dicts(data_spec_emp_exp6, update_dict.get('data_spec_emp_exp6', dict()), add_keys=True)
    if added_:
        added_list.extend(added_)
        warnings.warn('ADDED UPDATE ITEMS:')
        print(added_)

    data_spec['exp6'] = {'data_load_param': dict(), 'import_fn_dict': dict()}
    for key in ['import_responses_fn', 'import_participants_fn', 'calc_filter_criteria_fn', 'package_fn', 'followup_fn']:
        data_spec['exp6']['import_fn_dict'][key] = data_spec_emp_exp6.pop(key, None)
    data_spec['exp6']['data_load_param'] = data_spec_emp_exp6

    #### Exp 9

    from cam_import_empirical_data import import_responses_InversePlanning_Repu_widedf_exp9, import_participants_InversePlanning_Repu_widedf_exp9, calc_filter_criteria_InversePlanning_Repu_widedf_exp9, package_empdata_InversePlanning_Repu_widedf_exp9
    def valdataset(x): return (x >= 9 or x < 10)
    def valexcluded(x): return not x
    def val(x): return x == 'correct_response'
    def val3(x): return (x == 'correct_response' or x == '-1' or x == -1)
    data_spec_emp_exp9 = {
        'label': 'exp9',
        'description': '',
        'drop_practice_trial': True,
        'filter_fn': {  # acceptable values evaluate to True. Criteria not included here are ignored
            'Data_Set': valdataset,
            'Excluded': valexcluded,
            # 'subjectValidation1': lambda x: x,
            'val0(7510)': val,
            'val1(disdainful)': val,
            'val2(split)': val,
            'val3(three/rAIA)': val3,
            'val4(AF25HAS)': val,
        },
        'a1_labels': ['C', 'D'],
        'pots': None,
        'ncond': None,
        'print_responses': False,
        ### import_fn_dict
        'import_responses_fn': import_responses_InversePlanning_Repu_widedf_exp9,
        'import_participants_fn': import_participants_InversePlanning_Repu_widedf_exp9,
        'calc_filter_criteria_fn': calc_filter_criteria_InversePlanning_Repu_widedf_exp9,
        'package_fn': package_empdata_InversePlanning_Repu_widedf_exp9,
        'followup_fn': [print_rand_conds],
    }

    added_ = merge_dicts(data_spec_emp_exp9, update_dict.get('data_spec_emp_exp9', dict()), add_keys=True)
    if added_:
        added_list.extend(added_)
        warnings.warn('ADDED UPDATE ITEMS:')
        print(added_)

    data_spec['exp9'] = {'data_load_param': dict(), 'import_fn_dict': dict()}
    for key in ['import_responses_fn', 'import_participants_fn', 'calc_filter_criteria_fn', 'package_fn', 'followup_fn']:
        data_spec['exp9']['import_fn_dict'][key] = data_spec_emp_exp9.pop(key, None)
    data_spec['exp9']['data_load_param'] = data_spec_emp_exp9

    setattr(control_parameters, 'data_spec', data_spec)

    if added_list:
        print('---added_list:')
        pprint(added_list)

    if unused_list:
        print('---unused_list:')
        pprint(unused_list)

    diff_ = find_unmodified(update_dict, update_dict_original)
    if diff_:
        print('-------Unused updates:')
        pprint(diff_)

    if strict == 'strict':
        assert not unused_list
        assert not diff_, diff_
        assert not added_list
    elif strict == 'allow_add':
        assert not unused_list
        assert not diff_, diff_
        print('WARNING: Bypassing strict mode')
    else:
        if not unused_list or not diff_:
            print('--update dict--')
            pprint(update_dict)
            print('WARNING: BAD KEY __AND__ Bypassing strict mode')

    return control_parameters
