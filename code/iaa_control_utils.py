

class control_object():

    def __init__(self, environment, isInteractive=False):
        self.environment = environment
        self.isInteractive = isInteractive
        # self.control_dict = dict()
        self.paths = dict()
        self.plot_param = dict()
        self.plot_control = dict()
        self.empir_load_param = dict()
        self.data_spec = dict()
        self.wppl_model_spec = dict()
        self.wpplparam = dict()
        self.pytorch_spec = dict()
        # self.stan_spec = dict()
        self.cache = dict()
        # self.chosen_iaf_models = dict()
        # self.full_models = dict()
        # self.modelcomparison_param = dict()
        self.seed = None

    # def update(self, update_dict):
    #     self.control_dict.update(update_dict)

    # def set_(self, key, val):
    #     self.control_dict[key] = val

    # def get(self, key):
    #     return self.control_dict[key]

    # def gen_dict(self, keys_list):
    #     return {k: self.control_dict[k] for k in keys_list}

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

    def merge_dicts_(d, dnew, add_keys_):
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
                    yield from merge_dicts_(d[key], val, add_keys_)
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

    def find_unused(gen_obj, d, dnew):
        """e.g.
        dict1 = {'one': 'one', 'two': 'two', 'three': 'three', 'four':{'five':'five', 'six':'six'}}
        dict2 = {'one': 1, 'two': 2, 'four':{'five':5, 'seven':7}}
        aaa = find_unused(merge_dicts_(dict1, dict2), dict1, dict2)
        print(dict1)
        print(dict2)
        print(aaa)
        """

        def find_(dnew_):
            for key, val in dnew_.items():
                if isinstance(val, dict):
                    yield from find_(val)
                else:
                    if not val == 'used_by_merge_dicts':
                        yield from (key, val)

        _ = list(gen_obj)

        return list(find_(dnew))

    __keys_list__ = list()
    _ = list(merge_dicts_(dict_base, dict_update, add_keys))
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
    qqq = {'a': 1, 'b': 2, 'c': {'d': 1, 'e': 4}}
    qqq2 = {'a': 1, 'b': 2, 'c': {'d': None, 'e': 4}}
    find_unmodified(qqq, qqq2)
    """
    def find_(d1_, d0):
        for key, val in d0.items():
            if isinstance(val, dict):
                if key in d1_:
                    yield from find_(d1_[key], val)
            else:
                if d1_[key] == val and val == val:
                    _keys_list_.append((key, val))

    _keys_list_ = list()
    _ = list(find_(dmod, dorigin))

    return _keys_list_


# %%


def parse_param(update_dict_in=None):

    import os
    import sys
    from pathlib import Path
    from copy import deepcopy
    from pprint import pprint
    import warnings
    import time
    import numpy as np
    from webpypl_plotfun import init_plot_objects

    # ## plotting
    if update_dict_in is None:
        update_dict_in = dict()
    update_dict_original = deepcopy(update_dict_in)
    update_dict = deepcopy(update_dict_in)

    strict = update_dict.pop('strict', 'strict')
    _ = update_dict_original.pop('strict')

    seed = update_dict.get('seed', None)
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
    update_dict['seed'] = 'used'

    isInteractive = update_dict.get('isInteractive', False)

    if str(Path.home()) == '/Users/dae':
        environment = 'local'
        print(f'environment: {environment}')
    elif str(Path.home()) == '/home/daeda':
        environment = 'remotekernel'
        print(f'environment: {environment}; {len(os.sched_getaffinity(0))} CPU')
    else:
        environment = 'undetermined'
        print(f'environment: {environment}; {len(os.sched_getaffinity(0))} CPU; home: {Path.home()}')

    ###############

    control_parameters = control_object(environment, isInteractive)
    setattr(control_parameters, 'seed', seed)

    unused_list = list()
    added_list = list()

    #########
    ## cache
    #########

    runModel_ = True  # CACHE
    loadpickle_ = False  # CACHE
    hotwire_precached_data_ = False  # CACHE
    removeOldData_ = True  # CACHE
    saveOnExec_ = True  # CACHE

    cache_ = {
        'webppl': {
            'runModel': runModel_,
            'loadpickle': loadpickle_,
            'hotwire_precached_data': hotwire_precached_data_,
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

    #############################

    #########
    ## paths
    #########

    expdir = update_dict.pop('expdir', None)
    _ = update_dict_original.pop('expdir')

    dataOutBase = update_dict.pop('dataOutBase', None)
    _ = update_dict_original.pop('dataOutBase')

    if expdir is None:
        if control_parameters.environment in ['cluster', 'remotekernel']:
            expdir = Path('/om/user/daeda/ite_iaa/ite_gb_inverseappraisal/')
        elif control_parameters.environment == 'local':
            expdir = Path('/Users/dae/coding/-GitRepos/itegb_inferredappraisals/')

    if dataOutBase is None:
        if control_parameters.environment in ['cluster', 'remotekernel']:
            dataOutBase = Path(f'/om2/user/daeda/iaa_dataout/')
        elif control_parameters.environment == 'local':
            dataOutBase = expdir / f'dataOut/'

    codedir = expdir / 'code'
    assert codedir.is_dir()
    sys.path.append(str(codedir))

    #############################################################

    #########
    ## default values
    #########

    exp10exclusion = ['original', 'none', 'allquestions'][0]
    subjRespFilter = 'unfiltered'

    print_responses = False
    exp11exclusion = exp10exclusion

    wppl_model_spec_ = {
        'wppl_model_path': None,
        'prospect_fn': ['log', 'powerfn', 'nopsscale'][0],
        'refpoint_type': ['Norm', 'Gamma', 'None', 'Norm350'][0],
        'inf_param': ['rejection', 'MCMC1', 'incrementalMH1', 'rapidtest'][1],
        'repu_values_from': ['internal', 'empiricalKDE', 'empiricalExpectation', 'binary'][2],
        'prior_a2C_cmd': ['uniform', 0.6][1],
        'kde_width': 0.02,
        # 'affine_b': None,
        # 'logit_k': None,
        'prior_form': ['kde', 'type1', 'agg69'][0],
        'priors_only': ['full', 'priorOnly'][0],
        'cfa2_form': ['normedCFa1', 'unnoredCFa1'][0],
        ### passed to webppl
        ### used for dataprep / model choice
        ### Not currently functioning
        'pe_a2_valanced': 'valanced',  # Not currently functioning
        'pe_a2_scaling_included_in_wpplexport': ['Includeall'][0],  # Not currently functioning
    }

    setattr(control_parameters, 'wppl_model_spec', wppl_model_spec_)
    unused_ = merge_dicts(control_parameters.wppl_model_spec, update_dict.get('wppl_model_spec', dict()))
    if unused_:
        unused_list.extend(unused_)
        warnings.warn('UNUSED UPDATE ITEMS:')
        print(unused_)

    empir_load_param_ = {
        # 'pe_a2_scaled': ['none', 'all', 'lnpot', 'raw', 'unvalanced'][3],
        'exp10load': exp10exclusion,
        'exp11load': exp11exclusion,
        'subjRespFilter': subjRespFilter,
        'print_responses': print_responses
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

    inf_param_ = f"{control_parameters.wppl_model_spec['inf_param']}_"

    repu_values_from_ = f"_REPU{control_parameters.wppl_model_spec['repu_values_from']}"
    repu_values_from_key_ = {'internal': '', 'empiricalKDE': '_empRep', 'empiricalExpectation': '_EVempRep', 'binary': '_EVempRep'}
    repu_values_from_model_ = repu_values_from_key_[control_parameters.wppl_model_spec['repu_values_from']]

    print(control_parameters.wppl_model_spec['prior_a2C_cmd'])
    print(control_parameters.wppl_model_spec['prior_form'])
    if control_parameters.wppl_model_spec['prior_a2C_cmd'] == 'uniform':
        prior_a2C = 'uniform'
    elif control_parameters.wppl_model_spec['prior_a2C_cmd'] == 'split':
        prior_a2C = 'split'
    elif control_parameters.wppl_model_spec['prior_a2C_cmd'] == 'splitinv':
        prior_a2C = 'splitinv'
    else:
        prior_a2C = float(control_parameters.wppl_model_spec['prior_a2C_cmd'])
        # control_parameters.wppl_model_spec['prior_form'] = control_parameters.wppl_model_spec['prior_form']+'mixture'
    control_parameters.wppl_model_spec['prior_a2C'] = prior_a2C

    print(control_parameters.wppl_model_spec['prior_a2C_cmd'])
    print(control_parameters.wppl_model_spec['prior_form'])

    if control_parameters.wppl_model_spec['priors_only'] == 'priorOnly':
        priors_only_ = '_priors'
    else:
        priors_only_ = ''

    if control_parameters.wppl_model_spec['prior_form'] == 'kde':
        prior_form_ = '_kde'
    elif control_parameters.wppl_model_spec['prior_form'] == 'kdemixture':
        prior_form_ = '_kde_mixture'
    elif control_parameters.wppl_model_spec['prior_form'] == 'broadkdemixture':
        prior_form_ = '_broadkde_mixture'
    elif control_parameters.wppl_model_spec['prior_form'] == 'multivarkdemixture':
        prior_form_ = '_multivarkde_mixture'
    elif control_parameters.wppl_model_spec['prior_form'] == 'multivarkdemixtureLogit':
        prior_form_ = '_multivarkdeLogit_mixture'
    else:
        prior_form_ = ''

    cfa2_form_ = control_parameters.wppl_model_spec['cfa2_form']

    print(control_parameters.wppl_model_spec['prior_form'])
    # if control_parameters.wppl_model_spec['repu_values_from'] != 'internal':
    #     assert control_parameters.wppl_model_spec['prior_form'] == 'kdemixture'
    print(f"prior_form :: {prior_form_}")

    #########
    ## define names
    #########

    if control_parameters.wppl_model_spec['wppl_model_path'] is None:
        webppl_model_name_ = f"agentmodel_{prospect_fn_}{cfa2_form_}{prior_form_}{repu_values_from_model_}{priors_only_}.wppl"
    else:
        webppl_model_name_ = control_parameters.wppl_model_spec['wppl_model_path'].name
    datadir_name_ = f"data_{prospect_fn_}{inf_param_}{cfa2_form_}{prior_form_}{repu_values_from_}_psref{control_parameters.wppl_model_spec['refpoint_type']}_priora1{str(control_parameters.wppl_model_spec['prior_a2C']).replace('0.','')}{priors_only_}"

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
        'wpplDataCache': dataOutBase / exp_name_ / datadir_name_ / f"cacheddatain{exp10exclusion}{subjRespFilter}.pkl",
    }
    # paths['model'] = paths['code'] / exp_name_ / webppl_model_name_
    if control_parameters.wppl_model_spec['wppl_model_path'] is None:
        paths['model'] = paths['code'] / webppl_model_name_
    else:
        paths['model'] = control_parameters.wppl_model_spec['wppl_model_path']

    """
    runFromCompiled = False # if you want to run the model from an already compiled webppl script
    if runFromCompiled:
        paths['dataOut'] = Path(compiledPath).resolve().parent
        paths['executable'] = Path(compiledPath)
    ### If a different set of data are to be imported
    # paths["dataOut"] = Path('')
    if runFromCompiled:
        removeOldData, saveOnExec = False, False
        compiledPath = 'set to path of compiled webppl script'
    """

    paths['figsOut'] = paths['dataOut'] / 'figs'
    paths['byPot'] = paths['figsOut'] / 'byPotsize'

    # output directories
    paths['figsPub'] = paths['figsOut'] / 'pub'
    paths['txtPub'] = paths['figsOut'] / 'pub' / 'textvars'
    # input directories
    paths['stimuli'] = expdir / 'datain' / 'statics'
    paths['exp3xlsx'] = expdir / 'datain' / 'exp3' / 'trial_data_composite.csv'
    paths['subjectrackerexp3'] = expdir / 'datain' / 'exp3' / 'participant_data_composite.csv'
    paths['exp6xlsx'] = expdir / 'datain' / 'exp6' / 'trial_data_composite.csv'
    paths['subjectrackerexp6'] = expdir / 'datain' / 'exp6' / 'participant_data_composite.csv'
    paths['exp7xlsx'] = expdir / 'datain' / 'exp7' / 'trial_data_composite.csv'
    paths['subjectrackerexp7'] = expdir / 'datain' / 'exp7' / 'participant_data_composite.csv'
    paths['exp9xlsx'] = expdir / 'datain' / 'exp9' / 'trial_data_composite.csv'
    paths['subjectrackerexp9'] = expdir / 'datain' / 'exp9' / 'participant_data_composite.csv'
    paths['exp10xlsx'] = expdir / 'datain' / 'exp10' / 'trial_data_composite.csv'
    paths['subjectrackerexp10'] = expdir / 'datain' / 'exp10' / 'participant_data_composite.csv'
    paths['exp11xlsx'] = expdir / 'datain' / 'exp11' / 'trial_data_composite.csv'
    paths['subjectrackerexp11'] = expdir / 'datain' / 'exp11' / 'participant_data_composite.csv'

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

    print('Cache Settings::')
    pprint(control_parameters.cache)

    ########

    # setattr(control_parameters, 'chosen_iaf_models', dict())
    # added_ = merge_dicts(control_parameters.chosen_iaf_models, update_dict.get('chosen_iaf_models', dict()), add_keys=True)
    # if added_:
    #     added_list.extend(added_)
    #     warnings.warn('ADDED UPDATE ITEMS:')
    #     print(added_)

    # setattr(control_parameters, 'full_models', dict())
    # added_ = merge_dicts(control_parameters.full_models, update_dict.get('full_models', dict()), add_keys=True)
    # if added_:
    #     added_list.extend(added_)
    #     warnings.warn('ADDED UPDATE ITEMS:')
    #     print(added_)

    # setattr(control_parameters, 'modelcomparison_param', dict())
    # added_ = merge_dicts(control_parameters.modelcomparison_param, update_dict.get('modelcomparison_param', dict()), add_keys=True)
    # if added_:
    #     added_list.extend(added_)
    #     warnings.warn('ADDED UPDATE ITEMS:')
    #     print(added_)

    #######

    emoLabelsOrdered = ["Devastation", "Disappointment", "Contempt", "Disgust", "Envy", "Fury", "Annoyance", "Embarrassment", "Regret", "Guilt", "Confusion", "Surprise", "Sympathy", "Amusement", "Relief", "Respect", "Gratitude", "Pride", "Excitement", "Joy"]
    outcomes = ['CC', 'CD', 'DC', 'DD']
    data_spec = dict()

    #### Exp 10

    from iaa_import_empirical_data_wrapper import print_rand_conds, print_emp_responses_emotions
    from iaa_import_empirical_data_wrapper import import_responses_exp10, import_participants_exp10_exp11, calc_filter_criteria_exp10, package_empdata_exp10
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

    from iaa_import_empirical_data_wrapper import import_responses_exp11, import_participants_exp10_exp11, calc_filter_criteria_exp11, package_empdata_exp11
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

    from iaa_import_empirical_data_wrapper import import_responses_exp7, import_participants_exp7, calc_filter_criteria_exp7, package_empdata_exp7
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

    from iaa_import_empirical_data_wrapper import import_responses_InversePlanning_Base_widedf_exp6, import_participants_InversePlanning_Base_widedf_exp6, calc_filter_criteria_InversePlanning_Base_widedf_exp6, package_empdata_InversePlanning_Base_widedf_exp6
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

    from iaa_import_empirical_data_wrapper import import_responses_InversePlanning_Repu_widedf_exp9, import_participants_InversePlanning_Repu_widedf_exp9, calc_filter_criteria_InversePlanning_Repu_widedf_exp9, package_empdata_InversePlanning_Repu_widedf_exp9
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

    # print('update_dict_original')
    # print(update_dict_original['seed'])
    # print('update_dict')
    # print(update_dict['seed'])
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


# %%
