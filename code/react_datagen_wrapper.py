#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""react_datagen_wrapper.py
"""


def sbatch_webppl_datagen(cpar):
    import numpy as np
    from pathlib import Path
    import dill
    import subprocess
    import re

    # from iaa_gendata_webppl import gendata_webppl
    # gendata_webppl(update_dict)

    ### runs gendata_webppl() from iaa_gendata_webppl.py
    script_path = cpar.paths['code'] / "launch_iaawrapper_runwebppl.sbatch"

    pickle_dir = cpar.paths['dataOutBase'] / "temp_pickles"

    pickle_dir.mkdir(exist_ok=True, parents=True)

    file_unique = False
    while not file_unique:
        pickle_path = pickle_dir / f'pickle_for_webppl_{np.random.randint(0,10000)}.pkl'
        if not pickle_path.is_file():
            file_unique = True

    with open(pickle_path, 'wb') as f:
        dill.dump(cpar, f, protocol=-4)

    clout = subprocess.run(["sbatch", str(script_path), str(pickle_path)], capture_output=True)

    print(clout)

    depend = re.search(r'([0-9]+)', clout.stdout.decode("utf-8").strip()).group(0)

    return depend


def wrapper_gen_webppl_data(spec=None, hotpatch_cacheddata=False, projectbase_path=None, dataout_path=None):
    """
    """

    # %%

    from copy import deepcopy
    import warnings
    from pathlib import Path
    import numpy as np
    from iaa_control_utils import parse_param, merge_dicts
    from iaa_gendata_webppl import gendata_webppl

    constitutively_run_webppl = False
    if spec is None:
        spec = dict()

    prospect_fn = spec.get('prospect_fn', ['log', 'powerfn', 'nopsscale'][0])
    inf_param = spec.get('inf_param', ['rejection', 'MCMC1', 'incrementalMH1', 'meddebug', 'rapidtest'][1])
    cfa2_form = spec.get('cfa2_form', ['normedCFa1', 'unnoredCFa1'][0])
    prior_form = spec.get('prior_form', ['kde', 'kdemixture', 'broadkdemixture', 'multivarkdemixture', 'multivarkdemixtureLogit', 'type1', 'agg69'][3])
    prior_a2C_cmd = spec.get('prior_a2C_cmd', ['uniform', 'split', 'splitinv', 0.5, 0.4, 0.6, 0.1, 0.9, 0.45, 0.55, 0.35, 0.65, 0.01, 0.99, 1.0, 0.0][4])
    kde_width = spec.get('kde_width', [0, 0.01, 0.02, 0.04, 0.1][1])
    repu_values_from = spec.get('repu_values_from', ['internal', 'empiricalKDE', 'empiricalExpectation', 'binary'][2])
    refpoint_type = spec.get('refpoint_type', ['Power', 'Norm', 'Gamma', 'None', 'Norm350'][1])
    priors_only = spec.get('priors_only', ['full', 'priorOnly'][0])
    seed = spec.get('seed', None)

    prefix = spec.get('prefix', 'iaa')

    ### NOTE specifying model names overrides webppl spec above
    # webppl_model_name_ = f"agentmodel_{prospect_fn_}{cfa2_form_}{prior_form_}{repu_values_from_model_}{priors_only_}.wppl"
    # wpplmodelname = "agentmodel_normedCFa1_multivarkde_mixture_EVempRep"
    # if priors_only:
    #     wpplmodelname += "_priors"
    # wpplmodelpath = Path('.../ite_gb_inverseappraisal/code/....wppl')

    if prior_a2C_cmd == 'split':
        prior_a2C_cmd_name = 100
    elif prior_a2C_cmd == 'splitinv':
        prior_a2C_cmd_name = 200
    else:
        prior_a2C_cmd_name = prior_a2C_cmd

    exp_name = f'{prefix}_invpT-{prospect_fn}_{repu_values_from}_psref{refpoint_type}_{prior_form}_kde-{kde_width:0.3f}_mix-{prior_a2C_cmd_name*100:0.0f}'

    exp10exclusion = ['original', 'none', 'allquestions'][0]
    subjRespFilter = 'unfiltered'

    update_dict_shared = {
        'strict': 'allow_add',  # TEMP
        'expdir': projectbase_path,
        'dataOutBase': dataout_path,
        'exp_name': exp_name,
        'seed': None,
        'cache': {
            'webppl': {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotwire_precached_data': False,  # CACHE
            },
        },
        'empir_load_param': {
            'exp10load': exp10exclusion,
            'exp11load': exp10exclusion,
            'subjRespFilter': subjRespFilter,
        },
        'data_spec_emp_exp9': {
            'print_responses': False,
        },
        'data_spec_emp_exp11': {
            'print_responses': False,
        },
        'data_spec_emp_exp7': {
            'print_responses': False,
        },
        'wppl_model_spec': {
            'prospect_fn': prospect_fn,
            'inf_param': inf_param,
            'repu_values_from': repu_values_from,
            'cfa2_form': cfa2_form,
            'prior_form': prior_form,
            'kde_width': kde_width,
            'refpoint_type': refpoint_type,
            'prior_a2C_cmd': prior_a2C_cmd,
            'priors_only': priors_only,
        },
    }

    # init random state
    rng = np.random.default_rng(seed)

    update_dict_full = deepcopy(update_dict_shared)
    update_dict_full['seed'] = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
    cpar_full = parse_param(update_dict_full)
    depend_full_webppl = None

    update_dict_priors = deepcopy(update_dict_shared)
    unused_ = merge_dicts(update_dict_priors, {'wppl_model_spec': {'priors_only': 'priorOnly'}})
    if unused_:
        warnings.warn('UNUSED UPDATE ITEMS:')
        print(unused_)
    update_dict_priors['seed'] = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
    cpar_priorsOnly = parse_param(update_dict_priors)
    depend_priors_webppl = None

    ### Decision point ###
    constitutively_run_webppl
    environment = cpar_full.environment
    wpplDataCache_exists = cpar_full.paths['wpplDataCache'].is_file() and cpar_priorsOnly.paths['wpplDataCache'].is_file()
    wpplCpar_exists = (cpar_full.paths['wpplDataCache'].parent / 'cpar.dill').is_file() and (cpar_priorsOnly.paths['wpplDataCache'].parent / 'cpar.dill').is_file()

    if constitutively_run_webppl:  # run webppl regardless
        wpplDataCache_exists = False

    if not wpplDataCache_exists:
        # webppl needs to be run
        if environment == 'cluster':
            # run webppl in parallel with sbatch
            wppl_control = {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotwire_precached_data': False,  # CACHE
                'saveOnExec': True,
            }
            cpar_full.cache['webppl'].update(wppl_control)
            cpar_priorsOnly.cache['webppl'].update(wppl_control)

            print(f"\n---Running webppl with sbatch---\n")

            # full model
            depend_full_webppl = sbatch_webppl_datagen(cpar_full)
            # PriorsOnly model
            depend_priors_webppl = sbatch_webppl_datagen(cpar_priorsOnly)

            """ #exposition
            # sbatch_webppl_datagen calls:
            from iaa_gendata_webppl import gendata_webppl
            gendata_webppl(cpar_full)
            """

        elif environment == 'local':
            # run webppl in serial
            wppl_control = {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotwire_precached_data': False,  # CACHE
                'saveOnExec': False,
            }
            cpar_full.cache['webppl'].update(wppl_control)
            cpar_priorsOnly.cache['webppl'].update(wppl_control)
            print(f"\n---Running webppl in serial---\n")
            gendata_webppl(cpar_full)
            gendata_webppl(cpar_priorsOnly)
        else:
            wppl_control = {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotwire_precached_data': False,  # CACHE
                'saveOnExec': False,
            }
            cpar_full.cache['webppl'].update(wppl_control)
            cpar_priorsOnly.cache['webppl'].update(wppl_control)
            print(f"\n---environment not understood: {environment}---\n")
            # run webppl in serial
            print(f"\n---Running webppl in serial---\n")
            gendata_webppl(cpar_full)
            gendata_webppl(cpar_priorsOnly)

    else:  # if wpplDataCache_exists
        if wpplCpar_exists:  # if wpplDataCache_exists and wpplCpar_exists
            # bypass webppl and load cached data
            print(f"\n---Loading cached webppl data---\n")
            wppl_control = {
                'runModel': False,  # CACHE
                'loadpickle': True,  # CACHE
                'hotwire_precached_data': False,  # CACHE
                'saveOnExec': False,
            }
            cpar_full.cache['webppl'].update(wppl_control)
            cpar_priorsOnly.cache['webppl'].update(wppl_control)
            ### skip printing
            # gendata_webppl(cpar_full)
            # gendata_webppl(cpar_priorsOnly)
        else:  # if wpplDataCache_exists but not wpplCpar_exists
            if hotpatch_cacheddata:
                # hotpatch with previously cached data (generates a new cpar)
                print(f"\n---Running webppl with sbatch---\n")
                wppl_control = {
                    'runModel': False,  # CACHE
                    'loadpickle': True,  # CACHE
                    'hotwire_precached_data': True,  # CACHE
                    'saveOnExec': False,
                }
                cpar_full.cache['webppl'].update(wppl_control)
                cpar_priorsOnly.cache['webppl'].update(wppl_control)
                gendata_webppl(cpar_full)
                gendata_webppl(cpar_priorsOnly)
            else:
                raise NotImplementedError

    return dict(
        full=dict(cpar=cpar_full, depend=depend_full_webppl),
        priorsOnly=dict(cpar=cpar_priorsOnly, depend=depend_priors_webppl),
    )
