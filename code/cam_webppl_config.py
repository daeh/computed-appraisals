#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_webppl_config.py
"""


def sbatch_webppl_datagen(cpar):
    """
    runs getdata_webppl() from cam_webppl.py
    """
    import numpy as np
    from pathlib import Path
    import dill
    import subprocess
    import re

    script_path = cpar.paths['code'] / "launch_webppl.sbatch"

    log_dir_path = cpar.paths['dataOutBase'] / 'temp_sbatchlogs_wppl'
    log_dir_path.mkdir(parents=True, exist_ok=True)

    output_pattern = log_dir_path / "gendata_webppl_%J.txt"

    pickle_dir = cpar.paths['dataOutBase'] / "temp_pickles_wppl"
    pickle_dir.mkdir(exist_ok=True, parents=True)

    file_unique = False
    while not file_unique:
        pickle_path = pickle_dir / f'cpar_for_webppl_datagen_{np.random.randint(0,10000)}.dill'
        if not pickle_path.exists():
            file_unique = True

    with open(pickle_path, 'wb') as f:
        dill.dump(cpar, f, protocol=-4)

    cmd_list = [
        "sbatch",
        f"--output={str(output_pattern)}",
        f"--exclude=dgx001,dgx002,node017,node[031-077],node086,node[100-116]",  # gold
        ###
        # f"--exclude=node[017,078-094,097,098,100-116]", ### e5
        # f"--exclude=dgx001,dgx002,node017,node[031-077],node086,node[100-116]", ### gold
        # f"--exclude=node[100-116]", ### centos 7 on intel e5/gold
        str(script_path),
        str(script_path),
        str(pickle_path),
    ]

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
        raise ValueError("sbatch_webppl_datagen(): failed to get job id")

    return depend


def webppl_datagen_config(spec=None, hotpatch_cacheddata=False, projectbase_path=None, dataout_path=None):
    """
    """

    # %%

    from copy import deepcopy
    import warnings
    from pathlib import Path
    import numpy as np
    from cam_control_utils import parse_param, merge_dicts
    from cam_webppl import getdata_webppl

    constitutively_run_webppl = False
    if spec is None:
        spec = dict()

    prospect_fn = spec.get('prospect_fn', 'log')
    inf_param = spec.get('inf_param', ['MCMC3k', 'rapidtest'][0])
    prior_form = spec.get('prior_form', 'multivarkdemixture')
    prior_a2C_cmd = spec.get('prior_a2C_cmd', 0.4)
    kde_width = spec.get('kde_width', 0.01)
    generic_repu_values_from = spec.get('generic_repu_values_from', ['internal', 'empiricalExpectation'][0])
    distal_repu_values_from = spec.get('distal_repu_values_from', ['internal', 'empiricalExpectation'][1])
    refpoint_type = spec.get('refpoint_type', 'Norm')
    priors_only = spec.get('priors_only', ['full', 'priorOnly'][0])
    seed = spec.get('seed', None)

    prefix = spec.get('prefix', 'test')
    exp_name_default = f'{prefix}_invpT-{prospect_fn}_{generic_repu_values_from}-{distal_repu_values_from}_psref{refpoint_type}_{prior_form}_kde-{kde_width:0.3f}_mix-{prior_a2C_cmd*100:0.0f}'
    exp_name = spec.get('exp_name', exp_name_default)

    update_dict_shared = {
        'strict': 'strict',
        'expdir': projectbase_path,
        'dataOutBase': dataout_path,
        'exp_name': exp_name,
        'seed': None,
        'cache': {
            'webppl': {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotpatch_precached_data': False,  # CACHE
            },
        },
        'wppl_model_spec': {
            'prospect_fn': prospect_fn,
            'inf_param': inf_param,
            'generic_repu_values_from': generic_repu_values_from,
            'distal_repu_values_from': distal_repu_values_from,
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
    environment = cpar_full.environment
    wpplDataCache_exists = cpar_full.paths['wpplDataCache'].is_file() and cpar_priorsOnly.paths['wpplDataCache'].is_file()
    wpplCpar_exists = (cpar_full.paths['wpplDataCache'].parent / 'cpar.dill').is_file() and (cpar_priorsOnly.paths['wpplDataCache'].parent / 'cpar.dill').is_file()

    if constitutively_run_webppl:  # run webppl regardless
        wpplDataCache_exists = False

    if not wpplDataCache_exists:
        # webppl needs to be run
        if environment in ['cluster', 'remotekernel']:
            # run webppl in parallel with sbatch
            wppl_control = {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotpatch_precached_data': False,  # CACHE
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
            from cam_webppl import getdata_webppl
            getdata_webppl(cpar_full)
            """

        elif environment == 'local':
            # run webppl in serial
            wppl_control = {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotpatch_precached_data': False,  # CACHE
                'saveOnExec': False,
            }
            cpar_full.cache['webppl'].update(wppl_control)
            cpar_priorsOnly.cache['webppl'].update(wppl_control)
            print(f"\n---Running webppl in serial---\n")
            getdata_webppl(cpar_full)
            getdata_webppl(cpar_priorsOnly)

        else:
            wppl_control = {
                'runModel': True,  # CACHE
                'loadpickle': False,  # CACHE
                'hotpatch_precached_data': False,  # CACHE
                'saveOnExec': False,
            }
            cpar_full.cache['webppl'].update(wppl_control)
            cpar_priorsOnly.cache['webppl'].update(wppl_control)
            print(f"\n---environment not understood: {environment}---\n")
            # run webppl in serial
            print(f"\n---Running webppl in serial---\n")
            getdata_webppl(cpar_full)
            getdata_webppl(cpar_priorsOnly)

    else:  # if wpplDataCache_exists
        if wpplCpar_exists:  # if wpplDataCache_exists and wpplCpar_exists
            # bypass webppl and load cached data
            print(f"\n---Loading cached webppl data---\n")
            wppl_control = {
                'runModel': False,  # CACHE
                'loadpickle': True,  # CACHE
                'hotpatch_precached_data': False,  # CACHE
                'saveOnExec': False,
            }
            cpar_full.cache['webppl'].update(wppl_control)
            cpar_priorsOnly.cache['webppl'].update(wppl_control)
            ### skip printing
        else:  # if wpplDataCache_exists but not wpplCpar_exists
            if hotpatch_cacheddata:
                # hotpatch with previously cached data (generates a new cpar)
                print(f"\n---Hotpatching with cached webppl data---\n")
                wppl_control = {
                    'runModel': False,  # CACHE
                    'loadpickle': True,  # CACHE
                    'hotpatch_precached_data': True,  # CACHE
                    'saveOnExec': False,
                }
                cpar_full.cache['webppl'].update(wppl_control)
                cpar_priorsOnly.cache['webppl'].update(wppl_control)
                getdata_webppl(cpar_full, print_figs=False)
                getdata_webppl(cpar_priorsOnly, print_figs=False)
            else:
                raise NotImplementedError

    return dict(
        full=dict(cpar=cpar_full, depend=depend_full_webppl),
        priorsOnly=dict(cpar=cpar_priorsOnly, depend=depend_priors_webppl),
    )
