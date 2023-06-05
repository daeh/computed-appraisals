#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_webppl_gendata_wrapper.py
"""


class Game():

    def __init__(self, label=None, model_path=None, wpplparam=None, kde_data=None, kde_data_label=None, dataout_base_path=None, pots=None):
        from copy import deepcopy

        self.label = label
        self._model_path = model_path
        self._wpplparam_in = deepcopy(wpplparam)
        self.wpplparam_common = deepcopy(wpplparam)
        self.kde_data = deepcopy(kde_data)
        self.kde_data_label = kde_data_label
        self.pots = deepcopy(pots)
        self.dataout_base_path = dataout_base_path
        self.dataout_path = dataout_base_path / 'wppl_samples' / label
        self.executable_path = None

        kde_data_label = f'input_kde-data.json' if self.kde_data_label is None else f"input_kde-{self.kde_data_label}.json"

        self.kde_json_path = self.dataout_path / kde_data_label

        self.wpplparam_common['dataOut'] = str(self.dataout_path)
        self.wpplparam_common['path_to_kde'] = str(self.kde_json_path)

    def _compile(self):
        import subprocess
        import shutil

        self.executable_path = self.dataout_base_path / 'wppl_exe' / self._model_path.name
        self.executable_path.parent.mkdir(exist_ok=True, parents=True)

        ### copy webppl model to executable directory ###
        assert self._model_path.is_file(), f"Model not found: {self._model_path}"
        shutil.copy(self._model_path, self.executable_path)

        compiled_path = self.executable_path.with_suffix('.js')

        cmd_list = [
            "webppl", str(self.executable_path),
            "--require", "webppl-json",
            "--compile",
            "--out", str(compiled_path)]

        subprocess.run(cmd_list, capture_output=True, encoding='utf-8', check=True)

    def play(self, multithread=True, remove_existing_data=True, seed=None):
        from copy import deepcopy
        import numpy as np
        import pickle
        import json
        import time
        from datetime import timedelta

        if remove_existing_data and self.dataout_path.exists():
            import shutil
            shutil.rmtree(self.dataout_path)

        self.dataout_path.mkdir(parents=True, exist_ok=True)

        self._compile()

        assert not self.kde_json_path.exists()
        with open(self.kde_json_path, 'w') as f:
            json.dump(self.kde_data, f, separators=(',', ':'), indent=None)

        self.dataout_path.mkdir(parents=True, exist_ok=True)

        seed_inherited = True
        if seed is None:
            seed = int(str(int(time.time() * 10**6))[-9:])
            seed_inherited = False
        rng = np.random.default_rng(seed)
        seeds_ = rng.integers(low=1, high=np.iinfo(np.int32).max, size=len(self.pots), dtype=int)
        seeds = (int(s_) for s_ in seeds_)
        pot_seed_list = tuple(zip(self.pots, seeds))

        wppl_data_in = dict(
            model_path=str(self._model_path),
            executable_path=str(self.executable_path),
            wpplparam=self.wpplparam_common,
            _shellcmd=execute_webppl(self.executable_path, wpplparam=self.wpplparam_common, seed=-1, get_command_only=True),  # {**wpplparam_shared, 'pot': 'template'}
            seed_inherited=seed_inherited,
            seed=seed,
            pot_seeds=pot_seed_list,
            pots=self.pots,
            kde_data_path=str(self.kde_json_path),
            kde_data=self.kde_data,
        )

        with open(self.dataout_path / f"input_alldata.pkl", 'wb') as f:
            pickle.dump(wppl_data_in, f, protocol=-5)
        with open(self.dataout_path / f"input_alldata.json", 'w') as f:
            json.dump(wppl_data_in, f, indent=2)
        with open(self.dataout_path / f"input_seeds.json", 'w') as f:
            json.dump(pot_seed_list, f, indent=2)

        param_list = list()
        for pot_, wpplseed_ in pot_seed_list:
            wpplparam_ = deepcopy(self.wpplparam_common)
            wpplparam_['pot'] = pot_
            param_list.append(dict(seed=wpplseed_, wpplparam=wpplparam_))

        t0 = time.perf_counter()
        if multithread:
            from joblib import Parallel, delayed, cpu_count
            print(f"\nRunning {len(self.pots)} pots on {cpu_count()} CPU")
            with Parallel(n_jobs=min(len(self.pots), cpu_count())) as pool:
                sysout = pool(delayed(execute_webppl)(executable_path=self.executable_path, **param_) for param_ in param_list)
                shellout = sysout[0]
        else:
            for i_param, param_ in enumerate(param_list):
                sysout = execute_webppl(executable_path=self.executable_path, **param_)
                shellout = sysout
                print(f'{i_param + 1} finished, {len(param_list) - (i_param + 1)} remaining')
        elapsed_time = time.perf_counter() - t0
        print('\n\nWebPPL Finished, Execution Time:  {} ({:0.2f}s), {:0.4f} per cycle\n\n'.format(timedelta(seconds=elapsed_time), elapsed_time, elapsed_time / len(self.pots)))


def execute_webppl(executable_path=None, wpplparam=None, seed=None, get_command_only=False):
    import subprocess
    import json

    assert isinstance(seed, int)

    cmd_list = [
        "webppl", str(executable_path),
        "--random-seed", str(seed),
        "--require", "webppl-json",
        "--",
        "--param", json.dumps(wpplparam, separators=(',', ':'), indent=None)
    ]

    if get_command_only:
        return cmd_list

    return subprocess.run(cmd_list, capture_output=True, encoding='utf-8', check=True)


def import_ppl_data(cpar, game_full, game_distal_prior_flat, verbose=False):
    from cam_webppl_gendata_helpers import Timer
    from cam_import_empirical_data import importEmpirical_exp10_, importEmpirical_exp7_11_
    from cam_import_wppljson import importPPLmodel_, importPPLdata_parallel
    from joblib import Parallel, delayed, cpu_count

    ### import data

    t_import = Timer()
    t_import.start()

    ### load exp7 exp11
    ppldata, wpplparam = importPPLmodel_(game_full.dataout_path, game_full.wpplparam_common, game_full.pots, verbose)

    data_stats_7_11 = importEmpirical_exp7_11_(ppldata, cpar)

    t_import.lap('import 3 7 11')

    if cpar.data_spec['exp10']['data_load_param']['print_responses']:
        bypass_plotting = False
    else:
        bypass_plotting = True

    subject_stats_all_ = importEmpirical_exp10_(ppldata, cpar, stimulus='all', condition=None, update_ppldata=False, bypass_plotting=bypass_plotting)
    ppldata['subject_stats']['exp10all'] = subject_stats_all_

    t_import.lap('importEmpirical_exp10_')

    with Parallel(n_jobs=min(len(game_distal_prior_flat), cpu_count())) as pool:
        ppldata_list_loaded = pool(delayed(importPPLdata_parallel)(game_specific_, cpar) for game_specific_ in game_distal_prior_flat)

    t_import.lap('importPPLdata_parallel')

    distal_prior_ppldata = dict()
    for game_specific_ in game_distal_prior_flat:
        distal_prior_ppldata[game_specific_['stimid']] = dict(
            C=None,
            D=None,
        )
    for stim, a1, ppl_data_loaded, subject_stats in ppldata_list_loaded:
        distal_prior_ppldata[stim][a1] = ppl_data_loaded
        distal_prior_ppldata[stim][a1]['subject_stats_'] = subject_stats

    t_import.lap('distal_prior_ppldata')

    return ppldata, distal_prior_ppldata, wpplparam


def play_in_parallel(gameobj=None, stimid=None, a1=None, removeOldData=None, modelseed=None):
    gameobj.play(multithread=False, remove_existing_data=removeOldData, seed=modelseed)


def initialize_wrapper(cpar):
    import numpy as np
    import pickle
    import dill
    from copy import deepcopy
    import time
    from pprint import pprint
    from cam_webppl_gendata_helpers import Timer, gen_empir_kde_genericplayers_multivarkdemixture, gen_empir_kde_specificplayers_multivarkdemixture, cache_all_code
    from cam_import_empirical_data import importEmpirical_InversePlanning_exp6_exp9

    print('Cache Settings:')
    pprint(cpar.cache)

    environment = cpar.environment
    paths = cpar.paths
    runModel = cpar.cache['webppl']['runModel']
    loadpickle = cpar.cache['webppl']['loadpickle']
    hotpatch_precached_data = cpar.cache['webppl'].get('hotpatch_precached_data', False)
    removeOldData = cpar.cache['webppl']['removeOldData']
    saveOnExec = cpar.cache['webppl']['saveOnExec']
    empir_load_param = cpar.empir_load_param
    wppl_model_spec = cpar.wppl_model_spec
    verbose = cpar.plot_param['verbose']
    seed = getattr(cpar, 'seed', None)

    prior_form = wppl_model_spec['prior_form']
    generic_repu_values_from = wppl_model_spec['generic_repu_values_from']
    distal_repu_values_from = wppl_model_spec['distal_repu_values_from']
    inf_param = wppl_model_spec['inf_param']

    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
    rng = np.random.default_rng(seed)

    a1_labels = ['C', 'D']

    datacache = paths['wpplDataCache']

    if hotpatch_precached_data and datacache.is_file():
        runModel = False

    # %%

    if wppl_model_spec['prior_a2C'] in ['split', 'splitinv']:
        raise NotImplementedError

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

    if inf_param == 'MCMC3k':
        inferenceParam = {
            'm0': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # lvl0: base agent makes decision
            'm1': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # lvl2: reputation agent makes decision
            'm3': {"method": "MCMC", "samples": 1000, "lag": 3, "burn": 1000, "kernel": "MH"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "MCMC", "samples": 3000, "lag": 3, "burn": 3000, "kernel": "MH"}  # infer values of inverse appraisal features
        }
    elif inf_param == 'rejection':
        inferenceParam = {
            'm0': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # lvl0: base agent makes decision
            'm1': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # lvl2: reputation agent makes decision
            'm3': {"method": "rejection", "samples": 1000, "maxScore": 0, "incremental": "true"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "rejection", "samples": 3000, "maxScore": 0, "incremental": "true"}  # infer values of inverse appraisal features
        }
    elif inf_param == 'rapidtest':
        inferenceParam = {
            'm0': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # lvl0: base agent makes decision
            'm1': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # * lvl1: infer base weights given base agent's decision (these are basis for level 3 & 4 reputation utilities), # pd of estimation of other agent's choice given base agent's decision
            'm2': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # lvl2: reputation agent makes decision
            'm3': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"},  # * lvl3: infer weights of reputation agent's features, # pd of estimation of other agent's choice given reputation agent's decision
            'm4iaf': {"method": "rejection", "samples": 10, "maxScore": 0, "incremental": "true"}  # infer values of inverse appraisal features
        }
    else:
        raise ValueError(f"inf_param {inf_param} not recognized")

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
        'kde_width': wppl_model_spec['kde_width'],
        'prior_a2C': wppl_model_spec['prior_a2C'],
        'payoffMatrix': payoffMatrices['weakPD'],
    }

    if wppl_model_spec['refpoint_type'] == 'Norm':
        modelParam['baseRefPointDist'] = wppl_model_spec['refpoint_type']
        modelParam['baseRefPoint'] = {'Money': {'mu': 1000, 'sigma': 250}}
    else:
        raise ValueError(f"refpoint_type {wppl_model_spec['refpoint_type']} not recognized")

    t.lap('prior form start')

    ### load empirical data

    if prior_form == 'multivarkdemixture':
        kde_json_generic_, kde_json_generic_label_ = gen_empir_kde_genericplayers_multivarkdemixture(df_wide9=df_wide9, df_wide6=df_wide6, prior_a2C=wppl_model_spec['prior_a2C'], repu_values_from=generic_repu_values_from)
    else:
        raise ValueError(f"prior_form {prior_form} not recognized")

    t.lap('end prior form')

    ####

    ### initialize run
    wpplparam = {**modelParam, **inferenceParam}

    ### save cpar ###
    if runModel or hotpatch_precached_data:

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

    if runModel and saveOnExec:
        cache_all_code(code_dir=paths['code'], dump_dir=paths['dataOut'])

    modelseed = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
    pots_ = [2.0, 11, 25, 46, 77, 124, 194, 299, 457, 694, 1049, 1582, 2381, 3580, 5378, 8075, 12121, 18190, 27293, 40948, 61430, 92153, 138238, 207365]
    game_full = Game(label='generic', model_path=paths['model'], wpplparam=wpplparam, kde_data=kde_json_generic_, kde_data_label=kde_json_generic_label_, dataout_base_path=paths['dataOut'], pots=pots_)

    if runModel:
        if environment in ['local', 'cluster', 'remotekernel']:
            multithread_ = True
        else:
            multithread_ = False
        game_full.play(multithread=multithread_, remove_existing_data=removeOldData, seed=modelseed)
    del modelseed

    t.lap('game play')
    # %% #########################

    stimids = np.unique(df_wide9['face']).tolist()

    game_distal_prior_flat = list()
    for i_stim, stim in enumerate(stimids):
        for a1 in a1_labels:

            if prior_form == 'multivarkdemixture':
                kde_json_thisplayer_, kde_json_thisplayer_label_ = gen_empir_kde_specificplayers_multivarkdemixture(df_wide9=df_wide9, stimid=stim, a1=a1, repu_values_from=distal_repu_values_from)
            else:
                raise ValueError(f"prior_form {prior_form} not recognized")

            modelseed = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
            pots_ = [124.0, 694.0, 1582.0, 5378.0, 12121.0, 27293.0, 61430.0, 138238.0]
            game_distal_prior_flat.append(
                dict(
                    gameobj=Game(label=f'specific-{stim}-{a1}', model_path=paths['model'], wpplparam=wpplparam, kde_data=kde_json_thisplayer_, kde_data_label=kde_json_thisplayer_label_, dataout_base_path=paths['dataOut'], pots=pots_),
                    stimid=stim,
                    a1=a1,
                    modelseed=deepcopy(modelseed),
                    removeOldData=removeOldData,
                )
            )
            del modelseed

    if runModel:

        from joblib import Parallel, delayed, cpu_count
        print(f"\nRunning {len(game_distal_prior_flat)} specific player games on {cpu_count()} CPU")
        with Parallel(n_jobs=min(len(game_distal_prior_flat), cpu_count())) as pool:
            sysout = pool(delayed(play_in_parallel)(**game_specific_) for game_specific_ in game_distal_prior_flat)

    t.lap('json')
    # %% ########################

    print(f'\n---WebPPL Finished---\n')
    if loadpickle and datacache.is_file():
        print(f'Loading cached WebPPL data from {datacache}')
        with open(datacache, 'rb') as f:
            ppldata, distal_prior_ppldata, wpplparam = pickle.load(f)
    else:
        print(f'Caching WebPPL data')
        print(f'Starting to import WebPPL json data. Will be cached at: {datacache}')

        ppldata, distal_prior_ppldata, wpplparam = import_ppl_data(cpar, game_full, game_distal_prior_flat, verbose=False)

        # add inverse planning empirical ratings
        ppldata.update(inversePlanningDict)
        ppldata['subject_stats'].update(inversePlanningDict_subject_stats)

        if datacache.is_file():
            print(f'Removing cached data at {datacache}')
            datacache.unlink()
        with open(datacache, 'wb') as f:
            pickle.dump((ppldata, distal_prior_ppldata, wpplparam), f, protocol=-5)
        print(f'Cacheing data to {datacache} \n')

    t.lap('import')
    t.report()

    # %%

    setattr(cpar, 'wpplparam', wpplparam)

    return ppldata, distal_prior_ppldata
