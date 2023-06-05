#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_assemble_torch_models.py
"""


from cam_emotorch import EmoTorch
from cam_emotorch_utils import sbatch_torch_array


def make_seed_from_str_hash(string_decoded):
    import zlib
    import numpy as np
    max_int = np.iinfo(np.int32).max - 1
    string_encoded = string_decoded.encode(encoding='UTF-8')
    hash_val = zlib.crc32(string_encoded)
    while hash_val > max_int:
        hash_val -= max_int
    hash_val = int(hash_val)
    assert hash_val > 0 and hash_val <= max_int and isinstance(hash_val, int), f"hash_val: {hash_val}, type: {type(hash_val)}"
    return hash_val


def gen_cv_datafolds_trainongeneric_cvonspecific(specific_set_ids, n_tt_folds=2, seed=None):

    import numpy as np
    from copy import deepcopy
    from sklearn.model_selection import KFold

    import time
    seed_inherited = True
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
        seed_inherited = False

    rng = np.random.default_rng(seed)

    stimids_all = np.array(deepcopy(specific_set_ids))

    datafolds_tt = list()
    kf_tt_seed_ = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
    kf_tt = KFold(n_splits=n_tt_folds, random_state=kf_tt_seed_, shuffle=True)
    for tt_train_idx, tt_test_idx in kf_tt.split(stimids_all):
        stimid_tt_test = stimids_all[tt_test_idx]
        stimid_tt_train = stimids_all[tt_train_idx]

        datafolds_cv = list()
        datafolds_cv.append(dict(test=stimid_tt_train, train=list()))

        datafolds_tt.append(dict(test=stimid_tt_test, train=list(), cv=datafolds_cv))

    assert np.unique(np.concatenate([fold_['test'] for fold_ in datafolds_tt])).size == 20
    assert np.concatenate([fold_['test'] for fold_ in datafolds_tt]).size == 20

    return datafolds_tt


def assemble_cfg(model_spec):

    op_feature_selector = {
        'abslnpot': r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2lnpotunval)',
        'money': r'^(U\[baseMoney\]|PE\[baseMoney\])',
    }

    op_whitening = {
        'scaledSDmeanKept': {'with_std': True, 'with_mean': False},
    }

    _prospect_transform_kwargs_04 = {'alpha_': 0.4, 'beta_': 0.4, 'lambda_': 1.0, 'intercept': 0.0, 'noise_scale': 0.0}

    op_caf_prospect_scale = {
        'PSeven4': {'base_kwargs': _prospect_transform_kwargs_04, 'repu_kwargs': _prospect_transform_kwargs_04},
    }

    ###########

    model_type = model_spec['model_type']
    outpath = model_spec['outpath']

    #############
    feature_selector_label = 'abslnpot'
    feature_selector_label_post = 'caa'
    if model_type == 'money':
        feature_selector_label = 'money'
        feature_selector_label_post = 'money'
    #############
    whitening_label = 'scaledSDmeanKept'
    prospect_transform_label = 'PSeven4'
    scale_fn_label = 'ScalePEa2raw'
    data_prep_label = 'cv'
    #############

    whitening_param = op_whitening[whitening_label]
    prospect_param = op_caf_prospect_scale[prospect_transform_label]
    feature_selector = op_feature_selector[feature_selector_label]

    dataincache_shared_path = outpath / f"torch_datain_cache"

    #############

    reg_type = 'lasso'
    torch_model_ = 'lasso-mDsP'

    model_param = {'k': model_spec['logit_k'], 'laplace_scale': model_spec['laplace_scale']}
    optimization_param = {'iter': model_spec['niters'], 'seed': model_spec.get('seed', None)}

    pytorch_spec = dict()
    pytorch_spec['model_type'] = model_type
    pytorch_spec['reg_type'] = reg_type

    pytorch_spec['whitening'] = (whitening_label, whitening_param)
    pytorch_spec['prospect_param'] = (prospect_transform_label, prospect_param)
    pytorch_spec['feature_selector'] = (feature_selector_label_post, feature_selector)
    pytorch_spec['scale_transform_label'] = scale_fn_label
    pytorch_spec['thin_samples_factor'] = 8

    #################

    dout_base_dir_path = outpath

    run_prefix = f"mix{model_spec['imix']:02}_fold{model_spec['ittfold']:02}_cv{model_spec['icvfold']:02}_ri{model_spec['icvrinit']:02}"
    cv_suffix = f'{model_type}-PEpia2{feature_selector_label}_{prospect_transform_label}_{data_prep_label}'
    subrun_dir = f"{run_prefix}-{torch_model_}"

    dout_base_path = dout_base_dir_path / subrun_dir

    cfg = dict(
        dout_base_path=dout_base_path,
        figs_base_path=dout_base_dir_path.parent / 'figs_pytorch' / subrun_dir,
        dataincache_base_path=dataincache_shared_path / cv_suffix,
        ###
        model_id_brief=torch_model_,
        ###
        model_param=model_param,
        ###
        behavior='optimization',
        optimization_param=optimization_param,
        trackprogress=False,
        ###
        data_prep_label=data_prep_label,
        trainset=model_spec['trainset'],
        testset=model_spec['testset'],
        pytorch_spec=pytorch_spec,
        ###
        save_minimal=True,
        ####
        cpar_path_str=str(model_spec['cpar_path']),
        # cpar=cpar,
    )

    return cfg


def run_cv(outpath_base=None, cpar_path_full=None, cpar_path_priors=None, specificplayers_stimids=None, laplace_scales=None, logit_k=None, niters=None, n_tt_mixes=None, n_tt_folds=None, n_cv_random_reinits=None, seed=None, dependency=None):
    """train only on generic, cv on subset of specific players, leaveout remaining specific players for testing"""
    # %%

    '''
    cv:
        run all hyperparam sets
        train on generic, test on 15 specific players
        pick hyperparam based on prediction of 15 specific players
    tt:
        use selected hyperparam to train on generic, test on leftout 5 specific players
    '''

    import numpy as np
    from copy import deepcopy
    import pickle
    import dill

    assert cpar_path_full.is_file() and cpar_path_priors.is_file()

    datafolds_mixes = list()
    for i_tt_mix in range(n_tt_mixes):
        datafolds_mixes.append(gen_cv_datafolds_trainongeneric_cvonspecific(specificplayers_stimids, n_tt_folds=n_tt_folds, seed=make_seed_from_str_hash(f"{seed}_{i_tt_mix}_{n_tt_folds}")))

    outpath_base.mkdir(exist_ok=True, parents=True)

    model_param_list = list()
    for modelname in ['caaFull', 'invplanLesion', 'socialLesion']:

        if modelname == 'socialLesion':
            model_type = 'money'
        else:
            model_type = 'iaf'

        if modelname == 'invplanLesion':
            cpar_path = cpar_path_priors
        else:
            cpar_path = cpar_path_full

        for laplace_scale in laplace_scales:
            model_param_list.append(dict(
                outpath=outpath_base / 'torchcv' / modelname,
                cpar_path=cpar_path,
                model_type=model_type,
                logit_k=logit_k,
                laplace_scale=laplace_scale,
                niters=niters,
                imix=None,
                ittfold=None,
                icvfold=None,
                icvrinit=None,
                trainset=None,
                testset=None,
            ))

    # %%
    #################################
    ### launch cv jobs
    #################################

    modelspecs_cv_list = list()
    jobs_cv_list = list()
    for i_tt_mix in range(n_tt_mixes):
        for i_tt_fold in range(n_tt_folds):

            ### assemble models ###
            for i_cv_fold, stimid_cv_fold in enumerate(datafolds_mixes[i_tt_mix][i_tt_fold]['cv']):
                for model_param in model_param_list:
                    for i_cvrinit in range(n_cv_random_reinits):
                        seed_this_ = make_seed_from_str_hash(f"{seed}_{model_param['outpath'].name}_{model_param['laplace_scale']}_{i_tt_mix}_{i_tt_fold}_{i_cv_fold}_{i_cvrinit}")

                        cv_spec_ = dict(
                            imix=i_tt_mix,
                            ittfold=i_tt_fold,
                            icvfold=i_cv_fold,
                            icvrinit=i_cvrinit,
                            trainset=stimid_cv_fold['train'],
                            testset=stimid_cv_fold['test'],
                            seed=seed_this_,
                        )
                        model_spec = {**deepcopy(model_param), **cv_spec_}

                        cfg = assemble_cfg(model_spec)

                        modelspecs_cv_list.append(dict(
                            model_spec=deepcopy(model_spec),
                            cfg_=cfg,
                        ))

                        jobs_cv_list.append(cfg)

    with open(outpath_base / f'datafolds_nmix{n_tt_mixes}-nfold{n_tt_folds}.pkl', 'wb') as f:
        pickle.dump(datafolds_mixes, f, protocol=-5)

    with open(outpath_base / f'modelspecs-cv_nmix{n_tt_mixes}-nfold{n_tt_folds}-nricv{n_cv_random_reinits}.pkl', 'wb') as f:
        pickle.dump(modelspecs_cv_list, f, protocol=-5)

    '''exposition
    res0 = main_optimize(cfg)
    '''

    ### submit jobs ###

    print(f"launching {len(jobs_cv_list)} jobs")

    with open(cpar_path_full, 'rb') as f:
        cpar = dill.load(f)
    codedir_path = cpar.paths['code']
    dataoutbase_path = cpar.paths['dataOutBase']

    batch_every = 1000
    multithread_every = 2
    jobs_array = list()
    dependencies_out = list()
    while len(jobs_cv_list):
        jobgroup = list()
        while len(jobgroup) < multithread_every and len(jobs_cv_list):
            jobgroup.append(jobs_cv_list.pop(0))
        jobs_array.append(jobgroup)
        if len(jobs_array) == batch_every or len(jobs_cv_list) == 0:
            sbatch_out = sbatch_torch_array(
                jobs_array, behavior='optimize', codedir_path=codedir_path, dataoutbase_path=dataoutbase_path, dependency=dependency,
                job_name='camTorchCV',
                mem_per_job=9,
                time=5,
                exclude=None,
                partition=['use-everything', None, 'gablab'][0])
            dependencies_out.append(sbatch_out['dependency'])
            jobs_array = list()

    # %%
    return dependencies_out
    # %%


def relaunch_jobs_for_missing_cv_results(modelspecs_cv_path=None, dependency=None, run_missing=False):
    """
    find missing cv results, launch those jobs again
    there should be no running cv jobs
    """

    from pathlib import Path
    import pickle
    import dill
    import random

    with open(modelspecs_cv_path, 'rb') as f:
        modelspecs_cv_list = pickle.load(f)

    print(f"staring cv res scan")
    cfg_missing_list_started = list()
    cfg_missing_list_notstarted = list()
    multipletemp = list()
    for tmspec in modelspecs_cv_list:
        model_spec = tmspec['model_spec']
        resdir = Path(tmspec['cfg_']['dout_base_path']) / f"PTM-lasso-mDsP__k-{model_spec['logit_k']}_laplace_scale-{model_spec['laplace_scale']}"
        resfiles = list(resdir.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj.dill"))
        if len(resfiles) > 1:
            multipletemp.append(resfiles)
        if len(resfiles) == 0:
            naivefiles = list(resdir.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj-naive.dill"))
            if len(naivefiles) > 0:
                cfg_missing_list_started.append(assemble_cfg(model_spec))
            else:
                cfg_missing_list_notstarted.append(assemble_cfg(model_spec))

    random.shuffle(cfg_missing_list_started)
    random.shuffle(cfg_missing_list_notstarted)
    cfg_missing_list = [*cfg_missing_list_notstarted, *cfg_missing_list_started]

    print(f"found {len(multipletemp)} trajectories with multiple results")

    nmissing = len(cfg_missing_list)
    print(f"missing {nmissing} torch results of {len(modelspecs_cv_list)} ({len(modelspecs_cv_list) - nmissing} found)")

    dependencies_out = list()
    if cfg_missing_list and run_missing:
        print(f"launching {len(cfg_missing_list)} jobs")

        with open(modelspecs_cv_list[0]['cfg_']['_cpar_path'], 'rb') as f:
            cpar = dill.load(f)
        codedir_path = cpar.paths['code']
        dataoutbase_path = cpar.paths['dataOutBase']

        batch_every = 1000
        multithread_every = 2
        jobs_array = list()
        while len(cfg_missing_list):
            jobgroup = list()
            while len(jobgroup) < multithread_every and len(cfg_missing_list):
                jobgroup.append(cfg_missing_list.pop(0))
            jobs_array.append(jobgroup)
            if len(jobs_array) == batch_every or len(cfg_missing_list) == 0:
                sbatch_out = sbatch_torch_array(
                    jobs_array, behavior='optimize', codedir_path=codedir_path, dataoutbase_path=dataoutbase_path, dependency=dependency,
                    job_name='camTorchCV',
                    mem_per_job=9,
                    time=5,
                    exclude=None,
                    partition=['use-everything', None, 'gablab'][0])
                dependencies_out.append(sbatch_out['dependency'])
                jobs_array = list()

    # %%
    return nmissing, dependencies_out
    # %%


def run_tt(n_tt_mixes=None, n_tt_folds=None, n_tt_random_reinits=None, n_cv_random_reinits=None, laplace_scales=None, seed=None, niters=None, outpath_base=None, cpar_path_full=None, cpar_path_priors=None, dependency=None):

    # %%
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pickle
    import dill

    # %%
    #################################
    ### load cv results
    #################################

    with open(outpath_base / f'datafolds_nmix{n_tt_mixes}-nfold{n_tt_folds}.pkl', 'rb') as f:
        datafolds_mixes = pickle.load(f)

    # %%

    ### select best hyperparam ###

    hypparm_criterion_time = ['lastquarter', 'lastfifth', 'terminal'][0]

    model_names = ['caaFull', 'invplanLesion', 'socialLesion']

    # %%

    selected_res_log = dict()
    for modelname in model_names:
        selected_res_log[modelname] = dict()
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            if modelname == 'caaFull' and hypparm_criterion == 'hypc_absintens_bypotoutcome':
                pass
            else:
                selected_res_log[modelname][hypparm_criterion] = list()

    modelspecs_tt_list = list()
    jobs_tt_list = list()
    for modelname in model_names:
        for i_tt_mix in range(n_tt_mixes):
            print(f"model {modelname}, mix {i_tt_mix}")
            for i_tt_fold in range(n_tt_folds):

                stimid_tt_fold = datafolds_mixes[i_tt_mix][i_tt_fold]

                hypparm_search_path = outpath_base / 'torchcv' / modelname
                assert hypparm_search_path.is_dir()

                ### all random reinit results for this model-mix-fold
                eto_list = list(hypparm_search_path.glob(f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00_ri*-lasso-mDsP/PTM*/optimize_iter-*/EmoTorchObj.dill"))

                hypparm_list = list()
                for eto_path in eto_list:
                    eto = EmoTorch(verbose=False)
                    eto.load(eto_path)
                    progressdf = eto.optimized['op_progressdf'].drop_duplicates()

                    if hypparm_criterion_time == 'lastquarter':
                        ### median score of last quarter of iterations
                        scores_ = progressdf.iloc[-round(progressdf.shape[0] / 4):, :].median()
                    elif hypparm_criterion_time == 'lastfifth':
                        ### median score of last fifth of iterations
                        scores_ = progressdf.iloc[-round(progressdf.shape[0] / 5):, :].median()
                    else:
                        ### score of last iteration
                        scores_ = progressdf.iloc[-1:, :].median()

                    hypparm_list.append({**eto.model_param, **scores_})

                assert len(hypparm_list) == len(laplace_scales) * n_cv_random_reinits

                for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:

                    ### for the lesion models, they hyperparam is based on the score of the test metric.
                    ### for the full model, the hyperparam is based on the score of the deltas metric.
                    if modelname == 'caaFull' and hypparm_criterion == 'hypc_absintens_bypotoutcome':
                        continue

                    ### mean over random reinits
                    hypparm_criterion_ = {'hypc_absintens_bypotoutcome': 'test_ccc', 'hypc_deltas_byoutcome': 'deltas_ccc'}[hypparm_criterion]
                    hypparm_res = pd.DataFrame(hypparm_list).groupby(['laplace_scale']).agg(np.mean).reset_index().sort_values(hypparm_criterion_, ascending=False)

                    ### highest scoring hyperparam
                    selected_param_ = hypparm_res.iloc[0, :]
                    selected_res_log[modelname][hypparm_criterion].append(selected_param_)

                    #############################

                    selected_param = dict(
                        logit_k=selected_param_['k'],
                        laplace_scale=selected_param_['laplace_scale'],
                    )

                    if modelname == 'socialLesion':
                        model_type = 'money'
                    else:
                        model_type = 'iaf'

                    if modelname == 'invplanLesion':
                        cpar_path = cpar_path_priors
                    else:
                        cpar_path = cpar_path_full

                    for i_ttrinit in range(n_tt_random_reinits):

                        seed_this_ = make_seed_from_str_hash(f"tt-{seed}_{modelname}_{hypparm_criterion}_{i_tt_mix}_{i_tt_fold}_{i_ttrinit}")

                        shared_param = dict(
                            outpath=outpath_base / f'torchtt_{hypparm_criterion}_{hypparm_criterion_time}' / modelname,
                            cpar_path=cpar_path,
                            model_type=model_type,
                            niters=niters,
                        )
                        tt_spec_ = dict(
                            imix=i_tt_mix,
                            ittfold=i_tt_fold,
                            icvfold=0,
                            icvrinit=i_ttrinit,
                            trainset=stimid_tt_fold['train'],
                            testset=stimid_tt_fold['test'],
                            seed=seed_this_,
                        )
                        model_spec = {**shared_param, **selected_param, **tt_spec_}

                        cfg = assemble_cfg(model_spec)

                        modelspecs_tt_list.append(dict(
                            model_spec=model_spec,
                            cfg_=cfg,
                        ))

                        jobs_tt_list.append(cfg)

    # %%

    len(jobs_tt_list)
    #############################################
    for modelname in model_names:
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            if modelname == 'caaFull' and hypparm_criterion == 'hypc_absintens_bypotoutcome':
                pass
            else:
                selected_res_log[modelname][hypparm_criterion] = pd.concat(selected_res_log[modelname][hypparm_criterion], axis=1).T

    with open(outpath_base / f'modelspecs-tt_nmix{n_tt_mixes}-nfold{n_tt_folds}-nritt{n_tt_random_reinits}-nricv{n_cv_random_reinits}.pkl', 'wb') as f:
        pickle.dump(modelspecs_tt_list, f, protocol=-5)

    with open(outpath_base / f'torch_cvres_cache-nmix{n_tt_mixes}-nfold{n_tt_folds}-nricv{n_cv_random_reinits}.pkl', 'wb') as f:
        pickle.dump(selected_res_log, f, protocol=-5)

    # %%

    #######
    ### launch tt jobs
    #######
    print(f"launching {len(jobs_tt_list)} jobs")

    with open(cpar_path_full, 'rb') as f:
        cpar = dill.load(f)
    codedir_path = cpar.paths['code']
    dataoutbase_path = cpar.paths['dataOutBase']

    batch_every = 1000
    multithread_every = 2
    jobs_array = list()
    dependencies_out = list()
    while len(jobs_tt_list):
        jobgroup = list()
        while len(jobgroup) < multithread_every and len(jobs_tt_list):
            jobgroup.append(jobs_tt_list.pop(0))
        jobs_array.append(jobgroup)
        if len(jobs_array) == batch_every or len(jobs_tt_list) == 0:
            sbatch_out = sbatch_torch_array(
                jobs_array, behavior='optimize', codedir_path=codedir_path, dataoutbase_path=dataoutbase_path, dependency=dependency,
                job_name=f'camTorchTT',
                mem_per_job=9,
                time=4,
                exclude=None,
                partition=['use-everything', None, 'gablab'][0])
            dependencies_out.append(sbatch_out['dependency'])
            jobs_array = list()

    # %%
    return dependencies_out
    # %%


def relaunch_jobs_for_missing_tt_results(modelspecs_tt_path=None, dependency=None, run_missing=False):
    """
    find missing tt results, launch those jobs again
    there should be no running tt jobs
    """
    from pathlib import Path
    import pickle
    import dill
    import random

    with open(modelspecs_tt_path, 'rb') as f:
        modelspecs_tt_list = pickle.load(f)

    print(f"staring tt res scan")
    cfg_missing_list_started = list()
    cfg_missing_list_notstarted = list()
    multipletemp = list()
    for tmspec in modelspecs_tt_list:
        model_spec = tmspec['model_spec']
        resdir = Path(tmspec['cfg_']['dout_base_path']) / f"PTM-lasso-mDsP__k-{model_spec['logit_k']}_laplace_scale-{model_spec['laplace_scale']}"
        resfiles = list(resdir.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj.dill"))
        if len(resfiles) > 1:
            multipletemp.append(resfiles)
        if len(resfiles) == 0:
            naivefiles = list(resdir.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj-naive.dill"))
            if len(naivefiles) > 0:
                cfg_missing_list_started.append(assemble_cfg(model_spec))
            else:
                cfg_missing_list_notstarted.append(assemble_cfg(model_spec))

    random.shuffle(cfg_missing_list_started)
    random.shuffle(cfg_missing_list_notstarted)
    cfg_missing_list = [*cfg_missing_list_notstarted, *cfg_missing_list_started]

    print(f"found {len(multipletemp)} trajectories with multiple results")

    nmissing = len(cfg_missing_list)
    print(f"missing {nmissing} torch results of {len(modelspecs_tt_list)} ({len(modelspecs_tt_list) - nmissing} found)")

    dependencies_out = list()
    if cfg_missing_list and run_missing:
        print(f"launching {len(cfg_missing_list)} jobs")

        with open(modelspecs_tt_list[0]['cfg_']['_cpar_path'], 'rb') as f:
            cpar = dill.load(f)
        codedir_path = cpar.paths['code']
        dataoutbase_path = cpar.paths['dataOutBase']

        batch_every = 1000
        multithread_every = 2
        jobs_array = list()
        while len(cfg_missing_list):
            jobgroup = list()
            while len(jobgroup) < multithread_every and len(cfg_missing_list):
                jobgroup.append(cfg_missing_list.pop(0))
            jobs_array.append(jobgroup)
            if len(jobs_array) == batch_every or len(cfg_missing_list) == 0:
                sbatch_out = sbatch_torch_array(
                    jobs_array, behavior='optimize', codedir_path=codedir_path, dataoutbase_path=dataoutbase_path, dependency=dependency,
                    job_name='camTorchTT',
                    mem_per_job=9,
                    time=4,
                    exclude=None,
                    partition=['use-everything', None, 'gablab'][1])
                dependencies_out.append(sbatch_out['dependency'])
                jobs_array = list()

    # %%
    return nmissing, dependencies_out
    # %%


def cleanup_torch_runs(modelspecs_path=None):

    from pathlib import Path
    import pickle
    import itertools
    import shutil
    import numpy as np

    with open(modelspecs_path, 'rb') as f:
        modelspecs_cv_list = pickle.load(f)

    del_extra = list()
    del_incomplete = list()
    kept_res = list()
    for tmspec in modelspecs_cv_list:
        model_spec = tmspec['model_spec']
        resdir = Path(tmspec['cfg_']['dout_base_path']) / f"PTM-lasso-mDsP__k-{model_spec['logit_k']}_laplace_scale-{model_spec['laplace_scale']}"

        resfiles = list(resdir.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj.dill"))

        assert len(resfiles) > 0, f"no results found for {resdir}"

        if len(resfiles) > 1:
            resmultiple = dict(e5=list(), e5secondary=list(), gold=list(), amd=list(), other=list())
            for eto_path in resfiles:
                eto = EmoTorch(verbose=False)
                eto.load(eto_path)
                host_ = eto.optimized['host']
                if host_ in [f"node{i:03}" for i in np.arange(31, 78)]:
                    resmultiple['e5'].append(eto_path)
                elif host_ in ["dgx001", "dgx002", "node017"]:
                    resmultiple['e5secondary'].append(eto_path)
                elif host_ in [f"node{i:03}" for i in np.arange(78, 100)]:
                    resmultiple['gold'].append(eto_path)
                elif host_ in [f"node{i:03}" for i in np.arange(100, 117)]:
                    resmultiple['amd'].append(eto_path)
                else:
                    resmultiple['other'].append(eto_path)
                    print(f"unknown host {host_}")

            res_ordered = list(itertools.chain.from_iterable(resmultiple.values()))
            res_kept = res_ordered.pop(0)
            del_extra.extend(res_ordered)

        else:
            res_kept = resfiles[0]

        eto = EmoTorch(verbose=False)
        eto.load(res_kept)
        assert eto.optimized['op_dict']['A'].abs().sum().item() > 0.0
        kept_res.append(res_kept)

        naivefiles = list(resdir.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj-naive.dill"))

        for eto_path in naivefiles:
            if not (eto_path.parent / "EmoTorchObj.dill").is_file():
                del_incomplete.append(eto_path)

    if len(del_extra) > 0 or len(del_incomplete) > 0:
        print(f"Found {len(del_extra)} duplicate results")
        print(f"Found {len(del_incomplete)} incomplete results")

        res_dirs_temp = [str(eto_path.parent) for eto_path in kept_res]

        for eto_path in kept_res:
            assert str(eto_path.parent) in res_dirs_temp
        for eto_path in del_extra:
            assert str(eto_path.parent) not in res_dirs_temp
        for eto_path in del_incomplete:
            assert str(eto_path.parent) not in res_dirs_temp

        # %%

        print(f"Deleting {len(del_extra)} duplicate results")
        for eto_path in del_extra:
            shutil.rmtree(eto_path.parent)
        print(f"Deleting {len(del_incomplete)} incomplete results")
        for eto_path in del_incomplete:
            shutil.rmtree(eto_path.parent)
