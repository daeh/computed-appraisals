#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyscript.py
"""

import os
import sys
import argparse

try:
    if __IPYTHON__:  # type: ignore
        get_ipython().run_line_magic('matplotlib', 'inline')  # type: ignore
        get_ipython().run_line_magic('load_ext', 'autoreload')  # type: ignore
        get_ipython().run_line_magic('autoreload', '2')  # type: ignore
        isInteractive = True
except NameError:
    isInteractive = False


# %%


def sbatch_torch_array(job_array, behavior=None, job_name=None, use_everything=False, dependency=None):

    import numpy as np
    from pathlib import Path
    import dill
    import subprocess
    import re
    from pprint import pprint

    assert not (behavior is None), "behavior is required"

    data_cache_dir = Path("/om2/user/daeda/iaa_dataout/_sbatch_dumps_")
    if not data_cache_dir.is_dir():
        data_cache_dir.mkdir(parents=True, exist_ok=True)

    if job_name is None:
        job_name = "iaaTorchDefault"

    mem_per_job = 8

    pickle_dir_path = None
    file_unique = False
    while not file_unique:
        pickle_dir_path = data_cache_dir / f'{job_name}_-_{randomStringDigits()}'
        if not pickle_dir_path.exists():
            file_unique = True

    pickle_dir_path.mkdir(parents=True, exist_ok=True)
    pickle_path = pickle_dir_path / 'all_data.pkl'

    log_dir_path = pickle_dir_path / 'logs'
    log_dir_path.mkdir(parents=True, exist_ok=True)

    output_pattern = log_dir_path / f"{job_name}_%A_%a.txt"

    with open(pickle_path, 'wb') as f:
        dill.dump(job_array, f, protocol=-4)

    script_path = Path("/om/user/daeda/ite_iaa/ite_gb_inverseappraisal/code") / "launch_iaa21_pytorch_torchdata_wrapper.sbatch"

    cmd_list = ["sbatch", f"--array=1-{len(job_array)}", f"--mem={mem_per_job}GB", f"--job-name={job_name}", f"--output={str(output_pattern)}"]
    if use_everything:
        cmd_list.append("--partition=use-everything")
    if dependency is not None:
        cmd_list.append(f"--dependency=afterok:{dependency}")
    cmd_list.extend([str(script_path), str(pickle_path), behavior])

    clout = subprocess.run(cmd_list, capture_output=True, encoding='utf-8')

    error = False
    try:
        depend = re.search(r'([0-9]+)', clout.stdout.strip()).group(0)
        # depend_text_file = pickle_dir_path / f"{depend}.txt"
        # depend_text_file.write_text(f"{' '.join(cmd_list)}")
    except AttributeError:
        error = True
    finally:
        print(clout)
        print(clout.returncode)
        print(' '.join(clout.args))
        print(clout.stdout)
        print(clout.stderr)
    if error:
        raise ValueError("sbatch_torch_array(): failed to get job id")

    return dict(dependency=depend, clout=clout, cmd=' '.join(cmd_list))

##### util functions


def randomStringDigits(stringLength=8):
    """Generate a random string of letters and digits """
    import string
    import random
    lettersAndDigits = string.ascii_lowercase + string.digits
    return ''.join(random.choice(lettersAndDigits) for _ in range(stringLength))


def gen_seed():
    import random
    max_int = 2**31 - 1
    return random.randint(0, max_int - int(1e3))


##### data prep fun

def prospect_transform(X_, prospect_transform_param):
    """
    apply prospect_transform_param['base_kwargs'] to base features
    apply prospect_transform_param['repu_kwargs'] to reputation features
    apply prospect_transform_param['base_kwargs'] to PEa2pot (which scales linearly with pot size)
    do not apply any transform to 'PEa2lnpot', 'PEa2raw', 'PEa2unval'. 
    => 'PEa2lnpot' is already log1p transformed
    => 'PEa2raw', 'PEa2unval' are not scaled by pot size
    """
    import re
    prospect_transform_kwargs_dict_ = dict()
    for col in X_.columns.to_list():
        if bool(re.match(r"\S+\[b\S+\]", col)):  # base
            prospect_transform_kwargs_dict_[col] = prospect_transform_param['base_kwargs']
        elif bool(re.match(r"\S+\[r\S+\]", col)):  # repu
            prospect_transform_kwargs_dict_[col] = prospect_transform_param['repu_kwargs']
        elif col in ['PEa2pot']:
            prospect_transform_kwargs_dict_[col] = prospect_transform_param['base_kwargs']
        elif col in ['PEa2lnpot', 'PEa2raw', 'PEa2unval']:
            prospect_transform_kwargs_dict_[col] = None
        elif bool(re.match(r"^(pot|outcome)$", col)):  # skip
            prospect_transform_kwargs_dict_[col] = None
        else:
            assert False, f"prospect_transform_fn_(): {col} mismatch in {X_.columns.to_list()}"
    return prospect_transform_kwargs_dict_


def scale_iavariables(X_, scale_param):
    """
    scale base, reputation, and \pi_{a_2} features by scale_param['all'] (e.g. unit variance, mean kept)
    """
    import re

    scale_kwargs_dict_ = dict()
    for col in X_.columns.to_list():
        if bool(re.match(r"\S+\[b\S+\]", col)):  # base
            scale_kwargs_dict_[col] = scale_param['all']
        elif bool(re.match(r"\S+\[r\S+\]", col)):  # repu
            scale_kwargs_dict_[col] = scale_param['all']
        elif col in ['PEa2lnpot', 'PEa2pot', 'PEa2raw', 'PEa2unval']:
            scale_kwargs_dict_[col] = scale_param['all']
        elif bool(re.match(r"^(pot|outcome)$", col)):  # skip
            scale_kwargs_dict_[col] = None
        else:
            assert False, f"scale_features_1(): {col} mismatch in {X_.columns.to_list()}"
    return scale_kwargs_dict_


def scale_iavariables_butnot_pia2(X_, scale_param):
    """
    scale base and reputation features by scale_param['all'] (e.g. unit variance, mean kept)
    do not scale \pi_{a_2}
    """
    import re

    scale_kwargs_dict_ = dict()
    for col in X_.columns.to_list():
        if bool(re.match(r"\S+\[b\S+\]", col)):  # base
            scale_kwargs_dict_[col] = scale_param['all']
        elif bool(re.match(r"\S+\[r\S+\]", col)):  # repu
            scale_kwargs_dict_[col] = scale_param['all']
        elif col in ['PEa2lnpot', 'PEa2pot', 'PEa2raw', 'PEa2unval']:
            scale_kwargs_dict_[col] = None
        elif bool(re.match(r"^(pot|outcome)$", col)):  # skip
            scale_kwargs_dict_[col] = None
        else:
            assert False, f"scale_features_2(): {col} mismatch in {X_.columns.to_list()}"
    return scale_kwargs_dict_


"""
cfg specifies [data_prep_label, data_prep_fn]
for 'FullData', prep_model_data() is used
calls main_optimize(cfg)

main_optimize()
calls EmoStan.prep_data(cfg, ...)
calls EmoStan.optimize()

EmoStan.prep_data() 
calls gen_data(): cached_data_in = gen_data(cfg['cpar'])
    this loads the raw webppl data
calls cfg['data_prep_fn'] --> prep_model_data()
    data_dict = cfg['data_prep_fn'](data_generated)
    self.dim_param = data_dict['dim_param']
    self.dataprep_cfg_strs = data_dict['cfg_strs']
calls cache_data() to save stan_data to "/om2/user/daeda/iaa_dataout/_shared_cmdstan_cache_"

gen_data()
receives cpar
calls get_ppldata_cpar() from react_collect_pytorch_cvresults
calls model_assembler() from webpypl_emotionfunction_crossvalidation.py

prep_model_data()
receives data_in, which includes ['X_train_sets']
calls format_data_stan()
returns data_dict, which includes [data, dim_param]

format_data_stan()
calls transform_data_multifunction(index_pad=1, _thin_=...) from webpypl.py
transforms the X and Y data
returns [data_stan, data_transform, dim_param]

"""

#####


def prep_model_data_allplayers(ppldatasets, cpar=None, seed=None):
    """
    train on all players (generic and specific), limited pots
    test on all specific
    """

    import numpy as np
    import pandas as pd
    from webpypl import transform_data_multifunction
    from copy import deepcopy

    ####

    '''
    split generic data into 
    testpots / nontestpots
    
    split specific data into
    train/test
    cvtrain/cvtest
    '''

    prospect_transform_label, prospect_transform_param = cpar.pytorch_spec['prospect_param']
    prospect_transform_fn = prospect_transform

    scale_transform_label = cpar.pytorch_spec['scale_transform_label']
    scale_transform_suffix, scale_transform_param = cpar.pytorch_spec['whitening']
    if scale_transform_label == 'ScalePEa2raw':
        scale_transform_fn = scale_iavariables
    elif scale_transform_label == 'NoscalePEa2raw':
        scale_transform_fn = scale_iavariables_butnot_pia2

    dataprep_kwargs = {
        'thin': 8,
        'pre_opt_y_affine': None,
        'scale_param': {'fn': scale_transform_fn, 'param': {'all': scale_transform_param}},
        'prospect_transform_param': {
            'fn': prospect_transform_fn,
            'param': {
                'base_kwargs': prospect_transform_param['base_kwargs'],
                'repu_kwargs': prospect_transform_param['repu_kwargs'], }, }, }

    # n_tt_test = 0  # <10 test, 10 train>
    # n_cv_test = 20  # <5 cv test, 5 cv train>

    import time
    seed_inherited = True
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
        seed_inherited = False

    # ppldatasets.keys()

    X_generic = ppldatasets['generic']['X']
    Y_generic = ppldatasets['generic']['Y']

    generic_pots = sorted(X_generic['pot'].unique().tolist())
    specific_pots = sorted(ppldatasets['239_1']['X']['pot'].unique().tolist())
    assert len(specific_pots) == 8

    specific_set_ids = list()
    for stimid in ppldatasets.keys():
        if stimid != 'generic':
            specific_set_ids.append(stimid)
    specific_set_ids = sorted(specific_set_ids)

    stimidx = deepcopy(specific_set_ids)
    # rng = np.random.default_rng(seed)
    # rng.shuffle(stimidx)
    stimidx_tt_test = []
    stimidx_tt_train = []
    stimidx_cv_test = stimidx
    stimidx_cv_train = stimidx

    ##################### fit x transform #####################

    fit_scale_transform_x_data = list()
    fit_scale_transform_x_data.append(X_generic.loc[X_generic['pot'].isin(specific_pots), :])
    for stimid in stimidx_cv_train:
        fit_scale_transform_x_data.append(ppldatasets[stimid]['X'])
    data_transform = transform_data_multifunction(index_pad=0, _thin_=dataprep_kwargs['thin'], affine_intercept_=dataprep_kwargs['pre_opt_y_affine'], scale_param_=dataprep_kwargs['scale_param'], prospect_transform_param_=dataprep_kwargs['prospect_transform_param'], verbose=False)
    data_transform.fit_X_transform([pd.concat(fit_scale_transform_x_data)])

    ##################### CV train data #####################

    data_cv_train = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(specific_pots), :])
    data_cv_train['generic'] = dfs_

    for stimid in stimidx_cv_train:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_train[stimid] = dfs_

    ##################### CV test data #####################

    data_cv_test = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(specific_pots), :])
    data_cv_test['generic'] = dfs_

    for stimid in stimidx_cv_test:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_test[stimid] = dfs_

    ##################### make torch formatted data #####################

    for data_cv_ in [data_cv_train, data_cv_test]:
        for stimid, dfs_ in data_cv_.items():

            outcomes = ['CC', 'CD', 'DC', 'DD']
            pots_ = sorted(dfs_['x_pot_col'].unique().tolist())

            Yshort_ = dict()
            X_ = np.full([len(outcomes), len(pots_), np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < 4 outcome, . pot, 20 emotion>
            for i_outcome, outcome in enumerate(outcomes):
                Yshort_[outcome] = list()
                for i_pot, pot in enumerate(pots_):
                    idxer = ((dfs_['x_outcome_col'] == outcome) & (dfs_['x_pot_col'] == pot)).to_numpy()
                    X_[i_outcome, i_pot, :, :] = dfs_['X'][idxer, :]

                    idyer = ((dfs_['Ydf']['outcome'] == outcome) & (dfs_['Ydf']['pot'] == pot)).to_numpy()
                    Yshort_[outcome].append(dfs_['Y'][idyer, :])

            Xdims = dict(
                outcome=outcomes,
                pot=pots_,
                sample=np.unique(dfs_['Jx_sample']),
                iaf=dfs_['Xdf'].columns.drop(['pot', 'outcome']).tolist(),
            )
            Ydims = dict(
                outcome=outcomes,
                pot=pots_,
                response='variable',
                emotion=dfs_['Ydf'].columns.drop(['pot', 'outcome']).tolist(),
            )

            jx_map_list = list()
            jy_map_list = list()
            Xlong_ = np.full([np.unique(dfs_['Jx']).size, np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < . pot-outcome, 20 emotion>
            for jx in range(len(np.unique(dfs_['Jx']))):

                Xlong_[jx, :, :] = dfs_['X'][dfs_['Jx'] == jx, :]

                jx_map_list.append([np.array([jx]), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'pot'].unique(), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'outcome'].unique()])

                jy_map_list.append([np.array([jx]), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'pot'].unique(), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'outcome'].unique()])

            assert not np.any(np.isnan(X_))
            assert not np.any(np.isnan(Xlong_))

            data_cv_[stimid]['Xshort'] = X_
            data_cv_[stimid]['Xshortdims'] = Xdims
            data_cv_[stimid]['Yshort'] = Yshort_
            data_cv_[stimid]['Yshortdims'] = Ydims
            data_cv_[stimid]['Xlong'] = Xlong_
            data_cv_[stimid]['Jxmap'] = pd.DataFrame(np.concatenate(jx_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])
            data_cv_[stimid]['Jymap'] = pd.DataFrame(np.concatenate(jy_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])

    torchdata_cv_train = dict()
    for stimid, dfs_ in data_cv_train.items():
        torchdata_cv_train[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    torchdata_cv_test = dict()
    for stimid, dfs_ in data_cv_test.items():
        torchdata_cv_test[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    ############################# TT train data #####################
    # X_tt_test_dict['generic'] = X_generic.loc[~X_generic['pot'].isin(train_pots), :]
    # Y_tt_test_dict['generic'] = Y_generic.loc[~Y_generic['pot'].isin(train_pots), :]

    """
    right now, Y data stay raw in [0,1] space.
    X are transformed w/ standard scaler, prospect scale, etc. in data_transform
    mu_logit are transformed into mu_logistic with no affine
    a gaussian is placed on each mu_logistic, so while mu_logistic never reaches 0 or 1, the gaussian in which y is scored is unbounded.
    """

    return dict(train=torchdata_cv_train, test=torchdata_cv_test)


def prep_model_data_limitedpots(ppldatasets, cpar=None, seed=None):
    """
    train on generic, limited pots
    test on all specific
    """

    import numpy as np
    import pandas as pd
    from webpypl import transform_data_multifunction
    from copy import deepcopy

    ####

    '''
    split generic data into 
    testpots / nontestpots
    
    split specific data into
    train/test
    cvtrain/cvtest
    '''

    prospect_transform_label, prospect_transform_param = cpar.pytorch_spec['prospect_param']
    prospect_transform_fn = prospect_transform

    scale_transform_label = cpar.pytorch_spec['scale_transform_label']
    scale_transform_suffix, scale_transform_param = cpar.pytorch_spec['whitening']
    if scale_transform_label == 'ScalePEa2raw':
        scale_transform_fn = scale_iavariables
    elif scale_transform_label == 'NoscalePEa2raw':
        scale_transform_fn = scale_iavariables_butnot_pia2

    dataprep_kwargs = {
        'thin': 8,
        'pre_opt_y_affine': None,
        'scale_param': {'fn': scale_transform_fn, 'param': {'all': scale_transform_param}},
        'prospect_transform_param': {
            'fn': prospect_transform_fn,
            'param': {
                'base_kwargs': prospect_transform_param['base_kwargs'],
                'repu_kwargs': prospect_transform_param['repu_kwargs'], }, }, }

    n_tt_test = 0  # <10 test, 10 train>
    n_cv_test = 20  # <5 cv test, 5 cv train>

    import time
    seed_inherited = True
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
        seed_inherited = False

    # ppldatasets.keys()

    X_generic = ppldatasets['generic']['X']
    Y_generic = ppldatasets['generic']['Y']

    generic_pots = sorted(X_generic['pot'].unique().tolist())
    specific_pots = sorted(ppldatasets['239_1']['X']['pot'].unique().tolist())
    assert len(specific_pots) == 8

    specific_set_ids = list()
    for stimid in ppldatasets.keys():
        if stimid != 'generic':
            specific_set_ids.append(stimid)
    specific_set_ids = sorted(specific_set_ids)

    stimidx = deepcopy(specific_set_ids)
    # rng = np.random.default_rng(seed)
    # rng.shuffle(stimidx)
    stimidx_tt_test = stimidx[:n_tt_test]
    stimidx_tt_train = stimidx[n_tt_test:]
    stimidx_cv_test = stimidx_tt_train[:n_cv_test]
    stimidx_cv_train = stimidx_tt_train[n_cv_test:]

    ##################### fit x transform #####################

    fit_scale_transform_x_data = list()
    fit_scale_transform_x_data.append(X_generic.loc[X_generic['pot'].isin(specific_pots), :])

    for stimid in stimidx_cv_train:
        fit_scale_transform_x_data.append(ppldatasets[stimid]['X'])
    data_transform = transform_data_multifunction(index_pad=0, _thin_=dataprep_kwargs['thin'], affine_intercept_=dataprep_kwargs['pre_opt_y_affine'], scale_param_=dataprep_kwargs['scale_param'], prospect_transform_param_=dataprep_kwargs['prospect_transform_param'], verbose=False)
    data_transform.fit_X_transform([pd.concat(fit_scale_transform_x_data)])

    ##################### CV train data #####################

    data_cv_train = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(specific_pots), :])
    data_cv_train['generic'] = dfs_

    for stimid in stimidx_cv_train:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_train[stimid] = dfs_

    ##################### CV test data #####################

    data_cv_test = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(specific_pots), :])
    data_cv_test['generic'] = dfs_

    for stimid in stimidx_cv_test:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_test[stimid] = dfs_

    ##################### make torch formatted data #####################

    for data_cv_ in [data_cv_train, data_cv_test]:
        for stimid, dfs_ in data_cv_.items():

            outcomes = ['CC', 'CD', 'DC', 'DD']
            pots_ = sorted(dfs_['x_pot_col'].unique().tolist())

            Yshort_ = dict()
            X_ = np.full([len(outcomes), len(pots_), np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < 4 outcome, . pot, 20 emotion>
            for i_outcome, outcome in enumerate(outcomes):
                Yshort_[outcome] = list()
                for i_pot, pot in enumerate(pots_):
                    idxer = ((dfs_['x_outcome_col'] == outcome) & (dfs_['x_pot_col'] == pot)).to_numpy()
                    X_[i_outcome, i_pot, :, :] = dfs_['X'][idxer, :]

                    idyer = ((dfs_['Ydf']['outcome'] == outcome) & (dfs_['Ydf']['pot'] == pot)).to_numpy()
                    Yshort_[outcome].append(dfs_['Y'][idyer, :])

            Xdims = dict(
                outcome=outcomes,
                pot=pots_,
                sample=np.unique(dfs_['Jx_sample']),
                iaf=dfs_['Xdf'].columns.drop(['pot', 'outcome']).tolist(),
            )
            Ydims = dict(
                outcome=outcomes,
                pot=pots_,
                response='variable',
                emotion=dfs_['Ydf'].columns.drop(['pot', 'outcome']).tolist(),
            )

            jx_map_list = list()
            jy_map_list = list()
            Xlong_ = np.full([np.unique(dfs_['Jx']).size, np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < . pot-outcome, 20 emotion>
            for jx in range(len(np.unique(dfs_['Jx']))):

                Xlong_[jx, :, :] = dfs_['X'][dfs_['Jx'] == jx, :]

                jx_map_list.append([np.array([jx]), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'pot'].unique(), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'outcome'].unique()])

                jy_map_list.append([np.array([jx]), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'pot'].unique(), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'outcome'].unique()])

            assert not np.any(np.isnan(X_))
            assert not np.any(np.isnan(Xlong_))

            data_cv_[stimid]['Xshort'] = X_
            data_cv_[stimid]['Xshortdims'] = Xdims
            data_cv_[stimid]['Yshort'] = Yshort_
            data_cv_[stimid]['Yshortdims'] = Ydims
            data_cv_[stimid]['Xlong'] = Xlong_
            data_cv_[stimid]['Jxmap'] = pd.DataFrame(np.concatenate(jx_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])
            data_cv_[stimid]['Jymap'] = pd.DataFrame(np.concatenate(jy_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])

    torchdata_cv_train = dict()
    for stimid, dfs_ in data_cv_train.items():
        torchdata_cv_train[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    torchdata_cv_test = dict()
    for stimid, dfs_ in data_cv_test.items():
        torchdata_cv_test[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    ############################# TT train data #####################
    # X_tt_test_dict['generic'] = X_generic.loc[~X_generic['pot'].isin(train_pots), :]
    # Y_tt_test_dict['generic'] = Y_generic.loc[~Y_generic['pot'].isin(train_pots), :]

    """
    right now, Y data stay raw in [0,1] space.
    X are transformed w/ standard scaler, prospect scale, etc. in data_transform
    mu_logit are transformed into mu_logistic with no affine
    a gaussian is placed on each mu_logistic, so while mu_logistic never reaches 0 or 1, the gaussian in which y is scored is unbounded.
    """

    return dict(train=torchdata_cv_train, test=torchdata_cv_test)


def prep_model_data_allpots(ppldatasets, cpar=None, seed=None):
    """
    train on generic, all pots
    test on all specific
    """

    import numpy as np
    import pandas as pd
    from webpypl import transform_data_multifunction
    from copy import deepcopy

    ####

    '''
    split generic data into 
    testpots / nontestpots
    
    split specific data into
    train/test
    cvtrain/cvtest
    '''

    prospect_transform_label, prospect_transform_param = cpar.pytorch_spec['prospect_param']
    prospect_transform_fn = prospect_transform

    scale_transform_label = cpar.pytorch_spec['scale_transform_label']
    scale_transform_suffix, scale_transform_param = cpar.pytorch_spec['whitening']
    if scale_transform_label == 'ScalePEa2raw':
        scale_transform_fn = scale_iavariables
    elif scale_transform_label == 'NoscalePEa2raw':
        scale_transform_fn = scale_iavariables_butnot_pia2

    dataprep_kwargs = {
        'thin': 8,
        'pre_opt_y_affine': None,
        'scale_param': {'fn': scale_transform_fn, 'param': {'all': scale_transform_param}},
        'prospect_transform_param': {
            'fn': prospect_transform_fn,
            'param': {
                'base_kwargs': prospect_transform_param['base_kwargs'],
                'repu_kwargs': prospect_transform_param['repu_kwargs'], }, }, }

    n_tt_test = 0  # <10 test, 10 train>
    n_cv_test = 20  # <5 cv test, 5 cv train>

    import time
    seed_inherited = True
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
        seed_inherited = False

    # ppldatasets.keys()

    X_generic = ppldatasets['generic']['X']
    Y_generic = ppldatasets['generic']['Y']

    generic_pots = sorted(X_generic['pot'].unique().tolist())
    specific_pots = sorted(ppldatasets['239_1']['X']['pot'].unique().tolist())
    assert len(specific_pots) == 8

    specific_set_ids = list()
    for stimid in ppldatasets.keys():
        if stimid != 'generic':
            specific_set_ids.append(stimid)
    specific_set_ids = sorted(specific_set_ids)

    stimidx = deepcopy(specific_set_ids)
    # rng = np.random.default_rng(seed)
    # rng.shuffle(stimidx)
    stimidx_tt_test = stimidx[:n_tt_test]
    stimidx_tt_train = stimidx[n_tt_test:]
    stimidx_cv_test = stimidx_tt_train[:n_cv_test]
    stimidx_cv_train = stimidx_tt_train[n_cv_test:]

    ##################### fit x transform #####################

    fit_scale_transform_x_data = list()
    fit_scale_transform_x_data.append(X_generic)  # all pots
    for stimid in stimidx_cv_train:
        fit_scale_transform_x_data.append(ppldatasets[stimid]['X'])
    data_transform = transform_data_multifunction(index_pad=0, _thin_=dataprep_kwargs['thin'], affine_intercept_=dataprep_kwargs['pre_opt_y_affine'], scale_param_=dataprep_kwargs['scale_param'], prospect_transform_param_=dataprep_kwargs['prospect_transform_param'], verbose=False)
    data_transform.fit_X_transform([pd.concat(fit_scale_transform_x_data)])

    ##################### CV train data #####################

    data_cv_train = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic,
        Y_generic)
    data_cv_train['generic'] = dfs_

    for stimid in stimidx_cv_train:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_train[stimid] = dfs_

    ##################### CV test data #####################

    data_cv_test = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(specific_pots), :])
    data_cv_test['generic'] = dfs_

    for stimid in stimidx_cv_test:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_test[stimid] = dfs_

    ##################### make torch formatted data #####################

    for data_cv_ in [data_cv_train, data_cv_test]:
        for stimid, dfs_ in data_cv_.items():

            outcomes = ['CC', 'CD', 'DC', 'DD']
            pots_ = sorted(dfs_['x_pot_col'].unique().tolist())

            Yshort_ = dict()
            X_ = np.full([len(outcomes), len(pots_), np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < 4 outcome, . pot, 20 emotion>
            for i_outcome, outcome in enumerate(outcomes):
                Yshort_[outcome] = list()
                for i_pot, pot in enumerate(pots_):
                    idxer = ((dfs_['x_outcome_col'] == outcome) & (dfs_['x_pot_col'] == pot)).to_numpy()
                    X_[i_outcome, i_pot, :, :] = dfs_['X'][idxer, :]

                    idyer = ((dfs_['Ydf']['outcome'] == outcome) & (dfs_['Ydf']['pot'] == pot)).to_numpy()
                    Yshort_[outcome].append(dfs_['Y'][idyer, :])

            Xdims = dict(
                outcome=outcomes,
                pot=pots_,
                sample=np.unique(dfs_['Jx_sample']),
                iaf=dfs_['Xdf'].columns.drop(['pot', 'outcome']).tolist(),
            )
            Ydims = dict(
                outcome=outcomes,
                pot=pots_,
                response='variable',
                emotion=dfs_['Ydf'].columns.drop(['pot', 'outcome']).tolist(),
            )

            jx_map_list = list()
            jy_map_list = list()
            Xlong_ = np.full([np.unique(dfs_['Jx']).size, np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < . pot-outcome, 20 emotion>
            for jx in range(len(np.unique(dfs_['Jx']))):

                Xlong_[jx, :, :] = dfs_['X'][dfs_['Jx'] == jx, :]

                jx_map_list.append([np.array([jx]), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'pot'].unique(), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'outcome'].unique()])

                jy_map_list.append([np.array([jx]), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'pot'].unique(), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'outcome'].unique()])

            assert not np.any(np.isnan(X_))
            assert not np.any(np.isnan(Xlong_))

            data_cv_[stimid]['Xshort'] = X_
            data_cv_[stimid]['Xshortdims'] = Xdims
            data_cv_[stimid]['Yshort'] = Yshort_
            data_cv_[stimid]['Yshortdims'] = Ydims
            data_cv_[stimid]['Xlong'] = Xlong_
            data_cv_[stimid]['Jxmap'] = pd.DataFrame(np.concatenate(jx_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])
            data_cv_[stimid]['Jymap'] = pd.DataFrame(np.concatenate(jy_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])

    torchdata_cv_train = dict()
    for stimid, dfs_ in data_cv_train.items():
        torchdata_cv_train[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    torchdata_cv_test = dict()
    for stimid, dfs_ in data_cv_test.items():
        torchdata_cv_test[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    ############################# TT train data #####################
    # X_tt_test_dict['generic'] = X_generic.loc[~X_generic['pot'].isin(train_pots), :]
    # Y_tt_test_dict['generic'] = Y_generic.loc[~Y_generic['pot'].isin(train_pots), :]

    """
    right now, Y data stay raw in [0,1] space.
    X are transformed w/ standard scaler, prospect scale, etc. in data_transform
    mu_logit are transformed into mu_logistic with no affine
    a gaussian is placed on each mu_logistic, so while mu_logistic never reaches 0 or 1, the gaussian in which y is scored is unbounded.
    """

    return dict(train=torchdata_cv_train, test=torchdata_cv_test)


def prep_model_data_midhighpots(ppldatasets, cpar=None, seed=None):
    """
    train on generic, limited pots
    test on all specific
    """

    import numpy as np
    import pandas as pd
    from webpypl import transform_data_multifunction
    from copy import deepcopy

    ####

    '''
    split generic data into 
    testpots / nontestpots
    
    split specific data into
    train/test
    cvtrain/cvtest
    '''

    prospect_transform_label, prospect_transform_param = cpar.pytorch_spec['prospect_param']
    prospect_transform_fn = prospect_transform

    scale_transform_label = cpar.pytorch_spec['scale_transform_label']
    scale_transform_suffix, scale_transform_param = cpar.pytorch_spec['whitening']
    if scale_transform_label == 'ScalePEa2raw':
        scale_transform_fn = scale_iavariables
    elif scale_transform_label == 'NoscalePEa2raw':
        scale_transform_fn = scale_iavariables_butnot_pia2

    dataprep_kwargs = {
        'thin': 8,
        'pre_opt_y_affine': None,
        'scale_param': {'fn': scale_transform_fn, 'param': {'all': scale_transform_param}},
        'prospect_transform_param': {
            'fn': prospect_transform_fn,
            'param': {
                'base_kwargs': prospect_transform_param['base_kwargs'],
                'repu_kwargs': prospect_transform_param['repu_kwargs'], }, }, }

    n_tt_test = 0  # <10 test, 10 train>
    n_cv_test = 20  # <5 cv test, 5 cv train>

    import time
    seed_inherited = True
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
        seed_inherited = False

    # ppldatasets.keys()

    X_generic = ppldatasets['generic']['X']
    Y_generic = ppldatasets['generic']['Y']

    generic_pots = sorted(X_generic['pot'].unique().tolist())
    specific_pots = sorted(ppldatasets['239_1']['X']['pot'].unique().tolist())
    assert len(specific_pots) == 8

    specific_set_ids = list()
    for stimid in ppldatasets.keys():
        if stimid != 'generic':
            specific_set_ids.append(stimid)
    specific_set_ids = sorted(specific_set_ids)

    stimidx = deepcopy(specific_set_ids)
    # rng = np.random.default_rng(seed)
    # rng.shuffle(stimidx)
    stimidx_tt_test = stimidx[:n_tt_test]
    stimidx_tt_train = stimidx[n_tt_test:]
    stimidx_cv_test = stimidx_tt_train[:n_cv_test]
    stimidx_cv_train = stimidx_tt_train[n_cv_test:]

    ##################### fit x transform #####################

    train_pots = generic_pots[8:]

    fit_scale_transform_x_data = list()
    fit_scale_transform_x_data.append(X_generic.loc[X_generic['pot'].isin(train_pots), :])

    for stimid in stimidx_cv_train:
        fit_scale_transform_x_data.append(ppldatasets[stimid]['X'])
    data_transform = transform_data_multifunction(index_pad=0, _thin_=dataprep_kwargs['thin'], affine_intercept_=dataprep_kwargs['pre_opt_y_affine'], scale_param_=dataprep_kwargs['scale_param'], prospect_transform_param_=dataprep_kwargs['prospect_transform_param'], verbose=False)
    data_transform.fit_X_transform([pd.concat(fit_scale_transform_x_data)])

    ##################### CV train data #####################

    data_cv_train = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(train_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(train_pots), :])
    data_cv_train['generic'] = dfs_

    for stimid in stimidx_cv_train:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_train[stimid] = dfs_

    ##################### CV test data #####################

    data_cv_test = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(specific_pots), :])
    data_cv_test['generic'] = dfs_

    for stimid in stimidx_cv_test:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_test[stimid] = dfs_

    ##################### make torch formatted data #####################

    for data_cv_ in [data_cv_train, data_cv_test]:
        for stimid, dfs_ in data_cv_.items():

            outcomes = ['CC', 'CD', 'DC', 'DD']
            pots_ = sorted(dfs_['x_pot_col'].unique().tolist())

            Yshort_ = dict()
            X_ = np.full([len(outcomes), len(pots_), np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < 4 outcome, . pot, 20 emotion>
            for i_outcome, outcome in enumerate(outcomes):
                Yshort_[outcome] = list()
                for i_pot, pot in enumerate(pots_):
                    idxer = ((dfs_['x_outcome_col'] == outcome) & (dfs_['x_pot_col'] == pot)).to_numpy()
                    X_[i_outcome, i_pot, :, :] = dfs_['X'][idxer, :]

                    idyer = ((dfs_['Ydf']['outcome'] == outcome) & (dfs_['Ydf']['pot'] == pot)).to_numpy()
                    Yshort_[outcome].append(dfs_['Y'][idyer, :])

            Xdims = dict(
                outcome=outcomes,
                pot=pots_,
                sample=np.unique(dfs_['Jx_sample']),
                iaf=dfs_['Xdf'].columns.drop(['pot', 'outcome']).tolist(),
            )
            Ydims = dict(
                outcome=outcomes,
                pot=pots_,
                response='variable',
                emotion=dfs_['Ydf'].columns.drop(['pot', 'outcome']).tolist(),
            )

            jx_map_list = list()
            jy_map_list = list()
            Xlong_ = np.full([np.unique(dfs_['Jx']).size, np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < . pot-outcome, 20 emotion>
            for jx in range(len(np.unique(dfs_['Jx']))):

                Xlong_[jx, :, :] = dfs_['X'][dfs_['Jx'] == jx, :]

                jx_map_list.append([np.array([jx]), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'pot'].unique(), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'outcome'].unique()])

                jy_map_list.append([np.array([jx]), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'pot'].unique(), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'outcome'].unique()])

            assert not np.any(np.isnan(X_))
            assert not np.any(np.isnan(Xlong_))

            data_cv_[stimid]['Xshort'] = X_
            data_cv_[stimid]['Xshortdims'] = Xdims
            data_cv_[stimid]['Yshort'] = Yshort_
            data_cv_[stimid]['Yshortdims'] = Ydims
            data_cv_[stimid]['Xlong'] = Xlong_
            data_cv_[stimid]['Jxmap'] = pd.DataFrame(np.concatenate(jx_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])
            data_cv_[stimid]['Jymap'] = pd.DataFrame(np.concatenate(jy_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])

    torchdata_cv_train = dict()
    for stimid, dfs_ in data_cv_train.items():
        torchdata_cv_train[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    torchdata_cv_test = dict()
    for stimid, dfs_ in data_cv_test.items():
        torchdata_cv_test[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    ############################# TT train data #####################
    # X_tt_test_dict['generic'] = X_generic.loc[~X_generic['pot'].isin(train_pots), :]
    # Y_tt_test_dict['generic'] = Y_generic.loc[~Y_generic['pot'].isin(train_pots), :]

    """
    right now, Y data stay raw in [0,1] space.
    X are transformed w/ standard scaler, prospect scale, etc. in data_transform
    mu_logit are transformed into mu_logistic with no affine
    a gaussian is placed on each mu_logistic, so while mu_logistic never reaches 0 or 1, the gaussian in which y is scored is unbounded.
    """

    return dict(train=torchdata_cv_train, test=torchdata_cv_test)


def prep_model_data_cv(ppldatasets, trainstimid=None, teststimid=None, cpar=None):
    """
    train on generic, all pots
    test on specific
    """

    import numpy as np
    import pandas as pd
    from webpypl import transform_data_multifunction
    from copy import deepcopy

    # specific_set_ids = list()
    # for stimid in ppldatasets.keys():
    #     if stimid != 'generic':
    #         specific_set_ids.append(stimid)
    # specific_set_ids = sorted(specific_set_ids)

    ####

    '''
    split generic data into 
    testpots / nontestpots
    
    split specific data into
    train/test
    cvtrain/cvtest
    '''

    prospect_transform_label, prospect_transform_param = cpar.pytorch_spec['prospect_param']
    prospect_transform_fn = prospect_transform

    scale_transform_label = cpar.pytorch_spec['scale_transform_label']
    scale_transform_suffix, scale_transform_param = cpar.pytorch_spec['whitening']
    if scale_transform_label == 'ScalePEa2raw':
        scale_transform_fn = scale_iavariables
    elif scale_transform_label == 'NoscalePEa2raw':
        scale_transform_fn = scale_iavariables_butnot_pia2

    dataprep_kwargs = {
        'thin': 8,
        'pre_opt_y_affine': None,
        'scale_param': {'fn': scale_transform_fn, 'param': {'all': scale_transform_param}},
        'prospect_transform_param': {
            'fn': prospect_transform_fn,
            'param': {
                'base_kwargs': prospect_transform_param['base_kwargs'],
                'repu_kwargs': prospect_transform_param['repu_kwargs'], }, }, }

    # n_tt_test = 5  # <5 test, 15 train>
    # n_cv_test = 5  # <5 cv test, 10 cv train>

    # ppldatasets.keys()

    X_generic = ppldatasets['generic']['X']
    Y_generic = ppldatasets['generic']['Y']

    generic_pots = sorted(X_generic['pot'].unique().tolist())
    specific_pots = sorted(ppldatasets['239_1']['X']['pot'].unique().tolist())
    assert len(specific_pots) == 8

    # stimidx = deepcopy(specific_set_ids)
    # # rng = np.random.default_rng(seed)
    # # rng.shuffle(stimidx)
    # stimidx_tt_test = stimidx[:n_tt_test]
    # stimidx_tt_train = stimidx[n_tt_test:]
    # stimidx_cv_test = stimidx_tt_train[:n_cv_test]
    # stimidx_cv_train = stimidx_tt_train[n_cv_test:]

    stimidx_cv_train = trainstimid
    stimidx_cv_test = teststimid

    ##################### fit x transform #####################

    train_pots = generic_pots[8:]

    fit_scale_transform_x_data = list()
    fit_scale_transform_x_data.append(X_generic.loc[X_generic['pot'].isin(train_pots), :])

    for stimid in stimidx_cv_train:
        fit_scale_transform_x_data.append(ppldatasets[stimid]['X'])
    data_transform = transform_data_multifunction(index_pad=0, _thin_=dataprep_kwargs['thin'], affine_intercept_=dataprep_kwargs['pre_opt_y_affine'], scale_param_=dataprep_kwargs['scale_param'], prospect_transform_param_=dataprep_kwargs['prospect_transform_param'], verbose=False)
    data_transform.fit_X_transform(fit_scale_transform_x_data)

    ##################### CV train data #####################

    data_cv_train = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(train_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(train_pots), :])
    data_cv_train['generic'] = dfs_

    for stimid in stimidx_cv_train:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_train[stimid] = dfs_

    ##################### CV test data #####################

    data_cv_test = dict()
    mats_, dfs_ = data_transform.gen_x_y_pair(
        X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y_generic.loc[Y_generic['pot'].isin(specific_pots), :])
    data_cv_test['generic'] = dfs_

    for stimid in stimidx_cv_test:
        mats_, dfs_ = data_transform.gen_x_y_pair(ppldatasets[stimid]['X'], ppldatasets[stimid]['Y'])
        data_cv_test[stimid] = dfs_

    ##################### make torch formatted data #####################

    for data_cv_ in [data_cv_train, data_cv_test]:
        for stimid, dfs_ in data_cv_.items():

            outcomes = ['CC', 'CD', 'DC', 'DD']
            pots_ = sorted(dfs_['x_pot_col'].unique().tolist())

            Yshort_ = dict()
            X_ = np.full([len(outcomes), len(pots_), np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < 4 outcome, . pot, 20 emotion>
            for i_outcome, outcome in enumerate(outcomes):
                Yshort_[outcome] = list()
                for i_pot, pot in enumerate(pots_):
                    idxer = ((dfs_['x_outcome_col'] == outcome) & (dfs_['x_pot_col'] == pot)).to_numpy()
                    X_[i_outcome, i_pot, :, :] = dfs_['X'][idxer, :]

                    idyer = ((dfs_['Ydf']['outcome'] == outcome) & (dfs_['Ydf']['pot'] == pot)).to_numpy()
                    Yshort_[outcome].append(dfs_['Y'][idyer, :])

            Xdims = dict(
                outcome=outcomes,
                pot=pots_,
                sample=np.unique(dfs_['Jx_sample']),
                iaf=dfs_['Xdf'].columns.drop(['pot', 'outcome']).tolist(),
            )
            Ydims = dict(
                outcome=outcomes,
                pot=pots_,
                response='variable',
                emotion=dfs_['Ydf'].columns.drop(['pot', 'outcome']).tolist(),
            )

            jx_map_list = list()
            jy_map_list = list()
            Xlong_ = np.full([np.unique(dfs_['Jx']).size, np.unique(dfs_['Jx_sample']).size, dfs_['X'].shape[1]], np.nan, dtype=float)  # < . pot-outcome, 20 emotion>
            for jx in range(len(np.unique(dfs_['Jx']))):

                Xlong_[jx, :, :] = dfs_['X'][dfs_['Jx'] == jx, :]

                jx_map_list.append([np.array([jx]), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'pot'].unique(), dfs_['Xdf'].loc[dfs_['Jx'] == jx, 'outcome'].unique()])

                jy_map_list.append([np.array([jx]), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'pot'].unique(), dfs_['Ydf'].loc[dfs_['Jy'] == jx, 'outcome'].unique()])

            assert not np.any(np.isnan(X_))
            assert not np.any(np.isnan(Xlong_))

            data_cv_[stimid]['Xshort'] = X_
            data_cv_[stimid]['Xshortdims'] = Xdims
            data_cv_[stimid]['Yshort'] = Yshort_
            data_cv_[stimid]['Yshortdims'] = Ydims
            data_cv_[stimid]['Xlong'] = Xlong_
            data_cv_[stimid]['Jxmap'] = pd.DataFrame(np.concatenate(jx_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])
            data_cv_[stimid]['Jymap'] = pd.DataFrame(np.concatenate(jy_map_list, axis=1).T, columns=['J', 'pot', 'outcome'])

    torchdata_cv_train = dict()
    for stimid, dfs_ in data_cv_train.items():
        torchdata_cv_train[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    torchdata_cv_test = dict()
    for stimid, dfs_ in data_cv_test.items():
        torchdata_cv_test[stimid] = dict(
            Xshort=dfs_['Xshort'],
            Xshortdims=dfs_['Xshortdims'],
            Yshort=dfs_['Yshort'],
            Yshortdims=dfs_['Yshortdims'],
            Xlong=dfs_['Xlong'],
            Ylong=dfs_['Y'],
            Jy=dfs_['Jy'],
            Jxmap=dfs_['Jxmap'],
            Jymap=dfs_['Jymap'],
        )

    ############################# TT train data #####################
    # X_tt_test_dict['generic'] = X_generic.loc[~X_generic['pot'].isin(train_pots), :]
    # Y_tt_test_dict['generic'] = Y_generic.loc[~Y_generic['pot'].isin(train_pots), :]

    """
    right now, Y data stay raw in [0,1] space.
    X are transformed w/ standard scaler, prospect scale, etc. in data_transform
    mu_logit are transformed into mu_logistic with no affine
    a gaussian is placed on each mu_logistic, so while mu_logistic never reaches 0 or 1, the gaussian in which y is scored is unbounded.
    """

    return dict(train=torchdata_cv_train, test=torchdata_cv_test)


class EmoTorch():

    def __init__(self, verbose=True):

        self.cfg = None

        self.pickle_path = None

        self.dataout_base_path = None
        ###
        self.model_name_brief = None

        self.dim_param = None
        self.data_cache_path = None
        self.dataprep_cfg_strs = None

        self.model_param = None

        self.followup_fn_list = list()

        self.save_minimal = True

        self.verbose = verbose

    def get_torchdata(self):
        import pickle
        with open(self.data_cache_path, 'rb') as f:
            torch_data = pickle.load(f)
        return torch_data

    def dump(self, pickle_path):
        import dill
        self.pickle_path = pickle_path
        objdict = dict()
        for key in ['pickle_path', 'cfg', 'dataout_base_path', 'model_name_brief', 'dim_param', 'data_cache_path', 'dataprep_cfg_strs', 'model_param', 'optimized_cache_path', 'save_minimal']:
            objdict[key] = getattr(self, key)
        objdict['pickle_path'] = pickle_path
        with open(self.pickle_path, 'wb') as f:
            dill.dump(objdict, f, protocol=-4)
        print(f'dumped to {self.pickle_path}')

    def load(self, pickle_path):
        from pathlib import Path
        import pickle
        import dill
        if isinstance(pickle_path, str):
            pickle_path = Path(pickle_path)

        self.pickle_path = pickle_path

        if self.verbose:
            print(f'loading from {self.pickle_path}\n')
        with open(self.pickle_path, 'rb') as f:
            objdict = dill.load(f)

        for key, val in objdict.items():
            setattr(self, key, val)

        for key in ['optimized_cache_path']:
            if objdict[key] is not None:
                with open(objdict[key], 'rb') as f:
                    val = dill.load(f)
                # with open(val['fitobj_cache_path'], 'rb') as f:
                #     val['fitobj'] = pickle.load(f)
                setattr(self, key.replace('_cache_path', ''), val)
                if self.verbose:
                    print(f">>loaded {key.replace('_cache_path', '')}")

    def init_cfg(self, cfg, prefix=''):

        self.cfg = cfg
        self.model_name_brief = cfg['model_id_brief']
        self.model_param = cfg['model_param']

        ###########################

        cfg_strs = {'cfgstr': 'n'}
        self.dataprep_cfg_strs = cfg_strs

        _dataprep_param_str = "".join([f"_{k_}-{v_}" for k_, v_ in cfg_strs.items()])

        ###########################

        model_param_for_str = self.model_param
        model_param_str = "".join([f"_{k_}-{v_}" for k_, v_ in model_param_for_str.items()])

        prefix = f"{prefix}-" if prefix else prefix
        results_fname_base = f"{prefix}{self.model_name_brief}_{_dataprep_param_str}_{model_param_str}"

        self.dataout_base_path = cfg['dout_base_path'] / results_fname_base

        ###########################

        self.followup_fn_list = cfg.get('followup_fn_list', list())

        print(f"\n\n=== model initalized at {self.dataout_base_path} ===\n")

    def prep_data(self):

        # import numpy as np
        # from pathlib import Path
        from webpypl import cache_data, encode_number
        from react_collect_pytorch_cvresults import get_ppldata_cpar
        from webpypl_emotionfunction_crossvalidation import prep_generic_data_pair_, prep_specific_data_pair_
        from webpypl import getEmpiricalModel_pair_

        cfg = self.cfg
        cpar = cfg['cpar']
        data_prep_label = cfg['data_prep_label']
        shared_cache = cfg['dataincache_base_path']
        trainstimid = cfg['trainset']
        teststimid = cfg['testset']

        ###### get data ########

        ppldata, _, distal_prior_ppldata = get_ppldata_cpar(cpar)

        feature_selector_label, feature_selector = cpar.pytorch_spec['feature_selector']

        nobsdf_template = ppldata['empiricalEmotionJudgments']['nobs'].copy()

        ppldatasets = dict()

        composite_emodict_, composite_iafdict_ = prep_generic_data_pair_(ppldata)
        Y_full, X_full, _ = getEmpiricalModel_pair_(composite_emodict_, composite_iafdict_, feature_selector=feature_selector, return_ev=False)
        ppldatasets['generic'] = dict(
            X=X_full,
            Y=Y_full,
        )

        for stimid, ppldatadistal_ in distal_prior_ppldata.items():
            composite_emodict_, composite_iafdict_ = prep_specific_data_pair_(ppldatadistal_, nobsdf_template)
            Y_full, X_full, _ = getEmpiricalModel_pair_(composite_emodict_, composite_iafdict_, feature_selector=feature_selector, return_ev=False)
            ppldatasets[stimid] = dict(
                X=X_full,
                Y=Y_full,
            )

        print(f"generating data with >> {data_prep_label} <<")

        seed = None
        if data_prep_label == 'cv':
            data_dict = prep_model_data_cv(ppldatasets, trainstimid=trainstimid, teststimid=teststimid, cpar=cpar)
        elif data_prep_label == 'limitedpots':
            data_dict = prep_model_data_limitedpots(ppldatasets, cpar=cpar, seed=seed)
        elif data_prep_label == 'trainall':
            data_dict = prep_model_data_allplayers(ppldatasets, cpar=cpar, seed=seed)
        elif data_prep_label == 'allpots':
            data_dict = prep_model_data_allpots(ppldatasets, cpar=cpar, seed=seed)
        elif data_prep_label == 'midhighpots':
            data_dict = prep_model_data_midhighpots(ppldatasets, cpar=cpar, seed=seed)

        self.data_cache_path = cache_data(data_dict, outdir=shared_cache, fname_base='_data_cache', overwrite_existing=False)

    def optimize(self, fit_param_optimize=None, inits=None, trackprogress=None):
        import pickle
        import dill
        import time

        if fit_param_optimize is None:
            fit_param_optimize = dict()

        if trackprogress is None:
            trackprogress = self.cfg.get('trackprogress', True)

        torch_data = self.get_torchdata()
        ###

        trajectory_ = randomStringDigits(4)
        optimization_param_defaults = dict(iter=10, seed=gen_seed())
        optimization_param = {**optimization_param_defaults, **self.cfg.get('_optimize_param', dict()), **fit_param_optimize}
        niter = optimization_param['iter']

        ###

        tj_chain = list()

        res_type_str = 'optimize'
        res_specs_str = f"iter-{niter}_T-{trajectory_}"
        dout_path = self.dataout_base_path / f"{res_type_str}_{res_specs_str}"
        results_pickle_path = dout_path / 'res_op.dill'
        resultsnaive_pickle_path = results_pickle_path.parent / f"{results_pickle_path.with_suffix('').name}-naive.dill"

        tj_chain.append(dict(fn='optimize', tj=trajectory_, pklpath=results_pickle_path))
        res_str_dict = dict(res_type_str=res_type_str, res_specs_str=res_specs_str, trajectory=trajectory_)

        ######

        res = dict(
            kind='adam',
            desc='',
            ###
            model_name=self.model_name_brief,
            # model_hash=self.code_hash,
            # model_code=self.model_code,
            model_param=self.model_param,
            ###
            # op_df=fitobj.optimized_params_pd,
            op_dict=None,
            op_stats=None,
            op_appliedfit=None,
            op_progressdf=None,
            fit_path=dout_path,
            ###
            data_cache_path=self.data_cache_path,
            # fitobj_cache_path=fitobj_cache_path,
            ###
            optimization_param=optimization_param,
            inits=inits,
            res_str_dict=res_str_dict,
            ###
            trainset=self.cfg.get('trainset', None),
            testset=self.cfg.get('testset', None),
            ###
            et=0.0,
            trajectory=trajectory_,
            tj_chain=tj_chain,
            pickle_path=resultsnaive_pickle_path,
        )

        resultsnaive_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        if resultsnaive_pickle_path.is_file():
            resultsnaive_pickle_path.unlink()
        with open(resultsnaive_pickle_path, 'wb') as f:
            dill.dump(res, f, protocol=-5)
        self.optimized_cache_path = resultsnaive_pickle_path
        self.dump(results_pickle_path.parent / 'EmoTorchObj-naive.dill')

        ###

        ###
        print(f"---\nstarting optimization of {results_pickle_path}\n--fit param--vvv\n")
        print(optimization_param)
        print('\n--fit param--^^^')
        t0 = time.perf_counter()

        # from iaa21_pytorch_test import run
        # from iaa21_pytorch_horseshoe import run
        from iaa21_pytorch_lasso_meandatalp_sumparamlp_crossval import run

        model_results = run(datatrain=torch_data['train'], datatest=torch_data['test'], model_param=self.model_param, optimization_param=optimization_param, trackprogress=trackprogress, outpath=dout_path / 'figs')

        """
        model_results = run(datatrain=torch_data['train'], datatest=torch_data['test'], model_param=eto.model_param, optimization_param=dict(iter=50, seed=500), trackprogress=False, outpath=Path('/om2/user/daeda/iaa_dataout/ds21rs2_invpT-log_empiricalExpectation_psrefNorm_multivarkdemixture_kde-0.010_mix-40/torchtt/rapid') / 'figs')
        """

        # final_res = dict(
        #     learned_param=learned_param,
        #     stats=res_stats_final,
        #     appliedfit=res_data_final,
        #     progressdf=res_stats_by_iter_df,
        # )

        elapsed_time = time.perf_counter() - t0
        print(f'Run finished, et: {elapsed_time}')

        save_appliedfit_data = True

        res['pickle_path'] = results_pickle_path
        res['op_dict'] = model_results.get('learned_param', None)
        res['op_stats'] = model_results.get('stats', None)
        if save_appliedfit_data:
            res['op_appliedfit'] = model_results.get('appliedfit', None)
        res['op_progressdf'] = model_results.get('progressdf', None)
        # res['op_apply_fn'] = apply_fit_fn
        res['et'] = elapsed_time

        results_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        if results_pickle_path.is_file():
            results_pickle_path.unlink()
        with open(results_pickle_path, 'wb') as f:
            dill.dump(res, f, protocol=-5)
        self.optimized_cache_path = results_pickle_path
        self.optimized = res
        self.dump(results_pickle_path.parent / 'EmoTorchObj.dill')

    # def applyfit(self):

    #     torch_data = self.get_torchdata()

    #     # model_results = run(datatrain=torch_data['train'], datatest=torch_data['test'], model_param=self.model_param, optimization_param=optimization_param, trackprogress=trackprogress, outpath=dout_path / 'figs')
    #     from iaa21_pytorch_lasso_meandatalp_sumparamlp_crossval import apply_fit
    #     apply_fit(self.optimized['op_dict'], torch_data['test'])


#####################


def load_cfg(picklepath=None, jobnum=None):
    from pathlib import Path
    import dill
    from pprint import pprint

    job_num = jobnum - 1
    cfg_pickle_path = Path(picklepath)
    with open(cfg_pickle_path, 'rb') as f:
        cfg_pickle = dill.load(f)

    print(f"cfg_pickle length: {len(cfg_pickle)}")
    print(f"job_num: {job_num}")
    assert job_num >= 0 and job_num < len(cfg_pickle), f"job_num: {job_num}, len(cfg_pickle): {len(cfg_pickle)}"

    cfg = cfg_pickle[job_num]

    print(f"loaded #{job_num} from {cfg_pickle_path.name} at {cfg_pickle_path}")

    print('\n\nvvvvvvvv cfg -------------\n\n')
    pprint(cfg)
    print('\n\n^^^^^^^^ cfg -------------\n\n')

    return cfg


def main_optimize(cfg):
    eso = EmoTorch()

    prefix = 'PTM'
    eso.init_cfg(cfg, prefix=prefix)
    eso.prep_data()
    eso.optimize()


def main(**kwargs):
    from pathlib import Path
    import pickle

    behavior = kwargs.pop('behavior')
    picklepath = Path(kwargs['picklepath']) if isinstance(kwargs['picklepath'], str) else kwargs['picklepath']
    assert picklepath.is_file()
    jobnum = kwargs['jobnum']

    started_dir = picklepath.parent / 'started'
    finished_dir = picklepath.parent / 'finished'
    error_dir = picklepath.parent / 'error'

    started_dir.mkdir(parents=True, exist_ok=True)

    sbatch_started_path = started_dir / f"run-{jobnum}.pkl"

    with open(sbatch_started_path, 'wb') as f:
        pickle.dump(kwargs, f, protocol=-5)

    try:
        if behavior == 'optimize':

            cfg = load_cfg(picklepath=picklepath, jobnum=jobnum)

            main_optimize(cfg)

    except Exception as e:
        error_dir.mkdir(parents=True, exist_ok=True)
        with open(error_dir / f"run-{jobnum}.pkl", 'wb') as f:
            pickle.dump(kwargs, f, protocol=-5)
        sbatch_started_path.unlink()
        error_text_file = error_dir / f"run-{jobnum}.txt"
        print('ERROR:\n')
        print(e)
        error_text_file.write_text("Exception Occured: \n" + str(e))
        return e
    else:
        finished_dir.mkdir(parents=True, exist_ok=True)
        with open(finished_dir / f"run-{jobnum}.pkl", 'wb') as f:
            pickle.dump(kwargs, f, protocol=-5)
        sbatch_started_path.unlink()
        if not list(started_dir.glob('*')):
            started_dir.rmdir()
        return 0


# %% #####################


def wrapper_run_pytorch(cpar):

    import numpy as np
    from pathlib import Path

    op_feature_selector = {
        'none': r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\])',
        'all': r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2lnpot|PEa2pot|PEa2raw|PEa2unval)',
        'lnpot': r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2lnpot)',
        'pot': r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2pot)',
        'raw': r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2raw)',
        'unvalanced': r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2unval)',
        #####
        'money': r'^(U\[baseMoney\]|PE\[baseMoney\])',
    }

    op_scale_fn = {'ScalePEa2raw': scale_iavariables, 'NoscalePEa2raw': scale_iavariables_butnot_pia2}
    op_whitening = {
        'scaledSDmeanKept': {'with_std': True, 'with_mean': False},
        'scaledSDmeanRemoved': {'with_std': True, 'with_mean': True},
        'noWhitening': {'with_std': False, 'with_mean': False}
    }

    _prospect_transform_kwargs_05 = {'alpha_': 0.5, 'beta_': 0.5, 'lambda_': 1.0, 'intercept': 0.0, 'noise_scale': 0.0}
    _prospect_transform_kwargs_025 = {'alpha_': 0.25, 'beta_': 0.25, 'lambda_': 1.0, 'intercept': 0.0, 'noise_scale': 0.0}
    _prospect_transform_kwargs_log1p = {'log1p': True}

    op_iaf_prospect_scale = {
        'PSdiffRepu': {'base_kwargs': _prospect_transform_kwargs_05, 'repu_kwargs': _prospect_transform_kwargs_025},
        'PSdiffLogRepu': {'base_kwargs': _prospect_transform_kwargs_05, 'repu_kwargs': _prospect_transform_kwargs_log1p},
        'PSeven': {'base_kwargs': _prospect_transform_kwargs_05, 'repu_kwargs': _prospect_transform_kwargs_05},
    }

    ###########

    cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    paths = cpar.paths
    dataincache_shared_path = paths['dataOut'] / f"torch_datain_cache"

    ######## vvvvvvvvvvvvvvv

    baseresults_dir_name = f'pytorch_results'

    behavior = 'optimization'

    run_prefix = '1'

    nreps = 1

    #############

    # data_slice = ['train_on_generic'][0]
    model_type = ['iaf', 'money', 'iafev', 'iafyev'][0]
    whitening_label = 'scaledSDmeanKept'
    pe_a2_scaled = 'lnpot'
    prospect_transform_label = 'PSeven'
    scale_fn_label = 'ScalePEa2raw'
    data_prep_label = ['allpots', 'limitedpots', 'trainall', 'midhighpots'][3]

    #############

    debug_ = False
    # debug_ = True

    iters_list_ = [int(3000)]
    if debug_:
        iters_list_ = [int(200)]

    # reg_type = 'lasso'
    # model_list_ = ['lasso-meandata-sumparam']
    ###
    # reg_type = 'fhs'
    # model_list_ = ['fhs-rollback']
    ###
    reg_type = 'lasso'
    model_list_ = ['lasso-mDsP']

    modelparamsdict = dict(fhs=list(), hs=list(), lasso=list())

    for logit_k in [0.4]:
        for laplace_scale in [160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280., 290., 300.]:
            modelparamsdict['lasso'].append({'k': logit_k, 'laplace_scale': laplace_scale, })

    ######## ^^^^^^^^^^^^^^^^^

    dout_base_dir_path = paths['dataOut'] / baseresults_dir_name

    if model_type == 'money':
        pe_a2_scaled = 'money'

    cpar.pytorch_spec['model_type'] = model_type
    cpar.pytorch_spec['reg_type'] = reg_type
    cpar.pytorch_spec['whitening'] = (whitening_label, op_whitening[whitening_label])
    cpar.pytorch_spec['prospect_param'] = (prospect_transform_label, op_iaf_prospect_scale[prospect_transform_label])
    cpar.pytorch_spec['feature_selector'] = (pe_a2_scaled, op_feature_selector[pe_a2_scaled])
    cpar.pytorch_spec['scale_transform_label'] = scale_fn_label

    # cpar.pytorch_spec['data_slice'] = data_slice
    # cpar.pytorch_spec['prospect_transform_fn_'] = prospect_transform_fn
    # cpar.pytorch_spec['reg_param_list'] = regularization_param
    # prospect_transform_fn = prospect_transform
    # scale_fn = op_scale_fn[scale_fn_label]

    #################

    jobs_list = list()
    for niters_ in iters_list_:
        for model_id_brief in model_list_:

            if model_id_brief.startswith('fhs-'):
                modelparams_ = modelparamsdict['fhs']
            elif model_id_brief.startswith('hs-'):
                modelparams_ = modelparamsdict['hs']
            elif model_id_brief.startswith('lasso'):
                modelparams_ = modelparamsdict['lasso']
            elif model_id_brief.startswith('spikeslab'):
                modelparams_ = modelparamsdict['lasso']
            else:
                modelparams_ = dict()
                raise Exception

            for mp_ in modelparams_:
                for _ in range(nreps):

                    cv_suffix = f'{model_type}-PEpia2{pe_a2_scaled}_{prospect_transform_label}_{data_prep_label}'
                    subrun_dir = f"{run_prefix}-{model_id_brief}_{cv_suffix}_{behavior}"

                    # trajectory_id = randomStringDigits(4)
                    # followup_fn_list = model_assoc[model_id_brief]['followup']

                    ########################################################
                    model_param_shared = dict()

                    fit_param = {'iter': niters_}

                    model_param = {**mp_, **model_param_shared}
                    if debug_:
                        subrun_dir += '_DEBUG_'

                    dout_base_path = dout_base_dir_path / subrun_dir

                    cfg = dict(
                        dout_base_path=dout_base_path,
                        figs_base_path=dout_base_dir_path.parent / 'figs_pytorch' / subrun_dir,
                        dataincache_base_path=dataincache_shared_path / cv_suffix,
                        ###
                        model_id_brief=model_id_brief,
                        ###
                        model_param=model_param,
                        ###
                        _behavior=behavior,
                        _optimize_param=fit_param,
                        ###
                        data_prep_label=data_prep_label,
                        ###
                        save_minimal=True,
                        ####
                        # followup_fn_list=followup_fn_list,
                        ###
                        cpar=cpar,
                        ###
                    )
                    jobs_list.append(cfg)

    print(f'submiting {len(jobs_list)} jobs -- {"DEBUG" if debug_ else "full"}')
    sbatch_torch_array(jobs_list, 'optimize')

    from joblib import Parallel, delayed, cpu_count
    print(f"running {len(jobs_list)} on {cpu_count()} cores")
    with Parallel(n_jobs=min(len(jobs_list), cpu_count())) as pool:
        res = pool(delayed(main_optimize)(cfg_) for cfg_ in jobs_list)
    '''
    res0 = main_optimize(cfg)
    
    sbatch_torch_array(jobs_list, 'optimize')
    '''

    # sbout = sbatch_torch_array_memtest(jobs_list[:2])

    # return sbatch_torch_array(jobs_list, 'optimize')


# %% ######################


# %%

# data pre function preps cfg pickle and saves, submits sbatch job
### main script recieves path to pickle. that's it
### main function loads pickle, picks relevant job


def _cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    # parser.add_argument('-p', '--data-path', default='', type=str, help='model name without ext')
    parser.add_argument('-p', '--picklepath', default='', type=str, help='model name without ext')
    parser.add_argument('-t', '--jobnum', default=0, type=int, help='model name without ext')
    parser.add_argument('-b', '--behavior', default='optimize', type=str, help='model name without ext')
    # parser.add_argument('-i', '--niter', default='', type=int, help='model name without ext')
    qux_help = ("This argument will show its default in the help due to "
                "ArgumentDefaultsHelpFormatter")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    print(f'\n---Received {sys.argv} from shell---\n')
    exit_status = 1

    print(f"CC :: {os.getenv('CC')}")
    print(f"CXX :: {os.getenv('CXX')}")
    print(f"CXXFLAGS :: {os.getenv('CXXFLAGS')}")

    try:
        param_in = _cli()

        print(f'\nparam_in: {param_in}\n\n')

        exit_status = main(**param_in)

    except Exception as e:
        print(f'Got exception of type {type(e)}: {e}')
        print("Not sure what happened, so it's not safe to continue -- crashing the script!")
        sys.exit(1)
    finally:
        print(f"--iaa21_run_pytorch_torchdata() ended with exit code {exit_status}--")

    if exit_status == 0:
        print("--WRAPPER COMPLETED SUCCESSFULLY--")
    else:
        print(f"--SOME ISSUE, EXITING:: {exit_status}--")

    sys.exit(exit_status)
