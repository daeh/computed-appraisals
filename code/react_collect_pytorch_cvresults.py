#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""react_collect_pytorch_cvresults.py
"""

import numpy as np
from iaa21_run_pytorch_torchdata import EmoTorch, sbatch_torch_array


class DataGrabber:
    def __init__(self, data=None, specific_player_stimids=None, outcomes=None, pots=None, emotions=None):
        self.data = data
        self.specific_player_stimids = specific_player_stimids
        self.outcomes = outcomes
        self.pots = pots
        self.emotions = emotions
        #####
        self.hypparm_options = list(self.data.keys())
        self.model_names = list(self.data[self.hypparm_options[0]].keys())
        self.hypparm_criterion = None
        self.modelname = None
        self.evtype = None  # ev_bypotoutcome or deltas_byoutcome
        #####
        # self.datatype = None  # model or empir
        # self.dims = None

    def dump(self, outpath):
        import pickle
        if outpath.is_file():
            outpath.unlink()

        with open(outpath, 'wb') as f:
            pickle.dump(dict(data=self.data, specific_player_stimids=self.specific_player_stimids, outcomes=self.outcomes, pots=self.pots, emotions=self.emotions), f, protocol=-5)

    def set_spec(self, hypparm_criterion=None, modelname=None, evtype=None):
        assert hypparm_criterion in self.hypparm_options, 'hypparm_criterion not in hypparm_options'
        assert modelname in self.model_names
        assert evtype in ['ev_bypotoutcome', 'deltas_byoutcome']
        self.hypparm_criterion = hypparm_criterion
        self.modelname = modelname
        self.evtype = evtype  # ev_bypotoutcome or deltas_byoutcome

    # def report_spec(self):

    def get_data(self, ytype):
        return self.data[self.hypparm_criterion][self.modelname][self.evtype][ytype]

    def get_dims(self):
        data_ = self.get_data('model')

        if self.evtype == 'deltas_byoutcome':
            dims = dict(imix=list(range(data_.shape[0])), stimid=self.specific_player_stimids, outcome=self.outcomes, emotion=self.emotions)
        elif self.evtype == 'ev_bypotoutcome':
            dims = dict(imix=list(range(data_.shape[0])), stimid=self.specific_player_stimids, outcome=self.outcomes, pot=self.pots, emotion=self.emotions)

        return dims

    def get_learnedparam(self):
        return self.data[self.hypparm_criterion][self.modelname]['learned_param']


def load_tt_results(outpath_base, hypparmcriterion, hypparmcriterion_time, model_label, n_mixes, n_folds):
    tt_results = list()
    learned_param = list()
    for i_tt_mix in range(n_mixes):
        eso_paths_ttfolds = list()
        for i_tt_fold in range(n_folds):
            tt_search_path = outpath_base / f'torchtt_{hypparmcriterion}_{hypparmcriterion_time}' / model_label / f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00-lasso-mDsP_optimization"
            assert tt_search_path.is_dir()
            eto_list = list(tt_search_path.glob("PTM*/optimize_iter-*/EmoTorchObj.dill"))
            eso_paths_ttfolds.extend(eto_list)

        for eto_path in eso_paths_ttfolds:
            eto = EmoTorch(verbose=False)
            eto.load(eto_path)

            learned_param.append(dict(imix=i_tt_mix, ifold=i_tt_fold, learned_param=eto.optimized['op_dict']))

            for datatype in ['deltas', 'test_outcomepot']:
                for ytype in ['model', 'empir']:
                    for i_stimid, stimid in enumerate(eto.optimized['op_appliedfit']['test_outcomepot']['dims']['stimid']):
                        tt_results.append(dict(
                            datatype=datatype,
                            ytype=ytype,
                            imix=i_tt_mix,
                            stimid=stimid,
                            data=np.squeeze(eto.optimized['op_appliedfit'][datatype][ytype][i_stimid]),
                            eto_path=eto_path,
                        ))

    return tt_results, learned_param


def import_tt_results(outpath_base, specificplayers_stimids, hypparm_criterion_time, n_tt_mixes, n_tt_folds):

    ### load a random result in order to set the dims ###
    randetopath = list((outpath_base / f'torchtt_{"hypc_deltas_byoutcome"}_{hypparm_criterion_time}' / "iaaFull").glob("mix00_fold00_cv00-lasso-mDsP_optimization/PTM*/optimize_iter-*/EmoTorchObj.dill"))[0]
    eto = EmoTorch(verbose=False)
    eto.load(randetopath)
    outcomes = eto.optimized['op_appliedfit']['test_outcomepot']['dims']['outcome']
    pots = eto.optimized['op_appliedfit']['test_outcomepot']['dims']['pot']
    emotions = eto.optimized['op_appliedfit']['test_outcomepot']['dims']['emotion']

    n_outcomes = len(outcomes)
    n_emotions = len(emotions)
    n_pots = len(pots)

    # hypparm_criterion_time = ['terminal', 'lastquarter'][1]
    datacomp = dict()
    for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
        datacomp[hypparm_criterion] = dict()
        for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
            datacomp[hypparm_criterion][modelname] = dict()
            ttres, learned_param_res = load_tt_results(outpath_base, hypparm_criterion, hypparm_criterion_time, modelname, n_tt_mixes, n_tt_folds)

            ##########

            deltas_model_ = np.full([n_tt_mixes, len(specificplayers_stimids), n_outcomes, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specificplayers_stimids):
                for item in ttres:
                    if item['stimid'] == stimid and item['datatype'] == 'deltas' and item['ytype'] == 'model':
                        # aaa.append(item)
                        assert np.all(np.isnan(deltas_model_[item['imix'], i_stimid, :, :]))
                        deltas_model_[item['imix'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(deltas_model_))

            deltas_empir_ = np.full([n_tt_mixes, len(specificplayers_stimids), n_outcomes, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specificplayers_stimids):
                for item in ttres:
                    if item['stimid'] == stimid and item['datatype'] == 'deltas' and item['ytype'] == 'empir':
                        assert np.all(np.isnan(deltas_empir_[item['imix'], i_stimid, :, :]))
                        deltas_empir_[item['imix'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(deltas_empir_))

            evpotoutcome_model_ = np.full([n_tt_mixes, len(specificplayers_stimids), n_outcomes, n_pots, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specificplayers_stimids):
                for item in ttres:
                    if item['stimid'] == stimid and item['datatype'] == 'test_outcomepot' and item['ytype'] == 'model':
                        assert np.all(np.isnan(evpotoutcome_model_[item['imix'], i_stimid, :, :, :]))
                        evpotoutcome_model_[item['imix'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(evpotoutcome_model_))

            evpotoutcome_empir_ = np.full([n_tt_mixes, len(specificplayers_stimids), n_outcomes, n_pots, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specificplayers_stimids):
                for item in ttres:
                    if item['stimid'] == stimid and item['datatype'] == 'test_outcomepot' and item['ytype'] == 'empir':
                        assert np.all(np.isnan(evpotoutcome_empir_[item['imix'], i_stimid, :, :, :]))
                        evpotoutcome_empir_[item['imix'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(evpotoutcome_empir_))

            # hypparm_criterion
            # modelname
            # evtype
            datacomp[hypparm_criterion][modelname] = dict(
                deltas_byoutcome=dict(
                    model=deltas_model_,
                    empir=deltas_empir_,
                ),
                ev_bypotoutcome=dict(
                    model=evpotoutcome_model_,
                    empir=evpotoutcome_empir_,
                ),
                learned_param=learned_param_res,
            )

    return DataGrabber(data=datacomp, specific_player_stimids=specificplayers_stimids, outcomes=outcomes, pots=pots, emotions=emotions)


def load_tt_results_distabutions(outpath_base, hypparmcriterion, hypparmcriterion_time, model_label, n_mixes, n_folds, ppldata=None, subsample_frac=1):

    import pandas as pd
    from iaa21_pytorch_lasso_meandatalp_sumparamlp_crossval import apply_fit

    assert isinstance(subsample_frac, int), 'subsample_frac must be an integer'

    compiled_emo_pred_list = dict()
    for i_tt_mix in range(n_mixes):
        print(f"loading ttmix {i_tt_mix+1} / {n_mixes}")

        eso_paths_ttfolds = list()
        for i_tt_fold in range(n_folds):
            tt_search_path = outpath_base / f'torchtt_{hypparmcriterion}_{hypparmcriterion_time}' / model_label / f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00-lasso-mDsP_optimization"
            assert tt_search_path.is_dir()
            eto_list = list(tt_search_path.glob("PTM*/optimize_iter-*/EmoTorchObj.dill"))
            eso_paths_ttfolds.extend(eto_list)

        for eto_path in eso_paths_ttfolds:
            eto = EmoTorch(verbose=False)
            eto.load(eto_path)

            torch_data = eto.get_torchdata()
            fit_param = eto.optimized['op_dict']
            datatest = torch_data['test']
            for stimid, data in datatest.items():

                if stimid not in compiled_emo_pred_list:
                    compiled_emo_pred_list[stimid] = dict(CC=list(), CD=list(), DC=list(), DD=list())

                n_emotions = data['Ylong'].shape[1]
                yhat_ev = np.full([len(data['Xshortdims']['outcome']), len(data['Xshortdims']['pot']), n_emotions], np.nan, dtype=float)
                # y_ev = yhat_ev.copy()
                for i_outcome, outcome in enumerate(data['Xshortdims']['outcome']):
                    emodfs_list = list()
                    for i_pot, pot in enumerate(data['Xshortdims']['pot']):
                        yhat = np.squeeze(apply_fit(fit_param, np.expand_dims(data['Xshort'][i_outcome, i_pot, :, :], axis=0)))
                        # yhat_ev[i_outcome, i_pot, :] = np.mean(yhat, axis=0)

                        # y_ev[i_outcome, i_pot, :] = np.mean(data['Yshort'][outcome][i_pot], axis=0)
                        emodf_ = pd.DataFrame(yhat[0::subsample_frac], columns=data['Yshortdims']['emotion'])
                        emodf_['prob'] = emodf_.shape[0] ** -1
                        emodf_['pots'] = pot
                        compiled_emo_pred_list[stimid][outcome].append(emodf_.set_index('pots', inplace=False))

    compiled_emo_pred = dict()
    for stimid in compiled_emo_pred_list:
        composite_emodict = dict()
        composite_emodict['nobs'] = ppldata['empiricalEmotionJudgments']['nobs'].copy(deep=True)
        composite_emodict['nobs'].loc[:, :] = int(0)
        for outcome in compiled_emo_pred_list[stimid]:
            allsamples = pd.concat(compiled_emo_pred_list[stimid][outcome])

            columns_ = list()
            for label in allsamples.columns.to_list():
                if label != 'prob':
                    columns_.append(('emotionIntensities', label))
                else:
                    columns_.append(('prob', 'prob'))

            allsamples.columns = pd.MultiIndex.from_tuples(columns_)
            composite_emodict[outcome] = allsamples

            for pot in np.unique(ppldata['empiricalEmotionJudgments']['nobs'].index.get_level_values(0)):
                if pot in composite_emodict[outcome].index:
                    nobs = composite_emodict[outcome].loc[pot, :].shape[0]
                else:
                    nobs = 0
                composite_emodict['nobs'].loc[pot, outcome] = int(nobs)

        compiled_emo_pred[stimid] = composite_emodict

    return compiled_emo_pred


def gen_cv_datafolds_trainonsomespecific(specific_set_ids, n_tt_mixes=2, n_cv_mixes=2, n_tt_folds=4, n_cv_folds=3, seed=None):

    import numpy as np
    from copy import deepcopy
    from sklearn.model_selection import KFold

    import time
    seed_inherited = True
    if seed is None:
        seed = int(str(int(time.time() * 10**6))[-9:])
        seed_inherited = False

    rng = np.random.default_rng(seed)

    datafolds_mixes = list()
    for i_tt_mix in range(n_tt_mixes):

        stimids_all = np.array(deepcopy(specific_set_ids))

        datafolds_tt = list()
        kf_tt_seed_ = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
        kf_tt = KFold(n_splits=n_tt_folds, random_state=kf_tt_seed_, shuffle=True)
        for tt_train_idx, tt_test_idx in kf_tt.split(stimids_all):
            stimid_tt_test = stimids_all[tt_test_idx]
            stimid_tt_train = stimids_all[tt_train_idx]

            datafolds_cv = list()
            for i_cv_mix in range(n_cv_mixes):
                kf_cv_seed_ = int(rng.integers(low=1, high=np.iinfo(np.int32).max, dtype=int))
                kf_cv = KFold(n_splits=n_cv_folds, random_state=kf_cv_seed_, shuffle=True)
                for cv_train_idx, cv_test_idx in kf_cv.split(stimid_tt_train):
                    stimid_cv_test = stimid_tt_train[cv_test_idx]
                    stimid_cv_train = stimid_tt_train[cv_train_idx]

                    datafolds_cv.append(dict(test=stimid_cv_test, train=stimid_cv_train))

                # assert np.unique(np.concatenate([fold_['test'] for fold_ in datafolds_cv])).size == 15

            datafolds_tt.append(dict(test=stimid_tt_test, train=stimid_tt_train, cv=datafolds_cv))

        assert np.unique(np.concatenate([fold_['test'] for fold_ in datafolds_tt])).size == 20

        datafolds_mixes.append(datafolds_tt)

    return datafolds_mixes


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

    import numpy as np
    from pathlib import Path
    import dill

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

    op_scale_fn = {'ScalePEa2raw': None, 'NoscalePEa2raw': None}
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

    """
    model_spec = dict(
        outpath=outpath_base,
        cpar_path=cpar_path, 
        model_type=model_type, 
        logit_k=logit_k, 
        laplace_scale=laplace_scale,
        niters=niters,
        imix=None,
        ittfold=None,
        icvfold=None,
        trainset=None,
        testset=None,
    )
    """

    with open(model_spec['cpar_path'], 'rb') as f:
        cpar = dill.load(f)
    cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    model_type = model_spec['model_type']
    model_param = {'k': model_spec['logit_k'], 'laplace_scale': model_spec['laplace_scale']}
    fit_param = {'iter': model_spec['niters'], 'seed': model_spec.get('seed', None)}
    outpath = model_spec['outpath']

    # paths = cpar.paths
    dataincache_shared_path = outpath / f"torch_datain_cache"

    ######## vvvvvvvvvvvvvvv

    # baseresults_dir_name = f'torch_cv'

    behavior = 'optimization'

    nreps = 1

    #############

    # data_slice = ['train_on_generic'][0]
    # model_type = ['iaf', 'money', 'iafev', 'iafyev'][0]q
    whitening_label = 'scaledSDmeanKept'
    pe_a2_scaled = 'lnpot'
    prospect_transform_label = 'PSeven'
    scale_fn_label = 'ScalePEa2raw'
    data_prep_label = 'cv'  # ['allpots', 'limitedpots', 'trainall', 'midhighpots'][0]

    #############

    # debug_ = False
    # debug_ = True

    # iters_list_ = [int(3000)]
    # if debug_:
    #     iters_list_ = [int(200)]

    # reg_type = 'lasso'
    # model_list_ = ['lasso-meandata-sumparam']
    ###
    # reg_type = 'fhs'
    # model_list_ = ['fhs-rollback']
    ###
    reg_type = 'lasso'
    torch_model_ = 'lasso-mDsP'

    # modelparamsdict = dict(fhs=list(), hs=list(), lasso=list())

    # for logit_k in [0.4]:
    #     for laplace_scale in [160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280., 290., 300.]:
    #         modelparamsdict['lasso'].append({'k': logit_k, 'laplace_scale': laplace_scale, })

    ######## ^^^^^^^^^^^^^^^^^

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

    # for niters_ in iters_list_:
    #     for model_id_brief in model_list_:

    #         if model_id_brief.startswith('fhs-'):
    #             modelparams_ = modelparamsdict['fhs']
    #         elif model_id_brief.startswith('hs-'):
    #             modelparams_ = modelparamsdict['hs']
    #         elif model_id_brief.startswith('lasso'):
    #             modelparams_ = modelparamsdict['lasso']
    #         elif model_id_brief.startswith('spikeslab'):
    #             modelparams_ = modelparamsdict['lasso']
    #         else:
    #             modelparams_ = dict()
    #             raise Exception

    #         for mp_ in modelparams_:
    #             for _ in range(nreps):

    dout_base_dir_path = outpath

    run_prefix = f"mix{model_spec['imix']:02}_fold{model_spec['ittfold']:02}_cv{model_spec['icvfold']:02}"
    cv_suffix = f'{model_type}-PEpia2{pe_a2_scaled}_{prospect_transform_label}_{data_prep_label}'
    # subrun_dir = f"{run_prefix}-{torch_model_}_{cv_suffix}_{behavior}"
    subrun_dir = f"{run_prefix}-{torch_model_}_{behavior}"

    # trajectory_id = randomStringDigits(4)
    # followup_fn_list = model_assoc[model_id_brief]['followup']

    ########################################################
    # model_param_shared = dict()

    # model_param = {**mp_, **model_param_shared}
    # if debug_:
    #     subrun_dir += '_DEBUG_'

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
        _behavior=behavior,
        _optimize_param=fit_param,
        trackprogress=False,
        ###
        data_prep_label=data_prep_label,
        trainset=model_spec['trainset'],
        testset=model_spec['testset'],
        ###
        save_minimal=True,
        ####
        # followup_fn_list=followup_fn_list,
        ###
        cpar=cpar,
        ###
    )

    return cfg


############### plotting ##############################


def plot_webppl_data(cpar_path, plotParam=None):
    '''
    cfg = cfg_bs_emp
    seed = 1
    n_bootstrap = 100
    '''

    from react_collect_pytorch_cvresults import get_ppldata_cpar
    from iaa21_runwebppl_followup import followup_analyses
    from webpypl_plotfun import init_plot_objects
    import dill

    with open(cpar_path, 'rb') as f:
        cpar = dill.load(f)

    ###### get data ########

    ppldata, ppldata_exp3, distal_prior_ppldata = get_ppldata_cpar(cpar)

    if plotParam is None:
        plotParam = init_plot_objects(cpar.paths['figsOut'] / 'summary')

    followup_analyses(cpar, ppldata, ppldata_exp3, distal_prior_ppldata, plotParam=plotParam)

    #######
    # composite_inverse_planning_split_violin(ppldata, cpar.paths, plotParam)

    print('done printing webppldata')


def run_cv(base_path=None, cpar_path_full=None, cpar_path_priors=None, specificplayers_stimids=None, laplace_scales=None, logit_k=None, niters=None, n_tt_mixes=None, n_tt_folds=None, seed=None, dependency=None):
    """
    """
    # %%

    import numpy as np
    from copy import deepcopy
    import pickle

    assert cpar_path_full.is_file() and cpar_path_priors.is_file()

    """train on some of the specific players, test on leftout specific players"""
    # n_tt_mixes = 2
    # n_cv_mixes = 2
    # n_tt_folds = 5  # train on 16, test on 4
    # n_cv_folds = 2  # train on 8, test on 8
    # datafolds_mixes = gen_cv_datafolds_trainonsomespecific(specificplayers_stimids, n_mixes=n_mixes, tt_folds=tt_folds, cv_folds=cv_folds, seed=datafold_seed)

    """train only on generic, cv on some specific players, test on leftout specific players"""
    '''
    each cv:
        run all hyperparam sets
        train on generic, test on 15 specific players
    pick which hyperparam based on prediction of 15 specific players
    tt:
    use that bandwidth to train on generic, test on leftout 5 specific players
    '''

    rng_datafold = np.random.default_rng(seed)
    datafold_seeds = rng_datafold.integers(low=1, high=np.iinfo(np.int32).max, size=n_tt_mixes, dtype=int)
    datafolds_mixes = list()
    for datafold_seed_ in datafold_seeds:
        datafolds_mixes.append(gen_cv_datafolds_trainongeneric_cvonspecific(specificplayers_stimids, n_tt_folds=n_tt_folds, seed=datafold_seed_))

    outpath_base = base_path / f'torch_results-{n_tt_folds}'
    outpath_base.mkdir(exist_ok=True, parents=True)

    model_param_list = list()
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:

        if modelname == 'socialLesion':
            model_type = 'money'
            laplace_scale_list = laplace_scales
        else:
            model_type = 'iaf'
            laplace_scale_list = laplace_scales

        if modelname == 'invplanLesion':
            cpar_path = cpar_path_priors
        else:
            cpar_path = cpar_path_full

        for laplace_scale in laplace_scale_list:
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
                trainset=None,
                testset=None,
            ))

    # %%
    #################################
    ### launch cv jobs
    #################################

    run_sequentially = False
    mixes_to_run = np.arange(n_tt_mixes)
    # mixes_to_run = np.arange(20, n_tt_mixes)

    # batch_every = 2
    # ibatch = 0
    modelspecs_cv_list = list()
    jobs_list_cv = list()
    for i_tt_mix in np.arange(n_tt_mixes):
        rng_torch = np.random.default_rng(datafold_seeds[i_tt_mix])
        for i_tt_fold in range(n_tt_folds):

            ### assemble models ###
            for i_cv_fold, stimid_cv_fold in enumerate(datafolds_mixes[i_tt_mix][i_tt_fold]['cv']):
                for model_param in model_param_list:
                    cv_spec_ = dict(
                        imix=i_tt_mix,
                        ittfold=i_tt_fold,
                        icvfold=i_cv_fold,
                        trainset=stimid_cv_fold['train'],
                        testset=stimid_cv_fold['test'],
                        seed=int(rng_torch.integers(low=1, high=np.iinfo(np.int32).max, dtype=int)),
                    )
                    model_spec = {**deepcopy(model_param), **cv_spec_}
                    modelspecs_cv_list.append(model_spec)

                    if i_tt_mix in mixes_to_run:
                        cfg = assemble_cfg(model_spec)
                        jobs_list_cv.append(cfg)

    ### write datafold_seed to pickle file ###
    with open(outpath_base / f'datafolds_mixes-{n_tt_mixes}-{n_tt_folds}.pkl', 'wb') as f:
        pickle.dump(datafolds_mixes, f, protocol=-5)

    with open(outpath_base / f'sbatch_modelspecs_cv-{n_tt_mixes}-{n_tt_folds}.pkl', 'wb') as f:
        pickle.dump(modelspecs_cv_list, f, protocol=-5)

    ### submit jobs ###
    '''
    res0 = main_optimize(cfg)
    '''
    sbatch_out = sbatch_torch_array(jobs_list_cv, behavior='optimize', job_name='iaaTorchCV', use_everything=True, dependency=dependency)
    if run_sequentially:
        dependency = sbatch_out['dependency']

    # %%
    return dependency
    # %%


def relaunch_jobs_for_missing_results(outpath_base=None, n_tt_mixes=None, n_tt_folds=None, dependency=None, run_missing=False):

    import pickle

    ### find missing cv results, lauch those jobs again ###
    ### there should be no running cv jobs ###

    with open(outpath_base / f'sbatch_modelspecs_cv-{n_tt_mixes}-{n_tt_folds}.pkl', 'rb') as f:
        modelspecs_cv_list = pickle.load(f)

    model_spec_missing_list = list()
    multipletemp = list()
    for model_spec in modelspecs_cv_list:
        dir1 = model_spec['outpath'] / f"mix{model_spec['imix']:02}_fold{model_spec['ittfold']:02}_cv{model_spec['icvfold']:02}-lasso-mDsP_optimization"
        dir2 = dir1 / f"PTM-lasso-mDsP__cfgstr-n__k-{model_spec['logit_k']}_laplace_scale-{model_spec['laplace_scale']}"
        resfiles = list(dir2.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj.dill"))
        if len(resfiles) > 1:
            multipletemp.append(resfiles)
        if len(resfiles) == 0:
            model_spec_missing_list.append(model_spec)

    len(multipletemp)
    import shutil
    for model_spec_dup in multipletemp:
        for ii in np.arange(1, len(model_spec_dup)):
            shutil.rmtree(model_spec_dup[ii].parent)

    nmissing = len(model_spec_missing_list)

    dependency = None
    if model_spec_missing_list and run_missing:
        print(f"missing {nmissing} torch results")
        jobs_list_cv_missing = list()
        for model_spec in model_spec_missing_list:
            jobs_list_cv_missing.append(assemble_cfg(model_spec))
        sbatch_out = sbatch_torch_array(jobs_list_cv_missing, behavior='optimize', job_name='iaaTorchCV', use_everything=True, dependency=dependency)
        dependency = sbatch_out['dependency']

    # %%
    return nmissing, dependency
    # %%


def run_tt(n_tt_mixes, n_tt_folds, seed, niters, outpath_base=None, cpar_path_full=None, cpar_path_priors=None):

    # %%
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import pickle

    # %%

    # %%

    # %%
    #################################
    ### load cv results
    #################################

    with open(outpath_base / f'datafolds_mixes-{n_tt_mixes}-{n_tt_folds}.pkl', 'rb') as f:
        datafolds_mixes = pickle.load(f)

    # %%

    #############################################

    #### select best hyperparam based on fit to .... ####

    hypparm_criterion_time = ['terminal', 'lastquarter'][1]

    cvfolds_n_list = list()
    for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
        for i_tt_mix in range(n_tt_mixes):
            for i_tt_fold in range(n_tt_folds):
                stimid_tt_fold = datafolds_mixes[i_tt_mix][i_tt_fold]
                for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
                    hypparm_search_path = outpath_base / 'torchcv' / modelname / f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00-lasso-mDsP_optimization"
                    assert hypparm_search_path.is_dir()
                    eto_list = list(hypparm_search_path.glob("PTM*/optimize_iter-*/EmoTorchObj.dill"))
                    cvfolds_n_list.append(dict(model=modelname, criterion=hypparm_criterion, imix=i_tt_mix, ifold=i_tt_fold, n=len(eto_list)))

    missing_ = list()
    for item in cvfolds_n_list:
        if item['n'] < 18:
            missing_.append(item)
    len(missing_)

    #################

    ### TODO verify that eto_list contains all the hyperparm searchs that it needs
    collected_cv_res = list()
    for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
        for i_tt_mix in range(n_tt_mixes):
            print(f"mix {i_tt_mix}")
            for i_tt_fold in range(n_tt_folds):
                stimid_tt_fold = datafolds_mixes[i_tt_mix][i_tt_fold]
                for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
                    hypparm_search_path = outpath_base / 'torchcv' / modelname / f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00-lasso-mDsP_optimization"
                    assert hypparm_search_path.is_dir()
                    eto_list = list(hypparm_search_path.glob("PTM*/optimize_iter-*/EmoTorchObj.dill"))

                    hypparm_list = list()
                    for i_eto_path, eto_path in enumerate(eto_list):
                        eto = EmoTorch(verbose=False)
                        eto.load(eto_path)
                        progressdf = eto.optimized['op_progressdf'].drop_duplicates()

                        # hypparm_list.append({**eto.model_param, **eto.optimized['op_stats']})
                        collected_cv_res.append(dict(
                            hypparm_criterion=hypparm_criterion,
                            modelname=modelname,
                            i_tt_mix=i_tt_mix,
                            i_tt_fold=i_tt_fold,
                            i_eto_path=i_eto_path,
                            eto_path=eto_path,
                            model_param=eto.model_param,
                            progressdf=eto.optimized['op_progressdf'].drop_duplicates(),
                        ))

    # %%

    # input = {"A":"a", "B":"b", "C":"c"}
    # output = {k:v for (k,v) in input.items() if key_satifies_condition(k)}

    itermin = 1600
    itermax = 1800
    selected_hp = list()
    hypparm_criterion = 'hypc_deltas_byoutcome'
    modelname = 'iaaFull'
    for i_tt_mix in range(n_tt_mixes):
        for i_tt_fold in range(n_tt_folds):
            ### for this mix-fold, aggreage hyperparm results ###
            hypprm_select_list = list()
            for item in collected_cv_res:
                if item['modelname'] == modelname and item['hypparm_criterion'] == hypparm_criterion and item['i_tt_mix'] == i_tt_mix and item['i_tt_fold'] == i_tt_fold:
                    progressdf = item['progressdf']
                    hypprm_select_list.append(dict(
                        laplace_scale=item['model_param']['laplace_scale'],
                        # score=progressdf.iloc[-round(progressdf.shape[0] / 10):, :].median().loc['deltas_ccc']
                        score=progressdf.loc[(progressdf['iiter'] >= itermin) & (progressdf['iiter'] <= itermax), :].median().loc['deltas_ccc']
                    ))
            selected_hp.append(dict(
                imix=i_tt_mix,
                ifold=i_tt_fold,
                l1=pd.DataFrame(hypprm_select_list).sort_values('score', ascending=False)['laplace_scale'].iloc[0]
            ))

    pd.DataFrame(selected_hp)['l1'].value_counts()

    # %%

    selected_res_log = dict()
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
        selected_res_log[modelname] = dict()
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            selected_res_log[modelname][hypparm_criterion] = list()

    ### TODO fix seed so it can start anywhere
    rng_tt = np.random.default_rng(seed)
    modelspecs_tt_list = list()
    jobs_list_tt = list()
    for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
        for i_tt_mix in range(n_tt_mixes):
            for i_tt_fold in range(n_tt_folds):
                stimid_tt_fold = datafolds_mixes[i_tt_mix][i_tt_fold]
                for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
                    hypparm_search_path = outpath_base / 'torchcv' / modelname / f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00-lasso-mDsP_optimization"
                    assert hypparm_search_path.is_dir()
                    eto_list = list(hypparm_search_path.glob("PTM*/optimize_iter-*/EmoTorchObj.dill"))

                    hypparm_list = list()
                    for eto_path in eto_list:
                        eto = EmoTorch(verbose=False)
                        eto.load(eto_path)
                        progressdf = eto.optimized['op_progressdf'].drop_duplicates()
                        if hypparm_criterion_time == 'lastquarter':
                            scores_ = progressdf.iloc[-round(progressdf.shape[0] / 4):, :].median()
                        else:
                            scores_ = progressdf.iloc[-1:, :].median()

                        # hypparm_list.append({**eto.model_param, **eto.optimized['op_stats']})
                        hypparm_list.append({**eto.model_param, **scores_})
                    hypparm_criterion_ = {'hypc_absintens_bypotoutcome': 'test_ccc', 'hypc_deltas_byoutcome': 'deltas_ccc'}[hypparm_criterion]
                    hypparm_res = pd.DataFrame(hypparm_list).sort_values(hypparm_criterion_, ascending=False)

                    selected_param_ = hypparm_res.iloc[0, :]
                    selected_res_log[modelname][hypparm_criterion].append(selected_param_)
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
                        trainset=stimid_tt_fold['train'],
                        testset=stimid_tt_fold['test'],
                        seed=int(rng_tt.integers(low=1, high=np.iinfo(np.int32).max, dtype=int)),
                    )
                    model_spec = {**shared_param, **selected_param, **tt_spec_}
                    modelspecs_tt_list.append(model_spec)
                    cfg = assemble_cfg(model_spec)
                    jobs_list_tt.append(cfg)

    len(jobs_list_tt)
    #############################################
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            selected_res_log[modelname][hypparm_criterion] = pd.concat(selected_res_log[modelname][hypparm_criterion], axis=1).T
    selected_res_log['iaaFull']['hypc_absintens_bypotoutcome']
    selected_res_log['iaaFull']['hypc_deltas_byoutcome']
    selected_res_log['iaaFull']['hypc_deltas_byoutcome']['laplace_scale'].value_counts()
    selected_res_log['invplanLesion']['hypc_absintens_bypotoutcome']
    selected_res_log['invplanLesion']['hypc_deltas_byoutcome']
    selected_res_log['socialLesion']['hypc_absintens_bypotoutcome']
    selected_res_log['socialLesion']['hypc_deltas_byoutcome']

    with open(outpath_base / f'sbatch_modelspecs_tt-{n_tt_mixes}-{n_tt_folds}.pkl', 'wb') as f:
        pickle.dump(modelspecs_tt_list, f, protocol=-5)

    # %%
    #######
    ### launch tt jobs
    #######
    len(jobs_list_tt)
    sbatch_out = sbatch_torch_array(jobs_list_tt, behavior='optimize', job_name='iaaTorchTT', use_everything=True)

    dependency_out = sbatch_out['dependency']

    ### TODO pickle model_spec list and return dependancy
    # %%
    return dependency_out
    # %%


def relaunch_jobs_for_missing_results_tt(outpath_base=None, n_tt_mixes=None, n_tt_folds=None, dependency=None, run_missing=False, delete_extra=True):

    import pickle

    ### find missing cv results, lauch those jobs again ###
    ### there should be no running tt jobs ###

    with open(outpath_base / f'sbatch_modelspecs_tt-{n_tt_mixes}-{n_tt_folds}.pkl', 'rb') as f:
        modelspecs_tt_list = pickle.load(f)

    model_spec_missing_list = list()
    multipletemp = list()
    nfound = list()
    for model_spec in modelspecs_tt_list:
        dir1 = model_spec['outpath'] / f"mix{model_spec['imix']:02}_fold{model_spec['ittfold']:02}_cv{model_spec['icvfold']:02}-lasso-mDsP_optimization"
        # assert dir1.is_dir()
        dir2 = dir1 / f"PTM-lasso-mDsP__cfgstr-n__k-{model_spec['logit_k']}_laplace_scale-{model_spec['laplace_scale']}"
        # assert dir2.is_dir()
        resfiles = list(dir2.glob(f"optimize_iter-{model_spec['niters']}_T-*/EmoTorchObj.dill"))
        if len(resfiles) > 1:
            multipletemp.append(resfiles)
        if len(resfiles) == 0:
            model_spec_missing_list.append(model_spec)
        else:
            nfound.append(resfiles)
    len(nfound)

    len(multipletemp)
    if delete_extra:
        import shutil
        for model_spec_dup in multipletemp:
            for ii in np.arange(1, len(model_spec_dup)):
                shutil.rmtree(model_spec_dup[ii].parent)

    nmissing = len(model_spec_missing_list)
    print(f"missing {nmissing} torch results")

    dependency_out = None
    if model_spec_missing_list and run_missing:
        print(f"launch {nmissing} torch jobs")
        jobs_list_cv_missing = list()
        for model_spec in model_spec_missing_list:
            jobs_list_cv_missing.append(assemble_cfg(model_spec))
        sbatch_out = sbatch_torch_array(jobs_list_cv_missing, behavior='optimize', job_name='iaaTorchTT', use_everything=True, dependency=dependency)
        dependency_out = sbatch_out['dependency']

    # %%
    return nmissing, dependency_out
    # %%


def get_ppldata_cpar(cpar):
    import pickle

    cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    with open(cpar.paths['wpplDataCache'], 'rb') as f:
        ppldata, ppldata_exp3, distal_prior_ppldata, wpplparam = pickle.load(f)

    return ppldata, ppldata_exp3, distal_prior_ppldata


def get_ppldata_cfg(cfg):
    import pickle

    cpar = cfg['cpar']
    cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    with open(cpar.paths['wpplDataCache'], 'rb') as f:
        ppldata, ppldata_exp3, distal_prior_ppldata, wpplparam = pickle.load(f)

    return ppldata, ppldata_exp3, distal_prior_ppldata
