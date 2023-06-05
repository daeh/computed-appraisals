#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_collect_torch_results.py
"""


import numpy as np
from cam_emotorch import EmoTorch


class DataGrabber:

    def __init__(self, data=None, specific_player_stimids=None, outcomes=None, pots=None, emotions=None):
        self.data = data
        self.specific_player_stimids = specific_player_stimids
        self.outcomes = outcomes
        self.pots = pots
        self.emotions = emotions
        #####
        self.hypparm_options = list(self.data.keys())
        self.model_names = None
        self.hypparm_criterion = None
        self.modelname = None
        self.evtype = None  # ev_bypotoutcome or deltas_byoutcome

        model_names = list()
        for hypc_ in self.hypparm_options:
            for mn in list(self.data[hypc_].keys()):
                if mn not in model_names:
                    model_names.append(mn)

        self.model_names = model_names

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


def load_tt_results(outpath_base=None, hypparmcriterion=None, hypparmcriterion_time=None, model_label=None, n_mixes=None, n_folds=None, n_reinits=None):

    tt_results = list()
    learned_param = list()

    for i_tt_mix in range(n_mixes):
        for i_tt_reinit in range(n_reinits):
            for i_tt_fold in range(n_folds):
                tt_search_path = outpath_base / f'torchtt_{hypparmcriterion}_{hypparmcriterion_time}' / model_label
                assert tt_search_path.is_dir()

                eto_list = list(tt_search_path.glob(f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00_ri{i_tt_reinit:02}-lasso-mDsP/PTM*/optimize_iter-*/EmoTorchObj.dill"))
                assert len(eto_list) == 1
                eto_path = eto_list[0]

                eto = EmoTorch(verbose=False)
                eto.load(eto_path)

                learned_param.append(dict(
                    imix=i_tt_mix,
                    ireinit=i_tt_reinit,
                    ifold=i_tt_fold,
                    learned_param=eto.optimized['op_dict']
                ))

                for datatype in ['deltas', 'test_outcomepot']:
                    for ytype in ['model', 'empir']:
                        for i_stimid, stimid in enumerate(eto.optimized['op_appliedfit']['test_outcomepot']['dims']['stimid']):
                            tt_results.append(dict(
                                datatype=datatype,
                                ytype=ytype,
                                imix=i_tt_mix,
                                ireinit=i_tt_reinit,
                                stimid=stimid,
                                data=np.squeeze(eto.optimized['op_appliedfit'][datatype][ytype][i_stimid]),
                                eto_path=eto_path,
                            ))

    return tt_results, learned_param


def concat_random_reinits_and_mixes(nparray, n_tt_mixes):
    ### returns array where res[0] = <imix=0, ireint=0>, res[1] = <imix=0, ireint=1>, ... Same as np.concatenate(x, axis=0)
    return np.squeeze(np.concatenate(np.split(nparray, n_tt_mixes, axis=0), axis=1), axis=0)


def import_tt_results(outpath_base=None, specific_player_stimids=None, hypparm_criterion_time=None, n_tt_mixes=None, n_tt_folds=None, n_tt_random_reinits=None):

    ### load a random result in order to set the dims ###
    some_eto_path = list((outpath_base / f'torchtt_{"hypc_deltas_byoutcome"}_{hypparm_criterion_time}' / "caaFull" / "mix00_fold00_cv00_ri00-lasso-mDsP").glob("PTM*/optimize_iter-*/EmoTorchObj.dill"))[0]
    eto = EmoTorch(verbose=False)
    eto.load(some_eto_path)
    outcomes = eto.optimized['op_appliedfit']['test_outcomepot']['dims']['outcome']
    pots = eto.optimized['op_appliedfit']['test_outcomepot']['dims']['pot']
    emotions = eto.optimized['op_appliedfit']['test_outcomepot']['dims']['emotion']

    n_outcomes = len(outcomes)
    n_emotions = len(emotions)
    n_pots = len(pots)

    n_repeat_sets = n_tt_mixes * n_tt_random_reinits

    datacomp = dict()
    for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
        datacomp[hypparm_criterion] = dict()

        if hypparm_criterion == 'hypc_deltas_byoutcome':
            modelnames = ['caaFull', 'invplanLesion', 'socialLesion']
        elif hypparm_criterion == 'hypc_absintens_bypotoutcome':
            modelnames = ['invplanLesion', 'socialLesion']

        for modelname in modelnames:

            idx_used_all = list()

            datacomp[hypparm_criterion][modelname] = dict()
            ttres, learned_param_res = load_tt_results(outpath_base=outpath_base, hypparmcriterion=hypparm_criterion, hypparmcriterion_time=hypparm_criterion_time, model_label=modelname, n_mixes=n_tt_mixes, n_folds=n_tt_folds, n_reinits=n_tt_random_reinits)

            ##########

            deltas_model_ = np.full([n_tt_mixes, n_tt_random_reinits, len(specific_player_stimids), n_outcomes, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specific_player_stimids):
                for i_item, item in enumerate(ttres):
                    if (item['stimid'] == stimid and
                        item['datatype'] == 'deltas' and
                            item['ytype'] == 'model'):
                        idx_used_all.append(i_item)
                        assert np.all(np.isnan(deltas_model_[item['imix'], item['ireinit'], i_stimid, :, :]))
                        deltas_model_[item['imix'], item['ireinit'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(deltas_model_))

            deltas_empir_ = np.full([n_tt_mixes, n_tt_random_reinits, len(specific_player_stimids), n_outcomes, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specific_player_stimids):
                for i_item, item in enumerate(ttres):
                    if (item['stimid'] == stimid and
                        item['datatype'] == 'deltas' and
                            item['ytype'] == 'empir'):
                        idx_used_all.append(i_item)
                        assert np.all(np.isnan(deltas_empir_[item['imix'], item['ireinit'], i_stimid, :, :]))
                        deltas_empir_[item['imix'], item['ireinit'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(deltas_empir_))

            evpotoutcome_model_ = np.full([n_tt_mixes, n_tt_random_reinits, len(specific_player_stimids), n_outcomes, n_pots, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specific_player_stimids):
                for i_item, item in enumerate(ttres):
                    if (item['stimid'] == stimid and
                        item['datatype'] == 'test_outcomepot' and
                            item['ytype'] == 'model'):
                        idx_used_all.append(i_item)
                        assert np.all(np.isnan(evpotoutcome_model_[item['imix'], item['ireinit'], i_stimid, :, :, :]))
                        evpotoutcome_model_[item['imix'], item['ireinit'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(evpotoutcome_model_))

            evpotoutcome_empir_ = np.full([n_tt_mixes, n_tt_random_reinits, len(specific_player_stimids), n_outcomes, n_pots, n_emotions], np.nan, dtype=float)
            for i_stimid, stimid in enumerate(specific_player_stimids):
                for i_item, item in enumerate(ttres):
                    if (item['stimid'] == stimid and
                        item['datatype'] == 'test_outcomepot' and
                            item['ytype'] == 'empir'):
                        idx_used_all.append(i_item)
                        assert np.all(np.isnan(evpotoutcome_empir_[item['imix'], item['ireinit'], i_stimid, :, :, :]))
                        evpotoutcome_empir_[item['imix'], item['ireinit'], i_stimid, :, :] = item['data']
            assert not np.any(np.isnan(evpotoutcome_empir_))

            idx_used_vals_, idx_used_counts_ = np.unique(idx_used_all, return_counts=True)
            idx_used_count_vals, n_idx_used_counts_ = np.unique(idx_used_counts_, return_counts=True)
            assert idx_used_count_vals.size == 1 and idx_used_count_vals.item() == 1 and n_idx_used_counts_[0] == len(ttres)

            datacomp[hypparm_criterion][modelname] = dict(
                deltas_byoutcome=dict(
                    model=concat_random_reinits_and_mixes(deltas_model_, n_tt_mixes),
                    empir=concat_random_reinits_and_mixes(deltas_empir_, n_tt_mixes),
                ),
                ev_bypotoutcome=dict(
                    model=concat_random_reinits_and_mixes(evpotoutcome_model_, n_tt_mixes),
                    empir=concat_random_reinits_and_mixes(evpotoutcome_empir_, n_tt_mixes),
                ),
                learned_param=learned_param_res,
            )

    return DataGrabber(data=datacomp, specific_player_stimids=specific_player_stimids, outcomes=outcomes, pots=pots, emotions=emotions)


def load_tt_results_distributions(outpath_base=None, hypparmcriterion=None, hypparmcriterion_time=None, model_label=None, n_mixes=None, n_folds=None, n_reinits=None, nobsdf=None, subsample_frac=1):

    import pandas as pd
    from cam_pytorch_lasso import apply_fit

    assert isinstance(subsample_frac, int), 'subsample_frac must be an integer'

    tt_search_path = outpath_base / f'torchtt_{hypparmcriterion}_{hypparmcriterion_time}' / model_label
    assert tt_search_path.is_dir()

    compiled_emo_pred_lists = dict()
    for i_tt_mix in range(n_mixes):
        print(f"calculating emotion distribution data for mix{i_tt_mix:02}")
        for i_tt_fold in range(n_folds):
            for i_tt_reinit in range(n_reinits):

                eto_list = list(tt_search_path.glob(f"mix{i_tt_mix:02}_fold{i_tt_fold:02}_cv00_ri{i_tt_reinit:02}-lasso-mDsP/PTM*/optimize_iter-*/EmoTorchObj.dill"))
                assert len(eto_list) == 1
                eto_path = eto_list[0]

                eto = EmoTorch(verbose=False)
                eto.load(eto_path)

                torch_data = eto.get_torchdata()
                fit_param = eto.optimized['op_dict']
                datatest = torch_data['test']

                for stimid, data in datatest.items():

                    if stimid not in compiled_emo_pred_lists:
                        compiled_emo_pred_lists[stimid] = dict(CC=list(), CD=list(), DC=list(), DD=list())

                    for i_outcome, outcome in enumerate(data['Xshortdims']['outcome']):
                        for i_pot, pot in enumerate(data['Xshortdims']['pot']):
                            yhat = np.squeeze(apply_fit(fit_param, np.expand_dims(data['Xshort'][i_outcome, i_pot, :, :], axis=0)))
                            emodf_ = pd.DataFrame(yhat[0::subsample_frac], columns=data['Yshortdims']['emotion'])
                            emodf_['prob'] = emodf_.shape[0] ** -1
                            emodf_['pots'] = pot
                            compiled_emo_pred_lists[stimid][outcome].append(emodf_.set_index('pots', inplace=False))

    compiled_emo_pred = dict()
    for stimid in sorted(list(compiled_emo_pred_lists.keys())):
        print(f"compiling emotion distribution data for stim {stimid}")
        composite_emodict = dict()
        composite_emodict['nobs'] = nobsdf.copy(deep=True)
        composite_emodict['nobs'].loc[:, :] = int(0)
        for outcome in compiled_emo_pred_lists[stimid]:
            allsamples = pd.concat(compiled_emo_pred_lists[stimid][outcome])

            columns_ = list()
            for label in allsamples.columns.to_list():
                if label != 'prob':
                    columns_.append(('emotionIntensities', label))
                else:
                    columns_.append(('prob', 'prob'))

            allsamples.columns = pd.MultiIndex.from_tuples(columns_)
            composite_emodict[outcome] = allsamples

            for pot in np.unique(nobsdf.index.get_level_values(0)):
                if pot in composite_emodict[outcome].index:
                    nobs = composite_emodict[outcome].loc[pot, :].shape[0]
                else:
                    nobs = 0
                composite_emodict['nobs'].loc[pot, outcome] = int(nobs)

        compiled_emo_pred[stimid] = composite_emodict

    return compiled_emo_pred


def get_ppldata_from_cpar(cpar=None, cpar_path=None):
    import pickle

    assert (cpar is None or cpar_path is None) and not (cpar is None and cpar_path is None), 'either cpar or cpar_path must be specified'

    if cpar is None:
        import dill
        from pathlib import Path
        cpar_path = Path(cpar_path)
        assert cpar_path.is_file()
        with open(cpar_path, 'rb') as f:
            cpar = dill.load(f)

    cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    assert cpar.paths['wpplDataCache'].is_file()

    with open(cpar.paths['wpplDataCache'], 'rb') as f:
        ppldata, distal_prior_ppldata, wpplparam = pickle.load(f)

    return ppldata, distal_prior_ppldata
