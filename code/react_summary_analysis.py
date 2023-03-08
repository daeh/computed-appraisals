#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""react_summary_analysis.py
"""


from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from iaa_utils import concordance_corr_, adjusted_corr_
from webpypl import FTrz, FTzr, get_expected_vector_by_outcome
from webpypl_plotfun import init_plot_objects
from react_collect_pytorch_cvresults import DataGrabber, import_tt_results, load_tt_results_distabutions, assemble_cfg, get_ppldata_cpar
from iaa21_run_pytorch_torchdata import EmoTorch
from iaa21_bootstrap_pytorch_empirical import bootstrap_empirical
from iaa21_runwebppl_followup import followup_analyses
from iaa21_plot_pytorch_cvresults import get_iaf_names, format_iaf_labels_superphrase, format_iaf_labels_multiline, format_iaf_labels_superletter, format_iaf_labels_inlinechar, composite_emos_concord, composite_legend, composite_emos_summarybars, composite_Aweights, single_Aweights, new_emo_plot, new_emo_plot_iaf, composite_emos_concord_iaaonly, plotDM_handler


def compile_results(n_tt_mixes=None, n_tt_folds=None, seed=None, outpath_base=None, cpar_path_full=None, cpar_path_priors=None, specificplayers_desc_shorthand=None, cialpha=None):

    # %%

    specific_player_stimids = sorted(list(specificplayers_desc_shorthand.keys()))

    # %%
    ##############
    ### load tt results
    ##############

    ttdatadump_path = outpath_base / f'ttdatadump-{n_tt_mixes}-{n_tt_folds}.pkl'
    if ttdatadump_path.is_file():
        print(f"Loading tt results from {ttdatadump_path}")
        with open(ttdatadump_path, 'rb') as f:
            datgrb = DataGrabber(**pickle.load(f))
    else:
        print(f"Importing tt results and caching to {ttdatadump_path}")
        hypparm_criterion_time = ['terminal', 'lastquarter'][1]
        datgrb = import_tt_results(outpath_base, specific_player_stimids, hypparm_criterion_time, n_tt_mixes, n_tt_folds)
        datgrb.dump(ttdatadump_path)

    # %%
    ##############
    ### run empirical bootstrap
    ##############

    n_bootstrap = 1000
    bs_emp_pickle_path = outpath_base / f'bootstrap-empirical-{n_bootstrap}_cache.pkl'
    bs_emp_spec_ = dict(
        outpath=bs_emp_pickle_path.parent,
        cpar_path=cpar_path_full,
        model_type='empirical',
        logit_k=0.0,
        laplace_scale=0.0,
        niters=0,
        imix=0,
        ittfold=0,
        icvfold=0,
        trainset=[],
        testset=specific_player_stimids,
        seed=seed,
    )
    cfg_bs_emp = assemble_cfg(bs_emp_spec_)

    if bs_emp_pickle_path.is_file():
        with open(bs_emp_pickle_path, 'rb') as f:
            bs_ci_empirical = pickle.load(f)
    else:
        bs_ci_empirical, bs_res_empirical = bootstrap_empirical(cfg_bs_emp, n_bootstrap=n_bootstrap, seed=seed)
        bs_emp_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bs_emp_pickle_path, 'wb') as f:
            pickle.dump(bs_ci_empirical, f, protocol=-5)
        # with open(bs_emp_pickle_path.parent / f"{bs_emp_pickle_path.with_suffix('').name}-data.pkl", 'wb') as f:
        #     pickle.dump(bs_res_empirical, f, protocol=-5)
        # bs_res_empirical = None

    specific_player_order_ = dict()
    for stimid in specific_player_stimids:
        specific_player_order_[stimid] = bs_ci_empirical['deltas_byoutcome_player'][stimid]['adjusted_pearsonr']['pe']
    specific_player_order_tuple = sorted(specific_player_order_.items(), key=lambda x: x[1], reverse=True)
    specific_player_order = list(dict(specific_player_order_tuple).keys())

    # %%

    ##############
    ### learned parameters
    ##############

    datgrb.set_spec(hypparm_criterion='hypc_deltas_byoutcome', modelname='iaaFull', evtype='deltas_byoutcome')
    learned_param_deltas = datgrb.get_learnedparam()
    dim = datgrb.get_dims()
    ###
    Alist_deltas = [param['learned_param']['A'].numpy() for param in learned_param_deltas]
    A_deltas = np.stack(Alist_deltas)
    A_deltas_mean = np.mean(A_deltas, axis=0)

    emo_labels = dim['emotion']
    iaf_labels = get_iaf_names(cfg_bs_emp)

    A_deltas[0].shape
    A_deltas[:, 9, :]
    A_deltas[:, 9, -2]
    ### TODO plot hpd for guilt, envy

    A_deltas_mean_cizeroed = np.mean(A_deltas, axis=0)
    alphatrim = A_deltas.shape[0] * (cialpha / 2)
    alphatrim_int = int(alphatrim)
    assert alphatrim_int == alphatrim
    for i_emo in range(A_deltas.shape[1]):
        for i_feature in range(A_deltas.shape[2]):
            xxx = sorted(A_deltas[:, i_emo, i_feature].tolist())[alphatrim_int:-alphatrim_int]
            if np.min(xxx) <= 0 and np.max(xxx) >= 0:
                A_deltas_mean_cizeroed[i_emo, i_feature] = 0.0

    datgrb.set_spec(hypparm_criterion='hypc_absintens_bypotoutcome', modelname='iaaFull', evtype='ev_bypotoutcome')
    learned_param_absint = datgrb.get_learnedparam()
    ###
    Alist_absint = [param['learned_param']['A'].numpy() for param in learned_param_absint]
    A_absint = np.stack(Alist_absint)
    A_absint_mean_cizeroed = np.mean(A_absint, axis=0)
    alphatrim = A_absint.shape[0] * (cialpha / 2)
    alphatrim_int = int(alphatrim)
    for i_emo in range(A_absint.shape[1]):
        for i_feature in range(A_absint.shape[2]):
            xxx = sorted(A_absint[:, i_emo, i_feature].tolist())[alphatrim_int:-alphatrim_int]
            if np.min(xxx) <= 0 and np.max(xxx) >= 0:
                A_absint_mean_cizeroed[i_emo, i_feature] = 0.0

    ##############
    ### grand means
    ##############
    grand_emomeans_byoutcome = dict()
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
        grand_emomeans_byoutcome[modelname] = dict()
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            grand_emomeans_byoutcome[modelname][hypparm_criterion] = dict()
            datgrb.set_spec(hypparm_criterion=hypparm_criterion, modelname=modelname, evtype='ev_bypotoutcome')
            model_dat = datgrb.get_data('model')
            empir_dat = datgrb.get_data('empir')
            dim = datgrb.get_dims()
            for imix in dim['imix']:
                ymodel_ = model_dat[imix, :, :, :, :]
                yempir_ = empir_dat[imix, :, :, :, :]
                grand_emomeans_byoutcome[modelname][hypparm_criterion] = dict(
                    model=pd.DataFrame(np.mean(np.mean(ymodel_, axis=2), axis=0), index=dim['outcome'], columns=dim['emotion']),
                    empir=pd.DataFrame(np.mean(np.mean(yempir_, axis=2), axis=0), index=dim['outcome'], columns=dim['emotion'])
                )

    ##############
    ### deltas overall
    ##############

    stats_deltas_overall = dict()
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
        stats_deltas_overall[modelname] = dict()
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            stats_deltas_overall[modelname][hypparm_criterion] = dict()
            datgrb.set_spec(hypparm_criterion=hypparm_criterion, modelname=modelname, evtype='deltas_byoutcome')
            model_dat = datgrb.get_data('model')
            empir_dat = datgrb.get_data('empir')
            dim = datgrb.get_dims()
            ccc_list = list()
            pearr_list = list()
            for imix in dim['imix']:
                ymodel_ = model_dat[imix, :, :, :].flatten()
                yempir_ = empir_dat[imix, :, :, :].flatten()
                ccc_list.append(concordance_corr_(ymodel_, yempir_))
                pearr_list.append(pearsonr(ymodel_, yempir_)[0])
            stats_deltas_overall[modelname][hypparm_criterion]['ccc'] = ccc_list
            stats_deltas_overall[modelname][hypparm_criterion]['pearsonr'] = pearr_list

    dat_ = stats_deltas_overall['iaaFull']['hypc_deltas_byoutcome']['ccc']
    # alphatrim = len(dat_) * (cialpha / 2)
    # alphatrim_int = int(alphatrim)
    # dat_cibounded = sorted(dat_)[alphatrim_int:-alphatrim_int]
    # dat_cibounded_mean = FTzr(np.mean(FTrz(dat_cibounded)))
    # dat_cibounded_median = np.median(dat_cibounded)
    # pe_ = dat_cibounded_median
    # cil_ = np.min(dat_cibounded)
    # ciu_ = np.max(dat_cibounded)
    # print(f"{pe_:#.3}~=~[{cil_:#.3},~{ciu_:#.3}]")

    ##############
    ### deltas by player
    ##############

    stats_deltas_byplayer = dict()
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
        stats_deltas_byplayer[modelname] = dict()
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            stats_deltas_byplayer[modelname][hypparm_criterion] = dict()
            datgrb_spec = dict(
                hypparm_criterion=hypparm_criterion,
                modelname=modelname,
                evtype='deltas_byoutcome'
            )
            datgrb.set_spec(**datgrb_spec)
            model_dat = datgrb.get_data('model')
            empir_dat = datgrb.get_data('empir')
            dim = datgrb.get_dims()

            for i_stimid, stimid in enumerate(dim['stimid']):
                stats_deltas_byplayer[modelname][hypparm_criterion][stimid] = dict()
                ccc_list = list()
                pearr_list = list()
                adjpearr_list = list()
                for imix in dim['imix']:
                    ymodel_ = model_dat[imix, i_stimid, :, :].flatten()
                    yempir_ = empir_dat[imix, i_stimid, :, :].flatten()
                    ccc_list.append(concordance_corr_(ymodel_, yempir_))
                    pearr_list.append(pearsonr(ymodel_, yempir_)[0])
                    adjpearr_list.append(adjusted_corr_(ymodel_, yempir_, model_dat[imix, :, :, :].flatten()))
                stats_deltas_byplayer[modelname][hypparm_criterion][stimid]['ccc'] = ccc_list
                stats_deltas_byplayer[modelname][hypparm_criterion][stimid]['pearsonr'] = pearr_list
                stats_deltas_byplayer[modelname][hypparm_criterion][stimid]['adjusted_pearsonr'] = adjpearr_list

    #############################################

    ##############
    ### ev overall
    ##############

    stats_ev_overall = dict()
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
        stats_ev_overall[modelname] = dict()
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            stats_ev_overall[modelname][hypparm_criterion] = dict()
            datgrb.set_spec(hypparm_criterion=hypparm_criterion, modelname=modelname, evtype='ev_bypotoutcome')
            model_dat = datgrb.get_data('model')
            empir_dat = datgrb.get_data('empir')
            dim = datgrb.get_dims()
            ccc_list = list()
            pearr_list = list()
            for imix in dim['imix']:
                ymodel_ = model_dat[imix, :, :, :, :].flatten()
                yempir_ = empir_dat[imix, :, :, :, :].flatten()
                ccc_list.append(concordance_corr_(ymodel_, yempir_))
                pearr_list.append(pearsonr(ymodel_, yempir_)[0])
            stats_ev_overall[modelname][hypparm_criterion]['ccc'] = ccc_list
            stats_ev_overall[modelname][hypparm_criterion]['pearsonr'] = pearr_list

    ##############
    ### ev by emotion
    ##############

    stats_ev_byemotion = dict()
    for modelname in ['iaaFull', 'invplanLesion', 'socialLesion']:
        stats_ev_byemotion[modelname] = dict()
        for hypparm_criterion in ['hypc_absintens_bypotoutcome', 'hypc_deltas_byoutcome']:
            stats_ev_byemotion[modelname][hypparm_criterion] = dict()
            datgrb_spec = dict(
                hypparm_criterion=hypparm_criterion,
                modelname=modelname,
                evtype='ev_bypotoutcome'
            )
            datgrb.set_spec(**datgrb_spec)
            model_dat = datgrb.get_data('model')
            empir_dat = datgrb.get_data('empir')
            dim = datgrb.get_dims()

            for i_emo, emotion in enumerate(dim['emotion']):
                stats_ev_byemotion[modelname][hypparm_criterion][emotion] = dict()
                ccc_list = list()
                pearr_list = list()
                adjpearr_list = list()
                for imix in dim['imix']:
                    ymodel_ = model_dat[imix, :, :, :, i_emo].flatten()
                    yempir_ = empir_dat[imix, :, :, :, i_emo].flatten()
                    ccc_list.append(concordance_corr_(ymodel_, yempir_))
                    pearr_list.append(pearsonr(ymodel_, yempir_)[0])
                    adjpearr_list.append(adjusted_corr_(ymodel_, yempir_, model_dat[imix, :, :, :].flatten()))
                stats_ev_byemotion[modelname][hypparm_criterion][emotion]['ccc'] = ccc_list
                stats_ev_byemotion[modelname][hypparm_criterion][emotion]['pearsonr'] = pearr_list
                stats_ev_byemotion[modelname][hypparm_criterion][emotion]['adjusted_pearsonr'] = adjpearr_list

    #############################################

    #### make plot data ev overall ####
    bardata_list = list()
    plotspec = [
        ('iaaFull', 'hypc_absintens_bypotoutcome'),
        ('invplanLesion', 'hypc_absintens_bypotoutcome'),
        ('socialLesion', 'hypc_absintens_bypotoutcome'),
    ]
    for modelname, hypparm_criterion in plotspec:
        dat_ = stats_ev_overall[modelname][hypparm_criterion]['ccc']
        alphatrim = len(dat_) * (cialpha / 2)
        alphatrim_int = int(alphatrim)
        dat_cibounded = sorted(dat_)[alphatrim_int:-alphatrim_int]
        dat_cibounded_mean = FTzr(np.mean(FTrz(dat_cibounded)))
        dat_cibounded_median = np.median(dat_cibounded)
        pe_ = dat_cibounded_median
        cil_ = np.min(dat_cibounded)
        ciu_ = np.max(dat_cibounded)
        bardata_list.append(dict(
            model=modelname,
            xlabel='overall',
            pe=pe_,
            cil=cil_,
            ciu=ciu_,
        ))
    dat_ = bs_ci_empirical['ev_bypotoutcome_overall']['ccc']
    pe_ = dat_['pe']
    cil_ = dat_['ci'][0]
    ciu_ = dat_['ci'][1]
    bardata_list.append(dict(
        model='empirical',
        xlabel='overall',
        pe=pe_,
        cil=cil_,
        ciu=ciu_,
    ))
    bardatadf_evoverall = pd.DataFrame(bardata_list)

    #### make plot data deltas overall ####
    bardata_list = list()
    plotspec = [
        ('iaaFull', 'hypc_deltas_byoutcome'),
        ('invplanLesion', 'hypc_deltas_byoutcome'),
        ('socialLesion', 'hypc_deltas_byoutcome'),
    ]
    for modelname, hypparm_criterion in plotspec:
        dat_ = stats_deltas_overall[modelname][hypparm_criterion]['ccc']
        alphatrim = len(dat_) * (cialpha / 2)
        alphatrim_int = int(alphatrim)
        dat_cibounded = sorted(dat_)[alphatrim_int:-alphatrim_int]
        dat_cibounded_mean = FTzr(np.mean(FTrz(dat_cibounded)))
        dat_cibounded_median = np.median(dat_cibounded)
        pe_ = dat_cibounded_median
        cil_ = np.min(dat_cibounded)
        ciu_ = np.max(dat_cibounded)
        bardata_list.append(dict(
            model=modelname,
            xlabel='overall',
            pe=pe_,
            cil=cil_,
            ciu=ciu_,
        ))
    dat_ = bs_ci_empirical['deltas_byoutcome_overall']['ccc']
    pe_ = dat_['pe']
    cil_ = dat_['ci'][0]
    ciu_ = dat_['ci'][1]
    bardata_list.append(dict(
        model='empirical',
        xlabel='overall',
        pe=pe_,
        cil=cil_,
        ciu=ciu_,
    ))
    bardatadf_deltasoverall = pd.DataFrame(bardata_list)

    #### make plot data deltas by player ####
    bardata_list = list()
    plotspec = [
        ('iaaFull', 'hypc_deltas_byoutcome'),
        ('invplanLesion', 'hypc_deltas_byoutcome'),
        ('socialLesion', 'hypc_deltas_byoutcome'),
    ]
    for modelname, hypparm_criterion in plotspec:
        for stimid in specific_player_order:
            dat_ = stats_deltas_byplayer[modelname][hypparm_criterion][stimid]['adjusted_pearsonr']
            alphatrim = len(dat_) * (cialpha / 2)
            alphatrim_int = int(alphatrim)
            dat_cibounded = sorted(dat_)[alphatrim_int:-alphatrim_int]
            dat_cibounded_mean = FTzr(np.mean(FTrz(dat_cibounded)))
            dat_cibounded_median = np.median(dat_cibounded)
            pe_ = dat_cibounded_median
            cil_ = np.min(dat_cibounded)
            ciu_ = np.max(dat_cibounded)
            bardata_list.append(dict(
                model=modelname,
                xlabel=stimid,
                pe=pe_,
                cil=cil_,
                ciu=ciu_,
            ))
    for stimid in specific_player_order:
        dat_ = bs_ci_empirical['deltas_byoutcome_player'][stimid]['adjusted_pearsonr']
        pe_ = dat_['pe']
        cil_ = dat_['ci'][0]
        ciu_ = dat_['ci'][1]
        bardata_list.append(dict(
            model='empirical',
            xlabel=stimid,
            pe=pe_,
            cil=cil_,
            ciu=ciu_,
        ))
    bardatadf_deltasbyplayer = pd.DataFrame(bardata_list)

    #### make plot data ev by emotion ####
    emotion_order = dim['emotion']
    bardata_list = list()
    plotspec = [
        ('iaaFull', 'hypc_absintens_bypotoutcome'),
        ('invplanLesion', 'hypc_absintens_bypotoutcome'),
        ('socialLesion', 'hypc_absintens_bypotoutcome'),
    ]
    for modelname, hypparm_criterion in plotspec:
        for emotion in emotion_order:
            dat_ = stats_ev_byemotion[modelname][hypparm_criterion][emotion]['ccc']
            alphatrim = len(dat_) * (cialpha / 2)
            alphatrim_int = int(alphatrim)
            dat_cibounded = sorted(dat_)[alphatrim_int:-alphatrim_int]
            dat_cibounded_mean = FTzr(np.mean(FTrz(dat_cibounded)))
            dat_cibounded_median = np.median(dat_cibounded)
            pe_ = dat_cibounded_median
            cil_ = np.min(dat_cibounded)
            ciu_ = np.max(dat_cibounded)
            bardata_list.append(dict(
                model=modelname,
                xlabel=emotion,
                pe=pe_,
                cil=cil_,
                ciu=ciu_,
            ))
    for emotion in emotion_order:
        dat_ = bs_ci_empirical['ev_bypotoutcome_emotion'][emotion]['ccc']
        pe_ = dat_['pe']
        cil_ = dat_['ci'][0]
        ciu_ = dat_['ci'][1]
        bardata_list.append(dict(
            model='empirical',
            xlabel=emotion,
            pe=pe_,
            cil=cil_,
            ciu=ciu_,
        ))
    bardatadf_evbyemotion = pd.DataFrame(bardata_list)

    # %%

    res = dict(
        #### dimensions ####
        emo_labels=emo_labels,
        iaf_labels=iaf_labels,
        #### learned fit param ####
        A_deltas=A_deltas,
        A_deltas_mean=A_deltas_mean,
        A_deltas_mean_cizeroed=A_deltas_mean_cizeroed,
        A_absint_mean_cizeroed=A_absint_mean_cizeroed,
        #### stats dicts ####
        stats_ev_overall=stats_ev_overall,
        stats_ev_byemotion=stats_ev_byemotion,
        stats_deltas_overall=stats_deltas_overall,
        stats_deltas_byplayer=stats_deltas_byplayer,
        #### fit score ####
        emotion_order=emotion_order,
        specific_player_order=specific_player_order,
        grand_emomeans_byoutcome=grand_emomeans_byoutcome,
        bardatadf_evoverall=bardatadf_evoverall,
        bardatadf_evbyemotion=bardatadf_evbyemotion,
        bardatadf_deltasoverall=bardatadf_deltasoverall,
        bardatadf_deltasbyplayer=bardatadf_deltasbyplayer,
    )

    # %%

    return res


def plot_summary(res, n_tt_mixes=None, n_tt_folds=None, seed=None, outpath_base=None, cpar_path_full=None, cpar_path_priors=None, specificplayers_desc_shorthand=None, cialpha=None):

    # %%
    import numpy as np
    import dill
    from iaa21_bootstrap_pytorch_empirical import bootstrap_pe

    with open(cpar_path_full, 'rb') as f:
        cpar = dill.load(f)

    ppldata, ppldata_exp3, distal_prior_ppldata = get_ppldata_cpar(cpar)

    plotParam = init_plot_objects(outpath_base)
    save_text_var = plotParam['save_text_var']


    # %%

    ### unpack res ###

    #### dimensions ####
    emo_labels = res['emo_labels']
    iaf_labels = res['iaf_labels']
    #### learned fit param ####
    A_deltas = res['A_deltas']
    A_deltas_mean = res['A_deltas_mean']
    A_deltas_mean_cizeroed = res['A_deltas_mean_cizeroed']
    A_absint_mean_cizeroed = res['A_absint_mean_cizeroed']
    #### stats dicts ####
    stats_ev_overall = res['stats_ev_overall']
    stats_ev_byemotion = res['stats_ev_byemotion']
    stats_deltas_overall = res['stats_deltas_overall']
    stats_deltas_byplayer = res['stats_deltas_byplayer']
    #### fit score ####
    emotion_order = res['emotion_order']
    specific_player_order = res['specific_player_order']
    grand_emomeans_byoutcome = res['grand_emomeans_byoutcome']
    bardatadf_evoverall = res['bardatadf_evoverall']
    bardatadf_evbyemotion = res['bardatadf_evbyemotion']
    bardatadf_deltasoverall = res['bardatadf_deltasoverall']
    bardatadf_deltasbyplayer = res['bardatadf_deltasbyplayer']

    # %%

    hypc_ = 'hypc_absintens_bypotoutcome'
    for model_name, datadir_ in stats_ev_overall.items():
        for stat_ in datadir_[hypc_]:
            dat_ = datadir_[hypc_][stat_]
            alphatrim = len(dat_) * (cialpha / 2)
            alphatrim_int = int(alphatrim)
            dat_cibounded = sorted(dat_)[alphatrim_int:-alphatrim_int]
            dat_cibounded_mean = FTzr(np.mean(FTrz(dat_cibounded)))
            dat_cibounded_median = np.median(dat_cibounded)
            pe_ = dat_cibounded_median
            cil_ = np.min(dat_cibounded)
            ciu_ = np.max(dat_cibounded)
            save_text_var.write(f"{pe_:0.3f}~[{cil_:0.3f}, {ciu_:0.3f}]%", f"ev_overall-{model_name}_{stat_}.tex")

    hypc_ = 'hypc_deltas_byoutcome'
    for model_name, datadir_ in stats_deltas_overall.items():
        for stat_ in datadir_[hypc_]:
            dat_ = datadir_[hypc_][stat_]
            # dat_ = stats_deltas_overall['iaaFull']['hypc_deltas_byoutcome']['ccc']
            alphatrim = len(dat_) * (cialpha / 2)
            alphatrim_int = int(alphatrim)
            dat_cibounded = sorted(dat_)[alphatrim_int:-alphatrim_int]
            dat_cibounded_mean = FTzr(np.mean(FTrz(dat_cibounded)))
            dat_cibounded_median = np.median(dat_cibounded)
            pe_ = dat_cibounded_median
            cil_ = np.min(dat_cibounded)
            ciu_ = np.max(dat_cibounded)
            save_text_var.write(f"{pe_:0.3f}~[{cil_:0.3f}, {ciu_:0.3f}]%", f"deltas_overall-{model_name}_{stat_}.tex")

    # %%
    for emotion in emo_labels:
        single_Aweights(learned_param_A=A_deltas, emotion=emotion, emo_labels=emo_labels, iaf_labels=iaf_labels, cialpha=cialpha, fig_outpath=plotParam['figsOut'] / 'A-single-emos_hypc_deltas_byoutcome' / f'A-{emotion}.pdf', plotParam=plotParam)

    composite_Aweights(learned_param_Amean=A_deltas_mean, emo_labels=emo_labels, iaf_labels=iaf_labels, fig_outpath=plotParam['figsOut'] / 'A-unthresh_hypc_deltas_byoutcome.pdf', plotParam=plotParam)

    composite_Aweights(learned_param_Amean=A_deltas_mean_cizeroed, emo_labels=emo_labels, iaf_labels=iaf_labels, fig_outpath=plotParam['figsOut'] / f'A-ci-zeroed_hypc_deltas_byoutcome-{int((1-cialpha)*100)}CI.pdf', plotParam=plotParam)

    composite_Aweights(learned_param_Amean=A_absint_mean_cizeroed, emo_labels=emo_labels, iaf_labels=iaf_labels, fig_outpath=plotParam['figsOut'] / f'A-ci-zeroed_hypc_absintens_bypotoutcome-{int((1-cialpha)*100)}CI.pdf', plotParam=plotParam)

    # %%

    modellabels = ['iaaFull', 'invplanLesion', 'socialLesion']

    model_colors = {
        'iaaFull': 'cornflowerblue',
        'invplanLesion': 'dimgrey',
        'socialLesion': 'green',
    }

    labelkey = {
        'iaaFull': 'Inferred Appraisal',
        'invplanLesion': 'Inverse Plan. lesion',
        'socialLesion': 'Social lesion',
    }

    composite_emos_concord(bardatadf_deltasbyplayer, x_tick_order=specific_player_order, model_colors=model_colors, model_order=modellabels, plotParam=plotParam, x_tick_labels=specificplayers_desc_shorthand, fig_outpath=plotParam['figsOut'] / 'deltas_bars_byplayer.pdf')

    composite_emos_concord(bardatadf_evbyemotion, x_tick_order=emotion_order, model_colors=model_colors, model_order=modellabels, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emoev_bars_byemotion.pdf')

    ####

    model_colors_iaaonly = {
        'iaaFull': 'cornflowerblue',
        'invplanLesion': '#00000000',
        'socialLesion': '#00000000',
    }
    composite_emos_concord_iaaonly(bardatadf_evbyemotion, x_tick_order=emotion_order, model_colors=model_colors_iaaonly, model_order=modellabels, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'slides' / 'emoev_bars_byemotion_iaaonly.pdf')

    composite_legend(model_labels=labelkey, model_colors=model_colors, fig_outpath=plotParam['figsOut'] / 'legend.pdf', plotParam=plotParam)

    composite_emos_summarybars(bardatadf_evoverall, x_tick_order=['overall'], model_colors=model_colors, model_order=modellabels, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emoev_bars_overall.pdf')

    composite_emos_summarybars(bardatadf_deltasoverall, x_tick_order=['overall'], model_colors=model_colors, model_order=modellabels, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'deltas_bars_overall.pdf')

    # %%

    model_colors_iaaonly = {
        'iaaFull': 'cornflowerblue',
        'invplanLesion': '#00000000',
        'socialLesion': '#00000000',
    }
    emotion_order_subset = ['Envy', 'Fury', 'Guilt', 'Disappointment', 'Joy',
                            # 'Devastation', 'Contempt', 'Disgust', 'Annoyance', 'Embarrassment', 'Regret', 'Confusion', 'Surprise', 'Sympathy', 'Amusement', 'Relief', 'Respect', 'Gratitude', 'Pride', 'Excitement'
                            ]
    composite_emos_concord_iaaonly(bardatadf_evbyemotion, x_tick_order=emotion_order_subset, model_colors=model_colors_iaaonly, model_order=modellabels, plotParam=plotParam, ylim_fixed=(0, 1), fig_outpath=plotParam['figsOut'] / 'slides' / 'emoev_bars_byemotion_iaaonly_subset.pdf')

    model_colors_iaaonly = {
        'iaaFull': 'cornflowerblue',
        'invplanLesion': 'dimgrey',
        'socialLesion': '#00000000',
    }
    emotion_order_subset = ['Envy', 'Fury', 'Guilt', 'Disappointment', 'Joy',
                            # 'Devastation', 'Contempt', 'Disgust', 'Annoyance', 'Embarrassment', 'Regret', 'Confusion', 'Surprise', 'Sympathy', 'Amusement', 'Relief', 'Respect', 'Gratitude', 'Pride', 'Excitement'
                            ]
    composite_emos_concord_iaaonly(bardatadf_evbyemotion, x_tick_order=emotion_order_subset, model_colors=model_colors_iaaonly, model_order=modellabels, plotParam=plotParam, ylim_fixed=(0, 1), fig_outpath=plotParam['figsOut'] / 'slides' / 'emoev_bars_byemotion_iaaonly_subset2.pdf')

    model_colors_iaaonly = {
        'iaaFull': 'cornflowerblue',
        'invplanLesion': 'dimgrey',
        'socialLesion': 'green',
    }
    emotion_order_subset = ['Envy', 'Fury', 'Guilt', 'Disappointment', 'Joy',
                            # 'Devastation', 'Contempt', 'Disgust', 'Annoyance', 'Embarrassment', 'Regret', 'Confusion', 'Surprise', 'Sympathy', 'Amusement', 'Relief', 'Respect', 'Gratitude', 'Pride', 'Excitement'
                            ]
    composite_emos_concord_iaaonly(bardatadf_evbyemotion, x_tick_order=emotion_order_subset, model_colors=model_colors_iaaonly, model_order=modellabels, plotParam=plotParam, ylim_fixed=(0, 1), fig_outpath=plotParam['figsOut'] / 'slides' / 'emoev_bars_byemotion_iaaonly_subset3.pdf')

    # %%

    printFigList = plotParam['printFigList']
    # plt = plotParam['plt']

    # simmat_template = pd.DataFrame(pairwise_distances(compiled_distalplayer_empppldata['CC']['emotionIntensities'].to_numpy().T, metric='correlation'), index=emotions, columns=emotions)
    # simmat_template.loc[:, :] = 0.0

    figsout = list()

    ### within pot-outcome ###
    for outcome in plotParam['outcomes']:
        emodf = ppldata['empiricalEmotionJudgments'][outcome].loc[:, 'emotionIntensities']
        pot_idx = emodf.index.get_level_values(0).to_numpy()
        emotions = emodf.columns.to_list()
        simarray = np.full([np.unique(pot_idx).size, emodf.shape[1], emodf.shape[1]], np.nan, dtype=float)
        for i_pot, pot in enumerate(np.unique(pot_idx)):
            potdf = emodf.loc[pot_idx == pot, :]
            for i_emotion_row, emotion_row in enumerate(emotions):
                for i_emotion_col, emotion_col in enumerate(emotions):
                    simarray[i_pot, i_emotion_row, i_emotion_col] = pearsonr(potdf.loc[:, emotion_row].to_numpy(), potdf.loc[:, emotion_col].to_numpy())[0]
        simmat = pd.DataFrame(np.zeros([len(emotions), len(emotions)]), index=emotions, columns=emotions)
        for i_emotion_row, emotion_row in enumerate(emotions):
            for i_emotion_col, emotion_col in enumerate(emotions):
                if not np.allclose(simarray[:, i_emotion_row, i_emotion_col], 1):
                    simmat.loc[emotion_row, emotion_col] = FTzr(np.mean(FTrz(simarray[:, i_emotion_row, i_emotion_col])))
                else:
                    simmat.loc[emotion_row, emotion_col] = np.mean(simarray[:, i_emotion_row, i_emotion_col])
        figsout.append(plotDM_handler(simmat, show_labels=False, show_colorbar=False, invert_mask=True, plotParam=plotParam, outpath=plotParam['figsOut'] / 'simmat_extras' / f'simmat-empir-generic-withinpot-{outcome}.svg'))
        figsout.append(plotDM_handler(simmat, show_labels=False, show_colorbar=False, invert_mask=False, plotParam=plotParam, outpath=plotParam['figsOut'] / 'simmat_extras' / f'simmat-empir-generic-withinpot-{outcome}.pdf'))

    ### within outcome, across pots ###
    for outcome in plotParam['outcomes']:
        emodf = ppldata['empiricalEmotionJudgments'][outcome].loc[:, 'emotionIntensities']
        simmat = pd.DataFrame(np.zeros([emodf.shape[1], emodf.shape[1]]), index=emodf.columns.to_list(), columns=emodf.columns.to_list())
        for emotion_row in simmat.columns:
            for emotion_col in simmat.columns:
                simmat.loc[emotion_row, emotion_col] = pearsonr(emodf.loc[:, emotion_row].to_numpy(), emodf.loc[:, emotion_col].to_numpy())[0]
        # simmats[outcome] = simmat
        figsout.append(plotDM_handler(simmat, show_labels=True, show_colorbar=False, plotParam=plotParam, outpath=plotParam['figsOut'] / f'simmat-empir-generic-{outcome}.pdf'))

    ### across outcomes and pots ###
    outcome_dfs_ = list()
    for outcome in plotParam['outcomes']:
        emodf = ppldata['empiricalEmotionJudgments'][outcome].loc[:, 'emotionIntensities']
        outcome_dfs_.append(emodf)
    emodf = pd.concat(outcome_dfs_)
    simmat = pd.DataFrame(np.zeros([emodf.shape[1], emodf.shape[1]]), index=emodf.columns.to_list(), columns=emodf.columns.to_list())
    for emotion_row in simmat.columns:
        for emotion_col in simmat.columns:
            simmat.loc[emotion_row, emotion_col] = pearsonr(emodf.loc[:, emotion_row].to_numpy(), emodf.loc[:, emotion_col].to_numpy())[0]
    # simmats[outcome] = simmat
    figsout.append(plotDM_handler(simmat, show_labels=True, show_colorbar=True, plotParam=plotParam, outpath=plotParam['figsOut'] / f'simmat-empir-generic-overall.pdf'))

    # %%

    emodist_cachepath = outpath_base / 'emodist_cache.pkl'
    if emodist_cachepath.is_file():
        with open(emodist_cachepath, 'rb') as f:
            compiled_emodists = pickle.load(f)
    else:
        compiled_emodists = dict(
            hypc_absintens_bypotoutcome=dict(
                lastquarter=dict(
                    iaaFull=load_tt_results_distabutions(outpath_base, 'hypc_absintens_bypotoutcome', 'lastquarter', 'iaaFull', n_tt_mixes, n_tt_folds, ppldata=ppldata, subsample_frac=5)
                )
            ),
            hypc_deltas_byoutcome=dict(
                lastquarter=dict(
                    iaaFull=load_tt_results_distabutions(outpath_base, 'hypc_deltas_byoutcome', 'lastquarter', 'iaaFull', n_tt_mixes, n_tt_folds, ppldata=ppldata, subsample_frac=5)
                )
            ),
        )
        with open(emodist_cachepath, 'wb') as f:
            pickle.dump(compiled_emodists, f, protocol=-5)

    # %%

    ### get training data from some model
    torch_data_cachepath = outpath_base / 'torch_data_cache.pkl'
    if torch_data_cachepath.is_file():
        with open(torch_data_cachepath, 'rb') as f:
            torch_data = pickle.load(f)
    else:
        eto_path = list((outpath_base / 'torchtt_hypc_absintens_bypotoutcome_lastquarter' / 'iaaFull' / 'mix00_fold00_cv00-lasso-mDsP_optimization').rglob('EmoTorchObj.dill'))[0]
        eto = EmoTorch()
        eto.load(eto_path)
        torch_data = eto.get_torchdata()
        with open(torch_data_cachepath, 'wb') as f:
            pickle.dump(torch_data, f, protocol=-5)

    Xshort = torch_data['train']['generic']['Xshort']
    Xshortdims = torch_data['train']['generic']['Xshortdims']
    Xshortdims.keys()

    # %%

    iaf_labels_formatted_dict = dict()
    for label in iaf_labels:
        # iaf_labels_formatted_dict[label] = format_iaf_labels_inlinechar([label])[0]
        iaf_labels_formatted_dict[label] = format_iaf_labels_superphrase([label])[0]

    # %%

    ### within pot-outcome ###
    for i_outcome, outcome in enumerate(Xshortdims['outcome']):
        simarray = np.full([len(Xshortdims['pot']), len(Xshortdims['iaf']), len(Xshortdims['iaf'])], np.nan, dtype=float)
        for i_pot, pot in enumerate(Xshortdims['pot']):
            for i_iaf_row, iaf_row in enumerate(Xshortdims['iaf']):
                for i_iaf_col, iaf_col in enumerate(Xshortdims['iaf']):
                    x1 = Xshort[i_outcome, i_pot, :, i_iaf_row].flatten()
                    x2 = Xshort[i_outcome, i_pot, :, i_iaf_col].flatten()
                    if np.unique(x1).size == 1 or np.unique(x2).size == 1:
                        simarray[i_pot, i_iaf_row, i_iaf_col] = np.nan
                    else:
                        simarray[i_pot, i_iaf_row, i_iaf_col] = pearsonr(x1, x2)[0]

        simmat = pd.DataFrame(np.zeros([len(Xshortdims['iaf']), len(Xshortdims['iaf'])]), index=Xshortdims['iaf'], columns=Xshortdims['iaf'])
        for i_iaf_row, iaf_row in enumerate(Xshortdims['iaf']):
            for i_iaf_col, iaf_col in enumerate(Xshortdims['iaf']):
                if np.allclose(simarray[:, i_iaf_row, i_iaf_col], 1) or np.any(np.isnan(simarray[:, i_iaf_row, i_iaf_col])):
                    simmat.loc[iaf_row, iaf_col] = np.mean(simarray[:, i_iaf_row, i_iaf_col])
                else:
                    simmat.loc[iaf_row, iaf_col] = FTzr(np.mean(FTrz(simarray[:, i_iaf_row, i_iaf_col])))
        figsout.append(plotDM_handler(simmat, show_labels=False, show_colorbar=False, invert_mask=True, plotParam=plotParam, outpath=plotParam['figsOut'] / 'simmat_extras' / f'simmat-iaf-generic-withinpot-{outcome}.svg'))
        figsout.append(plotDM_handler(simmat, show_labels=False, show_colorbar=False, invert_mask=False, plotParam=plotParam, outpath=plotParam['figsOut'] / 'simmat_extras' / f'simmat-iaf-generic-withinpot-{outcome}.pdf'))

    ### across pot, within outcome ###
    for i_outcome, outcome in enumerate(Xshortdims['outcome']):
        simmat = pd.DataFrame(np.zeros([len(Xshortdims['iaf']), len(Xshortdims['iaf'])]), index=Xshortdims['iaf'], columns=Xshortdims['iaf'])
        for i_iaf_row, iaf_row in enumerate(Xshortdims['iaf']):
            for i_iaf_col, iaf_col in enumerate(Xshortdims['iaf']):
                x1 = Xshort[i_outcome, :, :, i_iaf_row].flatten()
                x2 = Xshort[i_outcome, :, :, i_iaf_col].flatten()
                if np.unique(x1).size == 1 or np.unique(x2).size == 1:
                    # print(f"{outcome} :: {iaf_row}")
                    simmat.loc[iaf_row, iaf_col] = np.nan
                else:
                    simmat.loc[iaf_row, iaf_col] = pearsonr(x1, x2)[0]
        figsout.append(plotDM_handler(simmat, label_rename_dict=iaf_labels_formatted_dict, show_labels=True, show_colorbar=False, plotParam=plotParam, outpath=plotParam['figsOut'] / f'simmat-iaf-generic-{outcome}.pdf'))

    simmat = pd.DataFrame(np.zeros([len(Xshortdims['iaf']), len(Xshortdims['iaf'])]), index=Xshortdims['iaf'], columns=Xshortdims['iaf'])
    for i_iaf_row, iaf_row in enumerate(Xshortdims['iaf']):
        for i_iaf_col, iaf_col in enumerate(Xshortdims['iaf']):
            x1 = Xshort[:, :, :, i_iaf_row].flatten()
            x2 = Xshort[:, :, :, i_iaf_col].flatten()
            if np.unique(x1).size == 1 or np.unique(x2).size == 1:
                # print(f"{outcome} :: {iaf_row}")
                simmat.loc[iaf_row, iaf_col] = np.nan
            else:
                simmat.loc[iaf_row, iaf_col] = pearsonr(x1, x2)[0]
    figsout.append(plotDM_handler(simmat, label_rename_dict=iaf_labels_formatted_dict, show_labels=True, show_colorbar=True, plotParam=plotParam, outpath=plotParam['figsOut'] / f'simmat-iaf-generic-overall.pdf'))

    figsout = printFigList(figsout, plotParam, save_kwargs=dict(transparent=True))

    # %%

    emotions_abbriv = {
        'Devastation': 'Devast',
        'Disappointment': 'Disap',
        'Contempt': 'Contem',
        'Disgust': 'Disgu',
        'Envy': 'Envy',
        'Fury': 'Fury',
        'Annoyance': 'Annoy',
        'Embarrassment': 'Embar',
        'Regret': 'Regret',
        'Guilt': 'Guilt',
        'Confusion': 'Confus',
        'Surprise': 'Surpri',
        'Sympathy': 'Sympa',
        'Amusement': 'Amuse',
        'Relief': 'Relief',
        'Respect': 'Respec',
        'Gratitude': 'Gratit',
        'Pride': 'Pride',
        'Excitement': 'Excite',
        'Joy': 'Joy',
    }

    compiled_pred_emodist = compiled_emodists['hypc_absintens_bypotoutcome']['lastquarter']['iaaFull']

    compiled_pred_emodist['246_1']['nobs']
    compiled_pred_emodist['246_1']['CC']

    compiled_distalplayer_empppldata = dict()
    for outcome in plotParam['outcomes']:
        ppld_list = list()
        for stimid in distal_prior_ppldata:
            ppld_list.append(distal_prior_ppldata[stimid][outcome[0]]['empiricalEmotionJudgments'][outcome])
        compiled_distalplayer_empppldata[outcome] = pd.concat(ppld_list)

    new_emo_plot(compiled_distalplayer_empppldata, emoevdf=grand_emomeans_byoutcome['iaaFull']['hypc_absintens_bypotoutcome']['empir'], scale_factor=13.0, bandwidth=0.07, emotions_abbriv=emotions_abbriv, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emo_EVscatter' / 'emo_EVscatter_specificplayers_empir.pdf')

    compiled_distalplayer_modelppldata = dict()
    for outcome in plotParam['outcomes']:
        ppld_list = list()
        for stimid in distal_prior_ppldata:
            ppld_list.append(compiled_pred_emodist[stimid][outcome])
        compiled_distalplayer_modelppldata[outcome] = pd.concat(ppld_list)

    new_emo_plot(compiled_distalplayer_modelppldata, emoevdf=grand_emomeans_byoutcome['iaaFull']['hypc_absintens_bypotoutcome']['model'], scale_factor=10.0, bandwidth=0.07, emotions_abbriv=None, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emo_EVscatter' / 'emo_EVscatter_specificplayers_model-absintensfit.pdf')

    # %%

    for stimid in distal_prior_ppldata:
        compiled_distalplayer_empppldata = dict()
        for outcome in plotParam['outcomes']:
            ppld_list = list()
            ppld_list.append(distal_prior_ppldata[stimid][outcome[0]]['empiricalEmotionJudgments'][outcome])
            compiled_distalplayer_empppldata[outcome] = pd.concat(ppld_list)

        emomeans_ = pd.concat([compiled_distalplayer_empppldata[outcome].groupby('pots').mean().mean() for outcome in plotParam['outcomes']], axis=1).T
        emomeans2_ = emomeans_.droplevel(axis=1, level=0)
        emomeans2_['outcome'] = plotParam['outcomes']
        emomeans2_.drop('prob', axis=1, inplace=True)
        emomeans2_.set_index('outcome', inplace=True)

        new_emo_plot(compiled_distalplayer_empppldata, emoevdf=emomeans2_, scale_factor=13.0, bandwidth=0.07, emotions_abbriv=emotions_abbriv, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emo_EVscatter' / f'emo_EVscatter_{stimid}_empir.pdf')

    compiled_pred_emodist = compiled_emodists['hypc_absintens_bypotoutcome']['lastquarter']['iaaFull']

    for stimid in compiled_pred_emodist:
        compiled_distalplayer_modelppldata = dict()
        for outcome in plotParam['outcomes']:
            ppld_list = list()
            ppld_list.append(compiled_pred_emodist[stimid][outcome])
            compiled_distalplayer_modelppldata[outcome] = pd.concat(ppld_list)

        emomeans_ = pd.concat([compiled_distalplayer_modelppldata[outcome].groupby('pots').mean().mean() for outcome in plotParam['outcomes']], axis=1).T
        emomeans2_ = emomeans_.droplevel(axis=1, level=0)
        emomeans2_['outcome'] = plotParam['outcomes']
        emomeans2_.drop('prob', axis=1, inplace=True)
        emomeans2_.set_index('outcome', inplace=True)

        new_emo_plot(compiled_distalplayer_modelppldata, emoevdf=emomeans2_, scale_factor=10.0, bandwidth=0.07, emotions_abbriv=None, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emo_EVscatter' / f'emo_EVscatter_{stimid}_model-absintensfit.pdf')

    # %%
    #### plotting EV of the deltas fit

    compiled_pred_emodist = compiled_emodists['hypc_deltas_byoutcome']['lastquarter']['iaaFull']

    for stimid in compiled_pred_emodist:
        compiled_distalplayer_modelppldata = dict()
        for outcome in plotParam['outcomes']:
            ppld_list = list()
            ppld_list.append(compiled_pred_emodist[stimid][outcome])
            compiled_distalplayer_modelppldata[outcome] = pd.concat(ppld_list)

        emomeans_ = pd.concat([compiled_distalplayer_modelppldata[outcome].groupby('pots').mean().mean() for outcome in plotParam['outcomes']], axis=1).T
        emomeans2_ = emomeans_.droplevel(axis=1, level=0)
        emomeans2_['outcome'] = plotParam['outcomes']
        emomeans2_.drop('prob', axis=1, inplace=True)
        emomeans2_.set_index('outcome', inplace=True)

        new_emo_plot(compiled_distalplayer_modelppldata, emoevdf=emomeans2_, scale_factor=10.0, bandwidth=0.07, emotions_abbriv=None, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emo_EVscatter' / f'emo_EVscatter_{stimid}_model-deltasfit.pdf')

    # %%

    # emo_expectations = calc_emo_ev_by_outcome(emodf)

    emotions = ppldata['empiricalEmotionJudgments']['CC']['emotionIntensities'].columns.to_list()
    outcomes = plotParam['outcomes']
    emomean_byoutcome_generic_empir = get_expected_vector_by_outcome(ppldata['empiricalEmotionJudgments'], emotions=emotions, outcomes=outcomes).set_index('outcome')
    ### TODO make sure this is returning the right value, correcting for number of observations

    new_emo_plot(ppldata['empiricalEmotionJudgments'], emoevdf=emomean_byoutcome_generic_empir, scale_factor=13.0, bandwidth=0.07, emotions_abbriv=emotions_abbriv, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emo_EVscatter' / 'emo_EVscatter_genericplayers_empir.pdf')

    new_emo_plot(ppldata['empiricalEmotionJudgments'], emoevdf=emomean_byoutcome_generic_empir, scale_factor=13.0, bandwidth=0.07, emotions_abbriv=None, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'emo_EVscatter' / 'emo_EVscatter_genericplayers_empir_long.pdf')

    # %%

    ### TODO get scatter plot of EV of the generic IAF

    ### get training data from some model
    # torch_data_cachepath = outpath_base / 'torch_data_cache.pkl'
    # if torch_data_cachepath.is_file():
    #     with open(outpath_base / 'torch_data_cache.pkl', 'rb') as f:
    #         torch_data = pickle.load(f)
    # else:
    #     eto_path = list((outpath_base / 'torchtt_hypc_absintens_bypotoutcome_lastquarter' / 'iaaFull' / 'mix00_fold00_cv00-lasso-mDsP_optimization').rglob('EmoTorchObj.dill'))[0]
    #     eto = EmoTorch()
    #     eto.load(eto_path)
    #     torch_data = eto.get_torchdata()
    #     with open(outpath_base / 'torch_data_cache.pkl', 'wb') as f:
    #         pickle.dump(torch_data, f, protocol=-5)

    Xshort = torch_data['train']['generic']['Xshort']
    Xshortdims = torch_data['train']['generic']['Xshortdims']
    Xshortdims.keys()

    Xshortdims['pot']
    len(Xshortdims['pot'])

    # torch_data['train']['generic'].keys()
    # torch_data['train']['generic']['Yshortdims'].keys()
    # len(torch_data['train']['generic']['Yshort']['CC'])

    # iaf_ev_list = list()
    # for i_outcome, outcome in enumerate(Xshortdims['outcome']):
    #     # for i_iaf, iaf in enumerate(Xshortdims['iaf']):
    #     Xshort.shape
    iaf_ev_byoutcome = np.squeeze(np.mean(np.mean(Xshort, axis=2, keepdims=True), axis=1, keepdims=True))
    iaf_ev_byoutcome_df = pd.DataFrame(iaf_ev_byoutcome, index=Xshortdims['outcome'], columns=Xshortdims['iaf'])

    fff = dict()
    for i_outcome, outcome in enumerate(Xshortdims['outcome']):
        iaflist = list()
        for i_pot, pot in enumerate(Xshortdims['pot']):
            ddd = pd.DataFrame(Xshort[i_outcome, i_pot, :, :], columns=Xshortdims['iaf'])
            ddd['pots'] = pot
            ddd['prob'] = ddd.shape[0]**-1
            iaflist.append(ddd)
        ddd2 = pd.concat(iaflist).set_index('pots')
        columns_ = list()
        for label in ddd2.columns.to_list():
            if label != 'prob':
                columns_.append(('emotionIntensities', label))
            else:
                columns_.append(('prob', 'prob'))
        ddd2.columns = pd.MultiIndex.from_tuples(columns_)
        fff[outcome] = ddd2

    iaf_labels_formatted_dict = dict()
    for label in iaf_labels:
        iaf_labels_formatted_dict[label] = format_iaf_labels_superphrase([label])[0]

    new_emo_plot_iaf(fff, emoevdf=iaf_ev_byoutcome_df, scale_factor=13.0, bandwidth=0.07, emotions_abbriv=iaf_labels_formatted_dict, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'iaf_EVscatter_genericplayers.pdf')

    iaf_labels_formatted_multiline_dict = dict()
    for label in iaf_labels:
        iaf_labels_formatted_multiline_dict[label] = format_iaf_labels_multiline([label])[0]

    new_emo_plot_iaf(fff, emoevdf=iaf_ev_byoutcome_df, scale_factor=13.0, bandwidth=0.07, emotions_abbriv=iaf_labels_formatted_multiline_dict, xrotate=False, plotParam=plotParam, fig_outpath=plotParam['figsOut'] / 'iaf_EVscatter_genericplayers_abbrev.pdf')

    # %%

    emotions = ppldata['empiricalEmotionJudgments']['CC']['emotionIntensities'].columns.to_list()

    ### plot guilt param
    # ### plot distribution of values
    # figout, axs = plt.subplots(2, 2, figsize=(12, 8))
    # for i_outcome, outcome in enumerate(outcomes):
    #     pot = list(dm_dict_model[outcome].keys())[-int(len(dm_dict_model[outcome].keys()) / 4)]
    #     ax = axs[[(0, 0), (0, 1), (1, 0), (1, 1)][i_outcome]]
    #     sns.distplot(dm_dict_model[outcome][pot].values[mask], kde=False, color='red', bins=np.linspace(-1, 1, 100), ax=ax)
    #     sns.distplot(dmemp_mean[outcome].values[mask], kde=False, color='blue', bins=np.linspace(-1, 1, 100), ax=ax)
    # figout.suptitle(f'distribution of dm correlation values (red=model,blue=mean emp [over pots, within outcome])\n{dsname} \n(${pot}) {outcome})')
    # figout.tight_layout()
    # figsout.append((modelfigsout / 'distribution_of_dm_correlations.pdf'.format(), figout))
    # plt.close(figout)

    # %%

    ###
    ### plot webppl data
    ###

    with open(cpar_path_priors, 'rb') as f:
        cpar_priors = dill.load(f)
    cpar_priors.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    print(f"-------followup_analyses cpar")
    followup_analyses(cpar, ppldata, ppldata_exp3, distal_prior_ppldata, plotParam=plotParam)

    ppldata_priors, _, _ = get_ppldata_cpar(cpar_priors)
    print(f"-------followup_analyses cpar_priors")
    followup_analyses(cpar_priors, *get_ppldata_cpar(cpar_priors), plotParam=plotParam)

    # %%

    #### get decision frequencies

    seed_bs_a1 = 1
    nbootstrap_samples_a1_MAP = 10000

    decisions = dict()
    for a1 in ['C', 'D']:
        decisions[a1] = dict()
        for stimid in distal_prior_ppldata:
            temp_freq = distal_prior_ppldata[stimid][a1]['level2'].reset_index()
            expected_prob = temp_freq.loc[temp_freq['level_1'] == a1, 'prob'].mean()
            decisions[a1][stimid] = expected_prob
            decision_prob_MAPs = np.array(list(decisions[a1].values()))
        seed_bs_a1 += 1
        rng_bs_a1 = np.random.default_rng(seed_bs_a1)
        median_pe, bs_median_ci = bootstrap_pe(decision_prob_MAPs, alpha=0.05, bootstrap_samples=nbootstrap_samples_a1_MAP, estimator=np.median, flavor='percentile', rng=rng_bs_a1)
        # bs_median_ci = bootstrap_median(decision_prob_MAPs, alpha=alpha_ci, bootstrap_samples=nbootstrap_samples_a1_MAP, atol=1e-08)
        plotParam['save_text_var'].write('{:0.0f}\% [{:0.0f}, {:0.0f}]%'.format(median_pe * 100, bs_median_ci[0] * 100, bs_median_ci[1] * 100), f'a1_frequency_level2_{a1}_alldistalstim.txt')
        plotParam['save_text_var'].write("{:0.0f}\%%".format(np.min(decision_prob_MAPs) * 100), f'a1_frequency_level2_{a1}_alldistalstim_min.txt')
        plotParam['save_text_var'].write("{:0.0f}\%%".format(np.max(decision_prob_MAPs) * 100), f'a1_frequency_level2_{a1}_alldistalstim_max.txt')

    seed_bs_a1 = 1
    for level_ in ['level0', 'level2']:
        for a1 in ['C', 'D']:
            temp_freq = ppldata[level_].reset_index()
            decision_prob_MAPs = temp_freq.loc[temp_freq['level_1'] == a1, 'prob'].to_numpy()
            seed_bs_a1 += 1
            rng_bs_a1 = np.random.default_rng(seed_bs_a1)
            a1_mean_pe, bs_a1_mean_ci = bootstrap_pe(decision_prob_MAPs, alpha=0.05, bootstrap_samples=nbootstrap_samples_a1_MAP, estimator=np.mean, flavor='percentile', rng=rng_bs_a1)
            # bs_median_ci = bootstrap_median(decision_prob_MAPs, alpha=alpha_ci, bootstrap_samples=nbootstrap_samples_a1_MAP, atol=1e-08)
            plotParam['save_text_var'].write('{:0.0f}\% [{:0.0f}, {:0.0f}]%'.format(a1_mean_pe * 100, bs_a1_mean_ci[0] * 100, bs_a1_mean_ci[1] * 100), f'a1_frequency_{level_}_{a1}.txt')

    # %%

    ppldata['level4IAF']['CC'].loc[:, ('compositeWeights', 'bAIA')].mean()
    ppldata['level4IAF']['DC'].loc[:, ('compositeWeights', 'bAIA')].mean()

    ppldata['level4IAF']['CD'].loc[:, ('compositeWeights', 'bAIA')].mean()
    ppldata['level4IAF']['DD'].loc[:, ('compositeWeights', 'bAIA')].mean()

    ppldata_priors['level3']['C']

    ppldata_priors['level4IAF']['CC'].loc[:, ('compositeWeights', 'bMoney')].mean()
    ppldata_priors['level4IAF']['DC'].loc[:, ('compositeWeights', 'bMoney')].mean()
    ppldata_priors['level4IAF']['CD'].loc[:, ('compositeWeights', 'bMoney')].mean()
    ppldata_priors['level4IAF']['DD'].loc[:, ('compositeWeights', 'bMoney')].mean()

    ppldata_priors['level4IAF']['CC'].loc[:, ('compositeWeights', 'bAIA')].mean()
    ppldata_priors['level4IAF']['DC'].loc[:, ('compositeWeights', 'bAIA')].mean()
    ppldata_priors['level4IAF']['CD'].loc[:, ('compositeWeights', 'bAIA')].mean()
    ppldata_priors['level4IAF']['DD'].loc[:, ('compositeWeights', 'bAIA')].mean()

    ppldata_priors['level4IAF']['CC'].loc[:, ('compositeWeights', 'bDIA')].mean()
    ppldata_priors['level4IAF']['DC'].loc[:, ('compositeWeights', 'bDIA')].mean()
    ppldata_priors['level4IAF']['CD'].loc[:, ('compositeWeights', 'bDIA')].mean()
    ppldata_priors['level4IAF']['DD'].loc[:, ('compositeWeights', 'bDIA')].mean()

    ppldata_priors['level4IAF']['CC'].loc[:, ('compositeWeights', 'rMoney')].mean()
    ppldata_priors['level4IAF']['DC'].loc[:, ('compositeWeights', 'rMoney')].mean()
    ppldata_priors['level4IAF']['CD'].loc[:, ('compositeWeights', 'rMoney')].mean()
    ppldata_priors['level4IAF']['DD'].loc[:, ('compositeWeights', 'rMoney')].mean()

    ppldata_priors['level4IAF']['CC'].loc[:, ('compositeWeights', 'rAIA')].mean()
    ppldata_priors['level4IAF']['DC'].loc[:, ('compositeWeights', 'rAIA')].mean()
    ppldata_priors['level4IAF']['CD'].loc[:, ('compositeWeights', 'rAIA')].mean()
    ppldata_priors['level4IAF']['DD'].loc[:, ('compositeWeights', 'rAIA')].mean()

    ppldata_priors['level4IAF']['CC'].loc[:, ('compositeWeights', 'rDIA')].mean()
    ppldata_priors['level4IAF']['DC'].loc[:, ('compositeWeights', 'rDIA')].mean()
    ppldata_priors['level4IAF']['CD'].loc[:, ('compositeWeights', 'rDIA')].mean()
    ppldata_priors['level4IAF']['DD'].loc[:, ('compositeWeights', 'rDIA')].mean()

    (1 - ppldata_priors['level4IAF']['CC'].loc[:, ('emotionIntensities', 'PEa2raw')]).mean()
    (1 - ppldata_priors['level4IAF']['DC'].loc[:, ('emotionIntensities', 'PEa2raw')]).mean()
    0 - ppldata_priors['level4IAF']['CD'].loc[:, ('emotionIntensities', 'PEa2raw')].mean()
    0 - ppldata_priors['level4IAF']['DD'].loc[:, ('emotionIntensities', 'PEa2raw')].mean()

    # %%

    for i_obslevel, obslevel in enumerate([0, 2]):
        # levellabel = {0: 'Base', 2: 'Reputation'}[obslevel]
        # ppldata['labels']['decisions']
        # data['prob']['C']
        aaa = ppldata['level{}'.format(obslevel)].groupby(level=1).mean()

        ppldata['level{}'.format(obslevel)]

    import numpy as np
    aaa = np.square(np.random.randn(2000, 50, 3))
    bbb = np.dstack([aaa, aaa, aaa, aaa])

    from scipy.special import logsumexp

    cc = np.mean(aaa, axis=2)
    dd = np.mean(cc, axis=1)
    ee = logsumexp(dd, axis=0) - np.log(aaa.shape[0])

    cc1 = np.mean(bbb, axis=2)
    dd1 = np.mean(cc1, axis=1)
    ee1 = logsumexp(dd1, axis=0) - np.log(bbb.shape[0])

    print(f"{ee}  vs  {ee1}")

    # %%
    # %%

    print('=====Finished plot summary====')
