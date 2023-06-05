#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_bootstrap_empirical.py
"""


def calc_bootstrap_ci(theta_hat, theta_tilde_samples, alpha=0.05, flavor='percentile'):
    '''
    theta_hat : statistic estimate from the original sample
    theta_tilde : statistic estimate from a bootstrap sample
    '''
    import numpy as np

    percentiles = (alpha * 100. / 2., 100. - alpha * 100. / 2.)

    if flavor == 'percentile':
        q_low, q_high = np.percentile(theta_tilde_samples, percentiles, axis=0)
        ci_ = np.array([q_low, q_high])

    elif flavor == 'basic':
        '''
        1 - alpha = P(q_{alpha/2} - theta_hat <= theta_tilde - theta_hat <= q_{1-alpha/2} - theta_hat)
                  = P(2*theta_hat - q_{1-alpha/2} <= theta <= 2*theta_hat - q_{alpha/2})
        '''
        q_low, q_high = np.percentile(theta_tilde_samples, percentiles, axis=0)
        bias = np.mean(theta_tilde_samples) - theta_hat
        ci_ = np.array([2 * theta_hat - q_high, 2 * theta_hat - q_low])

    return ci_


def bootstrap_pe(x, alpha=0.05, bootstrap_samples=1000, estimator=None, flavor='percentile', rng=None):
    from sklearn.utils import resample
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    seeds = rng.integers(low=1, high=np.iinfo(np.int32).max, size=bootstrap_samples)

    if flavor == 'percentile':
        '''
        theta_hat : statistic estimate from the original sample
        theta_tilde : statistic estimate from a bootstrap sample
        '''
        assert x.shape[0] > 1
        assert x.size == x.shape[0]

        theta_hat = estimator(x)

        theta_tilde_samples = np.full((bootstrap_samples,), np.nan)
        for ii in np.arange(bootstrap_samples):
            theta_tilde_samples[ii] = estimator(resample(x, replace=True, n_samples=x.shape[0], random_state=seeds[ii]))

        pe_ = theta_hat
        ci_ = calc_bootstrap_ci(theta_hat, theta_tilde_samples, alpha=alpha, flavor=flavor)

    elif flavor == 'basic':
        '''
        theta_hat : statistic estimate from the original sample
        theta_tilde : statistic estimate from a bootstrap sample
        bias : E[theta_tilde] - theta_hat
        1 - alpha = P(q_{alpha/2} - theta_hat <= theta_tilde - theta_hat <= q_{1-alpha/2} - theta_hat)
                  = P(2*theta_hat - q_{1-alpha/2} <= theta <= 2*theta_hat - q_{alpha/2})
        '''
        assert x.shape[0] > 1
        assert x.size == x.shape[0]

        theta_hat = estimator(x)

        theta_tilde_samples = np.full((bootstrap_samples,), np.nan)
        for ii in np.arange(bootstrap_samples):
            theta_tilde_samples[ii] = estimator(resample(x, replace=True, n_samples=x.shape[0], random_state=seeds[ii]))

        bias = np.mean(theta_tilde_samples) - theta_hat

        pe_ = theta_hat
        ci_ = calc_bootstrap_ci(theta_hat, theta_tilde_samples, alpha=alpha, flavor=flavor)

    return pe_, ci_


def bootstrap_empirical(cfg, n_bootstrap=1000, alpha=None, seed=None):

    import numpy as np
    from scipy.stats import pearsonr
    from cam_utils import concordance_corr_, adjusted_corr_
    from cam_emotorch import EmoTorch
    from cam_emotorch_utils import reformat_ppldata_allplayers

    eto = EmoTorch(verbose=False)
    eto.init_cfg(cfg)

    teststimid = cfg['testset']

    ###
    ### reformat webppl data
    ###

    feature_selector_label, feature_selector = cfg['pytorch_spec']['feature_selector']
    ppldatasets = reformat_ppldata_allplayers(cpar_path=cfg['cpar_path_str'], feature_selector=feature_selector, feature_selector_label=feature_selector_label)

    specific_pots = sorted(ppldatasets['239_1']['X']['pot'].unique().tolist())
    assert len(specific_pots) == 8

    ###
    ### collect train data for preprocessing transform
    ###

    X_generic = ppldatasets['generic']['X']
    Y_generic = ppldatasets['generic']['Y']

    ###
    ### fit preprocessing transform
    ###

    eto.preprocess_fit_transform([ppldatasets['generic']['X']], cache=False)

    ###
    ### collect test data
    ###

    ppldatasets_test = dict()
    ppldatasets_test['generic'] = dict(
        X=X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y=Y_generic.loc[Y_generic['pot'].isin(specific_pots), :],
    )
    for stimid in teststimid:
        ppldatasets_test[stimid] = ppldatasets[stimid]

    eto.preprocess_apply_transform(ppldatasets_test, label='test', cache=False)

    #################

    data_dict = eto.get_torchdata()

    rng = np.random.default_rng(seed)

    stimids_withgeneric = list(data_dict['test'].keys())  # generic, specplayer1, ...specplayer20
    assert len(stimids_withgeneric) == 21
    stimids_specific = [stimid for stimid in stimids_withgeneric if stimid != 'generic']
    assert len(stimids_specific) == 20

    outcomes = data_dict['test']['generic']['Yshortdims']['outcome']
    pots = data_dict['test']['generic']['Yshortdims']['pot']
    emotions = data_dict['test']['generic']['Yshortdims']['emotion']
    n_outcomes = len(outcomes)
    n_pots = len(pots)
    n_emotions = len(emotions)
    assert n_outcomes == 4
    assert n_pots == 8
    assert n_emotions == 20

    empirical_data_dict = dict()
    for stimid in stimids_withgeneric:
        empirical_data_dict[stimid] = data_dict['test'][stimid]['Yshort']

    observed_ev_bypotoutcome = np.full([len(stimids_withgeneric), n_outcomes, n_pots, n_emotions], np.nan, dtype=float)
    for i_stimid, stimid in enumerate(stimids_withgeneric):
        for i_outcome, outcome in enumerate(outcomes):
            for i_pot, pot in enumerate(pots):
                observed_ev_bypotoutcome[i_stimid, i_outcome, i_pot, :] = np.mean(empirical_data_dict[stimid][outcome][i_pot], axis=0)

    observed_ev_byoutcome = np.full([len(stimids_withgeneric), n_outcomes, n_emotions], np.nan, dtype=float)
    for i_stimid, stimid in enumerate(stimids_withgeneric):
        for i_outcome, outcome in enumerate(outcomes):
            observed_ev_byoutcome[i_stimid, i_outcome, :] = np.mean(observed_ev_bypotoutcome[i_stimid, i_outcome, :, :], axis=0)

    obs_data = dict()
    obs_data['ev_bypotoutcome_player'] = dict()
    obs_data['deltas_byoutcome_player'] = dict()
    for i_stimid, stimid in enumerate(stimids_withgeneric):
        if stimid != 'generic':
            obs_data['ev_bypotoutcome_player'][stimid] = observed_ev_bypotoutcome[i_stimid, :, :]
            ### average over pots ###
            ygeneric_ = observed_ev_byoutcome[[0], :, :]
            yspecific_ = observed_ev_byoutcome[[i_stimid], :, :]
            assert np.array_equal(ygeneric_.shape, yspecific_.shape)
            obs_data['deltas_byoutcome_player'][stimid] = np.squeeze(yspecific_ - ygeneric_)

    ####################

    print(f'starting bootstrap n={n_bootstrap}')
    bs_ev_array = np.full([n_bootstrap, len(stimids_withgeneric), n_outcomes, n_pots, n_emotions], np.nan, dtype=float)
    for i_stimid, stimid in enumerate(stimids_withgeneric):
        print(f'istim {i_stimid+1}/{len(stimids_withgeneric)}')
        for i_outcome, outcome in enumerate(outcomes):
            for i_pot, pot in enumerate(pots):
                y_ = empirical_data_dict[stimid][outcome][i_pot]
                for i_bs in range(n_bootstrap):
                    yidx = rng.choice(range(y_.shape[0]), replace=True, size=y_.shape[0])
                    # yidx = np.array(range(y_.shape[0]))
                    bs_ev_array[i_bs, i_stimid, i_outcome, i_pot, :] = np.mean(y_[yidx, :], axis=0)
    assert not np.any(np.isnan(bs_ev_array))
    print(f'finished bootstrap n={n_bootstrap}')

    bs_data = dict()
    bs_data['ev_bypotoutcome_player'] = dict()
    bs_data['deltas_byoutcome_player'] = dict()
    assert len(stimids_specific) == 20
    for i_stimid, stimid in enumerate(stimids_withgeneric):
        if stimid != 'generic':
            bs_data['ev_bypotoutcome_player'][stimid] = bs_ev_array[:, i_stimid, :, :, :]

            #### deltas by player ####

            ## average over pots ##

            ygeneric_ = np.mean(bs_ev_array[:, [0], :, :, :], axis=3, keepdims=True)  # calc bs delta relative to bootstrapped generic
            yspecific_ = np.mean(bs_ev_array[:, [i_stimid], :, :, :], axis=3, keepdims=True)

            bs_data['deltas_byoutcome_player'][stimid] = np.squeeze(yspecific_ - ygeneric_)

    ###
    ### calc stats ###
    ###

    bs_stats = dict()

    #### absolute intensities overall (by pot-outcome, across emotions, players) ####
    obsev_allplayers = np.stack(list(obs_data['ev_bypotoutcome_player'].values()), axis=0)
    bsev_allplayers = np.stack(list(bs_data['ev_bypotoutcome_player'].values()), axis=1)
    ccc_list = list()
    pearr_list = list()
    for i_bs in range(n_bootstrap):
        y_obs = obsev_allplayers[:, :, :, :].flatten()
        y_bs = bsev_allplayers[i_bs, :, :, :, :].flatten()
        ccc_list.append(concordance_corr_(y_obs, y_bs))
        pearr_list.append(pearsonr(y_obs, y_bs)[0])
    bs_stats['ev_bypotoutcome_overall'] = dict(
        ccc=ccc_list,
        pearsonr=pearr_list,
    )

    #### absolute intensities by emotion (by pot-outcome, across players) ####
    bs_stats['ev_bypotoutcome_emotion'] = dict()
    for i_emotion, emotion in enumerate(emotions):
        ccc_list = list()
        pearr_list = list()
        for i_bs in range(n_bootstrap):
            y_obs = obsev_allplayers[:, :, :, i_emotion].flatten()
            y_bs = bsev_allplayers[i_bs, :, :, :, i_emotion].flatten()
            ccc_list.append(concordance_corr_(y_obs, y_bs))
            pearr_list.append(pearsonr(y_obs, y_bs)[0])
        bs_stats['ev_bypotoutcome_emotion'][emotion] = dict(
            ccc=ccc_list,
            pearsonr=pearr_list,
        )

    #### deltas overall (by outcome, across players) ####
    obsdeltas_allplayers = np.stack(list(obs_data['deltas_byoutcome_player'].values()), axis=0)
    bsdeltas_allplayers = np.stack(list(bs_data['deltas_byoutcome_player'].values()), axis=1)
    ccc_list = list()
    pearr_list = list()
    for i_bs in range(n_bootstrap):
        y_obs = obsdeltas_allplayers[:, :, :].flatten()
        y_bs = bsdeltas_allplayers[i_bs, :, :, :].flatten()
        ccc_list.append(concordance_corr_(y_obs, y_bs))
        pearr_list.append(pearsonr(y_obs, y_bs)[0])
    bs_stats['deltas_byoutcome_overall'] = dict(
        ccc=ccc_list,
        pearsonr=pearr_list,
    )

    #### deltas by player (by outcome) ####
    obsdeltas_allplayers = np.stack(list(obs_data['deltas_byoutcome_player'].values()), axis=0)
    bs_stats['deltas_byoutcome_player'] = dict()
    for stimid in stimids_specific:
        ccc_list = list()
        pearr_list = list()
        adjpearr_list = list()
        for i_bs in range(n_bootstrap):
            y_obs = obs_data['deltas_byoutcome_player'][stimid][:, :].flatten()
            y_bs = bs_data['deltas_byoutcome_player'][stimid][i_bs, :, :].flatten()
            ccc_list.append(concordance_corr_(y_obs, y_bs))
            pearr_list.append(pearsonr(y_obs, y_bs)[0])
            adjpearr_list.append(adjusted_corr_(y_obs, y_bs, obsdeltas_allplayers.flatten()))
        bs_stats['deltas_byoutcome_player'][stimid] = dict(
            ccc=ccc_list,
            pearsonr=pearr_list,
            adjusted_pearsonr=adjpearr_list,
        )

    specific_player_order_ = dict()
    for stimid in stimids_specific:
        specific_player_order_[stimid] = np.median(bs_stats['deltas_byoutcome_player'][stimid]['ccc'])

    specific_player_order_tuple = sorted(specific_player_order_.items(), key=lambda x: x[1], reverse=True)
    specific_player_order = list(dict(specific_player_order_tuple).keys())

    ###
    ### calc bs ci ###
    ###

    bs_ci = dict()
    bs_ci['ev_bypotoutcome_overall'] = dict()
    for statistic, x_bs_vals in bs_stats['ev_bypotoutcome_overall'].items():
        x_bs_median = np.median(x_bs_vals)
        x_bs_ci = calc_bootstrap_ci(x_bs_median, x_bs_vals, alpha=alpha, flavor='percentile')
        bs_ci['ev_bypotoutcome_overall'][statistic] = dict(pe=x_bs_median, ci=x_bs_ci)

    bs_ci['deltas_byoutcome_overall'] = dict()
    for statistic, x_bs_vals in bs_stats['deltas_byoutcome_overall'].items():
        x_bs_median = np.median(x_bs_vals)
        x_bs_ci = calc_bootstrap_ci(x_bs_median, x_bs_vals, alpha=alpha, flavor='percentile')
        bs_ci['deltas_byoutcome_overall'][statistic] = dict(pe=x_bs_median, ci=x_bs_ci)

    bs_ci['ev_bypotoutcome_emotion'] = dict()
    for itemlabel, itemdata in bs_stats['ev_bypotoutcome_emotion'].items():
        bs_ci['ev_bypotoutcome_emotion'][itemlabel] = dict()
        for statistic, x_bs_vals in itemdata.items():
            x_bs_median = np.median(x_bs_vals)
            x_bs_ci = calc_bootstrap_ci(x_bs_median, x_bs_vals, alpha=alpha, flavor='percentile')
            bs_ci['ev_bypotoutcome_emotion'][itemlabel][statistic] = dict(pe=x_bs_median, ci=x_bs_ci)

    bs_ci['deltas_byoutcome_player'] = dict()
    for itemlabel, itemdata in bs_stats['deltas_byoutcome_player'].items():
        bs_ci['deltas_byoutcome_player'][itemlabel] = dict()
        for statistic, x_bs_vals in itemdata.items():
            x_bs_median = np.median(x_bs_vals)
            x_bs_ci = calc_bootstrap_ci(x_bs_median, x_bs_vals, alpha=alpha, flavor='percentile')
            bs_ci['deltas_byoutcome_player'][itemlabel][statistic] = dict(pe=x_bs_median, ci=x_bs_ci)

    return bs_ci, dict(bs_iterstats=bs_stats, specific_player_order=specific_player_order, bs_data=dict(observered=obs_data, bootstrapped=bs_data))
