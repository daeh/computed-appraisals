#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_webppl_utils.py
"""


def condensePPLdata(marginaldf, col_labels, bypassCheck=False):
    import numpy as np

    if not bypassCheck:
        np.testing.assert_approx_equal(marginaldf.prob.sum(), 1.0)

    return marginaldf.groupby(col_labels)['prob'].sum()


def marginalizeContinuous(dfin, featureList, bypassCheck=False):

    ### uses column labels
    df = dfin.copy(deep=True)
    ### drop the first column heiarchy if column multiindex
    if (isinstance(df.columns[0], list) or isinstance(df.columns[0], tuple)) and len(list(df.columns)[0]) > 1:
        df.columns = df.columns.droplevel(0)
    # avilableGroupings = list(df.columns[:-1])
    return condensePPLdata(df, featureList, bypassCheck)  # df.groupby(featureList)['prob'].sum()


def unweightProbabilities(dfin, nobs=None):
    import numpy as np
    from copy import deepcopy

    # convert series with index (e.g. from marginalizeContinuous) into df
    if len(dfin.shape) == 1:
        dfincopy = dfin.reset_index()
    else:
        dfincopy = deepcopy(dfin)

    if dfincopy.shape[0] == 0:
        np.testing.assert_equal(np.any(np.equal([nobs, nobs], [0, None])), True)
        dfincopy.loc[0, :] = np.full((1, dfincopy.shape[1]), np.nan, dtype=float)
        dfincopy.loc[0, 'prob'] = 0.0
        nobs = 1
        df = dfincopy
    else:
        # get index of non-zero probability Observations
        nzidx = np.squeeze(dfincopy.loc[:, 'prob'].values > 0)
        # drop zeros
        df = dfincopy.iloc[nzidx, :]  # return copy

    newprobs = np.squeeze(df.loc[:, 'prob'].values)

    # assert that df is normalized
    np.testing.assert_almost_equal(newprobs.sum(), 1.0)

    ## get lowest common denominator
    if nobs is None:
        ### this is bypassed for empty df since nobs is set to 1
        lcd = newprobs.min()
    else:
        lcd = 1 / nobs
    if not np.isclose(lcd, newprobs.max()):
        repfactor = newprobs / lcd
        repfactor_int = np.round(repfactor).astype(int)
        np.testing.assert_almost_equal(repfactor, np.round(repfactor), decimal=6)
        df = df.iloc[df.index.repeat(repfactor_int), :]
        df.loc[:, 'prob'] = lcd
        df.reset_index(drop=True, inplace=True)

    np.testing.assert_equal(df.shape[0], int(np.round(1 / lcd)))

    return df


def getEV(df, bypassCheck=False):
    import numpy as np
    if df.size:
        x = df.index
        p = df.values
        ev = np.inner(x, p)
        var = np.inner(p, np.square(x)) - np.inner(x, p)**2
    else:
        ev, var = np.nan, np.nan
    if not bypassCheck:
        np.testing.assert_almost_equal(p.sum(), 1.0)
    return ev, var


def marginalizeContinuousAcrossMultilevel(df, col_labels, nobs=None, set_prior=None, bypassCheck=False):
    '''
    concatonates MultiIndex column df according to first column level
    '''
    import numpy as np
    import pandas as pd

    mdf = df.loc[:, (*col_labels, 'prob')]
    topkeys = np.unique(mdf.index.get_level_values(0))
    tempdflist = [None] * len(topkeys)
    for i_key, key in enumerate(topkeys):
        tempdflist[i_key] = mdf.loc[key, ].copy(deep=True)

    if nobs is None:
        ### normalize assuming pot dfs represent the same number of observations (normalization for simulation, not empirical)
        nobs = np.array([1.0] * len(topkeys), dtype='float')

    ### if there are unequal observations per set, adjust probabilities accordingly

    nobs_type = {pd.core.series.Series: 'series', np.ndarray: 'ndarray', list: 'list'}[type(nobs)]

    performDEBUGcheck = False
    if set_prior is None:
        set_prior = np.full(len(nobs), 1 / len(nobs), dtype=float)
        performDEBUGcheck = True
    elif isinstance(set_prior, list):
        set_prior = np.array(set_prior)

    np.testing.assert_almost_equal(set_prior.sum(), np.round(set_prior.sum()))  # make sure it's an integer value
    if np.round(set_prior.sum()) > 1.0:
        set_prior = set_prior / np.round(set_prior.sum())  # normalize if sum is not already 1

    set_prob = set_prior * nobs / np.sum(nobs) * np.inner(set_prior, nobs / np.sum(nobs))**-1

    for i_key, key in enumerate(topkeys):
        if nobs_type == 'series':
            this_prob = set_prob.loc[key]
        elif nobs_type == 'ndarray':
            this_prob = set_prob[i_key]
        elif nobs_type == 'list':
            this_prob = set_prob[i_key]
        else:
            this_prob = set_prob[i_key]

        if performDEBUGcheck:
            if nobs_type == 'series':
                this_nobs = nobs.loc[key]
            elif nobs_type == 'ndarray':
                this_nobs = nobs[i_key]
            elif nobs_type == 'list':
                this_nobs = nobs[i_key]
            else:
                this_nobs = nobs[i_key]

            np.testing.assert_almost_equal(tempdflist[i_key].prob.multiply(this_prob).values, tempdflist[i_key].prob.multiply(this_nobs).divide(np.sum(nobs)).values)

        tempdflist[i_key].prob = tempdflist[i_key].prob.multiply(this_prob)

    final_expanded = pd.concat(tempdflist)
    assert final_expanded.shape[0] == mdf.shape[0]

    return condensePPLdata(final_expanded, col_labels, bypassCheck)
