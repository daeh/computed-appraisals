#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_import_wppljson_utils.py
"""


def importPPLdata(jsondata, supports):
    '''
    for importing data with a finite number of known supports (e.g. probability of a1)
    '''
    import numpy as np
    import pandas as pd

    probs = np.zeros(len(supports), dtype='float')
    for idx, obs in enumerate(supports):
        if obs in jsondata["support"]:
            probs[idx] = jsondata["probs"][jsondata["support"].index(obs)]
    df = pd.DataFrame(dict(support=supports, prob=probs))
    assert df.isnull().sum().sum() == 0
    df.set_index('support', inplace=True)
    np.testing.assert_almost_equal(df['prob'].sum(), 1.0, decimal=9, err_msg='probabilities do not sum to 1', verbose=True)
    return df


def importPPLdataDict(datain):
    import numpy as np
    import pandas as pd
    import itertools
    import sys

    error_ = False
    overflow_ = list()
    support = list()
    probs = list()
    featureList = list(datain['support'][0].keys())
    subfeatureList = list()
    for feature in featureList:
        subfeatureList.append(list(range(len(datain['support'][0][feature]))))
    for obs in range(len(datain['probs'])):
        supportRow = list()
        for feature in featureList:
            ### Test datatype ###
            for val_ in datain['support'][obs][feature]:
                if not (isinstance(val_, float) or isinstance(val_, int)) or val_ is None:
                    error_ = True
                    print('featureList')
                    print(featureList)
                    print('subfeatureList')
                    print(subfeatureList)
                assert not error_, f"val >>{val_}<< in feature >>{feature}<< in obs >>{obs}<< is not float/int but >>{type(val_)}<<"

                if np.abs(val_) > sys.float_info.max / 10.0 or (np.abs(val_) > 0 and np.abs(val_) < sys.float_info.min * 10.0):
                    overflow_.append(val_)
                    if np.abs(val_) > sys.float_info.max:
                        print(f"Numerical Warning, val >>{val_}<< in feature >>{feature}<< in obs >>{obs}<< exceeds system thresholds of ({sys.float_info.min}, {sys.float_info.max})")
                        val_ = np.sign(val_) * sys.float_info.max

            supportRow.append(datain['support'][obs][feature])
        support.append(list(itertools.chain(*supportRow)))
        probs.append(datain['probs'][obs])
    df = pd.DataFrame(support, columns=makeLabelHierarchy([featureList, subfeatureList]), dtype=float)
    se = pd.Series(probs, dtype=float)
    df[('prob', 'prob')] = se.values
    assert df.isnull().sum().sum() == 0
    assert not np.any(np.isnan(df.to_numpy()))

    return df, overflow_


def importPPLdataWithLinkerFn(jsondata, labels, repackageFn):
    import numpy as np
    import pandas as pd

    probs = np.array(jsondata['probs'], dtype='float')
    xs = np.full((len(jsondata['support']), len(labels)), np.nan, dtype=float)
    for i_obs, obs in enumerate(jsondata['support']):
        xs[i_obs, :] = repackageFn(obs)

    df = pd.DataFrame(data=np.insert(xs, xs.shape[1], probs, axis=1), columns=(np.append(labels, 'prob')))
    assert df.isnull().sum().sum() == 0
    return df


def makeLabelHierarchy(labels):
    import pandas as pd
    import itertools

    featureClassLabels = list()
    for idx, label in enumerate(labels[0]):
        featureClassLabels.append(list(itertools.repeat(label, len(labels[1][idx]))))
    arrays = [list(itertools.chain(*featureClassLabels)), list(itertools.chain(*labels[1]))]
    tuples = list(zip(*arrays))
    cols = pd.MultiIndex.from_tuples(tuples)
    return cols
