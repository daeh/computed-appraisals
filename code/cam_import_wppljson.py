#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_import_wppljson.py
"""


def _dictkeys_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def _agentRepackage(observation):
    import numpy as np
    import itertools
    return np.array(list(itertools.chain.from_iterable([observation['weights'], [observation['estimated_p2']]])), dtype=float)


def importPPLmodel_(modelDataPath, wpplparam, pots, verbose, suffix=''):
    import json
    import itertools
    import numpy as np
    import pandas as pd
    from pprint import pprint
    import warnings
    from cam_import_wppljson_utils import importPPLdata, importPPLdataDict, importPPLdataWithLinkerFn, makeLabelHierarchy

    with open(modelDataPath / 'paramin.json', mode='rt') as data_file:
        wpplparam_loaded = json.load(data_file)

    added, removed, modified, same = _dictkeys_compare(wpplparam, wpplparam_loaded)
    if 'pot' in added:
        added.remove('pot')

    with open(wpplparam_loaded['path_to_kde'], mode='rt') as data_file:
        json_passed_to_wppl = json.load(data_file)

    if len(added) > 0:
        warnings.warn('Keys added')
        print('Added:')
        for key in added:
            print(key)
    if len(removed) > 0:
        warnings.warn('Keys removed')
        print('Removed:')
        for key in removed:
            print(key, removed)
    if len(modified) > 0:
        warnings.warn('Keys modified')
        print('Modified:')
        for key in modified:
            print(key, modified[key])

    ppldata = {'pots' + suffix: pots}

    with open(modelDataPath / 'featureLabels.json', mode='rt') as data_file:
        jsonin = json.load(data_file)
    ppldata['labels' + suffix] = {'baseFeatures': jsonin[0],
                                  'reputationFeatures': jsonin[1],
                                  'combinedFeatures': list(itertools.chain.from_iterable(jsonin))}
    with open(modelDataPath / 'outcomeFeatureLabels.json', mode='rt') as data_file:
        ppldata['labels' + suffix]['outcomeFeatures'] = json.load(data_file)

    ppldata['labels' + suffix]['decisions'] = ['C', 'D']
    ppldata['labels' + suffix]['outcomes'] = ['CC', 'CD', 'DC', 'DD']

    ppldata['pots_byoutcome'] = {'all': pots}
    for outcome in ppldata['labels' + suffix]['outcomes']:
        ppldata['pots_byoutcome'][outcome] = pots

    if verbose:
        pprint(ppldata)

    # ### check for errors in log files
    for ipot, pot in enumerate(pots):
        data_file_path_ = modelDataPath / 'log_{0}USD_{1}.json'.format(int(pot) if int(pot) == pot else '{}'.format(pot).replace('.', 'c'), wpplparam['m4iaf']['method'])
        with open(data_file_path_, mode='rt') as data_file:
            jsonin = json.load(data_file)
            assert jsonin['numerical_errors'] == 0, f"Error: {jsonin['numerical_errors']} numerical errors found in {data_file_path_}"

    # ### level 0 & 2
    for obslevel in [0, 2]:
        tempdflist = [None] * len(pots)
        for ipot, pot in enumerate(pots):
            with open(modelDataPath / 'level{3}_{0}USD_{1}{2}_decisionFrequency.json'.format(int(pot) if int(pot) == pot else '{}'.format(pot).replace('.', 'c'), wpplparam['m{}'.format(obslevel)]['method'], wpplparam['m{}'.format(obslevel)]['samples'], obslevel), mode='rt') as data_file:
                jsonin = json.load(data_file)
            tempdflist[ipot] = importPPLdata(jsonin, ppldata['labels']['decisions'])
        ppldata['level{}'.format(obslevel) + suffix] = pd.concat(tempdflist, axis=0, keys=pots, names=['pots', None])
        ###
        nobsdf = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['decisions'])), wpplparam['m{}'.format(obslevel)]['samples']), index=pots, columns=ppldata['labels']['decisions'], dtype=int)
        nobsdf.index.set_names(['pots'], inplace=True)

    # ### level 1 & 3

    for i_obslevel, obslevel in enumerate([1, 3]):
        ppldata['level{}'.format(obslevel) + suffix] = dict()
        labellabel = ['baseFeatures', 'combinedFeatures'][i_obslevel]
        # feature_list = list(itertools.chain.from_iterable([ppldata['labels'][labellabel], ['pi_a2']]))
        if obslevel == 1:
            feature_list = [*ppldata['labels']['baseFeatures'], 'pi_a2_C']
        elif obslevel == 3:
            feature_list = ppldata['labels']['combinedFeatures']

        feature_list = [feature.replace('pi_a2_C', 'pi_a2') for feature in feature_list]

        iterables = [['feature'] * len(feature_list) + ['prob'], feature_list + ['prob']]
        md_columns = pd.MultiIndex.from_arrays(iterables)
        tempdfdict = dict()
        for decision in ppldata['labels' + suffix]['decisions']:
            tempdfdict[decision] = [None] * len(pots)
        for ipot, pot in enumerate(pots):
            with open(modelDataPath / 'level{3}_{0}USD_{1}{2}_observedAgents.json'.format(int(pot) if int(pot) == pot else '{}'.format(pot).replace('.', 'c'), wpplparam['m{}'.format(obslevel)]['method'], wpplparam['m{}'.format(obslevel)]['samples'], obslevel), mode='rt') as data_file:
                jsonin = json.load(data_file)
            for decision in ppldata['labels' + suffix]['decisions']:
                tempdfdict[decision][ipot] = importPPLdataWithLinkerFn(jsonin[decision], feature_list, _agentRepackage)
                np.testing.assert_array_equal(tempdfdict[decision][ipot].columns.to_numpy(), md_columns.get_level_values(1).to_numpy())
                tempdfdict[decision][ipot].columns = md_columns
        for decision in ppldata['labels' + suffix]['decisions']:
            ppldata['level{}'.format(obslevel) + suffix][decision] = pd.concat(tempdfdict[decision], axis=0, keys=pots, names=['pots', None])
        ppldata['level{}'.format(obslevel) + suffix]['labels'] = ppldata['labels'][labellabel]
        ###
        ppldata['level{}'.format(obslevel) + suffix]['nobs'] = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['decisions'])), wpplparam['m{}'.format(obslevel)]['samples']), index=pots, columns=ppldata['labels']['decisions'], dtype=int)
        ppldata['level{}'.format(obslevel) + suffix]['nobs'].index.set_names(['pots'], inplace=True)

    # ### level 4 inverse appraisal features
    ppldata['level4IAF' + suffix] = dict()
    featureClasses = ['compositeWeights', 'emotionIntensities']
    featureLabels = [ppldata['labels' + suffix]['combinedFeatures'], ppldata['labels']['outcomeFeatures']]
    tempdfdict = dict()
    for a1 in wpplparam['a1']:
        for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:
            tempdfdict[outcome] = [None] * len(pots)

    nobsdf = pd.DataFrame(data=np.zeros((len(pots), len(ppldata['labels']['outcomes']))), index=pots, columns=ppldata['labels']['outcomes'], dtype=int)
    nobsdf.index.set_names(['pots'], inplace=True)
    for ipot, pot in enumerate(pots):
        data_file_path_ = modelDataPath / 'level4IAF_{0}USD_{1}{2}_modelObservation_outcomeFeatures.json'.format(int(pot) if int(pot) == pot else '{}'.format(pot).replace('.', 'c'), wpplparam['m4iaf']['method'], wpplparam['m4iaf']['samples'])
        with open(data_file_path_, mode='rt') as data_file:
            jsonin = json.load(data_file)

        for a1 in wpplparam['a1']:
            for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:
                dataout, overflows_ = importPPLdataDict(jsonin[outcome])
                if overflows_:
                    print(f"\n\n---Warning: {len(overflows_)} numerical overflows in {data_file_path_}\n\n")
                dataout.columns = makeLabelHierarchy([featureClasses + ['prob'], featureLabels + [['prob']]])
                tempdfdict[outcome][ipot] = dataout
                nobsdf.loc[pot, outcome] = wpplparam['m4iaf']['samples']

    for a1 in wpplparam['a1']:
        for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:
            ppldata['level4IAF' + suffix][outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
    ppldata['labels' + suffix]['level4IAF'] = list(zip(featureClasses, featureLabels))
    ###
    ppldata['level4IAF' + suffix]['nobs'] = nobsdf

    if verbose:
        for a1 in wpplparam['a1']:
            for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:
                df = ppldata['level4IAF' + suffix][outcome]['emotionIntensities'].copy()
                print(f'---Outcome:: {outcome}')
                print('X_in range: ({:0.4f}, {:0.4f})'.format(np.min(df.values), np.max(df.values)))
                print('X_train range: ({:0.4f}, {:0.4f})'.format(np.min(ppldata['level4IAF' + suffix][outcome]['emotionIntensities'].values), np.max(ppldata['level4IAF' + suffix][outcome]['emotionIntensities'].values)))
                print(f'X_in sd max: {np.max(np.std(df.values, axis=0)):0.4}')
                print(f"X_train sd max: {np.max(np.std(ppldata['level4IAF'+suffix][outcome]['emotionIntensities'].values, axis=0)):0.4}")

    ppldata['wppl_datain'] = json_passed_to_wppl

    return ppldata, wpplparam


def importPPLdata_parallel(game_specific_, cpar):
    from cam_import_empirical_data import importEmpirical_exp10_

    gameobj = game_specific_['gameobj']
    stimid = game_specific_['stimid']
    a1 = game_specific_['a1']

    verbose = False

    stim_a1_ppldata, _ = importPPLmodel_(gameobj.dataout_path, gameobj.wpplparam_common, gameobj.pots, verbose)

    # Import distal prior emotion attributions
    # load empirical exp10
    subject_stats_ = importEmpirical_exp10_(stim_a1_ppldata, cpar, stimulus=stimid, condition=a1, update_ppldata=True, bypass_plotting=True)

    return (stimid, a1, stim_a1_ppldata, subject_stats_)
