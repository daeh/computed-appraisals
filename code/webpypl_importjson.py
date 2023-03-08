# coding: utf-8
# # Webppl play goldenballs


class Game():

    def __init__(self, label, wpplparam, dataOut_path, pots):
        self.wpplparam = wpplparam
        self.pots = pots
        self.label = label
        self.data_path_ = dataOut_path / label

    def play(self, paths, executable_path=None, environment='cluster', removeOldData=True, saveOnExec=True, seed=None):
        import numpy as np
        import pickle
        import json
        from webpypl import saveScript, compileModel, playGame
        import time
        from datetime import timedelta

        # paths[f'dataOut_{self.label}'] = self.data_path_
        if self.data_path_.exists() and removeOldData:
            import shutil
            shutil.rmtree(self.data_path_)
        self.data_path_.mkdir(parents=True, exist_ok=True)
        self.wpplparam['dataOut'] = str(self.data_path_)

        seed_inherited = True
        if seed is None:
            seed = int(str(int(time.time() * 10**6))[-9:])
            seed_inherited = False
        rng = np.random.default_rng(seed)
        seeds_ = rng.integers(low=1, high=np.iinfo(np.int32).max, size=len(self.pots), dtype=int)
        seeds = [int(s_) for s_ in seeds_]
        pot_seed_list = list(zip(self.pots, seeds))
        randseed_info = dict(
            seed=seed,
            seed_inherited=seed_inherited,
            pot_seed_list=pot_seed_list,
        )

        with open(self.data_path_ / f"wpplparam.pkl", 'wb') as f:
            pickle.dump(self.wpplparam, f, pickle.HIGHEST_PROTOCOL)
        with open(self.data_path_ / f"wpplparam.json", 'w') as f:
            json.dump(self.wpplparam, f, indent=2)
        with open(self.data_path_ / f"wpplparam_randseeds.json", 'w') as f:
            json.dump(randseed_info, f, indent=2)

        if saveOnExec:
            saveScript(paths)
        if executable_path is None:
            self.executable, compileout = compileModel(paths)
        else:
            self.executable = executable_path

        t0 = time.perf_counter()
        if environment in ['cluster', 'remotekernel']:
            from joblib import Parallel, delayed, cpu_count
            print('{} CPU'.format(cpu_count()))
            with Parallel(n_jobs=min(len(self.pots), cpu_count())) as pool:
                sysout = pool(delayed(playGame)(model=self.executable, param=self.wpplparam, pot=pot, seed=wpplseed) for pot, wpplseed in pot_seed_list)
                shellout = sysout[0]
        else:
            for i_pot, (pot, wpplseed) in enumerate(pot_seed_list):
                sysout = playGame(model=self.executable, param=self.wpplparam, pot=pot, seed=wpplseed)
                shellout = sysout
                print('{} finished, {} remaining'.format(i_pot + 1, len(self.pots) - (i_pot + 1)))
        elapsed_time = time.perf_counter() - t0
        print('\n\nWebPPL Execution Time:  {} ({:0.2f}s), {:0.4f} per cycle'.format(timedelta(seconds=elapsed_time), elapsed_time, elapsed_time / len(self.pots)))

    # def import_ppldata():

# ## import data


def importPPLmodel_(modelDataPath, wpplparam, pots, verbose, suffix=''):
    import json
    import itertools
    import numpy as np
    import pandas as pd
    from pprint import pprint
    from webpypl import gameData

    from webpypl import importPPLdata, importPPLdataDict, importPPLdataWithLinkerFn, makeLabelHierarchy

    def dict_compare(d1, d2):
        d1_keys = set(d1.keys())
        d2_keys = set(d2.keys())
        intersect_keys = d1_keys.intersection(d2_keys)
        added = d1_keys - d2_keys
        removed = d2_keys - d1_keys
        modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
        same = set(o for o in intersect_keys if d1[o] == d2[o])
        return added, removed, modified, same

    with open(modelDataPath / 'paramin.json', mode='rt') as data_file:
        wpplparam_loaded = json.load(data_file)

    added, removed, modified, same = dict_compare(wpplparam, wpplparam_loaded)
    if 'pot' in added:
        added.remove('pot')

    with open(wpplparam_loaded['path_to_kde'], mode='rt') as data_file:
        json_passed_to_wppl = json.load(data_file)

    import warnings
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
        nobsdf = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['decisions'])), wpplparam['m{}'.format(obslevel)]['samples']), index=pots, columns=ppldata['labels']['decisions'], dtype=np.int64)
        nobsdf.index.set_names(['pots'], inplace=True)
        # ppldata['level{}'.format(obslevel)]['nobs'] = nobsdf

    # ### level 1 & 3

    def agentRepackage(observation):
        import numpy as np
        import itertools

        return np.array(list(itertools.chain.from_iterable([observation['weights'], [observation['estimated_p2']]])), dtype=float)

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
                tempdfdict[decision][ipot] = importPPLdataWithLinkerFn(jsonin[decision], feature_list, agentRepackage)
                np.testing.assert_array_equal(tempdfdict[decision][ipot].columns.to_numpy(), md_columns.get_level_values(1).to_numpy())
                tempdfdict[decision][ipot].columns = md_columns
        for decision in ppldata['labels' + suffix]['decisions']:
            ppldata['level{}'.format(obslevel) + suffix][decision] = pd.concat(tempdfdict[decision], axis=0, keys=pots, names=['pots', None])
        ppldata['level{}'.format(obslevel) + suffix]['labels'] = ppldata['labels'][labellabel]
        ###
        ppldata['level{}'.format(obslevel) + suffix]['nobs'] = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['decisions'])), wpplparam['m{}'.format(obslevel)]['samples']), index=pots, columns=ppldata['labels']['decisions'], dtype=np.int64)
        ppldata['level{}'.format(obslevel) + suffix]['nobs'].index.set_names(['pots'], inplace=True)

    '''
    # ### level 1 & 3
    for pia2_level in [1,3]:
        ppldata['level{}pia2'.format(pia2_level)] = {}
        tempdf = {}
        for decision in ppldata['labels']['decisions']:
            tempdf[decision] = [None]*len(pots)
        for ipot,pot in enumerate(pots):
            with open(modelDataPath/'level{3}_{0}USD_{1}{2}_inferredEstimateOtherPlayer.json'.format(int(pot) if int(pot) == pot else '{}'.format(pot).replace('.','c'), wpplparam['m{}'.format(pia2_level)]['method'], wpplparam['m{}'.format(pia2_level)]['samples'], pia2_level), mode='rt') as data_file: jsonin = json.load(data_file)
            for decision in ppldata['labels']['decisions']:
                tempdf[decision][ipot] = importPPLdataContinuous2(jsonin[decision])
        for decision in ppldata['labels']['decisions']:
            ppldata['level{}pia2'.format(pia2_level)][decision] = pd.concat(tempdf[decision], axis=0, keys=pots, names=['pots',None])
    '''

    # ### level 4 inverse appraisal features
    ppldata['level4IAF' + suffix] = dict()
    featureClasses = ['compositeWeights', 'emotionIntensities']
    featureLabels = [ppldata['labels' + suffix]['combinedFeatures'], ppldata['labels']['outcomeFeatures']]
    tempdfdict = dict()
    for a1 in wpplparam['a1']:
        for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:  # ppldata['labels'+suffix]['outcomes']:
            tempdfdict[outcome] = [None] * len(pots)
    # nobsdf = np.full( (len(pots),len(ppldata['labels']['outcomes'])), wpplparam['m4iaf']['samples'] )
    nobsdf = pd.DataFrame(data=np.zeros((len(pots), len(ppldata['labels']['outcomes']))), index=pots, columns=ppldata['labels']['outcomes'], dtype=np.int64)
    nobsdf.index.set_names(['pots'], inplace=True)
    for ipot, pot in enumerate(pots):
        data_file_path_ = modelDataPath / 'level4IAF_{0}USD_{1}{2}_modelObservation_outcomeFeatures.json'.format(int(pot) if int(pot) == pot else '{}'.format(pot).replace('.', 'c'), wpplparam['m4iaf']['method'], wpplparam['m4iaf']['samples'])
        with open(data_file_path_, mode='rt') as data_file:
            jsonin = json.load(data_file)
        # for outcome in ppldata['labels'+suffix]['outcomes']:
        for a1 in wpplparam['a1']:
            for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:
                dataout, overflows_ = importPPLdataDict(jsonin[outcome])
                if overflows_:
                    print(f"\n\n---Warning: {len(overflows_)} numerical overflows in {data_file_path_}\n\n")
                dataout.columns = makeLabelHierarchy([featureClasses + ['prob'], featureLabels + [['prob']]])
                tempdfdict[outcome][ipot] = dataout
                nobsdf.loc[pot, outcome] = wpplparam['m4iaf']['samples']
    # for outcome in ppldata['labels'+suffix]['outcomes']:
    for a1 in wpplparam['a1']:
        for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:
            ppldata['level4IAF' + suffix][outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
    ppldata['labels' + suffix]['level4IAF'] = list(zip(featureClasses, featureLabels))
    ###
    ppldata['level4IAF' + suffix]['nobs'] = nobsdf

    ##########
    ### Apply prospect scaling to IAF
    ##########

    from webpypl import prospect_transform

    ppldata['prospect_fn'] = prospect_transform(alpha_=wpplparam.get('prospect_alpha', 1.0), beta_=wpplparam.get('prospect_beta', 1.0), lambda_=wpplparam.get('prospect_lambda', 1.0), intercept=0.0, noise_scale=0.0).transform
    ppldata['potspacing'] = prospect_transform(alpha_=wpplparam.get('prospect_alpha', 1.0), beta_=wpplparam.get('prospect_beta', 1.0), lambda_=wpplparam.get('prospect_lambda', 1.0), intercept=0.0, noise_scale=0.0).transform

    # for outcome in ppldata['labels']['outcomes']: ###wip
    for a1 in wpplparam['a1']:
        for outcome in {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[a1]:
            df = ppldata['level4IAF' + suffix][outcome]['emotionIntensities'].copy()
            # print(df.head())
            # ppldata['level4IAF'+suffix][outcome]['emotionIntensities'] = ppldata['prospect_fn'](df)
            # ppldata['level4IAF'+suffix][outcome]['emotionIntensities'] = df
            print(f'---Outcome:: {outcome}')
            print('X_in range: ({:0.4f}, {:0.4f})'.format(np.min(df.values), np.max(df.values)))
            print('X_train range: ({:0.4f}, {:0.4f})'.format(np.min(ppldata['level4IAF' + suffix][outcome]['emotionIntensities'].values), np.max(ppldata['level4IAF' + suffix][outcome]['emotionIntensities'].values)))
            ### DEBUG do max across outcomes
            print(f'X_in sd max: {np.max(np.std(df.values, axis=0)):0.4}')
            print(f"X_train sd max: {np.max(np.std(ppldata['level4IAF'+suffix][outcome]['emotionIntensities'].values, axis=0)):0.4}")

    import warnings  # wip
    warnings.warn('No prospect transformation being applied to the WEBPPL data')
    # for outcome in ppldata['labels']['outcomes']: ###wip
    #     df = ppldata['level4IAF'+suffix][outcome]['emotionIntensities'].copy()
    #     noise = 0#np.random.normal(loc=0.0, scale=3.0, size=df.shape)
    #     # alpha = 0.5
    #     # ppldata['level4IAF'][outcome]['emotionIntensities'] = np.power( np.exp(df), alpha )
    #     ppldata['level4IAF'+suffix][outcome]['emotionIntensities'] = ppldata['prospect_fn']( np.subtract(np.exp(df+noise), 1) )
    #
    # for outcome in ppldata_exp3['labels']['outcomes']: ###wip
    #     df = ppldata_exp3['level4IAF'][outcome]['emotionIntensities'].copy()
    #     noise = 0#np.random.normal(loc=0.0, scale=3.0, size=df.shape)
    #     # alpha = 0.5
    #     # ppldata_exp3['level4IAF'][outcome]['emotionIntensities'] = np.power( np.exp(df), alpha )
    #     ppldata_exp3['level4IAF'][outcome]['emotionIntensities'] = ppldata_exp3['prospect_fn']( np.subtract(np.exp(df+noise), 1) )

    # ### level 4 inverse appraisal features and emotions combined
    # ppldata['level4IAF_andemotions'+suffix] = {}
    # featureClasses = ['compositeWeights', 'emotionIntensities']
    # featureLabels = [ppldata['labels']['combinedFeatures'], list(itertools.chain(ppldata['labels']['outcomeFeatures'],ppldata['labels']['emotions_handcoded']))]

    # featureClasses = ['compositeWeights', 'inverseAppraisalFeatures', 'emotionIntensities'] ### important that order of ('inverseAppraisalFeatures', 'emotionIntensities') matches webppl export since identiy is encoded by position
    # featureLabels = [ppldata['labels'+suffix]['combinedFeatures'], ppldata['labels'+suffix]['outcomeFeatures'], ppldata['labels'+suffix]['emotions_handcoded']] ### same here

    # import warnings
    # warnings.warn('IAF not scaled for level4IAF_andemotions')

    '''
    ############## testing if this code gives different results
    with open('level4_{0}USD_{1}{2}_emotionPosteriors.json'.format(wpplparam['pot'], wpplparam['m4']['method'], wpplparam['m4']['samples']), mode='rt') as data_file:
        jsonin = json.load(data_file)
    ppldata['level4emotion'] = {}
    for outcome in ppldata['labels']['outcomes']:
        dataout = importPPLdataContinuous(jsonin[outcome])
        dataout.columns = ppldata['labels']['emotions_handcoded']+['prob']
        ppldata['level4emotion'][outcome] = dataout

    ppldata['level4emotion'][outcome].head()
    '''

    ppldata['obj'] = {}  # TEMP #wip
    ppldata['obj']['level4IAF' + suffix] = gameData(ppldata['level4IAF' + suffix])

    ppldata['wppl_datain'] = json_passed_to_wppl

    return ppldata, wpplparam


# ## import data
def importEmpirical_1_(ppldata, empDataPath, pots, verbose):
    import json
    import numpy as np
    import pandas as pd

    from webpypl import makeLabelHierarchy

    # ### empirical emotion judgments
    ppldata['labels']['emotions'] = ["Devastation", "Disappointment", "Contempt", "Disgust", "Envy", "Fury", "Annoyance", "Embarrassment", "Regret", "Guilt", "Confusion", "Surprise", "Sympathy", "Amusement", "Relief", "Respect", "Gratitude", "Pride", "Excitement", "Joy"]

    # ### empirical emotion judgments
    with open(empDataPath / 'emotionJudgementsObserved.json', mode='rt') as data_file:
        empiricalResponses = json.load(data_file)
    tempdfdict = {}
    ###
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['outcomes'])), np.nan), index=pots, columns=ppldata['labels']['outcomes'], dtype=np.int64)
    nobsdf.index.set_names(['pots'], inplace=True)
    for outcome in ppldata['labels']['outcomes']:
        tempdfdict[outcome] = [None] * len(pots)

    for item in empiricalResponses:
        outcome = item['outcome']
        pot = item['potUSD']
        temparray = [None] * len(ppldata['labels']['emotions'])
        for i_feature, feature in enumerate(ppldata['labels']['emotions']):
            responses = item[feature]
            temparray[i_feature] = responses
        np.testing.assert_array_equal(len(responses) > 0, True)
        prob = 1 / len(responses)
        temparray.append([prob] * len(responses))
        x = np.array(temparray)
        df = pd.DataFrame(data=x.T, columns=makeLabelHierarchy([['emotionIntensities'] + ['prob'], [ppldata['labels']['emotions']] + [['prob']]]))
        tempdfdict[outcome][pots.index(pot)] = df
        nobsdf.loc[pot, outcome] = len(responses)
    ppldata['empiricalEmotionJudgments'] = {}
    ppldata['empiricalEmotionJudgments']['nobs'] = nobsdf
    for outcome in ppldata['labels']['outcomes']:
        ppldata['empiricalEmotionJudgments'][outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
    ### take care of NaNs in the case that there are unequal numbers of responses
    nans = 0
    for outcome in ppldata['labels']['outcomes']:
        for pot in pots:
            nans += ppldata['empiricalEmotionJudgments'][outcome].loc[pot, 'prob'].isnull().sum().prob
            ppldata['empiricalEmotionJudgments'][outcome].loc[pot, 'prob'].fillna(value=0, axis=0, inplace=True)
    if ppldata['empiricalEmotionJudgments'][outcome].isnull().sum().sum() > 0:
        import warnings
        warnings.warn('Different number of responses; padding with NaNs')
        for outcome in ppldata['labels']['outcomes']:
            ppldata['empiricalEmotionJudgments'][outcome].fillna(value=10e10, inplace=True)
            warnings.warn('Replacing NaNs with 10e10')

    ####
    # par down the empirical emotions to match the model's
    ####
    # ppldata['labels']['emotions_hc_intersect'] = [value for value in ppldata['labels']['emotions_handcoded'] if value in ppldata['labels']['emotions']]

    # intersect_labels = ppldata['labels']['emotions_hc_intersect']+['prob']
    # tempdfdict = {}
    # for outcome in ppldata['labels']['outcomes']:
    #     tempdfdict[outcome] = ppldata['empiricalEmotionJudgments'][outcome].reindex(columns=intersect_labels,level=1)
    # ppldata['empiricalEmotionJudgments_hc_intersect'] = tempdfdict
    # ppldata['empiricalEmotionJudgments_hc_intersect']['nobs'] = ppldata['empiricalEmotionJudgments']['nobs'].copy()
    ########### end temp emotion subset

    # ### empirical emotion judgments
    dataIn = empDataPath / 'emotionJudgementsObserved_bySubject.json'
    with open(dataIn, mode='rt') as data_file:
        empiricalResponsesBySubject = json.load(data_file)
    df_list = [None] * len(empiricalResponsesBySubject)
    for iSubject, subjectData in enumerate(empiricalResponsesBySubject):
        tempdict = {}
        subID = subjectData['subjectId']
        nTrials = len(subjectData['pot'])
        prob = 1 / nTrials

        for iEmo, emoLabel in enumerate(ppldata['labels']['emotions_hc_intersect']):
            tempdict[('emotionIntensities', emoLabel)] = subjectData[emoLabel]
        tempdict[('stimulus', 'outcome')] = subjectData['outcome']
        tempdict[('stimulus', 'pot')] = subjectData['pot']
        tempdict[('subjectId', 'subjectId')] = [subID] * nTrials
        tempdict[('prob', 'prob')] = [prob] * nTrials

        df_list[iSubject] = pd.DataFrame.from_dict(tempdict)

    ppldata['empiricalEmotionJudgmentsBySubject_hc_intersect'] = df_list

    df_list = [None] * len(empiricalResponsesBySubject)
    for iSubject, subjectData in enumerate(empiricalResponsesBySubject):
        tempdict = {}
        subID = subjectData['subjectId']
        nTrials = len(subjectData['pot'])
        prob = 1 / nTrials

        for iEmo, emoLabel in enumerate(ppldata['labels']['emotions']):
            tempdict[('emotionIntensities', emoLabel)] = subjectData[emoLabel]
        tempdict[('stimulus', 'outcome')] = subjectData['outcome']
        tempdict[('stimulus', 'pot')] = subjectData['pot']
        tempdict[('subjectId', 'subjectId')] = [subID] * nTrials
        tempdict[('prob', 'prob')] = [prob] * nTrials

        df_list[iSubject] = pd.DataFrame.from_dict(tempdict)

    ppldata['empiricalEmotionJudgmentsBySubject'] = df_list


# ## import data
def importEmpirical_(ppldata, path_data, path_subjecttracker, verbose, suffix=''):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    from pprint import pprint

    #######
    ### Read in exp 3 data
    #######

    import warnings
    warnings.warn('Currently using mturk worker ID. Replace with hash')

    def unitscale(x): return int(x) / 48

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={"condition": str,
                                        "Version": str,
                                        "stimID": str,
                                        "stimulus": str,
                                        "decisionThis": str,
                                        "decisionOther": str,
                                        "pot": float,
                                        "subjectId": str,
                                        "gender": str,
                                        "Data_Set": float,
                                        "HITID": str},
                                 converters={
                                     "q1responseArray": unitscale,
                                     "q2responseArray": unitscale,
                                     "q3responseArray": unitscale,
                                     "q4responseArray": unitscale,
                                     "q5responseArray": unitscale,
                                     "q6responseArray": unitscale,
                                     "q7responseArray": unitscale,
                                     "q8responseArray": unitscale,
                                     "q9responseArray": unitscale,
                                     "q10responseArray": unitscale,
                                     "q11responseArray": unitscale,
                                     "q12responseArray": unitscale,
                                     "q13responseArray": unitscale,
                                     "q14responseArray": unitscale,
                                     "q15responseArray": unitscale,
                                     "q16responseArray": unitscale,
                                     "q17responseArray": unitscale,
                                     "q18responseArray": unitscale,
                                     "q19responseArray": unitscale,
                                     "q20responseArray": unitscale,
                                 },
                                 )

    jslabels = ["q1responseArray", "q2responseArray", "q3responseArray", "q4responseArray", "q5responseArray", "q6responseArray", "q7responseArray", "q8responseArray", "q9responseArray", "q10responseArray", "q11responseArray", "q12responseArray", "q13responseArray", "q14responseArray", "q15responseArray", "q16responseArray", "q17responseArray", "q18responseArray", "q19responseArray", "q20responseArray", ]
    emoLabels_original_adj = ["Annoyed", "Apprehensive", "Contemptuous", "Content", "Devastated", "Disappointed", "Disgusted", "Embarrassed", "Excited", "Furious", "Grateful", "Guilty", "Hopeful", "Impressed", "Jealous", "Joyful", "Proud", "Relieved", "Surprised", "Terrified", ]
    emoLabels_new_noun = ["Annoyance", "Apprehension", "Contempt", "Contentment", "Devastation", "Disappointment", "Disgust", "Embarrassment", "Excitement", "Fury", "Gratitude", "Guilt", "Hope", "Impressed", "Envy", "Joy", "Pride", "Relief", "Surprise", "Terror", ]

    rename_dict = dict(zip(jslabels, emoLabels_new_noun))

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis", "decisionOther"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    outcomekey = {'Split': {'Split': 'CC', 'Stole': 'CD'}, 'Stole': {'Split': 'DC', 'Stole': 'DD'}}
    outcome = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        outcome[idx] = outcomekey[datasheet_temp['decisionThis'][idx]][datasheet_temp['decisionOther'][idx]]
    datasheet_temp['outcome'] = pd.Series(outcome).astype(CategoricalDtype(ordered=False, categories=ppldata['labels']['outcomes']))

    def randcond(x):
        if x == 'NaN':
            x = np.nan
        return x

    datasheet_participants = pd.read_csv(path_subjecttracker,
                                         header=0, index_col=None,
                                         dtype={
                                             "subjectId": str,
                                             "validationRadio": str,
                                             "subjectValidation1": bool,
                                             "dem_gender": str,
                                             "dem_language": str,
                                             "val_recognized": str,
                                             "val_feedback": str,
                                             "Data_Set": float,
                                             "HITID": str,
                                             "HIT_Annot": str,
                                             "Excluded": bool, },
                                         converters={"randCondNum": randcond, },
                                         )

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    subjects_included = datasheet_participants.loc[((datasheet_participants["Data_Set"] >= 3) & (datasheet_participants["Data_Set"] < 4) & np.logical_not(datasheet_participants['Excluded']) & datasheet_participants['subjectValidation1']), "subjectId"]

    data_included = datasheet_temp.loc[(datasheet_temp['subjectId'].isin(subjects_included) & (datasheet_temp["Data_Set"] >= 3) & (datasheet_temp["Data_Set"] < 4)), :].copy()

    from pandas.api.types import CategoricalDtype
    for catfield in ["stimulus", "pot", "subjectId", "HITID", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    empemodata_bysubject_list = []
    for iSubject, subID in enumerate(subjects_included):
        subjectData = data_included.loc[(data_included["subjectId"] == subID), :]

        tempdict = {}
        nTrials = len(subjectData['pot'])
        prob = 1 / nTrials

        for feature in emoLabels_new_noun:
            tempdict[('emotionIntensities', feature)] = subjectData[feature]

        tempdict[('stimulus', 'outcome')] = subjectData['outcome']
        tempdict[('stimulus', 'pot')] = subjectData['pot']
        tempdict[('subjectId', 'subjectId')] = [subID] * nTrials
        tempdict[('prob', 'prob')] = [prob] * nTrials

        empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

    alldata = pd.concat(empemodata_bysubject_list)

    pots = np.unique(alldata[('stimulus', 'pot')])
    assert np.all(pots == data_included['pot'].cat.categories)

    ###
    tempdfdict = {}
    for outcome in ppldata['labels']['outcomes']:
        tempdfdict[outcome] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['outcomes'])), np.nan, dtype=int), index=pots, columns=ppldata['labels']['outcomes'], dtype=np.int64)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalEmotionJudgments = dict()
    for i_outcome, outcome in enumerate(ppldata['labels']['outcomes']):
        for i_pot, pot in enumerate(pots):
            df = alldata.loc[((alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot)), ['emotionIntensities', 'prob']]
            nobsdf.loc[pot, outcome] = df.shape[0]
            if df.shape[0] > 0:
                df[('prob', 'prob')] = 1 / df.shape[0]
                tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

        empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
    empiricalEmotionJudgments['nobs'] = nobsdf

    # np.sum( empiricalEmotionJudgments['CC']['prob'] )

    # np.sum( ppldata['empiricalEmotionJudgments']['CC']['prob'] )

    # np.unique(data_included['pot'])
    # data_included['pot'].cat.categories
    # data_included['stimulus'].cat.categories
    # data_included.head()

    ppldata['labels']['emotions' + suffix] = emoLabels_new_noun
    ppldata['empiricalEmotionJudgments' + suffix] = empiricalEmotionJudgments
    ppldata['empiricalEmotionJudgmentsBySubject' + suffix] = empemodata_bysubject_list

    # If need exp3 intersect hc
    ppldata['empiricalEmotionJudgments_hc_intersect' + suffix] = 'not included yet'
    ppldata['empiricalEmotionJudgmentsBySubject_hc_intersect' + suffix] = 'not included yet'
    # intersect_labels = ppldata['labels']['emotions_hc_intersect']+['prob']
    # tempdfdict = {}
    # for outcome in ppldata['labels']['outcomes']:
    #     tempdfdict[outcome] = ppldata['empiricalEmotionJudgments'][outcome].reindex(columns=intersect_labels,level=1)
    # ppldata['empiricalEmotionJudgments_hc_intersect'] = tempdfdict

    ppldata['pots' + suffix] = np.unique(alldata[('stimulus', 'pot')])

    ppldata['pots_byoutcome' + suffix] = {'all': ppldata['pots' + suffix]}
    for outcome in ppldata['labels']['outcomes']:
        ppldata['pots_byoutcome' + suffix][outcome] = np.unique(alldata.loc[(alldata[('stimulus', 'outcome')] == outcome), ('stimulus', 'pot')])


def response_filter(datasheet_in, datasheet_participants, emoLabels, bypass=False):
    import numpy as np
    from copy import deepcopy

    datasheet_temp = deepcopy(datasheet_in)

    neg_emos = ['Devastation', 'Disappointment', 'Fury', 'Annoyance']
    pos_emos = ['Relief', 'Excitement', 'Joy']
    joint_emos = [*neg_emos, *pos_emos]
    validation_valencecorr = np.zeros(datasheet_temp.shape[0], dtype=bool)
    validation_valencecorr_value = [''] * datasheet_temp.shape[0]

    if bypass:
        print(f"-- Bypassing Response Filter --")
    else:
        print(f"-- Response Filtering --")
        for iresp in range(datasheet_temp.shape[0]):

            resp = datasheet_temp.iloc[iresp, :]

            if resp['stimulus'] != '244_2':
                if resp.loc['outcome'] in ['CC', 'DC']:
                    if np.mean(resp.loc[neg_emos]) >= np.mean(resp.loc[pos_emos]) and resp.loc['pot'] > 2000:
                        validation_valencecorr[iresp] = True
                        validation_valencecorr_value[iresp] += f"[neg >= pos]"
                if resp.loc['outcome'] in ['CD', 'DD']:
                    if np.mean(resp.loc[pos_emos]) >= np.mean(resp.loc[neg_emos]):
                        validation_valencecorr[iresp] = True
                        validation_valencecorr_value[iresp] += f"[pos >= neg]"

                if np.max(resp.loc[emoLabels]) - np.min(resp.loc[emoLabels]) < 0.2:
                    validation_valencecorr[iresp] = True
                    validation_valencecorr_value[iresp] += f"[range {np.max(resp.loc[emoLabels]) - np.min(resp.loc[emoLabels]):0.2} < 0.2]"

                if resp.loc['outcome'] in ['CC', 'DC']:
                    if np.mean(resp.loc[neg_emos]) > 0.3 and resp.loc['pot'] > 2000:
                        validation_valencecorr[iresp] = True
                        validation_valencecorr_value[iresp] += f"[neg {np.mean(resp.loc[neg_emos]):0.2} > 0.3]"

                if resp.loc['outcome'] in ['CD', 'DD']:
                    if np.mean(resp.loc[pos_emos]) > 0.3:
                        validation_valencecorr[iresp] = True
                        validation_valencecorr_value[iresp] += f"[pos {np.mean(resp.loc[pos_emos]):0.2} > 0.3]"

    response_filter_fail = np.zeros_like(datasheet_participants['Excluded'], dtype=int)
    failing_participants = np.unique(datasheet_temp.loc[validation_valencecorr, 'subjectId'])

    datasheet_temp['validation_valencecorr'] = validation_valencecorr
    datasheet_temp['validation_valencecorr_value'] = validation_valencecorr_value

    for i_subid, subid in enumerate(failing_participants):
        response_filter_fail[datasheet_participants['subjectId'].to_numpy() == subid] = True
    s4 = np.logical_not(response_filter_fail)
    print(f"sums4 exp10 :: {np.sum(s4)}, dropping {np.sum(response_filter_fail)}")

    return s4, failing_participants, datasheet_temp


def print_subject_filter(subject_list, datasheet_temp, emoLabels, data_load_param, dataset='', excluded_elsewhere=None):
    from webpypl_plotfun import printFigList

    plt = data_load_param['plotParam']['plt']

    if excluded_elsewhere is None:
        excluded_elsewhere = list()

    savepath = data_load_param['print_responses_savepath']

    plt.close('all')
    excluded_participants_figs = list()

    neg_emos = ['Devastation', 'Disappointment', 'Fury', 'Annoyance']
    pos_emos = ['Relief', 'Excitement', 'Joy']
    joint_emos = [*neg_emos, *pos_emos]

    for i_subid, subid in enumerate(subject_list):
        tempdf = datasheet_temp.loc[datasheet_temp.loc[:, 'subjectId'] == subid, :]
        selected_emo_idx = list()
        for emol in joint_emos:
            selected_emo_idx.append(emoLabels.index(emol))
        figout, axes = plt.subplots(2, 2, figsize=(8, 4), constrained_layout=True, gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [2, 2]}, sharex=True, sharey=True)
        axs = axes.flatten()
        for i_outcome, outcome in enumerate(['CC', 'CD', 'DC', 'DD']):
            tempdf_outcome = tempdf.loc[tempdf.loc[:, 'outcome'] == outcome, :]
            titletxt = f"{outcome}"
            for iresp in range(tempdf_outcome.shape[0]):
                if tempdf_outcome.iloc[iresp, :]['stimulus'] != '244_2':
                    titletxt += f", {tempdf_outcome.iloc[iresp,:]['pot']}"
            for iresp in range(tempdf_outcome.shape[0]):
                if tempdf_outcome.iloc[iresp, :]['stimulus'] != '244_2':
                    linestyle = '--'
                    if tempdf_outcome.iloc[iresp, :]['validation_valencecorr']:
                        titletxt = '*' + titletxt + '\n' + tempdf_outcome.iloc[iresp, :]['validation_valencecorr_value']
                        # titletxt += f" *corr({tempdf_outcome.iloc[iresp,:]['validation_valencecorr_value'][0]:0.2})"
                        axs[i_outcome].scatter(selected_emo_idx, tempdf_outcome.iloc[iresp, :].loc[joint_emos], color=['green', 'blue', 'red', 'black'][i_outcome], alpha=0.5)
                        # assert tempdf_outcome.iloc[iresp,:]['validation_valencecorr_value'][0] == scipy.stats.pearsonr( tempdf_outcome.iloc[iresp,:]['emotionIntensities'].loc[neg_emos] + minor_var, tempdf_outcome.iloc[iresp,:]['emotionIntensities'].loc[pos_emos] + minor_var )[0]
                        linestyle = '-'
                    # else:
                    #     titletxt += f" corr({tempdf_outcome.iloc[iresp,:]['validation_valencecorr_value'][0]:0.2})"
                    axs[i_outcome].plot(range(len(emoLabels)), tempdf_outcome.iloc[iresp, :].loc[emoLabels], linestyle, color=['green', 'blue', 'red', 'black'][i_outcome], alpha=0.5)

            axs[i_outcome].axvline(x=emoLabels.index('Surprise'), color='k', linewidth=1, alpha=0.2)
            axs[i_outcome].axhline(y=0.3, color='k', linewidth=1, alpha=0.2)
            axs[i_outcome].axhline(y=0.7, color='k', linewidth=1, alpha=0.2)
            axs[i_outcome].set_title(titletxt, fontsize=9)
            axs[i_outcome].set_xticks(range(len(emoLabels)))
            axs[i_outcome].set_xticklabels(emoLabels, rotation=-35, ha='left', rotation_mode='anchor', fontsize=9)
            axs[i_outcome].set_ylim([-0.1, 1.1])

        subj_exluded_elsewhere = ''
        if subid in excluded_elsewhere:
            subj_exluded_elsewhere = 'x'

        plt.suptitle(f"{subj_exluded_elsewhere}  {subid} ({i_subid}/{len(subject_list)})", fontsize=9)
        excluded_participants_figs.append((savepath / f'{dataset}' / f'{subj_exluded_elsewhere}_{subid}.pdf', figout))

        plt.close(figout)

    _ = printFigList(excluded_participants_figs, data_load_param['plotParam'])


def importEmpirical_exp10_(ppldata, path_data, path_subjecttracker, stimulus=None, condition=None, emoLabels=None, data_load_param=False, suffix=''):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    from pprint import pprint

    #######
    ### Read in exp 10 data
    #######

    if stimulus in ['', 'all']:
        print(f"{stimulus} -- {condition}")
        assert condition is None

    import warnings
    warnings.warn('Currently using mturk worker ID. Replace with hash')

    def unitscale(x): return int(x) / 48
    def restore_quotes(x): return str(x).replace('non-profit', 'nonprofit').replace('-', '\"').replace('  ', ', ')

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype={
                                     "stimulus": str,
                                     "pronoun": str,
                                     # "desc": str,
                                     "decisionThis": str,
                                     "decisionOther": str,
                                     "pot": float,
                                     "respTimer": float,
                                     "gender": str,
                                     "subjectId": str,
                                     "Data_Set": float,
                                     "HITID": str},
                                 converters={
                                     "e_amusement": unitscale,
                                     "e_annoyance": unitscale,
                                     "e_confusion": unitscale,
                                     "e_contempt": unitscale,
                                     "e_devastation": unitscale,
                                     "e_disappointment": unitscale,
                                     "e_disgust": unitscale,
                                     "e_embarrassment": unitscale,
                                     "e_envy": unitscale,
                                     "e_excitement": unitscale,
                                     "e_fury": unitscale,
                                     "e_gratitude": unitscale,
                                     "e_guilt": unitscale,
                                     "e_joy": unitscale,
                                     "e_pride": unitscale,
                                     "e_regret": unitscale,
                                     "e_relief": unitscale,
                                     "e_respect": unitscale,
                                     "e_surprise": unitscale,
                                     "e_sympathy": unitscale,
                                     "desc": restore_quotes,
                                 },
                                 )

    # jslabels_alphabetical = ["e_amusement", "e_annoyance", "e_confusion", "e_contempt", "e_devastation", "e_disappointment", "e_disgust", "e_embarrassment", "e_envy", "e_excitement", "e_fury", "e_gratitude", "e_guilt", "e_joy", "e_pride", "e_regret", "e_relief", "e_respect", "e_surprise", "e_sympathy"]
    # emoLabels_alphabetical = ["Amusement", "Annoyance", "Confusion", "Contempt", "Devastation", "Disappointment", "Disgust", "Embarrassment", "Envy", "Excitement", "Fury", "Gratitude", "Guilt", "Joy", "Pride", "Regret", "Relief", "Respect", "Surprise", "Sympathy"]

    # rename_dict = dict(zip(jslabels_alphabetical,emoLabels_alphabetical))
    rename_dict_alphabetical = {
        "e_amusement": "Amusement",
        "e_annoyance": "Annoyance",
        "e_confusion": "Confusion",
        "e_contempt": "Contempt",
        "e_devastation": "Devastation",
        "e_disappointment": "Disappointment",
        "e_disgust": "Disgust",
        "e_embarrassment": "Embarrassment",
        "e_envy": "Envy",
        "e_excitement": "Excitement",
        "e_fury": "Fury",
        "e_gratitude": "Gratitude",
        "e_guilt": "Guilt",
        "e_joy": "Joy",
        "e_pride": "Pride",
        "e_regret": "Regret",
        "e_relief": "Relief",
        "e_respect": "Respect",
        "e_surprise": "Surprise",
        "e_sympathy": "Sympathy",
    }

    if emoLabels is None:
        emoLabels = [rename_dict_alphabetical[key] for key in rename_dict_alphabetical]

    ### reorder to match ppldata['lables']['emotions']
    rename_dict = dict()
    for emotion in emoLabels:
        for jsl, pyl in rename_dict_alphabetical.items():
            if pyl == emotion:
                rename_dict[jsl] = pyl
    assert len(rename_dict) == len(rename_dict_alphabetical)

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis", "decisionOther"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    outcomekey = {'Split': {'Split': 'CC', 'Stole': 'CD'}, 'Stole': {'Split': 'DC', 'Stole': 'DD'}}
    outcome = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        outcome[idx] = outcomekey[datasheet_temp['decisionThis'][idx]][datasheet_temp['decisionOther'][idx]]
    datasheet_temp['outcome'] = pd.Series(outcome).astype(CategoricalDtype(ordered=False, categories=ppldata['labels']['outcomes']))

    datasheet_participants = pd.read_csv(path_subjecttracker,
                                         header=0, index_col=None,
                                         dtype={
                                             "subjectId": str,
                                             "randCondNum": int,
                                             "validationRadio": str,
                                             "subjectValidation1": bool,
                                             "expTime_min": float,
                                             "minRespTime_sec": float,
                                             "iwould_large": int,
                                             "iwould_small": int,
                                             "iexpectOther_large": int,
                                             "iexpectOther_small": int,
                                             "dem_gender": str,
                                             "dem_language": str,
                                             "browser_version": str,
                                             "browser": str,
                                             "visible_area": str,
                                             "val_recognized": str,
                                             "val_familiar": str,
                                             "val_feedback": str,
                                             "Data_Set": float,
                                             "HITID": str,
                                             "HIT_Annot": str,
                                             "Excluded": bool,
                                             "val0(7510)": str,
                                             "val1(disdain)": str,
                                             "val2(jealousy)": str,
                                             "val3(AF25HAS)": str,
                                             "val4(steal)": str,
                                             "val5(pia2_D_a2_C)": str,
                                             "val6(pia2_D_a2_C)": str,
                                         }
                                         )

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    ########
    # Test that all subjects have same number of responses
    ########
    unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
    unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

    nresponses = list()
    for subject in unique_sub_batch2:
        nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
    assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"

    ########
    # Test that all responses are associated with a batch_2_ subject
    ########
    np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

    #########

    ##########
    ### Subject Filter
    ##########
    s1 = datasheet_participants["Data_Set"] >= 10
    s2 = np.logical_not(datasheet_participants['Excluded'])

    #### DEBUG
    import warnings

    for val_id in ["val0(7510)", "val1(disdain)", "val2(jealousy)", "val3(AF25HAS)", "val4(steal)", "val5(pia2_D_a2_C)", "val6(pia2_D_a2_C)"]:
        datasheet_participants[val_id].fillna('correct_response', inplace=True)

    validation_array = np.array([
        datasheet_participants["val0(7510)"] != 'correct_response',
        datasheet_participants["val1(disdain)"] != 'correct_response',
        datasheet_participants["val2(jealousy)"] != 'correct_response',
        datasheet_participants["val3(AF25HAS)"] != 'correct_response',
        datasheet_participants["val4(steal)"] != 'correct_response',
        datasheet_participants["val5(pia2_D_a2_C)"] != 'correct_response',
        datasheet_participants["val6(pia2_D_a2_C)"] != 'correct_response',
    ]).T

    ### WIP ###DEBUG
    exclusion = 'allquestions'
    update_ppldata = True
    if isinstance(data_load_param, dict):
        exclusion = data_load_param.get('exp10load', 'allquestions')
        update_ppldata = data_load_param.get('update_ppldata', True)
    if exclusion == 'allquestions':
        ### validate by all questions
        ### Filter by all validation question
        warnings.warn('VALIDATING ON ALL VALIDATION QUESTIONS -- STRINGENT')
        s3 = datasheet_participants['subjectValidation1']

    elif exclusion == 'original':
        #### validate by some questions
        ### Filter by original validation questions
        warnings.warn('VALIDATING ON A SUBSET OF VALIDATION QUESTIONS')
        filter_by_validation_questions = validation_array[:, 0:5].sum(axis=1) == 0
        s3 = filter_by_validation_questions

    elif exclusion == 'none':
        ### Bypass Filtering
        warnings.warn('BYPASSING VALIDATION FOR EXP10, ACCEPTING ALL SUBJECTS')
        ### bypass validation
        s3 = np.full_like(s1, True, dtype=np.bool)

    print(f"Exp10 Subjects Clearing s3 ({exclusion}): {s3.sum()}")  # TEMP
    ### DEBUG ^^^

    bypass_filter = False
    if isinstance(data_load_param, dict):
        if data_load_param.get('subjRespFilter', 'filtered') == 'unfiltered':
            bypass_filter = True
    print(f'********* 10 subjRespFilter {bypass_filter}')  # TEMP
    s4, failing_participants, datasheet_temp_temp = response_filter(datasheet_temp, datasheet_participants, emoLabels, bypass=bypass_filter)

    subjects_included = datasheet_participants.loc[(s1 & s2 & s3 & s4), "subjectId"]
    datasheet_participants['subjectValidation_included'] = (s1 & s2 & s3 & s4)

    if stimulus in ['all', '']:
        ### Drop practice trial
        a1 = datasheet_temp['stimulus'] != '244_2'
        a2 = np.full_like(a1, True, dtype=bool)
    else:
        assert stimulus != '244_2', "This is the practice trial"
        a1 = datasheet_temp['stimulus'] == stimulus
        a2 = datasheet_temp['decisionThis'] == {'C': 'Split', 'D': 'Stole'}[condition]
    a3 = datasheet_temp['subjectId'].isin(subjects_included)
    a4 = datasheet_temp["Data_Set"] >= 10

    grand_selector = (a1 & a2 & a3 & a4)

    # print(f'total number of responses: {datasheet_temp.shape[0]}')
    # print(f'a1:: {np.sum(a1)}')
    # print(f'a2:: {np.sum(a2)}')
    # print(f'a3:: {np.sum(a3)}')
    # print(f'a4:: {np.sum(a4)}')
    # print(f'grand_selector :: {np.sum(grand_selector)}')

    if grand_selector.sum() == 0:
        import warnings
        warnings.warn(f'No responses found for {stimulus}, {condition}')

    data_included = datasheet_temp.loc[grand_selector, :].copy()

    if 'print_responses_savepath' in data_load_param and stimulus in ['all', '']:
        print_responses = True
    else:
        print_responses = False
    if print_responses:
        additional_excluded_subjs = [subid for subid in failing_participants if subid in datasheet_participants.loc[s3, "subjectId"].to_numpy()]

        print_subject_filter(additional_excluded_subjs, datasheet_temp_temp, emoLabels, data_load_param, dataset=f'excluded_byresp/exp10_{len(additional_excluded_subjs)}', excluded_elsewhere=datasheet_participants.loc[np.logical_not(s3), "subjectId"].to_numpy())
        print_subject_filter(subjects_included, datasheet_temp_temp, emoLabels, data_load_param, dataset=f'included_final/exp10_{len(subjects_included)}')

    ### make categorical after selecting which data to include
    for catfield in ["stimulus", "pot", "subjectId", "HITID", "gender"]:
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    ### if not selecting my stimuli, make by-subject dicts
    if stimulus in ['all', '']:
        empemodata_bysubject_list = []
        for iSubject, subID in enumerate(subjects_included):
            subjectData = data_included.loc[(data_included['subjectId'] == subID), :]

            tempdict = {}
            nTrials = len(subjectData['pot'])
            prob = 1 / nTrials

            for feature in emoLabels:
                tempdict[('emotionIntensities', feature)] = subjectData[feature]

            tempdict[('stimulus', 'outcome')] = subjectData['outcome']
            tempdict[('stimulus', 'pot')] = subjectData['pot']
            tempdict[('subjectId', 'subjectId')] = subjectData['subjectId']
            tempdict[('prob', 'prob')] = [prob] * nTrials

            empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

        alldata = pd.concat(empemodata_bysubject_list)

        pots = np.unique(alldata[('stimulus', 'pot')])
        assert np.all(pots == data_included['pot'].cat.categories)

    else:
        emp_emo_data_bypot_list = list()
        for i_pot, pot in enumerate(data_included['pot'].cat.categories):
            potData = data_included.loc[(data_included['pot'] == pot), :]

            tempdict = {}
            nTrials = potData.shape[0]
            if nTrials > 0:
                prob = 1 / nTrials

                for feature in emoLabels:
                    tempdict[('emotionIntensities', feature)] = potData[feature]

                tempdict[('stimulus', 'outcome')] = potData['outcome']
                tempdict[('stimulus', 'pot')] = potData['pot']
                tempdict[('subjectId', 'subjectId')] = potData['subjectId']
                tempdict[('prob', 'prob')] = [prob] * nTrials

                emp_emo_data_bypot_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))
            else:
                columns_temp = [('emotionIntensities', feature) for feature in emoLabels] + [('stimulus', 'outcome'), ('stimulus', 'pot'), ('subjectId', 'subjectId'), ('prob', 'prob')]
                emp_emo_data_bypot_list.append(pd.DataFrame(columns=pd.MultiIndex.from_tuples(columns_temp)))

        alldata = pd.concat(emp_emo_data_bypot_list)  # WIP
        pots = np.unique(alldata[('stimulus', 'pot')])
        assert np.all(pots == data_included['pot'].cat.categories)

    ###
    tempdfdict = {}
    for outcome in ppldata['labels']['outcomes']:
        tempdfdict[outcome] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['outcomes'])), 0, dtype=int), index=pots, columns=ppldata['labels']['outcomes'], dtype=np.int64)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalEmotionJudgments = dict()
    if stimulus in ['all', '']:
        outcome_loop = ppldata['labels']['outcomes']
    else:
        outcome_loop = {'C': ['CC', 'CD'], 'D': ['DC', 'DD']}[condition]
    for outcome in outcome_loop:
        for i_pot, pot in enumerate(pots):
            df = alldata.loc[((alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot)), ['emotionIntensities', 'prob']]
            nobsdf.loc[pot, outcome] = df.shape[0]
            if df.shape[0] > 0:
                df[('prob', 'prob')] = 1 / df.shape[0]
                tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

        ### TEMP ###DEBUG skipping b/c there's not enough data
        if len(tempdfdict[outcome]) == 0:
            import warnings
            warnings.warn(f'Stim :: {stimulus} - {outcome} !!!! no data yet')
            empiricalEmotionJudgments[outcome] = 'no data yet'
        all_none = True
        for item in tempdfdict[outcome]:
            if item is not None:
                all_none = False
            break
        if not all_none:
            empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
        else:
            empiricalEmotionJudgments[outcome] = 'no data yet'
    empiricalEmotionJudgments['nobs'] = nobsdf

    if update_ppldata:
        ppldata['labels']['emotions' + suffix] = emoLabels
        ppldata['empiricalEmotionJudgments' + suffix] = empiricalEmotionJudgments
        if stimulus in ['all', '']:
            ppldata['empiricalEmotionJudgmentsBySubject' + suffix] = empemodata_bysubject_list

        ppldata['pots' + suffix] = np.unique(alldata[('stimulus', 'pot')])

        ppldata['pots_byoutcome' + suffix] = {'all': ppldata['pots' + suffix]}
        for outcome in ppldata['labels']['outcomes']:
            ppldata['pots_byoutcome' + suffix][outcome] = np.unique(alldata.loc[(alldata[('stimulus', 'outcome')] == outcome), ('stimulus', 'pot')])

    return data_included, datasheet_participants, alldata


def importEmpirical_exp7_11_old_(ppldata, path_data7, path_subjecttracker7, path_data11, path_subjecttracker11, data_load_param=False):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    from pprint import pprint

    """
    path_data7 = paths['exp7xlsx']
    path_subjecttracker7 = paths['subjectrackerexp7']
    path_data11 = paths['exp11xlsx']
    path_subjecttracker11 = paths['subjectrackerexp11']
    """

    def import_empirical_11(path_data, path_subjecttracker, emoLabels=None, data_load_param=False):
        from pandas.api.types import CategoricalDtype

        #######
        ### Read in exp 10 data
        #######

        debug = False

        import warnings
        warnings.warn('Currently using mturk worker ID. Replace with hash')

        def unitscale(x): return int(x) / 48
        def restore_quotes(x): return str(x).replace('non-profit', 'nonprofit').replace('-', '\"').replace('  ', ', ')

        datasheet_temp = pd.read_csv(path_data,
                                     header=0, index_col=None,
                                     dtype={
                                         "stimulus": str,
                                         "pronoun": str,
                                         # "desc": str,
                                         "decisionThis": str,
                                         "decisionOther": str,
                                         "pot": float,
                                         "respTimer": float,
                                         "gender": str,
                                         "subjectId": str,
                                         "Data_Set": float,
                                         "HITID": str},
                                     converters={
                                         "e_amusement": unitscale,
                                         "e_annoyance": unitscale,
                                         "e_confusion": unitscale,
                                         "e_contempt": unitscale,
                                         "e_devastation": unitscale,
                                         "e_disappointment": unitscale,
                                         "e_disgust": unitscale,
                                         "e_embarrassment": unitscale,
                                         "e_envy": unitscale,
                                         "e_excitement": unitscale,
                                         "e_fury": unitscale,
                                         "e_gratitude": unitscale,
                                         "e_guilt": unitscale,
                                         "e_joy": unitscale,
                                         "e_pride": unitscale,
                                         "e_regret": unitscale,
                                         "e_relief": unitscale,
                                         "e_respect": unitscale,
                                         "e_surprise": unitscale,
                                         "e_sympathy": unitscale,
                                         "desc": restore_quotes,
                                     },
                                     )

        # jslabels_alphabetical = ["e_amusement", "e_annoyance", "e_confusion", "e_contempt", "e_devastation", "e_disappointment", "e_disgust", "e_embarrassment", "e_envy", "e_excitement", "e_fury", "e_gratitude", "e_guilt", "e_joy", "e_pride", "e_regret", "e_relief", "e_respect", "e_surprise", "e_sympathy"]
        # emoLabels_alphabetical = ["Amusement", "Annoyance", "Confusion", "Contempt", "Devastation", "Disappointment", "Disgust", "Embarrassment", "Envy", "Excitement", "Fury", "Gratitude", "Guilt", "Joy", "Pride", "Regret", "Relief", "Respect", "Surprise", "Sympathy"]

        # rename_dict = dict(zip(jslabels_alphabetical,emoLabels_alphabetical))
        rename_dict_alphabetical = {
            "e_amusement": "Amusement",
            "e_annoyance": "Annoyance",
            "e_confusion": "Confusion",
            "e_contempt": "Contempt",
            "e_devastation": "Devastation",
            "e_disappointment": "Disappointment",
            "e_disgust": "Disgust",
            "e_embarrassment": "Embarrassment",
            "e_envy": "Envy",
            "e_excitement": "Excitement",
            "e_fury": "Fury",
            "e_gratitude": "Gratitude",
            "e_guilt": "Guilt",
            "e_joy": "Joy",
            "e_pride": "Pride",
            "e_regret": "Regret",
            "e_relief": "Relief",
            "e_respect": "Respect",
            "e_surprise": "Surprise",
            "e_sympathy": "Sympathy",
        }

        if emoLabels is None:
            emoLabels = [rename_dict_alphabetical[key] for key in rename_dict_alphabetical]

        ### reorder to match ppldata['lables']['emotions']
        rename_dict = dict()
        for emotion in emoLabels:
            for jsl, pyl in rename_dict_alphabetical.items():
                if pyl == emotion:
                    rename_dict[jsl] = pyl
        assert len(rename_dict) == len(rename_dict_alphabetical)

        datasheet_temp.rename(columns=rename_dict, inplace=True)

        for catfield in ["decisionThis", "decisionOther"]:
            datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

        outcomekey = {'Split': {'Split': 'CC', 'Stole': 'CD'}, 'Stole': {'Split': 'DC', 'Stole': 'DD'}}
        outcome = np.full_like(datasheet_temp['decisionThis'].values, 'none')
        for idx in range(datasheet_temp.shape[0]):
            outcome[idx] = outcomekey[datasheet_temp['decisionThis'][idx]][datasheet_temp['decisionOther'][idx]]
        datasheet_temp['outcome'] = pd.Series(outcome).astype(CategoricalDtype(ordered=False, categories=ppldata['labels']['outcomes']))

        datasheet_participants = pd.read_csv(path_subjecttracker,
                                             header=0, index_col=None,
                                             dtype={
                                                 "subjectId": str,
                                                 "randCondNum": int,
                                                 "validationRadio": str,
                                                 "subjectValidation1": bool,
                                                 "expTime_min": float,
                                                 "minRespTime_sec": float,
                                                 "iwould_large": int,
                                                 "iwould_small": int,
                                                 "iexpectOther_large": int,
                                                 "iexpectOther_small": int,
                                                 "dem_gender": str,
                                                 "dem_language": str,
                                                 "browser_version": str,
                                                 "browser": str,
                                                 "visible_area": str,
                                                 "val_recognized": str,
                                                 "val_familiar": str,
                                                 "val_feedback": str,
                                                 "Data_Set": float,
                                                 "HITID": str,
                                                 "HIT_Annot": str,
                                                 "Excluded": bool,
                                                 "val0(7510)": str,
                                                 "val1(disdain)": str,
                                                 "val2(jealousy)": str,
                                                 "val3(AF25HAS)": str,
                                                 "val4(steal)": str,
                                                 "val5(pia2_D_a2_C)": str,
                                                 "val6(pia2_D_a2_C)": str,
                                             }
                                             )

        for catfield in ["subjectId", "HITID", "dem_gender"]:
            datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

        ########
        # Test that all subjects have same number of responses
        ########
        unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
        unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

        nresponses = list()
        for subject in unique_sub_batch2:
            nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
        assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"

        data_stats = {
            'nsub_loaded': datasheet_participants.shape[0],
            'nresp_loaded': datasheet_temp.shape[0],
            'nresp_per_sub_retained': np.unique(nresponses)[0] - 1
        }

        ########
        # Test that all responses are associated with a batch_2_ subject
        ########
        np.testing.assert_array_equal(unique_sub_batch1, unique_sub_batch2, err_msg=f"Subjects don't match \nbatch_1:\n{datasheet_temp['subjectId']}\nbatch_2:\n{datasheet_participants['subjectId']}")

        #########

        ##########
        ### Subject Filter
        ##########
        s1 = datasheet_participants["Data_Set"] >= 11
        s2 = np.logical_not(datasheet_participants['Excluded'])

        #### DEBUG
        import warnings

        for val_id in ["val0(7510)", "val1(disdain)", "val2(jealousy)", "val3(AF25HAS)", "val4(steal)", "val5(pia2_D_a2_C)", "val6(pia2_D_a2_C)"]:
            datasheet_participants[val_id].fillna('correct_response', inplace=True)

        validation_array = np.array([
            datasheet_participants["val0(7510)"] != 'correct_response',
            datasheet_participants["val1(disdain)"] != 'correct_response',
            datasheet_participants["val2(jealousy)"] != 'correct_response',
            datasheet_participants["val3(AF25HAS)"] != 'correct_response',
            datasheet_participants["val4(steal)"] != 'correct_response',
            datasheet_participants["val5(pia2_D_a2_C)"] != 'correct_response',
            datasheet_participants["val6(pia2_D_a2_C)"] != 'correct_response',
        ]).T

        ### WIP ###DEBUG
        exclusion = 'original'
        if isinstance(data_load_param, dict):
            exclusion = data_load_param.get('exp11load', 'original')
        if exclusion == 'allquestions':
            ### validate by all questions
            ### Filter by all validation question
            warnings.warn('VALIDATING ON ALL VALIDATION QUESTIONS -- STRINGENT')
            s3 = datasheet_participants['subjectValidation1']

        elif exclusion == 'original':
            #### validate by some questions
            ### Filter by original validation questions
            warnings.warn('VALIDATING ON ORIGINAL SUBSET OF VALIDATION QUESTIONS')
            filter_by_validation_questions = validation_array[:, 0:5].sum(axis=1) == 0
            s3 = filter_by_validation_questions

        elif exclusion == 'none':
            ### Bypass Filtering
            warnings.warn('BYPASSING VALIDATION FOR EXP10, ACCEPTING ALL SUBJECTS')
            ### bypass validation
            s3 = np.full_like(s1, True, dtype=np.bool)

        print(f"Exp11 Subjects Clearing s3: {s3.sum()}")  # TEMP
        ### DEBUG ^^^

        bypass_filter = False
        if isinstance(data_load_param, dict):
            subjRespFilter = data_load_param.get('subjRespFilter', 'filtered')
            if subjRespFilter == 'unfiltered':
                bypass_filter = True
        print(f'********* 11 subjRespFilterBypass {bypass_filter}')  # TEMP
        s4, failing_participants, datasheet_temp_temp = response_filter(datasheet_temp, datasheet_participants, emoLabels, bypass=bypass_filter)

        subjects_included = datasheet_participants.loc[(s1 & s2 & s3 & s4), "subjectId"]
        datasheet_participants['subjectValidation_included'] = (s1 & s2 & s3 & s4)

        print(f"\nDATASET 11 RANDCONDS::\n")
        for i_cond in range(32):
            total_count = np.sum(np.abs(datasheet_participants['randCondNum']) == i_cond)
            valid_count = np.sum(np.abs(datasheet_participants.loc[datasheet_participants['subjectValidation_included'], 'randCondNum']) == i_cond)
            print(f'{i_cond}\t0\t{total_count}\t{valid_count}')
        print(f"\n")
        # print(f"Total passing validation exclusion:{exclusion} filter:{subjRespFilter} (exp 11): {np.sum(datasheet_participants['subjectValidation_included'])}, min cond val: {mmin}\n\n")
        mmin = np.unique(np.abs(datasheet_participants.loc[datasheet_participants['subjectValidation_included'], 'randCondNum']), return_counts=True)[1].min()
        mmax = np.unique(np.abs(datasheet_participants.loc[datasheet_participants['subjectValidation_included'], 'randCondNum']), return_counts=True)[1].max()
        print(f">> Total passing validation exclusion:{exclusion} filter:{subjRespFilter} (exp 11): {np.sum(datasheet_participants['subjectValidation_included'])}, min,max cond val: {mmin} / {mmax}\n")

        ### Drop practice trial
        a1 = datasheet_temp['stimulus'] != '244_2'
        a2 = np.full_like(a1, True, dtype=bool)
        a3 = datasheet_temp['subjectId'].isin(subjects_included)
        a4 = datasheet_temp["Data_Set"] >= 11

        grand_selector = (a1 & a2 & a3 & a4)

        # print(f'total number of responses: {datasheet_temp.shape[0]}')
        # print(f'a1:: {np.sum(a1)}')
        # print(f'a2:: {np.sum(a2)}')
        # print(f'a3:: {np.sum(a3)}')
        # print(f'a4:: {np.sum(a4)}')
        # print(f'grand_selector :: {np.sum(grand_selector)}')

        if grand_selector.sum() == 0:
            import warnings
            warnings.warn(f'No responses found')

        data_included = datasheet_temp.loc[grand_selector, :].copy()

        if 'print_responses_savepath' in data_load_param:
            print_responses = True
        else:
            print_responses = False
        if print_responses:
            additional_excluded_subjs = [subid for subid in failing_participants if subid in datasheet_participants.loc[s3, "subjectId"].to_numpy()]

            print_subject_filter(additional_excluded_subjs, datasheet_temp_temp, emoLabels, data_load_param, dataset=f'excluded_byresp/exp11_{len(additional_excluded_subjs)}', excluded_elsewhere=datasheet_participants.loc[np.logical_not(s3), "subjectId"].to_numpy())
            print_subject_filter(subjects_included, datasheet_temp_temp, emoLabels, data_load_param, dataset=f'included_final/exp11_{len(subjects_included)}')

        ### make categorical after selecting which data to include
        for catfield in ["stimulus", "pot", "subjectId", "HITID", "gender"]:
            data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

        empemodata_bysubject_list = []
        for iSubject, subID in enumerate(subjects_included):
            subjectData = data_included.loc[(data_included['subjectId'] == subID), :]

            tempdict = {}
            nTrials = len(subjectData['pot'])
            prob = 1 / nTrials

            for feature in emoLabels:
                tempdict[('emotionIntensities', feature)] = subjectData[feature]

            tempdict[('stimulus', 'outcome')] = subjectData['outcome']
            tempdict[('stimulus', 'pot')] = subjectData['pot']
            tempdict[('subjectId', 'subjectId')] = subjectData['subjectId']
            tempdict[('prob', 'prob')] = [prob] * nTrials

            empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

        alldata = pd.concat(empemodata_bysubject_list)

        pots = np.unique(alldata[('stimulus', 'pot')])
        assert np.all(pots == data_included['pot'].cat.categories)

        ###
        tempdfdict = {}
        for outcome in ppldata['labels']['outcomes']:
            tempdfdict[outcome] = [None] * len(pots)
        nobsdf = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['outcomes'])), 0, dtype=int), index=pots, columns=ppldata['labels']['outcomes'], dtype=np.int64)
        nobsdf.index.set_names(['pots'], inplace=True)

        empiricalEmotionJudgments = dict()
        for outcome in ['CC', 'CD', 'DC', 'DD']:
            for i_pot, pot in enumerate(pots):
                df = alldata.loc[((alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot)), ['emotionIntensities', 'prob']]
                nobsdf.loc[pot, outcome] = df.shape[0]
                if df.shape[0] > 0:
                    df[('prob', 'prob')] = 1 / df.shape[0]
                    tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

            ### TEMP ###DEBUG skipping b/c there's not enough data
            if len(tempdfdict[outcome]) == 0:
                import warnings
                warnings.warn(f'!!!! no data yet')
                empiricalEmotionJudgments[outcome] = 'no data yet'
            all_none = True
            for item in tempdfdict[outcome]:
                if item is not None:
                    all_none = False
                break
            if not all_none:
                empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
            else:
                empiricalEmotionJudgments[outcome] = 'no data yet'
        empiricalEmotionJudgments['nobs'] = nobsdf

        data_stats['nsub_retained'] = len(subjects_included)
        data_stats['nresp_retained'] = grand_selector.sum()
        data_stats['final_nobs'] = nobsdf.copy()
        data_stats['potential_problem'] = f" nresp/nsub = {grand_selector.sum()/len(subjects_included)}, expect {data_stats['nresp_per_sub_retained']}, grand_selector vs nobs: {grand_selector.sum()} vs {nobsdf.sum().sum()}"

        return emoLabels, empiricalEmotionJudgments, empemodata_bysubject_list, data_stats

    def import_empirical_7(path_data, path_subjecttracker, emoLabels=None, data_load_param=False):
        from pandas.api.types import CategoricalDtype

        import warnings
        warnings.warn('Currently using mturk worker ID. Replace with hash')

        def unitscale(x): return int(x) / 48

        datasheet_temp = pd.read_csv(path_data,
                                     header=0, index_col=None,
                                     dtype={"condition": str,
                                            "Version": str,
                                            "stimID": str,
                                            "stimulus": str,
                                            "decisionThis": str,
                                            "decisionOther": str,
                                            "pot": float,
                                            "randStimulusFace": str,
                                            "gender": str,
                                            "subjectId": str,
                                            "Data_Set": float,
                                            "HITID": str},
                                     converters={
                                         "q1responseArray": unitscale,
                                         "q2responseArray": unitscale,
                                         "q3responseArray": unitscale,
                                         "q4responseArray": unitscale,
                                         "q5responseArray": unitscale,
                                         "q6responseArray": unitscale,
                                         "q7responseArray": unitscale,
                                         "q8responseArray": unitscale,
                                         "q9responseArray": unitscale,
                                         "q10responseArray": unitscale,
                                         "q11responseArray": unitscale,
                                         "q12responseArray": unitscale,
                                         "q13responseArray": unitscale,
                                         "q14responseArray": unitscale,
                                         "q15responseArray": unitscale,
                                         "q16responseArray": unitscale,
                                         "q17responseArray": unitscale,
                                         "q18responseArray": unitscale,
                                         "q19responseArray": unitscale,
                                         "q20responseArray": unitscale,
                                     },
                                     )

        rename_dict_alphabetical = {
            "q1responseArray": "Amusement",
            "q2responseArray": "Annoyance",
            "q3responseArray": "Confusion",
            "q4responseArray": "Contempt",
            "q5responseArray": "Devastation",
            "q6responseArray": "Disappointment",
            "q7responseArray": "Disgust",
            "q8responseArray": "Embarrassment",
            "q9responseArray": "Envy",
            "q10responseArray": "Excitement",
            "q11responseArray": "Fury",
            "q12responseArray": "Gratitude",
            "q13responseArray": "Guilt",
            "q14responseArray": "Joy",
            "q15responseArray": "Pride",
            "q16responseArray": "Regret",
            "q17responseArray": "Relief",
            "q18responseArray": "Respect",
            "q19responseArray": "Surprise",
            "q20responseArray": "Sympathy",
        }

        ######
        if emoLabels is None:
            emoLabels = [rename_dict_alphabetical[key] for key in rename_dict_alphabetical]

        ### reorder to match ppldata['lables']['emotions']
        rename_dict = dict()
        for emotion in emoLabels:
            for jsl, pyl in rename_dict_alphabetical.items():
                if pyl == emotion:
                    rename_dict[jsl] = pyl
        assert len(rename_dict) == len(rename_dict_alphabetical)
        ######

        datasheet_temp.rename(columns=rename_dict, inplace=True)

        for catfield in ["decisionThis", "decisionOther"]:
            datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

        outcomekey = {'Split': {'Split': 'CC', 'Stole': 'CD'}, 'Stole': {'Split': 'DC', 'Stole': 'DD'}}
        outcome = np.full_like(datasheet_temp['decisionThis'].values, 'none')
        for idx in range(datasheet_temp.shape[0]):
            outcome[idx] = outcomekey[datasheet_temp['decisionThis'][idx]][datasheet_temp['decisionOther'][idx]]
        datasheet_temp['outcome'] = pd.Series(outcome).astype(CategoricalDtype(ordered=False, categories=ppldata['labels']['outcomes']))

        #################

        def randcond(x):
            if x == 'NaN':
                x = np.nan
            return x

        datasheet_participants = pd.read_csv(path_subjecttracker,
                                             header=0, index_col=None,
                                             dtype={
                                                 "subjectId": str,
                                                 "validationRadio": str,
                                                 "subjectValidation1": bool,
                                                 "dem_gender": str,
                                                 "dem_language": str,
                                                 "val_recognized": str,
                                                 "val_feedback": str,
                                                 "Data_Set": float,
                                                 "HITID": str,
                                                 "HIT_Annot": str,
                                                 "Excluded": bool, },
                                             converters={"randCondNum": randcond, },
                                             )

        for catfield in ["subjectId", "HITID", "dem_gender"]:
            datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

        ########
        # Test that all subjects have same number of responses
        ########
        unique_sub_batch2 = np.unique(datasheet_participants['subjectId'].values)
        unique_sub_batch1 = np.unique(datasheet_temp['subjectId'].values)

        nresponses = list()
        for subject in unique_sub_batch2:
            nresponses.append(np.sum(datasheet_temp['subjectId'] == subject))
        assert len(np.unique(nresponses)) == 1, "subjects have different numbers of responses"
        data_stats = {
            'nsub_loaded': datasheet_participants.shape[0],
            'nresp_loaded': datasheet_temp.shape[0],
            'nresp_per_sub_retained': np.unique(nresponses)[0]
        }
        #########

        bypass_filter = False
        if isinstance(data_load_param, dict):
            if data_load_param.get('subjRespFilter', 'filtered') == 'unfiltered':
                bypass_filter = True
        print(f'********* 7 subjRespFilter {bypass_filter}')  # TEMP
        s4, failing_participants, datasheet_temp_temp = response_filter(datasheet_temp, datasheet_participants, emoLabels, bypass=bypass_filter)

        s3 = ((datasheet_participants["Data_Set"] >= 7) & (datasheet_participants["Data_Set"] < 8) & np.logical_not(datasheet_participants['Excluded']) & datasheet_participants['subjectValidation1'])
        subject_selector = (s3 & s4)
        subjects_included = datasheet_participants.loc[subject_selector, "subjectId"]
        datasheet_participants['subjectValidation_included'] = subject_selector

        data_included = datasheet_temp.loc[(datasheet_temp['subjectId'].isin(subjects_included) & (datasheet_temp["Data_Set"] >= 7) & (datasheet_temp["Data_Set"] < 8)), :].copy()

        if 'print_responses_savepath' in data_load_param:
            print_responses = True
        else:
            print_responses = False
        if print_responses:
            additional_excluded_subjs = [subid for subid in failing_participants if subid in datasheet_participants.loc[s3, "subjectId"].to_numpy()]

            print_subject_filter(additional_excluded_subjs, datasheet_temp_temp, emoLabels, data_load_param, dataset=f'excluded_byresp/exp7_{len(additional_excluded_subjs)}', excluded_elsewhere=datasheet_participants.loc[np.logical_not(s3), "subjectId"].to_numpy())
            print_subject_filter(subjects_included, datasheet_temp_temp, emoLabels, data_load_param, dataset=f'included_final/exp7_{len(subjects_included)}')

        from pandas.api.types import CategoricalDtype
        for catfield in ["stimulus", "pot", "subjectId", "HITID", "gender"]:
            data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

        # emoLabels = [rename_dict[f] for f in rename_dict]
        empemodata_bysubject_list = []
        for iSubject, subID in enumerate(subjects_included):
            subjectData = data_included.loc[(data_included["subjectId"] == subID), :]

            tempdict = {}
            nTrials = len(subjectData['pot'])
            prob = 1 / nTrials

            for feature in emoLabels:
                tempdict[('emotionIntensities', feature)] = subjectData[feature]

            tempdict[('stimulus', 'outcome')] = subjectData['outcome']
            tempdict[('stimulus', 'pot')] = subjectData['pot']
            tempdict[('subjectId', 'subjectId')] = [subID] * nTrials
            tempdict[('prob', 'prob')] = [prob] * nTrials

            empemodata_bysubject_list.append(pd.DataFrame.from_dict(tempdict).reset_index(drop=True))

        alldata = pd.concat(empemodata_bysubject_list)

        pots = np.unique(alldata[('stimulus', 'pot')])
        assert np.all(pots == data_included['pot'].cat.categories)

        ###
        tempdfdict = {}
        for outcome in ppldata['labels']['outcomes']:
            tempdfdict[outcome] = [None] * len(pots)
        nobsdf = pd.DataFrame(data=np.full((len(pots), len(ppldata['labels']['outcomes'])), np.nan, dtype=int), index=pots, columns=ppldata['labels']['outcomes'], dtype=np.int64)
        nobsdf.index.set_names(['pots'], inplace=True)

        empiricalEmotionJudgments = dict()
        for i_outcome, outcome in enumerate(ppldata['labels']['outcomes']):
            for i_pot, pot in enumerate(pots):
                df = alldata.loc[((alldata[('stimulus', 'outcome')] == outcome) & (alldata[('stimulus', 'pot')] == pot)), ['emotionIntensities', 'prob']]
                nobsdf.loc[pot, outcome] = df.shape[0]
                if df.shape[0] > 0:
                    df[('prob', 'prob')] = 1 / df.shape[0]
                    tempdfdict[outcome][i_pot] = df.reset_index(inplace=False, drop=True)

            empiricalEmotionJudgments[outcome] = pd.concat(tempdfdict[outcome], axis=0, keys=pots, names=['pots', None])
        empiricalEmotionJudgments['nobs'] = nobsdf

        data_stats['nsub_retained'] = len(subjects_included)
        data_stats['nresp_retained'] = alldata.shape[0]
        data_stats['final_nobs'] = nobsdf.copy()
        data_stats['potential_problem'] = f" nresp/nsub = {alldata.shape[0]/len(subjects_included)}, expect {data_stats['nresp_per_sub_retained']}, alldata.shape[0] vs nobs: {alldata.shape[0]} vs {nobsdf.sum().sum()}"
        data_stats['s4'] = s4
        data_stats['s3'] = s3
        data_stats['subjectValidation1'] = datasheet_participants['subjectValidation1']
        data_stats['Excluded'] = datasheet_participants['Excluded']

        return emoLabels, empiricalEmotionJudgments, empemodata_bysubject_list, data_stats

    emoLabelsOrdered = ["Devastation", "Disappointment", "Contempt", "Disgust", "Envy", "Fury", "Annoyance", "Embarrassment", "Regret", "Guilt", "Confusion", "Surprise", "Sympathy", "Amusement", "Relief", "Respect", "Gratitude", "Pride", "Excitement", "Joy"]

    update_ppldata = True
    if isinstance(data_load_param, dict):
        update_ppldata = data_load_param.get('update_ppldata', True)

    emoLabels7, empiricalEmotionJudgments7, empemodata_bysubject_list7, data_stats7 = import_empirical_7(path_data7, path_subjecttracker7, emoLabels=emoLabelsOrdered, data_load_param=data_load_param)

    emoLabels11, empiricalEmotionJudgments11, empemodata_bysubject_list11, data_stats11 = import_empirical_11(path_data11, path_subjecttracker11, emoLabels=emoLabelsOrdered, data_load_param=data_load_param)

    empiricalEmotionJudgments_combined = dict()
    nobsdf = empiricalEmotionJudgments7['nobs'].copy()
    allpots = list()
    for outcome in ppldata['labels']['outcomes']:

        # for pot in empiricalEmotionJudgments11[outcome].index.get_level_values('pots'):
        empiricalEmotionJudgments_combined_lists = list()
        for pot in np.unique(empiricalEmotionJudgments7[outcome].index.get_level_values('pots')):
            potidx7 = empiricalEmotionJudgments7[outcome].index.get_level_values('pots') == pot
            if pot in empiricalEmotionJudgments11[outcome].index.get_level_values('pots'):
                potidx11 = empiricalEmotionJudgments11[outcome].index.get_level_values('pots') == pot

                tempdf7 = empiricalEmotionJudgments7[outcome].loc[potidx7, :]
                tempdf11 = empiricalEmotionJudgments11[outcome].loc[potidx11, :]
                tempdf = pd.concat([tempdf7, tempdf11])
                tempdf.loc[:, ('prob', 'prob')] = tempdf.shape[0]**-1
                nobsdf.loc[pot, outcome] = tempdf.shape[0]
            else:
                tempdf = empiricalEmotionJudgments7[outcome].loc[potidx7, :]

            empiricalEmotionJudgments_combined_lists.append(tempdf)

        empiricalEmotionJudgments_combined[outcome] = pd.concat(empiricalEmotionJudgments_combined_lists)

    empiricalEmotionJudgments_combined['nobs'] = nobsdf

    ####################################################################### exp 7 ^^^^^

    if update_ppldata:
        ppldata['labels']['emotions'] = emoLabelsOrdered
        ppldata['empiricalEmotionJudgments'] = empiricalEmotionJudgments_combined
        ppldata['empiricalEmotionJudgmentsBySubject'] = empemodata_bysubject_list7 + empemodata_bysubject_list11

        ppldata['pots'] = empiricalEmotionJudgments_combined['nobs'].index.to_list()

        ppldata['pots_byoutcome'] = {'all': ppldata['pots']}
        for outcome in ppldata['labels']['outcomes']:
            ppldata['pots_byoutcome'][outcome] = empiricalEmotionJudgments_combined['nobs'].index[empiricalEmotionJudgments_combined['nobs'][outcome] > 0].to_list()

        if not 'subject_stats' in ppldata:
            ppldata['subject_stats'] = dict()
        ppldata['subject_stats']['exp7'] = data_stats7
        ppldata['subject_stats']['exp11'] = data_stats11
        ppldata['subject_stats']['exp711cat'] = {'nobsdf': nobsdf.copy()}

    return empiricalEmotionJudgments_combined


def importEmpirical_InversePlanning_Base_widedf_exp6_(a1_labels, path_data, path_subjecttracker):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype

    #######
    ### Read in exp 6.2 data
    #######

    def unitscale(x): return int(x) / 48

    # #### subject data

    dtype_dict = {
        "condition": str,
        "Version": str,
        "stimID": str,
        "stimulus": str,
        "decisionThis": str,
        "decisionOther": str,
        "pot": float,
        "randStimulusFace": str,
        "BTS_actual_otherDecisionConfidence": int,
        "subjectId": str,
        "gender": str,
        "Data_Set": float,
        "HITID": str
    }

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype=dtype_dict,
                                 converters={
                                     "q1responseArray": unitscale,
                                     "q2responseArray": unitscale,
                                     "q3responseArray": unitscale},
                                 )

    assert max(datasheet_temp['q1responseArray']) == 1
    assert min(datasheet_temp['q1responseArray']) == 0

    rename_dict = {
        "q1responseArray": "bMoney",
        "q2responseArray": "bAIA",
        "q3responseArray": "bDIA",
        "BTS_actual_otherDecisionConfidence": "pi_a2"}

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    decision_this_key = {'Split': 'C', 'Stole': 'D'}
    decision_this = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        decision_this[idx] = decision_this_key[datasheet_temp['decisionThis'][idx]]
    datasheet_temp['a_1'] = pd.Series(decision_this).astype(CategoricalDtype(ordered=False, categories=a1_labels))

    # ######## trial data

    dtype_dict_participants = {
        "subjectId": str,
        # "randCondNum": int,
        "validationRadio": str,
        "subjectValidation1": bool,
        "dem_gender": str,
        "dem_language": str,
        "val_recognized": str,
        "val_feedback": str,
        "Data_Set": float,
        "HITID": str,
        "HIT_Annot": str,
        "Excluded": bool
    }

    def randcond(xin):
        if xin == 'NaN':
            x = np.nan
        else:
            assert isinstance(xin, (int, float))
            x = int(xin)
        return x

    datasheet_participants = pd.read_csv(
        path_subjecttracker,
        header=0,
        index_col=None,
        dtype=dtype_dict_participants,
        converters={"randCondNum": randcond}
    )

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    subjects_included = datasheet_participants.loc[((datasheet_participants["Data_Set"] > 6) & (datasheet_participants["Data_Set"] < 7) & np.logical_not(datasheet_participants['Excluded']) & datasheet_participants['subjectValidation1']), "subjectId"]  # DEBUG

    data_included = datasheet_temp.loc[datasheet_temp['subjectId'].isin(subjects_included), :].copy()

    data_included.rename(columns={'stimulus': 'face'}, inplace=True)  # DEBUG

    for catfield in ["face", "pot", "subjectId", "HITID", "gender"]:  # DEBUG
        data_included[catfield] = data_included[catfield].astype(CategoricalDtype(ordered=False))

    features = ["bMoney", "bAIA", "bDIA", "pi_a2"]  # NB ordering

    shorthand = {
        "bMoney": "getting money",
        "bAIA": "not getting too much",
        "bDIA": "not getting too little",
        "pi_a2": "p_1's expectation of a_2"}

    shorthand_list = [
        "getting money\n(bMoney)",
        "not getting too much\n(bAIA)",
        "not getting too little\n(bDIA)",
        "belief about $a_2$\n($\pi_{a2}$)"]

    data_reduced_wide = data_included[list(shorthand.keys()) + ['a_1', 'pot', 'face', 'subjectId']]

    data_reduced_wide_sorted = data_reduced_wide.sort_values(['a_1', 'pot', 'face'], ascending=[1, 1, 1], inplace=False)

    return data_reduced_wide_sorted.reset_index(inplace=False, drop=True), features, shorthand, shorthand_list


def importEmpirical_InversePlanning_Repu_widedf_exp9_(a1_labels, path_data, path_subjecttracker):
    import numpy as np
    import pandas as pd
    from pandas.api.types import CategoricalDtype
    from pprint import pprint

    #######
    ### Read in exp 9 data
    #######

    def unitscale(x): return int(x) / 48
    def restore_quotes(x): return str(x).replace('non-profit', 'nonprofit').replace('-', '\"').replace('  ', ', ')
    # #### subject data

    dtype_dict = {
        "stimulus": str,
        "pronoun": str,
        # "desc": str,
        "decisionThis": str,
        "pot": float,
        "respTimer": float,
        "BTS_actual_otherDecisionConfidence": int,
        "gender": str,
        "subjectId": str,
        "Data_Set": float,
        "HITID": str
    }

    datasheet_temp = pd.read_csv(path_data,
                                 header=0, index_col=None,
                                 dtype=dtype_dict,
                                 converters={
                                     "q_bMoney_Array": unitscale,
                                     "q_rMoney_Array": unitscale,
                                     "q_bAIA_Array": unitscale,
                                     "q_rAIA_Array": unitscale,
                                     "q_bDIA_Array": unitscale,
                                     "q_rDIA_Array": unitscale,
                                     "desc": restore_quotes,
                                 },
                                 )

    rename_dict = {
        "q_bMoney_Array": "bMoney",
        "q_bAIA_Array": "bAIA",
        "q_bDIA_Array": "bDIA",
        "q_rMoney_Array": "rMoney",
        "q_rAIA_Array": "rAIA",
        "q_rDIA_Array": "rDIA",
        "BTS_actual_otherDecisionConfidence": "pi_a2"}

    datasheet_temp.rename(columns=rename_dict, inplace=True)

    for catfield in ["decisionThis"]:
        datasheet_temp[catfield] = datasheet_temp[catfield].astype(CategoricalDtype(ordered=False, categories=['Split', 'Stole']))

    decision_this_key = {'Split': 'C', 'Stole': 'D'}
    decision_this = np.full_like(datasheet_temp['decisionThis'].values, 'none')
    for idx in range(datasheet_temp.shape[0]):
        decision_this[idx] = decision_this_key[datasheet_temp['decisionThis'][idx]]
    datasheet_temp['a_1'] = pd.Series(decision_this).astype(CategoricalDtype(ordered=False, categories=a1_labels))

    # ######## trial data

    dtype_dict_participants = {
        "subjectId": str,
        "randCondNum": int,
        "validationRadio": str,
        "subjectValidation1": bool,
        "expTime_min": float,
        "minRespTime_sec": float,
        "dem_gender": str,
        "dem_language": str,
        "browser_version": str,
        "browser": str,
        "visible_area": str,
        "val_recognized": str,
        "val_feedback": str,
        "Data_Set": float,
        "HITID": str,
        "HIT_Annot": str,
        "Excluded": bool
    }

    datasheet_participants = pd.read_csv(
        path_subjecttracker,
        header=0,
        index_col=None,
        dtype=dtype_dict_participants)

    for catfield in ["subjectId", "HITID", "dem_gender"]:
        datasheet_participants[catfield] = datasheet_participants[catfield].astype(CategoricalDtype(ordered=False))

    subjects_included = datasheet_participants.loc[(np.logical_not(datasheet_participants['Excluded']) & datasheet_participants['subjectValidation1']), "subjectId"]

    data_included = datasheet_temp.loc[datasheet_temp['subjectId'].isin(subjects_included), :].copy()

    data_included.rename(columns={'stimulus': 'face'}, inplace=True)

    # drop the practice trial
    data_reduced_wide_temp = data_included[data_included.face != '244_2'].copy()  # Specific to experiment9

    for catfield in ["face", "pot", "subjectId", "HITID", "gender"]:
        data_reduced_wide_temp[catfield] = data_reduced_wide_temp[catfield].astype(CategoricalDtype(ordered=False))

    features = ["bMoney", "bAIA", "bDIA", "rMoney", "rAIA", "rDIA", "pi_a2"]  # NB ordering

    shorthand = {
        "bMoney": "getting money",
        "rMoney": "reputation for not prioritizing money",
        "bAIA": "not getting too much",
        "rAIA": "reputation for being considerate",
        "bDIA": "not getting too little",
        "rDIA": "reputation for being competitive",
        "pi_a2": "$p_1$'s expectation of $a_2=D$"}

    shorthand_list = [
        "getting money\n(bMoney)",
        "reputation for not prioritizing money\n(rMoney)",
        "not getting too much\n(bAIA)",
        "reputation for being considerate\n(rAIA)",
        "not getting too little\n(bDIA)",
        "reputation for being competitive\n(rDIA)",
        "belief about $a_2$\n($\pi_{a_2=D}$)"]

    # shorthand_list = [
    #     "getting money\n($\omega_{b,Money}$)",
    #     "not prioritizing money\n($\omega_{r,Money}$)",
    #     "not getting too much\n($\omega_{b,AIA}$)",
    #     "being considerate\n($\omega_{r,AIA}$)",
    #     "not getting too little\n($\omega_{b,DIA}$)",
    #     "being competitive\n($\omega_{r,DIA}$)",
    #     "belief about $a_2$\n({})".format(r'$\pi_{a2}$')]

    data_reduced_wide = data_reduced_wide_temp[list(shorthand.keys()) + ['a_1', 'pot', 'face', 'desc', 'subjectId']]

    data_reduced_wide_sorted = data_reduced_wide.sort_values(['a_1', 'pot', 'face'], ascending=[1, 1, 1], inplace=False)

    ### WIP vvvv
    # data_stats['nsub_retained'] = len(subjects_included)
    # data_stats['nresp_retained'] = grand_selector.sum()
    # data_stats['final_nobs'] = nobsdf.copy()
    # data_stats['potential_problem'] = f" nresp/nsub = {grand_selector.sum()/len(subjects_included)}, expect {data_stats['nresp_per_sub_retained']}, grand_selector vs nobs: {grand_selector.sum()} vs {nobsdf.sum().sum()}"
    ### WIP ^^^^

    return data_reduced_wide_sorted.reset_index(inplace=False, drop=True), features, shorthand, shorthand_list


def importEmpirical_InversePlanning_(a1_labels, path_data, path_subjecttracker, verbose, suffix=''):
    import pandas as pd
    import numpy as np

    ################
    ####### Make WebPPL style df
    ################

    if suffix == 'BaseGeneric':  # experiment6.2
        df_wide, feature_list, shorthand, shorthand_list = importEmpirical_InversePlanning_Base_widedf_exp6_(a1_labels, path_data, path_subjecttracker)
    elif suffix == 'RepuSpecific':  # experiment9
        df_wide, feature_list, shorthand, shorthand_list = importEmpirical_InversePlanning_Repu_widedf_exp9_(a1_labels, path_data, path_subjecttracker)
    else:
        raise Exception

    pots = np.unique(df_wide['pot'])
    assert np.all(pots == df_wide['pot'].cat.categories)

    #####

    tempdfdict = {}
    for a_1 in a1_labels:
        tempdfdict[a_1] = [None] * len(pots)
    nobsdf = pd.DataFrame(data=np.full((len(pots), len(a1_labels)), np.nan, dtype=int), index=pots, columns=a1_labels, dtype=np.int64)
    nobsdf.index.set_names(['pots'], inplace=True)

    empiricalPreferenceJudgments = dict()

    # ##

    iterables = [['feature'], feature_list]
    idx_a1 = [(df_wide['a_1'] == a_1) for a_1 in a1_labels]
    idx_pot = [(df_wide['pot'] == pot) for pot in pots]
    for i_a1, a_1 in enumerate(a1_labels):
        for i_pot, pot in enumerate(pots):
            df = df_wide.loc[idx_a1[i_a1] & idx_pot[i_pot], feature_list]
            np.testing.assert_array_equal(df.columns.to_list(), feature_list)
            df.columns = pd.MultiIndex.from_product(iterables)
            nobsdf.loc[pot, a_1] = df.shape[0]
            if df.shape[0] > 0:
                df[('prob', 'prob')] = 1 / df.shape[0]
                tempdfdict[a_1][i_pot] = df.reset_index(inplace=False, drop=True)

        empiricalPreferenceJudgments[a_1] = pd.concat(tempdfdict[a_1], axis=0, keys=pots, names=['pots', None])
    empiricalPreferenceJudgments['nobs'] = nobsdf

    ################
    ####### Assign to ppldata dict
    ################

    dataout = {'empiricalInverseJudgments_' + suffix: empiricalPreferenceJudgments}
    dataout['pots_' + suffix] = pots
    dataout['pots_byoutcome_' + suffix] = {'all': pots}
    for i_a1, a_1 in enumerate(a1_labels):
        dataout['pots_byoutcome_' + suffix][a_1] = np.unique(df_wide.loc[idx_a1[i_a1], 'pot'])

    # ppldata['empiricalInverseJudgments_'+suffix] = empiricalPreferenceJudgments

    return dataout
