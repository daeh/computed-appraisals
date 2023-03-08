
def prep_generic_data_pair_(ppldata):
    import numpy as np
    import pandas as pd
    from webpypl import unweightProbabilities

    composite_training_emodict = dict()
    composite_training_emodict['nobs'] = ppldata['empiricalEmotionJudgments']['nobs'].copy(deep=True)
    composite_training_emodict['nobs'].loc[:, :] = np.nan

    composite_training_iafdict = dict()
    composite_training_iafdict['nobs'] = ppldata['level4IAF']['nobs'].copy(deep=True)
    composite_training_iafdict['nobs'].loc[:, :] = int(0)

    for outcome in ppldata['labels']['outcomes']:
        new_emp_data_list = list()
        new_iaf_data_list = list()

        for pot in ppldata['empiricalEmotionJudgments']['nobs'].index.get_level_values(0):
            df_temp_generic = unweightProbabilities(ppldata['empiricalEmotionJudgments'][outcome].loc[pot, :], ppldata['empiricalEmotionJudgments']['nobs'].loc[pot, outcome])
            df_temp_generic['pots'] = pot
            new_emp_data_list.append(df_temp_generic.set_index('pots'))

            df_temp_genericiaf = unweightProbabilities(ppldata['level4IAF'][outcome].loc[pot, :], ppldata['level4IAF']['nobs'].loc[pot, outcome])
            df_temp_genericiaf['pots'] = pot
            new_iaf_data_list.append(df_temp_genericiaf.set_index('pots'))

        composite_training_emodict[outcome] = pd.concat(new_emp_data_list)
        composite_training_iafdict[outcome] = pd.concat(new_iaf_data_list)

        for pot in np.unique(ppldata['empiricalEmotionJudgments']['nobs'].index.get_level_values(0)):
            nobs = composite_training_emodict[outcome].loc[pot, :].shape[0]
            composite_training_emodict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
            composite_training_emodict['nobs'].loc[pot, outcome] = nobs

            nobs = composite_training_iafdict[outcome].loc[pot, :].shape[0]
            composite_training_iafdict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
            composite_training_iafdict['nobs'].loc[pot, outcome] = nobs

    return composite_training_emodict, composite_training_iafdict


def prep_specific_data_pair_(distal_ppldata, nobsdf_template):
    import numpy as np
    import pandas as pd
    from webpypl import unweightProbabilities

    composite_training_emodict = dict()
    composite_training_emodict['nobs'] = nobsdf_template.copy(deep=True)
    composite_training_emodict['nobs'].loc[:, :] = np.nan

    composite_training_iafdict = dict()
    composite_training_iafdict['nobs'] = nobsdf_template.copy(deep=True)
    composite_training_iafdict['nobs'].loc[:, :] = int(0)

    for a1 in ['C', 'D']:
        for a2 in ['C', 'D']:
            outcome = f'{a1}{a2}'
            new_emp_data_list = list()
            new_iaf_data_list = list()

            for obspot in distal_ppldata[a1]['empiricalEmotionJudgments']['nobs'].index:
                assert obspot in composite_training_emodict['nobs'].index.get_level_values(0)

            for pot in composite_training_emodict['nobs'].index.get_level_values(0):

                ### add distal prior empirical ratings
                if pot in distal_ppldata[a1]['empiricalEmotionJudgments']['nobs'].index:
                    df_temp_distal = unweightProbabilities(distal_ppldata[a1]['empiricalEmotionJudgments'][outcome].loc[pot, :], distal_ppldata[a1]['empiricalEmotionJudgments']['nobs'].loc[pot, outcome])
                    df_temp_distal['pots'] = pot
                    new_emp_data_list.append(df_temp_distal.set_index('pots'))

                    df_temp_iaf = unweightProbabilities(distal_ppldata[a1]['level4IAF'][outcome].loc[pot, :], distal_ppldata[a1]['level4IAF']['nobs'].loc[pot, outcome])
                    df_temp_iaf['pots'] = pot
                    new_iaf_data_list.append(df_temp_iaf.set_index('pots'))

            composite_training_emodict[outcome] = pd.concat(new_emp_data_list)
            composite_training_iafdict[outcome] = pd.concat(new_iaf_data_list)

            for pot in np.unique(composite_training_emodict[outcome].index.get_level_values(0)):
                nobs = composite_training_emodict[outcome].loc[pot, :].shape[0]
                composite_training_emodict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
                composite_training_emodict['nobs'].loc[pot, outcome] = nobs

                nobs = composite_training_iafdict[outcome].loc[pot, :].shape[0]
                composite_training_iafdict[outcome].loc[pot, ('prob', 'prob')] = nobs**-1
                composite_training_iafdict['nobs'].loc[pot, outcome] = nobs

    return composite_training_emodict, composite_training_iafdict
