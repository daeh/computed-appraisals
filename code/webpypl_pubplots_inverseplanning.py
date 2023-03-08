

def make_wide_df_for_IP(invplandf):
    import pandas as pd
    from webpypl import unweightProbabilities

    nobsdf = invplandf['nobs']
    a1s = list(nobsdf.columns)
    df_array = list()
    for a1 in a1s:
        pots_by_a1_temp = nobsdf.index[nobsdf[a1] > 0]
        for i_pot, pot in enumerate(pots_by_a1_temp):
            data_slice = invplandf[a1].loc[pot, slice('feature', 'prob')]
            nobs = nobsdf[a1].loc[pot]
            # emodict_temp1['prob']['prob'][0] = emodict_temp1['prob']['prob'][0]**-1
            data_slice_unweighted = unweightProbabilities(data_slice, nobs=nobs)

            data_slice_unweighted.columns = data_slice_unweighted.columns.droplevel(0)
            df_out = data_slice_unweighted
            df_out['pot'] = pot
            df_out['a_1'] = a1
            df_array.append(df_out)

    return pd.concat(df_array)


def convert_model_level1_level3_widedf_to_longdf_(df_wide, features):
    import numpy as np
    import pandas as pd

    df_wide_ = df_wide.drop(columns=['prob'])

    df_array = list()
    for a1 in ['C', 'D']:
        for pot in np.unique(df_wide_['pot'].values):
            df_temp = df_wide_.loc[(df_wide_['a_1'] == a1) & (df_wide_['pot'] == pot), :]
            for feature in features:
                weight_vector = df_temp[feature].values
                df_array.append(pd.DataFrame(data=np.array([np.full_like(weight_vector, a1, dtype=object), np.full_like(weight_vector, feature, dtype=object), weight_vector]).T, columns=['a_1', 'feature', 'weight']))
    df_plot_temp = pd.concat(df_array).reset_index(drop=True)

    return df_plot_temp


def aggregate_df6_df9(df_wide9, df_wide6):
    import numpy as np
    import pandas as pd

    def remap_pia2(emp_):
        assert emp_ in [0, 1, 2, 3, 4, 5]
        remap = np.array([11, 9, 7, 5, 3, 1]) / 12
        return remap[emp_]

    df_wide9_ = df_wide9.copy()
    df_wide6_ = df_wide6.copy()

    df_array = list()
    df_array9 = list()
    df_array6 = list()

    remap_ = np.vectorize(remap_pia2)

    for a1 in ['C', 'D']:
        for pot in np.unique(df_wide9_['pot'].values):
            df_temp = df_wide9_.loc[(df_wide9_['a_1'] == a1) & (df_wide9_['pot'] == pot), :]
            for feature in df_wide9_.columns[~df_wide9_.columns.isin(['pot', 'face', 'a_1', 'desc', 'subjectId'])].to_list():
                if feature == 'pi_a2':
                    weight_vector = remap_(df_temp[feature].values)
                else:
                    weight_vector = df_temp[feature].values

                df_cat = pd.DataFrame(data=np.array([np.full_like(weight_vector, a1, dtype=object), np.full_like(weight_vector, feature, dtype=object)]).T, columns=['a_1', 'feature'])
                df_cat['weight'] = weight_vector
                df_array.append(df_cat)
                df_array9.append(df_cat)

    for a1 in ['C', 'D']:
        for pot in np.unique(df_wide6_['pot'].values):
            df_temp = df_wide6_.loc[(df_wide6_['a_1'] == a1) & (df_wide6_['pot'] == pot), :]
            for feature in df_wide6_.columns[~df_wide6_.columns.isin(['pot', 'face', 'a_1', 'desc', 'subjectId'])].to_list():
                if feature == 'pi_a2' and df_temp[feature].shape[0] > 0:
                    weight_vector = remap_(df_temp[feature].values)
                else:
                    weight_vector = df_temp[feature].values
                # df_cat = pd.DataFrame(data=np.array([np.full_like(weight_vector,a1,dtype=object), np.full_like(weight_vector,feature,dtype=object), weight_vector]).T, columns=['a_1','feature','weight'])
                ### preserve float type of weights
                df_cat = pd.DataFrame(data=np.array([np.full_like(weight_vector, a1, dtype=object), np.full_like(weight_vector, feature, dtype=object)]).T, columns=['a_1', 'feature'])
                df_cat['weight'] = weight_vector
                df_array.append(df_cat)
                df_array6.append(df_cat)

    df_long_agg69 = pd.concat(df_array).reset_index(drop=True)
    df_long9 = pd.concat(df_array9).reset_index(drop=True)
    df_long6 = pd.concat(df_array6).reset_index(drop=True)

    df9_dict = dict()
    for stimid in np.unique(df_wide9_['face'].values):
        df_array9_bystim = list()
        for a1 in ['C', 'D']:
            for pot in np.unique(df_wide9_['pot'].values):
                df_temp = df_wide9_.loc[(df_wide9_['face'] == stimid) & (df_wide9_['a_1'] == a1) & (df_wide9_['pot'] == pot), :]
                for feature in df_wide9_.columns[~df_wide9_.columns.isin(['face', 'pot', 'a_1', 'desc', 'subjectId'])].to_list():
                    if feature == 'pi_a2' and df_temp[feature].shape[0] > 0:
                        weight_vector = remap_(df_temp[feature].values)
                    else:
                        weight_vector = df_temp[feature].values

                    # df_array9_bystim.append( pd.DataFrame(data=np.array([np.full_like(weight_vector,stimid,dtype=object), np.full_like(weight_vector,a1,dtype=object), np.full_like(weight_vector,feature,dtype=object), weight_vector]).T, columns=['face','a_1','feature','weight']) )
                    df_temp2 = pd.DataFrame(data=np.array([np.full_like(weight_vector, a1, dtype=object), np.full_like(weight_vector, feature, dtype=object)]).T, columns=['a_1', 'feature'])
                    df_temp2['weight'] = weight_vector
                    df_array9_bystim.append(df_temp2.copy())

        df9_dict[stimid] = pd.concat(df_array9_bystim).reset_index(drop=True)

    return df_long_agg69, df9_dict, df_long6
