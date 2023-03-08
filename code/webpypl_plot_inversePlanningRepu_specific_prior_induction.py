

def convert_exp9_widedf_to_longdf_(df_wide, features):
    import numpy as np
    import pandas as pd

    assert min(df_wide['pi_a2']) == 0
    assert max(df_wide['pi_a2']) == 5
    df_array = list()
    for stim in np.unique(df_wide['face']):
        for a1 in ['C', 'D']:
            df_temp = df_wide.loc[(df_wide['face'] == stim) & (df_wide['a_1'] == a1), :]
            for feature in features:
                weight_vector = df_temp[feature].values
                if feature == 'pi_a2':
                    weight_vector = weight_vector / 5.0
                df_array.append(pd.DataFrame(data=np.array([np.full_like(weight_vector, stim, dtype=object), np.full_like(weight_vector, a1, dtype=object), np.full_like(weight_vector, feature, dtype=object), weight_vector]).T, columns=['face', 'a_1', 'feature', 'weight']))
    df_plot_temp = pd.concat(df_array).reset_index(drop=True)

    stim_desc_dict = dict()
    for stim in np.unique(df_wide['face']):
        desc_temp = np.unique(df_wide.loc[(df_wide['face'] == stim), 'desc'])
        assert len(desc_temp) == 1
        stim_desc_dict[stim] = desc_temp[0]

    return df_plot_temp, stim_desc_dict


def convert_exp6_widedf_to_longdf_(df_wide, features):
    import numpy as np
    import pandas as pd

    assert min(df_wide['pi_a2']) == 0
    assert max(df_wide['pi_a2']) == 5
    df_array = list()
    for stim in np.unique(df_wide['face']):
        for a1 in ['C', 'D']:
            df_temp = df_wide.loc[(df_wide['face'] == stim) & (df_wide['a_1'] == a1), :]
            for feature in features:
                weight_vector = df_temp[feature].values
                if feature == 'pi_a2':
                    weight_vector = weight_vector / 5.0
                df_array.append(pd.DataFrame(data=np.array([np.full_like(weight_vector, stim, dtype=object), np.full_like(weight_vector, a1, dtype=object), np.full_like(weight_vector, feature, dtype=object), weight_vector]).T, columns=['face', 'a_1', 'feature', 'weight']))
    df_plot_temp = pd.concat(df_array).reset_index(drop=True)

    return df_plot_temp


def plot_inversePlanning_PublicGame_PriorInduction(df_plot_temp, shorthand_list, plotParam, paths, title='', suffix='', hatch=True):

    import matplotlib.patches as patches
    import matplotlib.cbook as cbook
    import matplotlib.gridspec as gridspec

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    ### ALL

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0.025, hspace=0.05)

    scale = 1
    figout = plt.figure(figsize=(16 * scale, 4 * scale))
    ax1 = plt.subplot(gs1[0])
    # ax1 = figout.add_subplot(1,2,1)

    with cbook.get_sample_data(paths['stimuli'] / f'generic_avatar_male.png') as image_file:
        image = plt.imread(image_file)

    im = ax1.imshow(image)
    # patch = patches.BoxStyle("Round", pad=0.2, transform=ax.transData)
    # im.set_clip_path(patch)

    ax1.axis('off')
    ax1.set_title(f'all stim', fontsize=15)

    # ax2 = figout.add_subplot(1,2,2)
    ax2 = plt.subplot(gs1[1])

    sns.boxplot(x="feature", y="weight", hue="a_1", data=df_plot_temp, ax=ax2,
                # width=1,
                notch=True,
                hue_order=['C', 'D'],
                palette=dict(D='dimgrey', C='cornflowerblue'))

    # make grouped stripplot
    sns.stripplot(x="feature", y="weight", hue="a_1", data=df_plot_temp, ax=ax2,
                  hue_order=['C', 'D'],
                  jitter=True,
                  dodge=True,
                  marker='o',
                  alpha=0.1,
                  color='black')

    sns.pointplot(x="feature", y="weight", hue="a_1", data=df_plot_temp, ax=ax2,
                    hue_order=['C', 'D'],
                    jitter=True,
                    dodge=0.44,
                    join=False,
                    marker='.',
                    scale=0.5,
                    palette=dict(D='red', C='red'),
                    ci=None)

    print(ax2.get_xlim())
    ax2.set_xlim((-0.5, 6.5))

    # sns.pointplot(x="feature", y="weight", hue="a_1", data=df_plot_means)
    # sns.pointplot(x="feature", y="weight", hue="a_1", data=df_plot_means)

    ax2.set_title(f'all stim :: {title}', fontsize=16 * scale)
    ax2.tick_params(axis="both", labelsize=15)
    ax2.xaxis.label.set_size(15)
    ax2.yaxis.label.set_size(15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax2.set_xticklabels(shorthand_list, rotation=-30, ha='left')

    # for i_patch,patch in enumerate(ax2.artists):
    #     print(patch.__dict__)### TEMP

    if hatch:
        reputation_idx = [2, 3, 6, 7, 10, 11]
        for i_patch, patch in enumerate(ax2.artists):
            if i_patch in reputation_idx:
                patch.set_hatch("//")

    plt.show()

    figsout_return = (paths['figsOut'] / 'distal_prior_induction' / f'inversePlanningFeatures_boxplot_ALLCAT{suffix}.pdf', figout, False)

    plt.close(figout)

    return figsout_return


def plot_inversePlanning_PublicGame_PriorInduction_byStim(df_plot_temp, shorthand_list, stim_desc_dict, plotParam, paths, plot_rain=False):
    import numpy as np

    import matplotlib.patches as patches
    import matplotlib.cbook as cbook
    import matplotlib.gridspec as gridspec

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    figsout = list()

    for stim in np.unique(df_plot_temp['face']):
        # for stim in [np.unique(df['stimulus'])[0]]:
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.025, hspace=0.05)

        scale = 1
        figout = plt.figure(figsize=(16 * scale, 4 * scale))
        ax1 = plt.subplot(gs1[0])
        # ax1 = figout.add_subplot(1,2,1)

        with cbook.get_sample_data(paths['stimuli'] / f'{stim}.png') as image_file:
            image = plt.imread(image_file)

        im = ax1.imshow(image)
        # patch = patches.BoxStyle("Round", pad=0.2, transform=ax.transData)
        # im.set_clip_path(patch)

        ax1.axis('off')

        ncount = (df_plot_temp.loc[((df_plot_temp['face'] == stim) & (df_plot_temp['feature'] == 'bMoney') & (df_plot_temp['a_1'] == 'C')), 'weight'].shape[0],  # WIP
                  df_plot_temp.loc[((df_plot_temp['face'] == stim) & (df_plot_temp['feature'] == 'bMoney') & (df_plot_temp['a_1'] == 'D')), 'weight'].shape[0])
        # ax1.set_title('{} n{}'.format(stim, ncount), fontsize=15)

        # ax2 = figout.add_subplot(1,2,2)
        ax2 = plt.subplot(gs1[1])

        sns.boxplot(x="feature", y="weight", hue="a_1", data=df_plot_temp[df_plot_temp['face'] == stim], ax=ax2,
                    notch=True, showfliers=False, bootstrap=10000,
                    palette=dict(D='dimgrey', C='cornflowerblue'))

        if plot_rain:
            ### make grouped stripplot
            sns.stripplot(x="feature", y="weight", hue="a_1", data=df_plot_temp[df_plot_temp['face'] == stim], ax=ax2,
                          jitter=True,
                          dodge=True,
                          marker='o',
                          alpha=0.5,
                          color='black')

            sns.pointplot(x="feature", y="weight", hue="a_1", data=df_plot_temp, ax=ax2,
                          jitter=True,
                          dodge=0.44,
                          join=False,
                          marker='.',
                          scale=0.5,
                          palette=dict(D='red', C='red'),
                          ci=None)

        ax2.set_xlim((-0.5, 6.5))
        ax2.set_title(f'{stim_desc_dict[stim]}', fontsize=16 * scale, fontweight='bold')
        ax2.tick_params(axis="both", labelsize=15)
        ax2.xaxis.label.set_size(15)
        ax2.yaxis.label.set_size(15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax2.set_xticklabels(shorthand_list, rotation=-30, ha='left')

        reputation_idx = [2, 3, 6, 7, 10, 11]
        for i_patch, patch in enumerate(ax2.artists):
            if i_patch in reputation_idx:
                # patch.set_hatch("///")
                patch.set_hatch("xx")

        # ax.artists
        # font
        # font_scale
        # .set_yticklabels(ppldata['labels']['emotions_hc_intersect'],weight='bold')
        # verdana
        # axarr[1].set_ylabel('MSE', fontsize=11*scale)
        # igout.suptitle('{} (loss = {:+.4f})'.format(feature,errors[i_feature,:,:].sum()), fontsize=15, y=1.02, fontweight='bold')

        # ax2.font_scale(2)
        plt.show()

        figsout.append((paths['figsOut'] / 'distal_prior_induction' / 'feature_weights_boxplot' / 'inversePlanningFeatures_boxplot_{}.pdf'.format(stim), figout, False))

        plt.close(figout)

    return figsout


def fit_priors_exp6_(a1_labels, df_wide, feature_list):
    import numpy as np
    import scipy.stats

    df_wide_reduced = df_wide.drop(['subjectId', 'face'], axis=1).sort_values(['a_1', 'pot'], ascending=[1, 1])

    empratings = dict()
    for feature in feature_list:
        empratings[feature] = dict()
        for a1 in a1_labels:
            empratings[feature][a1] = df_wide_reduced.loc[df_wide_reduced['a_1'] == a1, feature].values
        empratings[feature]['marginal'] = df_wide_reduced[feature].values

    ### Calculate the prior parameters

    ### Center the values in their bins
    features_betas = feature_list.copy()
    features_betas.remove('pi_a2')

    n = 49
    rescaled_min = 1 / (2 * n)
    rescaled_max = 1 - 1 / (2 * n)
    def rescale_intensities_(x): return (2 * (n - 1) * x + 1) / (2 * n)

    prior_param = dict()
    for feature in features_betas:
        prior_param[feature] = dict()
        for marginal in ['marginal', 'C', 'D']:
            ### Center the values in their bins
            rescaled_intensities = rescale_intensities_(empratings[feature][marginal])

            a, b, _, _ = scipy.stats.beta.fit(rescaled_intensities, floc=0, fscale=1)
            prior_param[feature][marginal] = (a, b)

    feature = 'pi_a2'
    prior_param[feature] = dict()
    n = 6
    rescaled_min = 0
    rescaled_max = 5
    for marginal in ['marginal', 'C', 'D']:
        density, values = np.histogram(empratings[feature][marginal], bins=np.arange(0, n + 1) - 0.5, density=True)

        prior_param[feature][marginal] = density

    return prior_param, rescale_intensities_, empratings


def fit_priors_exp9_(a1_labels, df_wide, feature_list):
    import numpy as np
    import scipy.stats

    df_wide_reduced = df_wide.drop(['subjectId', 'desc'], axis=1).sort_values(['face', 'a_1', 'pot'], ascending=[1, 1, 1])

    empratings = dict()
    for stim in df_wide_reduced['face'].cat.categories.to_list():
        empratings[stim] = dict()
        for feature in feature_list:
            empratings[stim][feature] = dict()
            for a1 in a1_labels:
                empratings[stim][feature][a1] = df_wide_reduced.loc[(df_wide_reduced['face'] == stim) & (df_wide_reduced['a_1'] == a1), feature].values
            empratings[stim][feature]['marginal'] = df_wide_reduced.loc[df_wide_reduced['face'] == stim, feature].values

    ### Calculate the prior parameters

    ### Center the values in their bins
    n = 49
    rescaled_min = 1 / (2 * n)
    rescaled_max = 1 - 1 / (2 * n)
    def rescale_intensities_(x): return (2 * (n - 1) * x + 1) / (2 * n)

    features_betas = feature_list.copy()
    features_betas.remove('pi_a2')
    distal_prior_param = dict()
    for stim in df_wide_reduced['face'].cat.categories.to_list():
        distal_prior_param[stim] = dict()
        for feature in features_betas:
            distal_prior_param[stim][feature] = dict()

            for marginal in ['marginal', 'C', 'D']:
                ### Center the values in their bins
                rescaled_intensities = rescale_intensities_(empratings[stim][feature][marginal])

                # max(empratings[stim][feature]['marginal'])
                # min(empratings[stim][feature]['marginal'])

                # max(rescaled_intensities) == 1 - 1/(2*n)
                # min(rescaled_intensities) == 1/(2*n)

                a, b, _, _ = scipy.stats.beta.fit(rescaled_intensities, floc=0, fscale=1)
                distal_prior_param[stim][feature][marginal] = (a, b)

                # print(f'For {stim}, beta param :: ({a},{b})')

    feature = 'pi_a2'
    n = 6
    rescaled_min = 0
    rescaled_max = 5
    for stim in df_wide_reduced['face'].cat.categories.to_list():
        distal_prior_param[stim][feature] = dict()
        for marginal in ['marginal', 'C', 'D']:
            density, values = np.histogram(empratings[stim][feature][marginal], bins=np.arange(0, n + 1) - 0.5, density=True)

            distal_prior_param[stim][feature][marginal] = density

    return distal_prior_param, rescale_intensities_, empratings


def plot_prior_exp9(df_wide, stim_desc_dict9, distal_prior_param, rescale_intensities_, empratings, paths, display, plotParam):
    import matplotlib.gridspec as gridspec
    import matplotlib.cbook as cbook
    import numpy as np
    import scipy.stats

    def histOutline(dataIn, *args, **kwargs):
        import numpy as np

        (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

        stepSize = binsIn[1] - binsIn[0]

        bins = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
        data = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
        for bb in range(len(binsIn)):
            bins[2 * bb + 1] = binsIn[bb]
            bins[2 * bb + 2] = binsIn[bb] + stepSize
            if bb < len(histIn):
                data[2 * bb + 1] = histIn[bb]
                data[2 * bb + 2] = histIn[bb]

        bins[0] = bins[1]
        bins[-1] = bins[-2]
        data[0] = 0
        data[-1] = 0

        return (bins, data)

    ### DEBUG get these variables passed in rather than reading from disk again vvvv
    df_wide_reduced = df_wide.drop(['subjectId', 'desc'], axis=1).sort_values(['face', 'a_1', 'pot'], ascending=[1, 1, 1])
    # ^^^^^^^

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    n = 49
    rescaled_min = 1 / (2 * n)
    rescaled_max = 1 - 1 / (2 * n)

    edges = np.arange(0, n + 1, 1) / n

    x = np.linspace(rescaled_min, rescaled_max, 100)

    #####

    figsout = list()

    plot_marginal = False  # Whether to plot the sum of C and D
    # for i_stim,stim in enumerate([df_wide_reduced['face'].cat.categories.to_list()[0]]):
    for stim in df_wide_reduced['face'].cat.categories.to_list():

        gs1 = gridspec.GridSpec(2, 4)
        gs1.update(wspace=0.18, hspace=.45)

        scale = 1
        figout = plt.figure(figsize=(18 * scale, 6 * scale))

        ax1 = plt.subplot(gs1[0])

        with cbook.get_sample_data(paths['stimuli'] / f'{stim}.png') as image_file:
            image = plt.imread(image_file)

        im = ax1.imshow(image)

        ax1.axis('off')
        c_count = len(empratings[stim]['bMoney']['C'])
        d_count = len(empratings[stim]['bMoney']['D'])
        ax1.set_title('{} n(C={}, D={})'.format(stim, c_count, d_count), fontsize=15)

        # gsmap = {'bMoney':1, 'bAIA':2, 'bDIA':3, 'rMoney':5, 'rAIA':6, 'rDIA':7, 'pi_a2':4}
        gsmap = {'bMoney': 1, 'bAIA': 2, 'bDIA': 3, 'rMoney': 5, 'rAIA': 6, 'rDIA': 7}
        for feature in list(gsmap.keys()):
            ax = plt.subplot(gs1[gsmap[feature]])

            if plot_marginal:
                a, b = distal_prior_param[stim][feature]['marginal']
                y = scipy.stats.beta.pdf(x, a, b)
                ax.plot(x, y, 'k-', lw=3, label='prior')

            ac, bc = distal_prior_param[stim][feature]['C']
            ax.plot(x, scipy.stats.beta.pdf(x, ac, bc), '-', color='cornflowerblue', lw=2, label='C')
            ad, bd = distal_prior_param[stim][feature]['D']
            ax.plot(x, scipy.stats.beta.pdf(x, ad, bd), '-', color='k', lw=2, label='D')

            ax.hist(rescale_intensities_(empratings[stim][feature]['D']), bins=edges, color='k', alpha=0.5)
            ax.hist(rescale_intensities_(empratings[stim][feature]['C']), bins=edges, color='cornflowerblue', alpha=0.5)

            # sns.distplot(rescale_intensities_(empratings[stim][feature]['D']), rug=True, hist=False, rug_kws={'color': 'dimgrey'}, kde_kws={'color': 'dimgrey', 'lw': 2})

            ax.set_xlim((0, 1))

            ax.set_title(f'{feature}')

        # plt.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)
        pia2_edges = np.arange(-0.5, 6.5, 1)

        ax = plt.subplot(gs1[4])

        if plot_marginal:
            (bins, counts) = histOutline(empratings[stim]['pi_a2']['marginal'], **{'bins': pia2_edges})
            ax.plot(bins, counts, 'k-')

        (bins, counts) = histOutline(empratings[stim]['pi_a2']['C'], **{'bins': pia2_edges})
        ax.plot(bins, counts, '-', color='cornflowerblue')
        ax.hist(empratings[stim]['pi_a2']['C'], bins=pia2_edges, color='cornflowerblue', alpha=0.5)

        (bins, counts) = histOutline(empratings[stim]['pi_a2']['D'], **{'bins': pia2_edges})
        ax.plot(bins, counts, '-', color='dimgrey')
        ax.hist(empratings[stim]['pi_a2']['D'], bins=pia2_edges, color='dimgrey', alpha=0.5)

        ax.axvline(x=2.5, linewidth=1, color='r')
        ax.set_xlim([min(pia2_edges), max(pia2_edges)])
        ax.set_title(r'$\pi_{a2}$')
        ax.set_xticks([0, 5])
        ax.set_xticklabels(['0', '1'])
        ax.set_xlabel('P($a_2$ = D)')

        plt.suptitle(stim_desc_dict9[stim])

        plt.show()

        figsout.append((paths['figsOut'] / 'distal_prior_induction' / 'prior_function_plots' / 'inversePlanningFeatures_learnedDists_{}.pdf'.format(stim), figout, False))

        plt.close(figout)

    return figsout


def plot_prior_betafunction(df_wide, generic_prior_param, rescale_intensities_, paths, display, plotParam):
    import matplotlib.gridspec as gridspec
    import matplotlib.cbook as cbook
    import numpy as np
    import scipy.stats

    def histOutline(dataIn, *args, **kwargs):
        import numpy as np

        (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

        stepSize = binsIn[1] - binsIn[0]

        bins = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
        data = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
        for bb in range(len(binsIn)):
            bins[2 * bb + 1] = binsIn[bb]
            bins[2 * bb + 2] = binsIn[bb] + stepSize
            if bb < len(histIn):
                data[2 * bb + 1] = histIn[bb]
                data[2 * bb + 2] = histIn[bb]

        bins[0] = bins[1]
        bins[-1] = bins[-2]
        data[0] = 0
        data[-1] = 0

        return (bins, data)

    ### DEBUG get these variables passed in rather than reading from disk again vvvv
    df_wide_reduced = df_wide.drop(['subjectId', 'face'], axis=1).sort_values(['a_1', 'pot'], ascending=[1, 1])
    # ^^^^^^^

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    prior_fit = {
        'bMoney': [1.7197083773927708, 0.772231557418525],
        'bAIA': [0.18912803454985763, 0.9088211809964003],
        'bDIA': [1.5630042102551907, 0.6495103627265263]
    }

    n = 49
    rescaled_min = 1 / (2 * n)
    rescaled_max = 1 - 1 / (2 * n)

    edges = np.arange(0, n + 1, 1) / n

    x = np.linspace(rescaled_min, rescaled_max, 100)

    #####

    plot_marginal = True  # Whether to plot the sum of C and D
    # for i_stim,stim in enumerate([df_wide_reduced['face'].cat.categories.to_list()[0]]):
    # for stim in df_wide_reduced['face'].cat.categories.to_list():

    gs1 = gridspec.GridSpec(2, 4)
    gs1.update(wspace=0.18, hspace=.45)

    scale = 1
    figout = plt.figure(figsize=(18 * scale, 6 * scale))

    ax1 = plt.subplot(gs1[0])

    with cbook.get_sample_data(paths['stimuli'] / f'generic_avatar_male.png') as image_file:
        image = plt.imread(image_file)

    im = ax1.imshow(image)

    ax1.axis('off')
    c_count = (df_wide_reduced['a_1'] == 'C').sum()
    d_count = (df_wide_reduced['a_1'] == 'D').sum()
    ax1.set_title('n(C={}, D={})'.format(c_count, d_count), fontsize=15)

    gsmap = {'bMoney': 1, 'bAIA': 2, 'bDIA': 3}  # , 'rMoney':5, 'rAIA':6, 'rDIA':7}
    for feature in list(gsmap.keys()):
        ax = plt.subplot(gs1[gsmap[feature]])

        if plot_marginal:
            a, b = generic_prior_param[feature]['marginal']
            y = scipy.stats.beta.pdf(x, a, b)
            ax.plot(x, y, 'k-', lw=3, label='prior')

        ### plot comparison to earlier fitting
        a, b = prior_fit[feature]
        y = scipy.stats.beta.pdf(x, a, b)
        ax.plot(x, y, 'k--', lw=3, label='fit prior')

        ac, bc = generic_prior_param[feature]['C']
        print('generic_prior_param[feature][C]:  ')
        print(generic_prior_param[feature]['C'])
        ax.plot(x, scipy.stats.beta.pdf(x, ac, bc), '-', color='cornflowerblue', lw=2, label='C')
        ad, bd = generic_prior_param[feature]['D']
        ax.plot(x, scipy.stats.beta.pdf(x, ad, bd), '-', color='dimgrey', lw=2, label='D')

        # ax.hist(rescale_intensities_( df_wide_reduced.loc[ df_wide_reduced['a_1'] == 'C', feature ] ), bins=edges, color='cornflowerblue', density=True, alpha=0.5)
        # ax.hist(rescale_intensities_( df_wide_reduced.loc[ df_wide_reduced['a_1'] == 'D', feature ] ), bins=edges, color='k', density=True, alpha=0.5)

        # sns.distplot(rescale_intensities_(empratings[stim][feature]['D']), rug=True, hist=False, rug_kws={'color': 'dimgrey'}, kde_kws={'color': 'dimgrey', 'lw': 2})

        ax.set_xlim((0, 1))

        ax.set_title(f'{feature}')

    # plt.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)
    pia2_edges = np.arange(-0.5, 6.5, 1)

    ax = plt.subplot(gs1[4])

    if plot_marginal:
        (bins, counts) = histOutline(df_wide_reduced.loc[:, 'pi_a2'], **{'bins': pia2_edges})
        ax.plot(bins, counts, 'k-')

    (bins, counts) = histOutline(df_wide_reduced.loc[df_wide_reduced['a_1'] == 'C', 'pi_a2'], **{'bins': pia2_edges})
    ax.plot(bins, counts, '-', color='cornflowerblue')
    ax.hist(df_wide_reduced.loc[df_wide_reduced['a_1'] == 'C', 'pi_a2'], bins=pia2_edges, color='cornflowerblue', alpha=0.5)

    (bins, counts) = histOutline(df_wide_reduced.loc[df_wide_reduced['a_1'] == 'D', 'pi_a2'], **{'bins': pia2_edges})
    ax.plot(bins, counts, '-', color='dimgrey')
    ax.hist(df_wide_reduced.loc[df_wide_reduced['a_1'] == 'D', 'pi_a2'], bins=pia2_edges, color='dimgrey', alpha=0.5)

    ax.axvline(x=2.5, linewidth=1, color='r')
    ax.set_xlim([min(pia2_edges), max(pia2_edges)])
    ax.set_title(r'$\pi_{a2}$')
    ax.set_xticks([0, 5])
    ax.set_xticklabels(['0', '1'])
    ax.set_xlabel('P($a_2$ = D)')

    # plt.suptitle(stim_desc_dict9[stim])

    plt.show()

    figout_tuple = (paths['figsOut'] / 'distal_prior_induction' / 'comparison_learned_vs_summed_prior_inversePlanningFeatures_learnedDists.pdf', figout, False)

    # plt.close(figout)

    return figout_tuple


def get_empirical_inverse_planning_priors_OLD_(a1_labels, paths):
    from webpypl_importjson import importEmpirical_InversePlanning_Base_widedf_exp6_, importEmpirical_InversePlanning_Repu_widedf_exp9_

    '''Maybe ? Update to use ppldata data (i.e. store df_wides in ppldata)'''
    ### DEBUG get these variables passed in rather than reading from disk again vvvv
    df_wide6, feature_list6, shorthand6, shorthand_list6 = importEmpirical_InversePlanning_Base_widedf_exp6_(a1_labels, paths['exp6xlsx'], paths['subjectrackerexp6'])

    df_wide9, feature_list9, shorthand9, shorthand_list9 = importEmpirical_InversePlanning_Repu_widedf_exp9_(a1_labels, paths['exp9xlsx'], paths['subjectrackerexp9'])
    ### ^^^^^^

    generic_prior_param, _, empratings6 = fit_priors_exp6_(a1_labels, df_wide6, feature_list6)

    distal_prior_param, rescale_intensities_, empratings9 = fit_priors_exp9_(a1_labels, df_wide9, feature_list9)

    return df_wide9, shorthand9, shorthand_list9, distal_prior_param, empratings9, df_wide6, shorthand6, shorthand_list6, generic_prior_param, empratings6, rescale_intensities_


def get_empirical_inverse_planning_priors_(invPlanExtras_BaseGeneric, invPlanExtras_RepuSpecific):

    df_wide6, feature_list6, shorthand6, shorthand_list6 = invPlanExtras_BaseGeneric['df_wide'].copy(), invPlanExtras_BaseGeneric['feature_list'], invPlanExtras_BaseGeneric['shorthand'], invPlanExtras_BaseGeneric['shorthand_list']

    df_wide9, feature_list9, shorthand9, shorthand_list9 = invPlanExtras_RepuSpecific['df_wide'].copy(), invPlanExtras_RepuSpecific['feature_list'], invPlanExtras_RepuSpecific['shorthand'], invPlanExtras_RepuSpecific['shorthand_list']

    a1_labels = ['C', 'D']

    generic_prior_param, _, empratings6 = fit_priors_exp6_(a1_labels, df_wide6, feature_list6)

    distal_prior_param, rescale_intensities_, empratings9 = fit_priors_exp9_(a1_labels, df_wide9, feature_list9)

    return df_wide9, shorthand9, shorthand_list9, distal_prior_param, empratings9, df_wide6, shorthand6, shorthand_list6, generic_prior_param, empratings6, rescale_intensities_


def plot_inversePlanning_PublicGame_PriorInduction_byStim_cdc(df_plot_temp, shorthand_list, stim_desc_dict, plotParam, paths, hatch=True, nbootstraps=1000):
    import numpy as np

    import matplotlib.patches as patches
    import matplotlib.cbook as cbook
    import matplotlib.gridspec as gridspec

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    figsout = list()

    for stim in np.unique(df_plot_temp['face']):
        # for stim in [np.unique(df['stimulus'])[0]]:
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.025, hspace=0.05)

        scale = 1
        figout = plt.figure(figsize=(16 * scale, 4 * scale))
        ax1 = plt.subplot(gs1[0])
        # ax1 = figout.add_subplot(1,2,1)

        ### DEBUG make path relative
        with cbook.get_sample_data(paths['stimuli'] / f'{stim}.png') as image_file:
            image = plt.imread(image_file)

        im = ax1.imshow(image)
        # patch = patches.BoxStyle("Round", pad=0.2, transform=ax.transData)
        # im.set_clip_path(patch)

        ax1.axis('off')

        ncount = (df_plot_temp.loc[((df_plot_temp['face'] == stim) & (df_plot_temp['feature'] == 'bMoney') & (df_plot_temp['a_1'] == 'C')), 'weight'].shape[0],
                  df_plot_temp.loc[((df_plot_temp['face'] == stim) & (df_plot_temp['feature'] == 'bMoney') & (df_plot_temp['a_1'] == 'D')), 'weight'].shape[0])
        # ax1.set_title('{} n{}'.format(stim, ncount), fontsize=15)

        # ax2 = figout.add_subplot(1,2,2)
        ax2 = plt.subplot(gs1[1])

        sns.boxplot(x="feature", y="weight", hue="a_1", data=df_plot_temp[df_plot_temp['face'] == stim], ax=ax2,
                    notch=hatch, showfliers=False, bootstrap=nbootstraps,
                    palette=dict(D='dimgrey', C='cornflowerblue'))

        # make grouped stripplot
        # sns.stripplot(x="feature", y="weight", hue="a_1", data=df_plot_temp[df_plot_temp['face']==stim], ax=ax2,
        #             jitter=True,
        #             dodge=True,
        #             marker='o',
        #             alpha=0.5,
        #             color='black')

        # sns.pointplot(x="feature", y="weight", hue="a_1", data=df_plot_temp, ax=ax2,
        #             jitter=True,
        #             dodge=0.44,
        #             join=False,
        #             marker='.',
        #             scale=0.5,
        #             palette=dict(D='red', C='red'),
        #             ci=None)

        ax2.set_xlim((-0.5, 6.5))
        ax2.set_title(f'{stim_desc_dict[stim]}', fontsize=24 * scale, fontweight='bold')
        ax2.tick_params(axis="both", labelsize=15)
        ax2.xaxis.label.set_size(15)
        ax2.yaxis.label.set_size(15)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # g._legend.remove()
        ax2.get_legend().remove()

        ax2.set_xticklabels(shorthand_list, rotation=-35, horizontalalignment='left', rotation_mode='anchor')

        reputation_idx = [2, 3, 6, 7, 10, 11]
        for i_patch, patch in enumerate(ax2.artists):
            if i_patch in reputation_idx:
                patch.set_hatch("///")

        ax2.set_xlabel('')
        ax2.set_ylabel('Weight', fontsize=18, fontweight='bold')
        # ax.artists
        # font
        # font_scale
        # .set_yticklabels(ppldata['labels']['emotions_hc_intersect'],weight='bold')
        # verdana
        # axarr[1].set_ylabel('MSE', fontsize=11*scale)
        # igout.suptitle('{} (loss = {:+.4f})'.format(feature,errors[i_feature,:,:].sum()), fontsize=15, y=1.02, fontweight='bold')

        # ax2.font_scale(2)
        plt.show()

        figsout.append((paths['figsOut'] / 'distal_prior_induction' / 'feature_weights_boxplot_cdc' / 'inversePlanningFeatures_boxplot_{}.pdf'.format(stim), figout, False))

        plt.close(figout)

    return figsout


def plot_inversePlanning_PublicGame_PriorInduction_byStim_thin(df_plot_temp, shorthand_list, stim_desc_dict, plotParam, paths, plot_rain=False):
    import numpy as np

    import matplotlib.patches as patches
    import matplotlib.cbook as cbook
    import matplotlib.gridspec as gridspec

    #### plot pia2 as squares
    #  https://matplotlib.org/3.1.1/gallery/statistics/errorbars_and_boxes.html#sphx-glr-gallery-statistics-errorbars-and-boxes-py

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    figsout = list()

    for stim in np.unique(df_plot_temp['face'][:1]):
        # for stim in [np.unique(df['stimulus'])[0]]:
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(wspace=0.325, hspace=0.05)

        scale = 1
        figout = plt.figure(figsize=(9 * scale, 4 * scale), constrained_layout=True)
        ax1 = plt.subplot(gs1[0])
        # ax1 = figout.add_subplot(1,2,1)

        with cbook.get_sample_data(paths['stimuli'] / f'{stim}.png') as image_file:
            image = plt.imread(image_file)

        im = ax1.imshow(image)
        # patch = patches.BoxStyle("Round", pad=0.2, transform=ax.transData)
        # im.set_clip_path(patch)

        ax1.axis('off')

        ncount = (df_plot_temp.loc[((df_plot_temp['face'] == stim) & (df_plot_temp['feature'] == 'bMoney') & (df_plot_temp['a_1'] == 'C')), 'weight'].shape[0],  # WIP
                  df_plot_temp.loc[((df_plot_temp['face'] == stim) & (df_plot_temp['feature'] == 'bMoney') & (df_plot_temp['a_1'] == 'D')), 'weight'].shape[0])
        # ax1.set_title('{} n{}'.format(stim, ncount), fontsize=15)

        # ax2 = figout.add_subplot(1,2,2)
        ax2 = plt.subplot(gs1[1])

        sns.boxplot(x="feature", y="weight", hue="a_1", data=df_plot_temp[df_plot_temp['face'] == stim], ax=ax2,
                    notch=True, showfliers=False, bootstrap=10000,
                    width=.5,
                    linewidth=0,
                    whiskerprops=dict(linestyle='-', linewidth=1),
                    palette=dict(D='dimgrey', C='cornflowerblue'))

        if plot_rain:
            ### make grouped stripplot
            sns.stripplot(x="feature", y="weight", hue="a_1", data=df_plot_temp[df_plot_temp['face'] == stim], ax=ax2,
                          jitter=True,
                          dodge=True,
                          marker='o',
                          alpha=0.5,
                          color='black')

            sns.pointplot(x="feature", y="weight", hue="a_1", data=df_plot_temp, ax=ax2,
                          jitter=True,
                          dodge=0.44,
                          join=False,
                          marker='.',
                          scale=0.5,
                          palette=dict(D='red', C='red'),
                          ci=None)

        # rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
        # ax.add_patch(rect)

        ax2.set_xlim((-0.5, 6.5))
        ax2.set_title(f'{stim_desc_dict[stim]}', fontsize=18 * scale, fontweight='bold')
        ax2.tick_params(axis="both", labelsize=18)
        ax2.xaxis.label.set_size(18)
        ax2.yaxis.label.set_size(18)
        ax2.set_ylim((0, 1))
        ax2.set_xlabel(None)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        ax2.set_xticklabels(shorthand_list, rotation=-30, ha='left', rotation_mode='anchor')

        reputation_idx = [2, 3, 6, 7, 10, 11]
        for i_patch, patch in enumerate(ax2.artists):
            if i_patch in reputation_idx:
                # patch.set_hatch("///")
                patch.set_hatch("xxxx")

        # ax.artists
        # font
        # font_scale
        # .set_yticklabels(ppldata['labels']['emotions_hc_intersect'],weight='bold')
        # verdana
        # axarr[1].set_ylabel('MSE', fontsize=11*scale)
        # igout.suptitle('{} (loss = {:+.4f})'.format(feature,errors[i_feature,:,:].sum()), fontsize=15, y=1.02, fontweight='bold')

        # ax2.font_scale(2)
        plt.show()

        figsout.append((paths['figsOut'] / 'distal_prior_induction' / 'feature_weights_boxplot_thin' / 'inversePlanningFeatures_boxplot_{}.pdf'.format(stim), figout, False))

        plt.close(figout)

    return figsout


def plot_inversePlanning_PriorInductionsCompared_wrapper(df_wide9, shorthand9, shorthand_list9, distal_prior_param, empratings9, df_wide6, shorthand6, shorthand_list6, generic_prior_param, rescale_intensities_, paths, display, plotParam):

    df_long6 = convert_exp6_widedf_to_longdf_(df_wide6, list(shorthand6.keys()))

    df_long9, stim_desc_dict9 = convert_exp9_widedf_to_longdf_(df_wide9, list(shorthand9.keys()))

    # ##

    figsout = list()

    figsout.append(plot_prior_betafunction(df_wide6, generic_prior_param, rescale_intensities_, paths, display, plotParam))

    figsout.append(plot_prior_exp9(df_wide9, stim_desc_dict9, distal_prior_param, rescale_intensities_, empratings9, paths, display, plotParam))

    figsout.append(plot_inversePlanning_PublicGame_PriorInduction(df_long9, shorthand_list9, plotParam, paths, title='PublicGame with Distal Prior Information', suffix='_exp9', hatch=True))

    figsout.append(plot_inversePlanning_PublicGame_PriorInduction(df_long6, shorthand_list6, plotParam, paths, title='AnonymousGame', suffix='_exp6', hatch=False))

    figsout.append(plot_inversePlanning_PublicGame_PriorInduction_byStim(df_long9, shorthand_list9, stim_desc_dict9, plotParam, paths))

    figsout.append(plot_inversePlanning_PublicGame_PriorInduction_byStim_thin(df_long9, shorthand_list9, stim_desc_dict9, plotParam, paths))

    figsout.append(plot_inversePlanning_PublicGame_PriorInduction_byStim_cdc(df_long9, shorthand_list9, stim_desc_dict9, plotParam, paths, hatch=True, nbootstraps=nbootstraps))

    return figsout
