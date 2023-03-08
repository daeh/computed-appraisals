# coding: utf-8
# # Webppl play goldenballs

def run_inversePlanningAnalysis(ppldata, paths, display, plot_param):
    # ## data
    import numpy as np

    from webpypl import getEV, marginalizeContinuous
    from webpypl_plotfun import plotDecisionStacked

    pots = ppldata['pots']
    isInteractive = plot_param['isInteractive']
    verbose = plot_param['verbose']
    showAllFigs = plot_param['showAllFigs']
    printFigsByPotsize = plot_param['printFigsByPotsize']
    plt = plot_param['plt']
    sns = plot_param['sns']

    if printFigsByPotsize:
        paths['byPot'] = paths['figsOut'] / 'byPotsize'
        paths['byPot'].mkdir(exist_ok=True)

    # %%
    # # Level 0 & 2 - agent decision
    # *frequency of decision*

    plt.close('all')
    figsout = list()

    for i_obslevel, obslevel in enumerate([0, 2]):
        levellabel = {0: 'Base', 2: 'Reputation'}[obslevel]

        ### plot
        fig = plt.figure(figsize=(2, 3))
        ax = fig.add_subplot(1, 1, 1)
        title = {'label': '{} agent decision posterior \n(decision collapsed across pot sizes)'.format(levellabel)}
        ax = plotDecisionStacked(sns, ppldata['level{}'.format(obslevel)].groupby(level=1).mean(), ppldata['labels']['decisions'], display, title, ax=ax)

        figsout.append((paths['figsOut'] / 'inversePlanning' / 'decision{}.pdf'.format(levellabel), fig, False))
        plt.close(fig)

    for i_obslevel, obslevel in enumerate([0, 2]):
        levellabel = {0: 'Base', 2: 'Reputation'}[obslevel]

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)

        condColors = [display['colors']['a_1']['C']]
        idx_quad = 0
        labels = {'x': 'ln(pot)', 'y': 'Frequency', 'title': {'label': 'Cooperation Frequency - {} Agent'.format(levellabel)}}
        x = np.log(ppldata['level{}'.format(obslevel)].index.levels[0])
        y = ppldata['level{}'.format(obslevel)].loc[(slice(None), slice('C')), :].prob.values

        # fit with np.polyfit
        m, b = np.polyfit(x, y, 1)

        ax.plot(x.values, y, '.', color=condColors[idx_quad], alpha=.6)

        ax.plot(x.values, (m * x + b).values, color=condColors[idx_quad], alpha=.6)
        ax.plot(0, y.mean(), '.', color=condColors[idx_quad], markersize=20, alpha=0.6)
        ax.set_ylim((0, 1))
        ax.set_xlim((0, 14))
        ax.set_xlabel(labels['x'])
        ax.set_ylabel(labels['y'])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.text(0.2, .96 - 0.05 * idx_quad, '{}: mean={:0.2f} m={:0.2f}'.format('C', y.mean(), m), color=condColors[idx_quad], size=11)
        plt.title(**labels['title'])

        figsout.append((paths['figsOut'] / 'inversePlanning' / f'decision_bypot_{levellabel}.pdf', fig, False))
        plt.close(fig)

    # %%

    suffix = '_BaseGeneric'  # formally '_exp6'

    d = dict()
    v = dict()
    se = dict()

    for i_a1, decision in enumerate(ppldata['labels']['decisions']):
        d[decision] = np.full((len(ppldata['labels']['baseFeatures']), len(ppldata['pots_byoutcome' + suffix][decision])), np.nan)
        v[decision] = np.full_like(d[decision], np.nan)
        se[decision] = np.full_like(d[decision], np.nan)
        df = ppldata['empiricalInverseJudgments' + suffix][decision]
        for i_pot, pot in enumerate(ppldata['pots_byoutcome' + suffix][decision]):
            for i_feature, feature in enumerate(ppldata['labels']['baseFeatures']):
                d[decision][i_feature, i_pot], v[decision][i_feature, i_pot] = getEV(marginalizeContinuous(df.loc[pot, ], [feature]))
                se[decision][i_feature, i_pot] = np.sqrt(v[decision][i_feature, i_pot]) / np.sqrt(df.loc[pot, ].shape[0] - 1)

    scale = 1

    x1 = np.linspace(np.min(ppldata['pots_byoutcome' + suffix]['all']), np.max(ppldata['pots_byoutcome' + suffix]['all']), 100)
    y1 = x1**0.17

    feature_colors = {'bMoney': 'green', 'bAIA': 'blue', 'bDIA': 'red'}
    for decision in ppldata['labels']['decisions']:
        fig = plt.figure(figsize=(10 * scale, 5 * scale))
        ax = fig.add_subplot(1, 1, 1)

        for i_feature, feature in enumerate(ppldata['labels']['baseFeatures']):
            ax.plot(np.log1p(ppldata['pots_byoutcome' + suffix][decision]), d[decision][i_feature, ], color=feature_colors[feature], alpha=.6)
            ax.errorbar(np.log1p(ppldata['pots_byoutcome' + suffix][decision]), d[decision][i_feature, ], se[decision][i_feature, ], color=feature_colors[feature], alpha=.6, linewidth=5)
        ax.set_title('$EV[\omega_b|a_1={}]$'.format(decision))
        ax.legend(ppldata['labels']['baseFeatures'])
        ax.set_ylabel('$\omega_b$')
        ax.set_ylim((0, 1))
        ax.set_xlabel('ln(pot+1)')

        figsout.append((paths['figsOut'] / 'inversePlanning' / 'empiricalPreferences-{}.pdf'.format(decision), fig))
        plt.close('all')

    return figsout


def run_inversePlanningAnalysis_marginalPlots(datain, ppldataLabels, save_path, display, plot_param):
    ### NOTE not used

    # ## data
    import numpy as np
    import pandas as pd
    # import sympy as sp
    import itertools
    # ## i/o
    # ## interaction
    from pprint import pprint

    from webpypl import unweightProbabilities

    isInteractive = plot_param['isInteractive']
    verbose = plot_param['verbose']
    showAllFigs = plot_param['showAllFigs']
    printFigsByPotsize = plot_param['printFigsByPotsize']
    plt = plot_param['plt']
    sns = plot_param['sns']

    plt.close('all')
    figsout = list()

    featlabels = datain['labels'] + ['pi_a2']

    expandeddf = dict()

    ### plot heatmaps of posteriors
    for decision in ppldataLabels['decisions']:
        mdf = datain[decision]
        topkeys = np.unique(mdf.index.get_level_values(0))
        tempdflist = [None] * len(topkeys)
        nobslist = [None] * len(topkeys)
        for i_key, key in enumerate(topkeys):
            tempdflist[i_key] = mdf.loc[key, ]
            nobslist[i_key] = datain['nobs'][decision].loc[key, ]
        tempconcat = pd.concat(tempdflist)

        ### normalize
        tempconcat.prob = tempconcat.prob.divide(tempconcat.prob.sum())

        tempconcat
        assert tempconcat.columns.nlevels == 2
        tempconcat.columns = tempconcat.columns.droplevel(level=0)

        expandeddf[decision] = unweightProbabilities(tempconcat, nobs=np.sum(nobslist))

    for i_feature, feature in enumerate(featlabels):
        fig, ax = plt.subplots()

        for i_decision, decision in enumerate(ppldataLabels['decisions']):
            ### 5-10s per loop

            ax = sns.distplot(expandeddf[decision][feature], kde_kws={"color": display['colors']['a_1'][decision], "lw": 3, "label": "KDE", "alpha": .2, "shade": True}, hist_kws={"histtype": "step", "linewidth": 3, "alpha": .7, "color": display['colors']['a_1'][decision]})

            ax.scatter(expandeddf[decision][feature].mean(), 0, marker='o', alpha=1, facecolors=display['colors']['a_1'][decision], edgecolors=display['colors']['a_1'][decision], linewidth=1)

        ax.set_title('P( {} | {} )'.format(feature, r'$a_1$'))
        ax.set_xlabel('{}'.format(feature))
        figsout.append((save_path / 'marginalDist_{}.pdf'.format(featlabels[i_feature]), fig, False))
        plt.close(fig)

    ### ~270s

    for decision in ppldataLabels['decisions']:

        slices = list(itertools.combinations(range(0, len(featlabels)), 2))

        for thisslice in slices:

            ax = sns.JointGrid(x=featlabels[thisslice[0]], y=featlabels[thisslice[1]], data=expandeddf[decision], space=0, dropna=True, xlim=(0, 1), ylim=(0, 1)).set_axis_labels('P( {} | {} )'.format(featlabels[thisslice[0]], decision), 'P( {} | {} )'.format(featlabels[thisslice[1]], decision))
            ax = ax.plot_joint(sns.kdeplot, cmap="Blues_d", n_levels=10, shade=True)
            ax = ax.plot_marginals(sns.kdeplot, shade=True)

            fig = ax.fig

            figsout.append((save_path / 'marginalHeatmap_{}_vs_{}_{}.pdf'.format(featlabels[thisslice[0]], featlabels[thisslice[1]], decision), fig, False))

            plt.close('all')

    return figsout


def genPosterFigs_pia2(dfin, paths, display, datatitle, plot_param, invert=False):
    import numpy as np
    from webpypl import marginalizeContinuousAcrossMultilevel

    """
    ## figsout.append( genPosterFigs_pia2(ppldata['level1'], paths, display, 'Base Model', plot_param) )
    display, plot_param = plot_param['display_param'], plot_param
    dfin, datatitle  = ppldata['level1'], 'Base Model'
    """

    isInteractive = plot_param['isInteractive']
    verbose = plot_param['verbose']
    showAllFigs = plot_param['showAllFigs']
    plt = plot_param['plt']
    sns = plot_param['sns']

    figsout = list()
    save_path = paths['figsOut'] / 'inversePlanning'

    #########
    # %%
    ### plot marginal posteriors VS EMPIRICAL

    cColor = 'cornflowerblue'
    dColor = 'dimgrey'
    a1_color = {'C': cColor, 'D': dColor}

    scale = 2
    barscale = 1
    lw = 0

    fig = plt.figure(figsize=(2.5 * scale, 1 * scale))
    ax = fig.add_subplot(1, 1, 1)
    for decision in ['C', 'D']:
        xmark = 0.5

        # x = [1/12,3/12,5/12,7/12,9/12,11/12]

        dftemp = dfin[decision].copy()
        x1 = np.unique(np.around(dftemp[('feature', 'pi_a2')].to_numpy(), decimals=4))
        x = np.arange(x1.size) / x1.size
        dftemp.loc[:, ('feature', 'pi_a2')] = np.around(dftemp[('feature', 'pi_a2')].to_numpy(), decimals=14)
        assert dftemp.columns.nlevels == 2
        dftemp.columns = dftemp.columns.droplevel(level=0)
        marginaldf = marginalizeContinuousAcrossMultilevel(dftemp, ['pi_a2'], nobs=dfin['nobs'][decision], bypassCheck=False)  # N.B. column labels must be tuple. If only one, must have terminal comma, e.g. ('col1',)

        ### normalized pdf histogram

        nbins = x.size + 1
        edges = np.linspace(0, 1, nbins, endpoint=True)

        # adjust edges of outline
        edges1 = edges.copy()
        edges1[0] = edges1[0] - 0.1
        edges1[-1] = edges1[-1] + 0.1

        if invert:
            weights = marginaldf.to_numpy()[::-1]
        else:
            weights = marginaldf.to_numpy()

        ax.hist(x, bins=edges, weights=weights, density=False, facecolor=a1_color[decision], linewidth=0, alpha=0.25, label=decision)
        ax.hist(x, bins=edges1, weights=weights, density=False, edgecolor=a1_color[decision], linewidth=2 * scale, histtype='step', label=decision)

    ax.set_ylabel('probability', fontsize=28).set_color('black')
    ax.set_ylim([0, .5])

    ax.set_xticks([0, .5, 1])
    ax.set_xlim([0 - .0015, 1.0015])

    ax.set_xticklabels(['0', '0.5', '1'])

    ax.set_xlabel(r'$\pi_{a_2}$ for $a_2=C$', fontsize=28)

    if xmark is not None:
        ax.plot([xmark, xmark], [0, .5], '--', color='black', alpha=1, lw=1)

    ax.grid(False)

    ax.tick_params(axis='both', labelsize=28)

    figsout.append((save_path / f"otherDecision_{datatitle.replace(' ', '-')}.pdf", fig, False))

    plt.close(fig)

    return figsout


def inverse_planning_posterfigs_wrapper(ppldata, paths, display, plot_param):

    figsout = list()

    figsout.append(genPosterFigs_pia2(ppldata['level1'], paths, display, 'Base Model', plot_param))
    figsout.append(genPosterFigs_pia2(ppldata['empiricalInverseJudgments_BaseGeneric'], paths, display, 'Base Empirical', plot_param, invert=True))

    figsout.append(genPosterFigs_pia2(ppldata['level3'], paths, display, 'Repu Model', plot_param))
    figsout.append(genPosterFigs_pia2(ppldata['empiricalInverseJudgments_RepuSpecific'], paths, display, 'Repu Empirical', plot_param, invert=True))

    return figsout
