#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_plotfn_inverseplanning.py
"""


def plotDecisionStacked(sns, data, labels, display, title, ax=None):

    bp = sns.barplot(x=[0], y=[1], color=display['colors']['a_1']['D'], ax=ax)
    bp = sns.barplot(x=[0], y=[data['prob']['C']], color=display['colors']['a_1']['C'], ax=ax)

    bp.set_xticklabels([])

    bp.annotate("%.2f" % data['prob']['D'], (0, 1.), ha='center', va='center', color=display['colors']['a_1']['D'], rotation=0, xytext=(0, 10), textcoords='offset points')

    bp.annotate("%.2f" % data['prob']['C'], (0, 0.), ha='center', va='center', color=display['colors']['a_1']['C'], rotation=0, xytext=(0, -10), textcoords='offset points')

    bp.set_ylim(0, 1)
    bp.set(xlabel='decision', ylabel='frequency')
    bp.set_title(**title, y=1.16)

    return bp


def run_inversePlanningAnalysis(ppldata, paths, display, plot_param):

    import numpy as np
    from cam_webppl_utils import getEV, marginalizeContinuous

    printFigsByPotsize = plot_param['printFigsByPotsize']
    plt = plot_param['plt']
    sns = plot_param['sns']

    if printFigsByPotsize:
        paths['byPot'] = paths['figsOut'] / 'byPotsize'
        paths['byPot'].mkdir(exist_ok=True)

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

    suffix = '_BaseGeneric'

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


def genPosterFigs_pia2(dfin, paths, display_, datatitle, plot_param, invert=False):

    import numpy as np
    from cam_webppl_utils import marginalizeContinuousAcrossMultilevel

    plt = plot_param['plt']

    figsout = list()
    save_path = paths['figsOut'] / 'inversePlanning'

    ### plot marginal posteriors VS EMPIRICAL

    cColor = 'cornflowerblue'
    dColor = 'dimgrey'
    a1_color = {'C': cColor, 'D': dColor}

    scale = 2

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
        marginaldf = marginalizeContinuousAcrossMultilevel(dftemp, ['pi_a2'], nobs=dfin['nobs'][decision], bypassCheck=False)  # N.B. column labels must be iterable

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
