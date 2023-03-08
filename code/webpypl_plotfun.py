# coding: utf-8
# # Webppl play goldenballs

class SaveTextVar():
    def __init__(self, save_path):
        self.save_path = save_path

        if not self.save_path.is_dir():
            if self.save_path.exists():
                raise Exception(f'There is something here {self.save_path}')
            else:
                self.save_path.mkdir(parents=True)

    def write(self, txt, fname):
        fpath = self.save_path / fname
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        if fpath.exists() and fpath.is_file():
            fpath.unlink()
        fpath.write_text(txt)


def printFigList(figlist, plot_param, save_kwargs=None):
    """
    takes printFigList(figslist, plt) or printFigList(figslist, dict(plt=plt))
    (fpath,fig,showFig,dupPath_)
    """
    from collections.abc import Iterable
    import itertools
    import shutil

    def flatten(l):
        from collections.abc import Iterable
        import itertools
        for el in l:
            if isinstance(el, itertools.chain):
                yield from (flatten(list(el)))
            elif isinstance(el, (Iterable, itertools.chain)) and not isinstance(el, (str, bytes, tuple, type(None))):
                yield from flatten(el)
            else:
                yield el

    def show_figure(fig, plt):
        """
        create a dummy figure and use its
        manager to display 'fig'
        """
        dummy = plt.figure()
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)

    if save_kwargs is None:
        save_kwargs = dict()

    if isinstance(plot_param, dict):
        plt = plot_param['plt']
    else:
        plt = plot_param

    if isinstance(figlist, (Iterable, itertools.chain)):
        flatlist = list(flatten(figlist))
    else:
        flatlist = list()

    listidx = list(range(len(flatlist)))

    print(f'Printing {len(flatlist)} figures')

    plt.close('all')

    printed, duplicates, overwritten, unprintable = list(), list(), list(), list()
    for ii, item in enumerate(flatlist):
        if isinstance(item, tuple) and len(item) in [2, 3, 4] and item is not None:
            showFig = False
            dupPath_ = None
            if len(item) == 2:
                fpath, fig = item
            elif len(item) == 3:
                fpath, fig, showFig = item
            elif len(item) == 4:
                fpath, fig, showFig, dupPath_ = item
            else:
                fpath = None
                raise TypeError

            if isinstance(fpath, str):
                from pathlib import Path
                print('OSPATH ERROR')
                print(fpath)
                fpath = Path(fpath)
            directory = fpath.resolve().parent
            filename = fpath.name
            extension = fpath.suffix

            if not directory.exists():
                print('Creating directory: {}'.format(str(directory)))
                directory.mkdir(parents=True, exist_ok=True)
            if fpath.exists():
                overwritten.append(str(fpath))
            show_figure(fig, plt)

            try:
                plt.savefig(fpath, format=extension[1::], bbox_inches='tight', **save_kwargs)
            except RuntimeError:
                print(f"\n\nERROR -- PRINT FAILED for :: {fpath}\n\n")
                unprintable.append(item)
                plt.close('all')
            if not showFig:
                plt.close(fig)

            if not dupPath_ is None:
                dupPath = dupPath_.resolve()

                if not dupPath.suffix:
                    ### if directory
                    dubDirectory = dupPath
                    dupFile = dubDirectory / filename
                else:
                    ### if file
                    dubDirectory = dupPath.parent
                    dupFile = dupPath.name

                if not fpath.exists():
                    from warnings import warn
                    warn(f'Cannot copy non-existant file {str(fpath)}')
                else:
                    if not dubDirectory.exists():
                        dubDirectory.mkdir(parents=True)

                    shutil.copy(fpath, dubDirectory / dupFile)

            print(f'({ii+1}/{len(flatlist)})\t{fpath.name}\tsaved to\t{str(fpath)}')
            duplicates.append(str(fpath)) if str(fpath) in printed else printed.append(str(fpath))
        else:
            print('ERROR{')
            print(type(item))
            print(item)
            print('}')

    if len(overwritten) > 0:
        print(f'\n{len(overwritten)} files overwritten:')
        _ = [print(item) for item in overwritten if item not in duplicates]
    if len(duplicates) > 0:
        import warnings
        msg = f'\n{len(duplicates)} duplicate figures printed:'
        print(msg)
        for item in duplicates:
            print(item)
        warnings.warn(msg)

    if len(unprintable) == 0:
        print('Figure Printing Complete')
    else:
        import warnings
        warnings.warn(f'printFigList Errors :: {len(unprintable)} figures returned')
    return unprintable


def get_plt(isInteractive=False):
    from copy import deepcopy
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    ### notebook plot style settings
    # https://seaborn.pydata.org/tutorial/aesthetics.html

    # display_param_['mplParam'].update({
    #     'font.size': 8,
    #     'text.usetex': True,
    #     'text.latex.preamble': r'\usepackage{amsfonts}'
    # })

    mplParam = {
        'axes.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.format': 'pdf',
        ###
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{bm,microtype}\usepackage{amsfonts}\usepackage{wasysym}',
    }
    sns_context = {
        'context': 'paper'
    }
    sns_style = {
        'style': 'whitegrid',
        'rc': {
            'figure.facecolor': 'white'
        }
    }

    sns.set_style("white")
    matplotlib.rcParams.update(mplParam)
    sns.set_context(**sns_context)
    sns.set_style(**sns_style)

    display_param_ = dict(
        mplParam=mplParam,
        mplParam_full=deepcopy(plt.rcParams),
        snsParam=dict(context=sns_context, style=sns_style)
    )

    if not isInteractive:
        matplotlib.use('pdf')

    return plt, sns, isInteractive, display_param_


def init_plot_objects(summary_results_base, isInteractive=None):
    import numpy as np

    if isInteractive is None:
        isInteractive = False
        try:
            if __IPYTHON__:  # type: ignore
                isInteractive = True
        except NameError:
            isInteractive = False

    paths = {
        'figsOut': summary_results_base / 'figs',
        'figsPub': summary_results_base / 'figsPub',
        'varsPub': summary_results_base / 'textvars',
    }
    # for dirpath in paths.values():
    #     dirpath.mkdir(parents=True, exist_ok=True)

    outcomes = ['CC', 'CD', 'DC', 'DD']

    emotion_neworder = [
        'Disappointment',
        'Annoyance',
        'Devastation',
        'Fury',
        'Contempt',
        'Regret',
        'Disgust',
        'Confusion',
        'Embarrassment',
        'Envy',
        'Guilt',
        'Joy',
        'Excitement',
        'Relief',
        'Amusement',
        'Pride',
        'Surprise',
        'Sympathy',
        'Gratitude',
        'Respect', ]

    colors = {
        'a_1': {'C': 'cornflowerblue', 'D': 'dimgray'},
        'outcome-std': dict(CC='#4da64d', CD='#4d4dff', DC='#ff4d4d', DD='#4d4d4d'),
        'outcome-bright': dict(CC='#61E066', CD='#61C2E0', DC='#E06190', DD='#A3A3A3'),
        'outcome-desat': dict(CC='#8AE08D', CD='#8BCCE0', DC='#E08BAA', DD='#B5B5B5'),
    }

    plt, sns, isInteractive, display_param_ = get_plt(isInteractive=False)

    display_param_['colors'] = colors

    plot_param_ = {
        'plt': plt,
        'sns': sns,
        'isInteractive': isInteractive,
        'verbose': False,
        'showAllFigs': False,
        'bypass_print': False,
        'display_param': display_param_,
        'printFigList': printFigList,
        'save_text_var': SaveTextVar(paths['varsPub']),
        'figsOut': paths['figsOut'],
        'figsPub': paths['figsPub'],
        'varsPub': paths['varsPub'],
        'emotion_neworder': emotion_neworder,
        'outcomes': outcomes,
    }

    return plot_param_


def plotMarginalsContinuousComparisonGrouped(axes1, axes2, df_model, df_empirical, outcome, edges, plotModelEV=True, xmark=None):
    import numpy as np
    import scipy.stats

    x_model = df_model.index
    p_model = df_model.values
    EV_model = np.inner(x_model, p_model)  # expected value
    Var_model = np.inner(p_model, np.square(x_model)) - np.inner(x_model, p_model)**2

    x_empirical = df_empirical.index
    p_empirical = df_empirical.values
    EV_empirical = np.inner(x_empirical, p_empirical)  # expected value
    Var_empirical = np.inner(p_empirical, np.square(x_empirical)) - np.inner(x_empirical, p_empirical)**2

    ### normalized pdf histogram

    n_model, bins_model, rectangles_model = axes1.hist(x_model, bins=edges, weights=p_model, density=True, facecolor=(0, 0, 0, 0.5), edgecolor='black', linewidth=0)
    n_empirical, bins_empirical, rectangles_empirical = axes2.hist(x_empirical, bins=edges, weights=p_empirical, density=True, facecolor=(0, 0, 1, 0.5), edgecolor='black', linewidth=0)

    axes2.tick_params(axis='y', colors='blue')
    axes2.grid(False)

    ### overlay histogram edge
    axes2.hist(x_empirical, bins=edges, weights=p_empirical, density=True, facecolor="none", edgecolor='green', linewidth=2)
    axes1.hist(x_model, bins=edges, weights=p_model, density=True, facecolor="none", edgecolor='black', linewidth=2)

    axes1.set_xlim((edges[0], edges[-1]))

    ymax = max([axes1.get_ylim()[1], axes2.get_ylim()[1]])
    axes1.set_ylim((0, ymax))
    axes2.set_ylim((0, ymax))

    ### plot origin
    ylim = axes1.get_ylim()
    if xmark is not None:
        axes1.plot([xmark, xmark], ylim, '--', color='black', alpha=1, lw=2)
    axes1.set_ylim(ylim)
    axes1.set_xlabel('intensity').set_color('black')
    axes1.set_ylabel('probability density').set_color('black')

    import scipy.stats
    n_emp_norm = np.multiply(n_empirical, np.full(n_empirical.shape, 1.0 / n_empirical.sum(), dtype=float))  # n_empirical.size
    n_mod_norm = np.multiply(n_model, np.full(n_model.shape, 1.0 / n_model.sum(), dtype=float))  # n_model.size
    temp_D_KL = scipy.stats.entropy(n_model, n_emp_norm, base=2)

    ### plot expected value
    if plotModelEV:
        axes1.plot([EV_model, EV_model], ylim, color='red', alpha=.8, lw=1.5)
        axes1.text(EV_model, ylim[1] * 1.02, '{:0.2f} ± {:0.2f}'.format(EV_model, np.sqrt(Var_model)), color='red')
    axes2.plot([EV_empirical, EV_empirical], ylim, color='blue', alpha=.8, lw=1.5)
    axes2.text(EV_empirical, ylim[1] * 1.08, '{:0.2f} ± {:0.2f}'.format(EV_empirical, np.sqrt(Var_empirical)), color='blue')

    axes1.set_title('{top}: {exp} = {dkl:0.2f}'.format(top=outcome, dkl=temp_D_KL, exp=r'$D_{KL}(model||empirical)$'), y=1.12)

    return axes1, axes2


def plotMarginalBars(axes, data, display, title):

    ### switch column order
    dftemp = data[list(reversed(data.columns.tolist()))]

    sbp1 = dftemp.plot.bar(stacked=True, color=display['colors']['weight_bernoulli'], ax=axes)

    handles, labels = sbp1.get_legend_handles_labels()
    # labels2 = [l.replace('prob', 'weight=') for l in labels]
    labels2 = labels
    axes.legend(handles[::-1], labels2[::-1], title='weight', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    axes.set_ylabel('probability')
    axes.set_title(**title, y=1.08)
    axes.set_ylim([0, 1])

    axes.tick_params(axis='x', pad=20)
    axes.xaxis.labelpad = 15

    ### 1 values (bottom)
    for rect in sbp1.patches[0: int(len(sbp1.patches) / 2)]:
        sbp1.annotate("%.3g" % rect.get_height(), (rect.get_x() + rect.get_width() / 2., 0.),
                      ha='center', va='center', color=display['colors']['weight_bernoulli'][0], rotation=0, xytext=(0, -10),
                      textcoords='offset points')

    ### 0 values (top)
    for rect in sbp1.patches[int(len(sbp1.patches) / 2): int(len(sbp1.patches))]:
        sbp1.annotate("%.3g" % rect.get_height(), (rect.get_x() + rect.get_width() / 2., 1.),
                      ha='center', va='center', color=display['colors']['weight_bernoulli'][1], rotation=0, xytext=(0, 10),
                      textcoords='offset points')

    return axes


def plotMarginalsContinuous(axes1, df, title, nbins=10, xlim=None, xmark=None):
    import numpy as np

    x = df.index
    p = df.values

    EV = np.inner(x, p)  # expected value
    Var = np.inner(p, np.square(x)) - np.inner(x, p)**2

    ### normalized pdf histogram

    if xlim is None:
        edges = np.linspace(x.min(), x.max(), nbins, endpoint=True)
    else:
        edges = np.linspace(xlim[0], xlim[1], nbins, endpoint=True)
    n, bins, rectangles = axes1.hist(x, bins=edges, weights=p, density=True, facecolor="grey", edgecolor='black', linewidth=0)

    ### plot origin
    ylim = axes1.get_ylim()
    if xmark is not None:
        axes1.plot([xmark, xmark], ylim, '--', color='black', alpha=1, lw=2)
    axes1.set_ylim(ylim)
    axes1.set_xlabel('intensity').set_color('black')
    axes1.set_ylabel('probability density').set_color('black')

    if xlim is not None:
        axes1.set_xlim(xlim)
    '''
    axes2 = axes1.twinx()

    ### points
    axes2.plot(x, p, 'ro', ms=10, alpha=.1)
    # axes2.set_ylabel('probability (points)').set_color('red')
    axes2.tick_params(axis='y', colors='red')
    axes2.grid(False)
    '''
    ### overlay histogram edge
    axes1.hist(x, bins=edges, weights=p, density=True, facecolor="none", edgecolor='black', linewidth=2)

    ### plot expected value
    axes1.plot([EV, EV], ylim, color='red', alpha=.8, lw=1.5)
    axes1.text(EV, ylim[1] * 1.02, '{:0.2f} ± {:0.2f}'.format(EV, np.sqrt(Var)), color='red')

    axes1.set_title(**title, y=1.08)

    return axes1


def plotPotRegressions(axes1, x, Y, conditions, displayparam, labels={'x': '', 'y': '', 'title': {'label': ''}}, ylim=(0, 1)):
    import numpy as np

    steps = np.linspace(Y.max(), Y.min(), 20)

    if Y.min() > min(ylim) and Y.max() < max(ylim):
        axes1.set_ylim(ylim)
    else:
        axes1.set_ylim((Y.min(), Y.max()))

    for idx_cond, condition in enumerate(conditions):

        y = Y[idx_cond, :]
        m, b = np.polyfit(x, y, 1)

        axes1.scatter(x, y, marker='o', alpha=0.3, facecolors=displayparam['face'][idx_cond], edgecolors=displayparam['edge'][idx_cond], linewidth=displayparam['edgewidth'])
        axes1.plot(x, m * x + b, displayparam['line'], color=displayparam['edge'][idx_cond], alpha=0.3)
        axes1.scatter(0, y.mean(), s=200, marker='o', alpha=0.3, facecolors=displayparam['mean'][idx_cond], edgecolors=displayparam['edge'][idx_cond], linewidth=displayparam['edgewidth'])

        axes1.set_xlim((0, 14))
        axes1.set_xlabel(labels['x'])
        axes1.set_ylabel(labels['y'])
        # (axes1.get_ylim()[1]-.4)-((axes1.get_ylim()[1] - axes1.get_ylim()[0])/100*idx_cond*(axes1.get_ylim()[1]-axes1.get_ylim()[0]))

        if displayparam['showr']:
            axes1.text(0.25,
                       steps[idx_cond + 2],
                       '{}: mean={:0.2f} m={:0.2f}'.format(condition, y.mean(), m), color=displayparam['edge'][idx_cond], size=11)
        axes1.set_title(**labels['title'])

    return axes1


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


def plotMarginalHeatmap(plt, sns, data, slices, labels, display, suptitle):
    import numpy as np
    import pandas as pd

    support = np.array(data.index.tolist())

    ### set display matrix size
    maxcol = 6
    if len(slices) % maxcol > 0:
        dispcolumns = []
        for j in range(maxcol, 0, -1):
            if len(slices) % j == 0:
                dispcolumns.append(j)
    else:
        dispcolumns = [maxcol]

    figdims = 2  # in inches

    dispcol = dispcolumns[0]
    disprow = int(np.ceil(len(slices) / dispcolumns[0]))

    # pprint('{0} rows x {1} col'.format(disprow, dispcol))
    fig, axs = plt.subplots(nrows=disprow, ncols=dispcol, figsize=(figdims * dispcol + (6 / 4) * dispcol, figdims * disprow + 1), squeeze=False)  # , sharey=True, sharex=True
    plt.tight_layout()
    fig.suptitle(**suptitle, fontsize=16, y=1.08)

    for j, con in enumerate(slices):

        a = data.index.tolist()
        b = support[:, con].tolist()

        data2 = data.assign(sliced=pd.Series(b).values)
        data2.sliced = data2.sliced.apply(tuple)

        df = data2.groupby('sliced').agg({'prob': sum})

        c = df.index.tolist()
        d = {labels[con[0]]: [x[0] for x in c],
             labels[con[1]]: [x[1] for x in c]}
        data4 = df.assign(**d)

        g = labels[con[0]]
        h = labels[con[1]]
        result = data4.pivot(index=g, columns=h, values='prob')
        (ii, jj) = np.unravel_index(j, (disprow, dispcol))

        axs[ii, jj].set_aspect('equal')
        # cbarviz = True if jj + 1 == dispcol else False # only display heatbar once per row
        cbarviz = True
        hm1 = sns.heatmap(result, annot=True, fmt=".3g", vmin=0, vmax=1, square=True, ax=axs[ii, jj], cbar=cbarviz)  # cmap='RdBu_r'viridis, vmin=0, vmax=2
        hm1.invert_yaxis()

        ### axs[ii,jj].set_title('test{0}'.format(j), y=1.0) # subplot title

    # plt.show()
    return plt


def plotHeatmapDistOverPots_prep_emoDict_(emoDictIAF, i_feature=None, feature=None, outcomes=None, pots=None, transformation_fn=None, transformation_label=None):
    from webpypl import marginalizeContinuous

    all_x = list()
    all_p = list()
    temp_marginals = {'x': dict(), 'p': dict()}
    for i_outcome, outcome in enumerate(outcomes):
        temp_marginals['x'][outcome] = [None] * len(pots)
        temp_marginals['p'][outcome] = [None] * len(pots)
        for i_pot, pot in enumerate(pots):
            temp_marginaldf = marginalizeContinuous(emoDictIAF[outcome].loc[pot, ], [feature])

            if transformation_fn is None:
                xvals_ = temp_marginaldf.index.to_numpy()
                pvals_ = temp_marginaldf.to_numpy()
            else:
                xvals_ = transformation_fn(temp_marginaldf.index.to_numpy())
                pvals_ = temp_marginaldf.to_numpy()
            all_x.extend(xvals_.tolist())
            all_p.extend(pvals_.tolist())
            temp_marginals['x'][outcome][i_pot] = xvals_
            temp_marginals['p'][outcome][i_pot] = pvals_

    return dict(temp_marginals=temp_marginals, all_x=all_x, all_p=all_p, i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=transformation_fn, transformation_label=transformation_label)


def plotHeatmapDistOverPots_prep_preppedData_(feature_df, i_feature=None, feature=None, transformation_fn=None, transformation_label=None):
    import numpy as np

    fdf = feature_df
    pot_outcome_pairs = fdf.loc[:, ('pot', 'outcome')].copy().drop_duplicates().sort_values(by=['outcome', 'pot']).reset_index(drop=True)

    outcomes = pot_outcome_pairs['outcome'].unique()
    pots = np.unique(pot_outcome_pairs['pot'])

    all_x = list()
    all_p = list()
    temp_marginals = {'x': dict(), 'p': dict()}
    for i_outcome, outcome in enumerate(outcomes):
        temp_marginals['x'][outcome] = [None] * len(pots)
        temp_marginals['p'][outcome] = [None] * len(pots)
        for i_pot, pot in enumerate(pots):
            temp_marginal_values = fdf.loc[(fdf['outcome'] == outcome) & (fdf['pot'] == pot), feature]

            if transformation_fn is None:
                xvals_ = temp_marginal_values.to_numpy()
                pvals_ = np.full_like(xvals_, xvals_.shape[0]**-1)
            else:
                xvals_ = transformation_fn(temp_marginal_values.to_numpy())
                pvals_ = np.full_like(xvals_, xvals_.shape[0]**-1)
            all_x.extend(xvals_.tolist())
            all_p.extend(pvals_.tolist())
            temp_marginals['x'][outcome][i_pot] = xvals_
            temp_marginals['p'][outcome][i_pot] = pvals_

    return dict(temp_marginals=temp_marginals, all_x=all_x, all_p=all_p, i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=transformation_fn, transformation_label=transformation_label)


def plotHeatmapDistOverPots_pub(temp_marginals=None, all_x=None, all_p=None, i_feature=None, feature=None, outcomes=None, pots=None, transformation_fn=None, transformation_label=None, figoutpath=None, datarange=(0, 0), dynamicRange=True, color_transform=None, renormalize_color_within_outcome=False, dispZeros=True, plot_param=None):
    import numpy as np

    plt = plot_param['plt']
    sns = plot_param['sns']

    plt.close('all')
    figout = plt.figure(figsize=(8, 4 * 1))

    totalrange = (0, 0)
    if dynamicRange:
        observation_range = (np.min(all_x), np.max(all_x))
        datarange = (min(np.min(all_x), datarange[0]), max(np.max(all_x), datarange[1]))
        if datarange[0] >= 0.0 and datarange[0] < 0.001 and datarange[1] > 0.999 and datarange[1] <= 1.0:
            datarange = (0.0, 1.0)
        totalrange = (min(totalrange[0], datarange[0]), max(totalrange[1], datarange[1]))

    all_x_unique = np.unique(all_x)

    # nbins_ = 10
    # datarange = (-10, 10)
    # edges_ = np.linspace(datarange[0], datarange[1], nbins_ + 1, endpoint=True)

    nbins_ = 12
    edges_ = np.linspace(datarange[0], datarange[1], nbins_ + 1, endpoint=True)
    step_ = np.abs(edges_[1] - edges_[0])
    edges_up = list()
    edges_down = list()
    if edges_[-1] > 0.0:
        edges_up = np.arange(0.0 - step_ / 2, edges_[-1] + 3 * step_ / 2, step_)
    if edges_[0] < 0.0:
        edges_down = np.flip(np.arange(0.0 + step_ / 2, edges_[0] - 3 * step_ / 2, -1 * step_))
    edges = np.unique(np.hstack([edges_down, edges_up]))
    nbins = edges.size - 1

    # print("edges_")
    # print(edges_)
    # print("edges_up")
    # print(edges_up)
    # print("edges_down")
    # print(edges_down)
    # print(f"{feature}")
    # print(edges)
    hist_m = edges[1] - edges[0]
    hist_b = (-edges[0]) / hist_m

    axes = list()
    if color_transform is None:
        def color_transform(x): return x

    H_total, _ = np.histogram(all_x, weights=all_p, bins=edges, density=True)

    vmax_ = np.sort(H_total.flatten())[-2] if renormalize_color_within_outcome else None

    hists = dict()
    evs = dict()
    ranges = dict()
    for i_outcome, outcome in enumerate(outcomes):
        feature_hist = np.full((nbins, len(pots)), np.nan, dtype=float)
        feature_ev = np.full((len(pots),), np.nan, dtype=float)
        feature_range = np.full((len(pots), 2), np.nan, dtype=float)
        for i_pot, pot in enumerate(pots):
            feature_hist[:, i_pot], _ = np.histogram(temp_marginals['x'][outcome][i_pot], weights=temp_marginals['p'][outcome][i_pot], bins=edges, density=True)
            feature_ev[i_pot] = np.inner(temp_marginals['x'][outcome][i_pot], temp_marginals['p'][outcome][i_pot])
            feature_range[i_pot, 0] = np.min(temp_marginals['x'][outcome][i_pot])
            feature_range[i_pot, 1] = np.max(temp_marginals['x'][outcome][i_pot])
        if np.all(feature_range == 0):
            feature_hist = np.full_like(feature_hist, 0.0)
        assert not np.any(np.isnan(feature_hist)), f"{feature} - {outcome} ({transformation_label}) :: {np.sum(np.isnan(feature_hist))}, \n\nMEANS \n\n {feature_ev} \n\nRANGE\n\n {feature_range}"

        hists[outcome] = color_transform(feature_hist)
        evs[outcome] = feature_ev
        ranges[outcome] = feature_range

    axes = list()
    for i_outcome, outcome in enumerate(outcomes):
        axes.append(figout.add_subplot(2, 2, i_outcome + 1))

        # import matplotlib.colors as colors
        # cmap_ = colors.ListedColormap(['blue', 'white', 'black'])

        # x_unique = np.unique(hists[outcome])
        # boundaries = [0.0, x_unique[1] / 3, x_unique[1] / 2, np.max(hists[outcome])]
        # norm = colors.BoundaryNorm(boundaries, cmap_.N, clip=True)

        cmap_ = sns.color_palette("Greys", as_cmap=False, n_colors=100)
        # cmap_.set_bad("red")
        from matplotlib.colors import LogNorm, SymLogNorm
        aaa = np.extract(np.unique(hists[outcome]) > 0, np.unique(hists[outcome]))
        if aaa.size == 0:
            vmin_ = 0.000000001
        else:
            vmin_ = np.max([np.min(aaa), 0.000000001])
        vmax_ = np.max([hists[outcome].max().max(), 2 * vmin_])
        log_norm = LogNorm(vmin=vmin_, vmax=vmax_)

        # np.unique(hists[outcome])
        # log_norm = SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0.0, vmax=hists[outcome].max().max(), base=10)

        # print(f"OUTCOME {outcome}")
        # print(f"vmin {np.unique(hists[outcome])[1]}")
        # print(f"vmax {hists[outcome].max().max()}")
        # print(hists[outcome])

        axes[-1] = sns.heatmap(hists[outcome], xticklabels=False, yticklabels=True, cmap=cmap_, norm=log_norm, cbar=False, ax=axes[-1], linecolor='white', linewidths=0.5)

        axes[-1].set_facecolor("#CDE7F0")

        def linequ(x): return x / hist_m + hist_b

        axes[-1].scatter(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), s=30, marker='o', alpha=1, facecolors='none', edgecolors='white', linewidth=1, zorder=26)
        axes[-1].scatter(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), s=25, marker='o', alpha=1, facecolors='none', edgecolors='orange', linewidth=1, zorder=27)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), alpha=1, color='w', linewidth=1.2, zorder=25)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), alpha=1, color='orange', linewidth=1, zorder=31)
        # axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 0]), alpha=1, color='w', linewidth=1.2, zorder=22)
        # axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 0]), alpha=1, color='g', linewidth=0.8, zorder=24)
        # axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 1]), alpha=1, color='w', linewidth=1.2, zorder=21)
        # axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 1]), alpha=1, color='g', linewidth=0.8, zorder=23)

        if dispZeros:
            axes[-1].scatter(np.arange(len(pots)) + 0.5, np.full((1, len(pots)), linequ(0)), s=25, marker='.', alpha=1, facecolors='red')

        axes[-1].invert_yaxis()

        axes[-1].set_title(f'{outcome}', fontsize=18, color="black", y=1.1, pad=-8)

    for iax in [0, 2]:
        # axes[iax].set_yticks([-0.5, nbins + 1])
        # axes[iax].set_yticklabels([f"{datarange[0]:0.2f}", f"{datarange[1]:0.2f}"], rotation=0)
        # axes[-1].set_yticks([0,0.5,1,nbins,nbins+1])
        axes[iax].set_yticks([])

        axes[iax].tick_params(axis="y", direction="out", length=16, width=2, color="turquoise")
        # axes[-1].tick_params(axis="x", labelsize=18, labelrotation=-35, labelcolor="blue")
    for iax in [1, 3]:
        # axes[iax].set_yticks([-0.5, nbins + 1])
        # axes[iax].set_yticklabels([f"", f""])
        axes[iax].set_yticks([])

    if transformation_label is None:
        transformation_label = 'no_prospect_transform'

    fname_base = 'ColorNormed' if renormalize_color_within_outcome else "heatmapFeature"

    return (figoutpath / f"{transformation_label}-{fname_base}" / f'heatmapFeature_{transformation_label}_{fname_base}_{i_feature}-{feature}.pdf', figout)


def plotHeatmapDistOverPots(temp_marginals=None, all_x=None, all_p=None, i_feature=None, feature=None, outcomes=None, pots=None, transformation_fn=None, transformation_label=None, figoutpath=None, datarange=(0, 0), dynamicRange=True, color_transform=None, renormalize_color_within_outcome=False, dispZeros=True, plot_param=None):
    import numpy as np

    plt = plot_param['plt']
    sns = plot_param['sns']

    figsout = list()
    totalrange = (0, 0)

    plt.close('all')
    figout = plt.figure(figsize=(8, 4 * 1))

    if dynamicRange:
        observation_range = (np.min(all_x), np.max(all_x))
        datarange = (min(np.min(all_x), datarange[0]), max(np.max(all_x), datarange[1]))
        if datarange[0] >= 0.0 and datarange[0] < 0.001 and datarange[1] > 0.999 and datarange[1] <= 1.0:
            datarange = (0.0, 1.0)
        totalrange = (min(totalrange[0], datarange[0]), max(totalrange[1], datarange[1]))
        # print(f"--{feature}")
        # print(f"Observed range: {observation_range[0]:0.3f} -> {observation_range[1]:0.3f}")
        # print(f"Hist range: {datarange[0]:0.3f} -> {datarange[1]:0.3f}")

    sub_title_amend_str = ''
    # if edges[1] == edges[0] or hist_m == 0 or datarange[0] == datarange[1]:
    if observation_range[0] == observation_range[1]:
        print(f"---ERROR: NO DATA FOR :: {feature}")
        print(f"---{feature}")
        print(f"Observed range: {observation_range[0]:0.3f} -> {observation_range[1]:0.3f}")
        print(f"Hist range: {datarange[0]:0.3f} -> {datarange[1]:0.3f}")
        print(f"Range: {datarange[0] - datarange[1]}\n")
        datarange = (-1, 1)
        sub_title_amend_str = '*NO DATA'
    if datarange[0] == datarange[1]:  # only occures if observation_range is not (0,0)
        datarange = observation_range

    nbins = 12
    edges = np.linspace(datarange[0], datarange[1], nbins + 1, endpoint=True)
    hist_m = edges[1] - edges[0]
    hist_b = (-datarange[0]) / hist_m

    axes = list()
    if color_transform is None:
        def color_transform(x): return x

    H_total, _ = np.histogram(all_x, weights=all_p, bins=edges, density=True)
    # print('H_total')
    # print(H_total)

    # vmax_ = np.percentile(H_total.flatten(), 90)
    vmax_ = np.sort(H_total.flatten())[-2] if renormalize_color_within_outcome else None

    hists = dict()
    evs = dict()
    ranges = dict()
    title_amend = dict()
    for i_outcome, outcome in enumerate(outcomes):
        title_str_ = ''
        feature_hist = np.full((nbins, len(pots)), np.nan, dtype=float)
        feature_ev = np.full((len(pots),), np.nan, dtype=float)
        feature_range = np.full((len(pots), 2), np.nan, dtype=float)
        for i_pot, pot in enumerate(pots):
            feature_hist[:, i_pot], _ = np.histogram(temp_marginals['x'][outcome][i_pot], weights=temp_marginals['p'][outcome][i_pot], bins=edges, density=True)
            feature_ev[i_pot] = np.inner(temp_marginals['x'][outcome][i_pot], temp_marginals['p'][outcome][i_pot])
            feature_range[i_pot, 0] = np.min(temp_marginals['x'][outcome][i_pot])
            feature_range[i_pot, 1] = np.max(temp_marginals['x'][outcome][i_pot])
        if np.all(feature_range == 0):
            feature_hist = np.full_like(feature_hist, 0.0)
            print(f"WARNING {feature} : {outcome} ({transformation_label}) -- THIS FEATURE HAS NO VALUES")
            title_str_ = '*no values'
        assert not np.any(np.isnan(feature_hist)), f"{feature} - {outcome} ({transformation_label}) :: {np.sum(np.isnan(feature_hist))}, \n\nMEANS \n\n {feature_ev} \n\nRANGE\n\n {feature_range}"

        hists[outcome] = color_transform(feature_hist)
        evs[outcome] = feature_ev
        ranges[outcome] = feature_range
        title_amend[outcome] = title_str_

    hhh = np.stack(list(hists.values()))  # (4, 12, 24) shape

    max_hval_by_bin = np.max(hhh, axis=(0, 2))
    # print(max_hval_by_bin)

    for i_outcome, outcome in enumerate(outcomes):
        axes.append(figout.add_subplot(2, 2, i_outcome + 1))

        axes[-1] = sns.heatmap(hists[outcome], xticklabels=False, yticklabels=True, cmap="coolwarm", cbar=False, ax=axes[-1], vmin=0.0, vmax=vmax_)  # vmin=None, vmax=None, cmap=None ### vmin=0,

        def linequ(x): return x / hist_m + hist_b

        axes[-1].scatter(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), s=30, marker='o', alpha=1, facecolors='none', edgecolors='white', linewidth=1, zorder=26)
        axes[-1].scatter(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), s=25, marker='o', alpha=1, facecolors='none', edgecolors='orange', linewidth=1, zorder=27)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), alpha=1, color='w', linewidth=1.2, zorder=25)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), alpha=1, color='k', linewidth=1, zorder=31)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 0]), alpha=1, color='w', linewidth=1.2, zorder=22)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 0]), alpha=1, color='g', linewidth=0.8, zorder=24)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 1]), alpha=1, color='w', linewidth=1.2, zorder=21)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(ranges[outcome][:, 1]), alpha=1, color='g', linewidth=0.8, zorder=23)

        if dispZeros:
            axes[-1].scatter(np.arange(len(pots)) + 0.5, np.full((1, len(pots)), linequ(0)), s=25, marker='.', alpha=1, facecolors='black')

        axes[-1].invert_yaxis()
        # # axes.set_title('{}'.format(outcome))

        # axes[-1].plot([0,1,5,10],[0,1,10,5])
        if renormalize_color_within_outcome:
            axes[-1].set_title(f'{outcome}{title_amend[outcome]}({np.max(hists[outcome]):0.2f}/{vmax_:0.2f}={np.max(hists[outcome])/vmax_:0.2f})')
        else:
            axes[-1].set_title(f'{outcome}{title_amend[outcome]} ({np.max(hists[outcome]):0.2f})')

    for iax in [0, 2]:
        axes[iax].set_yticks([-0.5, nbins])
        axes[iax].set_yticklabels([f"{datarange[0]:0.2f}", f"{datarange[1]:0.2f}"], rotation=0)
        # axes[-1].set_yticks([0,0.5,1,nbins,nbins+1])

        axes[iax].tick_params(axis="y", direction="out", length=16, width=2, color="turquoise")
        # axes[-1].tick_params(axis="x", labelsize=18, labelrotation=-35, labelcolor="blue")
    for iax in [1, 3]:
        axes[iax].set_yticks([-0.5, nbins])
        axes[iax].set_yticklabels([f"", f""])

    if transformation_label is None:
        transformation_label = 'no_prospect_transform'

    if transformation_fn is None:
        transformation_applied = 'NO prospect transform'
    else:
        transformation_applied = transformation_label

    fname_base = 'ColorNormed' if renormalize_color_within_outcome else "heatmapFeature"

    plt.suptitle(f"{feature}{sub_title_amend_str}\n{transformation_applied}\npseudo-$\sigma^2={np.var(all_x):3.0}$, $\mu={np.inner(all_x, all_p):3.0}$; {observation_range[0]:3.0f}->{observation_range[1]:3.0f}", y=1.23)
    # plt.show()

    return (figoutpath / f"{transformation_label}-{fname_base}" / f'heatmapFeature_{transformation_label}_{fname_base}_{i_feature}-{feature}.pdf', figout)


def plotModelComparisonScatter_old(axes1, X, Y, labels, colors):  # DEBUG
    import numpy as np

    x = X.flatten()
    y = Y.flatten()

    sqrerr = np.power(np.subtract(x, y), 2)
    SSres = np.sum(sqrerr)
    SStot = np.sum(np.power(np.subtract(y, np.mean(y)), 2))
    rmse = np.sqrt(np.mean(sqrerr))
    r2 = 1 - SSres / SStot

    axes1.scatter(x, y, c=colors / 255.0)
    axes1.plot([0, 1], [0, 1], '--', color='black', alpha=1, lw=1)
    axes1.set_ylim((0, 1))
    axes1.set_xlim((0, 1))
    # axes1.axis('equal')
    if labels is not None:
        axes1.text(0.01, .96 - 0.05 * 1, '{}={:0.3f},  RMSE={:0.3f}'.format(r'R$^2$', r2, rmse), color='k', size=11)
        axes1.set_xlabel(labels['x'])
        axes1.set_ylabel(labels['y'])
        axes1.set_title(**labels['title'])
    else:
        axes1.set_xlabel('')
        axes1.set_ylabel('')
        axes1.set_title('')

    return axes1


def plotModelComparisonScatter(ax, data=None, model=None, labels={'data': '', 'model': '', 'title': {'label': ''}}, colors=None):
    '''
    A data set has n values marked y_1,...,y_n (collectively known as yi or as a vector y = [y1,...,yn]T), each associated with a predicted (or modeled) value f1,...,fn (known as fi, or sometimes ŷi, as a vector f).
    Define the residuals as ei = yi − fi (forming a vector e)
    '''
    import numpy as np
    import pandas as pd
    import scipy.stats
    from webpypl import modelStats_

    np.testing.assert_equal(data.shape, model.shape)

    if isinstance(data, pd.DataFrame) or isinstance(model, pd.Series):
        y = data.values.flatten()
        f = model.values.flatten()
    else:
        y = data.flatten()
        f = model.flatten()

    if colors is None:
        colors = np.full((1, 4), np.array([0, 0, 0, 255 / 3]) / 255)
        aval = None
    else:
        aval = 1 / (np.max([len(y) / 100, 3]))
    # elif isinstance(colors, str):
    #     colors = colors
    # else:
    #     if isinstance(colors, pd.DataFrame) or isinstance(colors, pd.Series):
    #         colors = colors.values.flatten()
    #     else:
    #         colors = colors.flatten()

    model_stats = modelStats_(data=data, model_predictions=model)

    ax.scatter(y, f, c=colors, alpha=aval)
    ax.plot([0, 1], [0, 1], '--', color='black', alpha=1, lw=1)
    ax.axis('equal')
    if labels is not None:
        ax.text(0.01, .96 - 0.05 * 1, '{}={:0.3f},  RMSE={:0.3f}, r={:0.3f}'.format(r'R$^2$', model_stats['R2'], model_stats['RMSE'], model_stats['pearson_r']), color='k', size=11)
        ax.set(xlabel=labels['data'], ylabel=labels['model'])
        ax.set_title(**labels['title'])
    else:
        ax.set(xlabel='', ylabel='', title='')
    ax.set(ylim=(0, 1), xlim=(0, 1))

    return ax, model_stats


'''
def plotMarginalHeatmapContinuous(df, title):
    # import numpy as np
    
    # takes dataframe with two index columns of support observations and one column of probabilities (the values)
    fig = plt.figure(figsize=(4, 4))
    axes = fig.add_subplot(1,1,1)

    xs = np.array(df.index.tolist()).T

    histo = plt.hist2d(xs[0], xs[1], bins=20, range=None, normed=False, weights=df, cmin=None, cmax=None, hold=None, data=None)

    axes.set_xlabel(df.index.names[0])
    axes.set_ylabel(df.index.names[1])

    axes.set_title(title)


def plotDecision(data, labels, display, title):

    bp = sns.barplot(data.index, data.prob, order=labels, palette=display['colors']['decision'])
    # bp = sns.barplot("support", "prob", data=ppldata['level0'].reset_index(), order=ppldata['labels']['decisions'], palette=clrs)

    for rect in bp.patches:
        bp.annotate("%.2f" % rect.get_height(), (rect.get_x() + rect.get_width() / 2., rect.get_height()),
                     ha='center', va='center', color='0.2', rotation=0, xytext=(0, 10),
                     textcoords='offset points')

    # for rect in bp.patches:
    #     bp.annotate("%.2f" % rect.get_height(), (rect.get_x() + rect.get_width() / 2., rect.get_height()), ha='center', va='center', fontsize=14, color='gray',
    #     rotation=0, xytext=(0, 10), textcoords='offset points')

    # for rect in bp.patches:
    #     bp.text(rect.get_x() + rect.get_width()/2, rect.get_height(), "%.2f" % rect.get_height(), ha='center', va='bottom')

    # for rect, val in zip(bp.patches,y):
    #     bp.text(rect.get_x() + rect.get_width()/2, val, "%.2f" % val, ha='center', va='bottom')

    bp.set_ylim(0, 1)
    bp.set(xlabel='decision', ylabel='frequency')
    bp.set_title(**title, y=1.08)
    return plt
'''


def makeColorbarHorizontal(axis):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    from mpl_toolkits.axes_grid1.colorbar import colorbar
    # split axes of heatmap to put colorbar
    ax_divider = make_axes_locatable(axis)
    # define size and padding of axes for colorbar
    cax = ax_divider.append_axes('bottom', size='4%', pad='2%')
    # make colorbar for heatmap.
    # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
    colorbar(axis.get_children()[0], cax=cax, orientation='horizontal', ticks=[-1, 0, 1])
    # locate colorbar ticks
    cax.xaxis.set_ticks_position('bottom')

# def colorbar2(mappable):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("bottom", size="5%", pad=0.05)
#     return fig.colorbar(mappable, cax=cax)


def fixed_size_axes(fig, axessize):
    # https://matplotlib.org/gallery/axes_grid1/demo_fixed_size_axes.html
    from mpl_toolkits.axes_grid1 import Divider, Size
    from mpl_toolkits.axes_grid1.mpl_axes import Axes

    # fig = plt.figure(1, (6, 6))

    # The first items are for padding and the second items are for the axes.
    # sizes are in inch.

    h = [Size.Fixed(1.), Size.Fixed(axessize[0])]
    v = [Size.Fixed(5.), Size.Fixed(axessize[1])]

    divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
    # the width and height of the rectangle is ignored.

    ax = Axes(fig, divider.get_position())
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    fig.add_axes(ax)

    return ax


def plotDM_handler(dm, ax, sns):
    import numpy as np
    from webpypl_analysis_rsa import plotDM
    from webpypl_plotfun import makeColorbarHorizontal

    assert (len(np.unique(dm.shape)) == 1)

    mask_allemotions_tril1 = np.zeros((dm.shape[0], dm.shape[0]), dtype=bool)
    mask_allemotions_tril1[np.triu_indices_from(mask_allemotions_tril1, k=1)] = True

    ax = plotDM(ax, sns, dm, mask=mask_allemotions_tril1, vmin=-1, vmax=1, vcenter=0, cmap='YlGnBu')
    # ax = plotDM(ax, sns, dm, mask=mask_allemotions_tril1, vmin=-1, vmax=1, vcenter=0, cmap=sns.diverging_palette(220, 10, as_cmap=True))
    makeColorbarHorizontal(ax)


def plotDM_rescaleModel_handler(dm, ax, sns):
    import numpy as np
    from webpypl import FTrz
    from webpypl_analysis_rsa import plotDM
    from webpypl_plotfun import makeColorbarHorizontal

    assert (len(np.unique(dm.shape)) == 1)

    dmz = dm.copy()

    mask = np.ones(dmz.shape, dtype=bool)
    np.fill_diagonal(mask, False)

    dmz_values = np.full_like(dmz.values, dmz.values[0, 0])

    dmz_values[mask] = FTrz(dmz.values[mask])

    dmz.loc[:, :] = dmz_values

    # sns.heatmap(dmz, vmin=dmz.min().min(), vmax=dmz.max().max(), center=0., cmap=sns.diverging_palette(220, 10, as_cmap=True))

    mask_allemotions_tril1 = np.zeros((dm.shape[0], dm.shape[0]), dtype=bool)
    mask_allemotions_tril1[np.triu_indices_from(mask_allemotions_tril1, k=1)] = True

    ax = plotDM(ax, sns, dmz, mask=mask_allemotions_tril1, vmin=dmz.min().min(), vmax=dmz.max().max(), vcenter=0, cmap=sns.diverging_palette(220, 10, as_cmap=True))
    makeColorbarHorizontal(ax)


def plotGroupedBars(datain, colorin, major_labels, ax, width=0.35, space=0.35):
    import numpy as np

    n_major = datain.shape[0]
    n_minor = datain.shape[1]
    xloc = np.zeros((n_major, n_minor))
    iloc = 0
    for i_major in range(n_major):
        for i_minor in range(n_minor):
            xloc[i_major, i_minor] = iloc
            iloc += width
        iloc += width

    major_ticks = xloc[:, 0] + (xloc[:, -1] - xloc[:, 0]) / 2

    for i_x in range(n_major):
        for i_y in range(n_minor):
            ax.bar(xloc[i_x, i_y], datain[i_x, i_y], width, color=colorin[i_x, i_y])
    ax.set(xticks=major_ticks, xticklabels=major_labels)


def plotGroupedBarsHorizontal(datain, colorin, major_labels, ax, width=0.4, space=0.35):
    import numpy as np

    n_major = datain.shape[0]
    n_minor = datain.shape[1]
    xloc = np.zeros((n_major, n_minor))
    iloc = 0
    for i_major in range(n_major):
        for i_minor in range(n_minor):
            xloc[i_major, i_minor] = iloc
            iloc += width
        iloc += space

    major_ticks = xloc[:, 0] + (xloc[:, -1] - xloc[:, 0]) / 2

    for i_x in range(n_major):
        for i_y in range(n_minor):  # range(n_minor-1,-1,-1):
            ax.barh(xloc[i_x, (n_minor - 1) - i_y], datain[i_x, i_y], width, color=colorin[i_x, i_y], linewidth=0)
    ax.set(yticks=major_ticks, yticklabels=major_labels)
    ax.invert_yaxis()
    ax.set_ylim(-space, iloc - width)


def plot_emo_comparison_scatter(baseline_ev, new_ev, emolabels, outcomes, display, ax, scale=2):

    # plotrange = [-.75, len(emolabels)-.25]
    ### these should be 2d arrays of EV, E.G.
    assert len(baseline_ev.shape) == 2
    assert len(new_ev.shape) == 2
    assert new_ev.shape[0] == len(emolabels)
    assert new_ev.shape[1] == len(outcomes)
    for i_feature, feature in enumerate(emolabels):
        for i_outcome, outcome in enumerate(outcomes):
            # if i_feature == 0:
            #     labeltitle = outcome
            # else:
            #     labeltitle = None
            color = display['colors']['outcomedict'][outcome]
            # ax.arrow(i_feature+[-1.5, -.5, .5, 1.5][i_outcome]/10.0, baseline_ev[i_emotion,i_outcome,:].mean(), 1, 1) #delta_ev[i_feature,i_outcome,:].mean()
            # ax.arrow(i_feature+[-1.5, -.5, .5, 1.5][i_outcome]/10.0, expectedValues[i_feature,i_outcome], color, alpha=.7, markersize=30*scale/2,label=labeltitle)
            # ax.arrow(i_feature,i_outcome,1,1,color=color)
            # ax.annotate("", xy=(i_feature,i_outcome), xytext=(1,1), arrowprops=dict(arrowstyle="->"), color='yellow')
            # ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
            # ax.quiver(i_feature+[-1.5, -.5, .5, 1.5][i_outcome]/10.0, baseline_ev[i_emotion,i_outcome,:].mean(), 0, delta_ev[i_feature,i_outcome,:].mean(), color=color)

            # ax.arrow(i_feature+[-1.5, -.5, .5, 1.5][i_outcome]/10.0, baseline_ev[i_emotion,i_outcome,:].mean(), 0, delta_ev[i_feature,i_outcome,:].mean(), color=color)
            # ax.plot(i_feature+[-1.5, -.5, .5, 1.5][i_outcome]/10.0, baseline_ev[i_emotion,i_outcome,:].mean(), color=color)
            x_coord = i_feature + [-1.5, -.5, .5, 1.5][i_outcome] / 10.0

            generic_intensity = baseline_ev[i_feature, i_outcome]
            distial_prior_intensity = new_ev[i_feature, i_outcome]

            ### delta line
            ax.plot([x_coord, x_coord], [generic_intensity, distial_prior_intensity], lw=2, color=color, alpha=0.5, zorder=1)  # on bottom

            ### white circles
            ax.scatter(x_coord, generic_intensity, 80, color='white', marker='o', alpha=1, zorder=2)  # mask gridlines and difference line

            ### white squares
            ax.scatter(x_coord, distial_prior_intensity, 150, color='white', marker='o', alpha=1, zorder=3, edgecolor='none')  # mask delta lines

            ### open circles (baseline)
            ax.scatter(x_coord, generic_intensity, 80, color=color, marker='o', alpha=.6, facecolor='white', zorder=4)

            ### colored squares
            ax.scatter(x_coord, distial_prior_intensity, 150, color=color, marker='o', alpha=.6, zorder=10, edgecolor='none')  # on top

    ax.set_xlim((-0.75, 19.75))
    ax.set_ylim((0, 1))
    ax.set_ylabel('Intensity', fontsize=11 * scale)
    ax.tick_params(axis='both', labelsize=11 * scale)
    # ax.tick_params(axis='x',rotation=-30)
    ax.set_xticks(range(len(emolabels)))
    ax.set_xticklabels(emolabels, fontdict={'fontsize': 11 * scale, 'horizontalalignment': 'left'}, rotation=-35, rotation_mode='anchor')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    # plt.xticks(range(len(emolabels)), np.array(list(range(len(emolabels))))+1, ha='left')

    return ax
