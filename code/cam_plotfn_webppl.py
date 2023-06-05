#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_plotfn_webppl.py
"""


def plotHeatmapDistOverPots_prep_emoDict_(emoDictIAF, i_feature=None, feature=None, outcomes=None, pots=None, transformation_fn=None, transformation_label=None):
    from cam_webppl_utils import marginalizeContinuous

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


def plotHeatmapDistOverPots_pub(temp_marginals=None, all_x=None, all_p=None, i_feature=None, feature=None, outcomes=None, pots=None, transformation_fn=None, transformation_label=None, figoutpath=None, datarange=(0, 0), dynamicRange=True, color_transform=None, renormalize_color_within_outcome=False, dispZeros=True, linewidth=0.5, plot_param=None):
    import numpy as np
    from matplotlib.colors import LogNorm, SymLogNorm

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

        cmap_ = sns.color_palette("Greys", as_cmap=False, n_colors=100)

        vals_ = np.extract(np.unique(hists[outcome]) > 0, np.unique(hists[outcome]))
        if vals_.size == 0:
            vmin_ = 0.000000001
        else:
            vmin_ = np.max([np.min(vals_), 0.000000001])
        vmax_ = np.max([hists[outcome].max().max(), 2 * vmin_])
        log_norm = LogNorm(vmin=vmin_, vmax=vmax_)

        axes[-1] = sns.heatmap(hists[outcome], xticklabels=False, yticklabels=True, cmap=cmap_, norm=log_norm, cbar=False, ax=axes[-1], linecolor='white', linewidths=linewidth)

        axes[-1].set_facecolor("#CDE7F0")

        def linequ(x): return x / hist_m + hist_b

        axes[-1].scatter(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), s=30, marker='o', alpha=1, facecolors='none', edgecolors='white', linewidth=1, zorder=26)
        axes[-1].scatter(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), s=25, marker='o', alpha=1, facecolors='none', edgecolors='orange', linewidth=1, zorder=27)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), alpha=1, color='w', linewidth=1.2, zorder=25)
        axes[-1].plot(np.arange(len(pots)) + 0.5, linequ(evs[outcome]), alpha=1, color='orange', linewidth=1, zorder=31)

        if dispZeros:
            axes[-1].scatter(np.arange(len(pots)) + 0.5, np.full((1, len(pots)), linequ(0)), s=25, marker='.', alpha=1, facecolors='red')

        axes[-1].invert_yaxis()

        axes[-1].set_title(f'{outcome}', fontsize=18, color="black", y=1.1, pad=-8)

    for iax in [0, 2]:
        axes[iax].set_yticks([])
        axes[iax].tick_params(axis="y", direction="out", length=16, width=2, color="turquoise")
    for iax in [1, 3]:
        axes[iax].set_yticks([])

    if transformation_label is None:
        transformation_label = 'no_prospect_transform'

    fname_base = 'ColorNormed' if renormalize_color_within_outcome else "heatmapFeature"

    return (figoutpath / f"{transformation_label}-{fname_base}" / f'heatmapFeature_{transformation_label}_{fname_base}_{i_feature}-{feature}.pdf', figout)


def plot_inverse_planning_kde_base_split_HIST_(plotParam, inv_planning_df_, inv_planning_baseline_df_=None, feature_list=None, width=0.7, ax_=None):
    import numpy as np
    from matplotlib.collections import PolyCollection

    plt = plotParam['plt']
    sns = plotParam['sns']

    if not feature_list:
        feature_list = np.unique(inv_planning_df_['feature'].values)
    inv_planning_df = inv_planning_df_.astype({'weight': 'float'})

    df_ = inv_planning_df.loc[inv_planning_df['feature'].isin([feature for feature in feature_list if feature not in 'pi_a2']), :]
    vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=df_, ax=ax_,
                        order=feature_list,
                        hue_order=['C', 'D'],
                        split=True,
                        palette=dict(C='cornflowerblue', D='dimgrey'),
                        width=width,
                        bw='scott', cut=0, scale='area',
                        linewidth=0.0,
                        saturation=1,
                        inner=None)
    ax_.set_ylim([-0.0, 1.0])

    for i_feature, feature in enumerate(feature_list):
        for a1 in ['C', 'D']:
            xcoord = i_feature + {'C': -width / 4, 'D': width / 4}[a1]
            yval = np.mean(inv_planning_df.loc[(inv_planning_df['feature'] == feature) & (inv_planning_df['a_1'] == a1), 'weight'])
            color_ = dict(C='cornflowerblue', D='dimgrey')[a1]
            ax_.plot([xcoord - width / 6, xcoord + width / 6], [yval, yval], color=color_, linewidth=2)

    ######
    ## plot histogram
    ######
    plot_histogram = False
    if plot_histogram:
        for i_feature, feature in enumerate(feature_list):
            for a1 in ['C', 'D']:
                yvals = inv_planning_df.loc[(inv_planning_df['feature'] == feature) & (inv_planning_df['a_1'] == a1), 'weight']
                bins_ = np.arange(0, 1 + 2 / 48, 1 / 48) - 1 / (48 * 2)

                hist_unnormed, be_ = np.histogram(yvals, bins=bins_, density=False)
                hist = hist_unnormed / np.max(hist_unnormed)

                xcoord = i_feature + {'C': -width / 4, 'D': width / 4}[a1]

                x_base = i_feature
                for i_binval, binval in enumerate(np.arange(0, 1 + 1 / 48, 1 / 48)):
                    xdelta = (width / 2) * hist[i_binval]
                    if a1 == 'C':
                        ax_.plot([i_feature, i_feature - xdelta], [binval, binval], linewidth=2, color='cornflowerblue', alpha=1)
                    else:
                        ax_.plot([i_feature, i_feature + xdelta], [binval, binval], linewidth=2, color='dimgrey', alpha=1)

    #####

    if 'pi_a2' in feature_list:
        kwargs = dict()
        kwargs.update(transform=ax_.transAxes, clip_on=False)
        xcoord_ = feature_list.index('pi_a2')

        for a1 in ['C', 'D']:
            xvals, ycounts = np.unique(inv_planning_df.loc[(inv_planning_df['a_1'] == a1) & (inv_planning_df['feature'] == 'pi_a2'), 'weight'], return_counts=True)
            ycounts_total = ycounts.sum()

            radii = (ycounts / ycounts_total)

            xcoord = xcoord_ + {'C': -width / 4, 'D': width / 4}[a1]
            for ixval, xval in enumerate(xvals):
                marker_style = dict(color='none', linestyle=':', marker='o', markersize=50 * radii[ixval], markerfacecoloralt={'C': 'cornflowerblue', 'D': 'dimgrey'}[a1], alpha=0.5)
                ax_.plot(xcoord_, xval, fillstyle={'C': 'right', 'D': 'left'}[a1], **marker_style)

    for art in ax_.get_children():
        if isinstance(art, PolyCollection):
            art.set_alpha(0.3)
            art.set_zorder(3)

    ax_.set_xlim([-0.5, len(feature_list) - 0.5])

    fakegrid_kwrgs = {'color': 'lightgrey', 'linewidth': 1.5, 'alpha': 1}
    for xcord in np.arange(1.5, ax_.get_xlim()[1], 2):
        ax_.plot([xcord, xcord], [-0.14, 1], linestyle='-', clip_on=False, **fakegrid_kwrgs)

    fakegrid_kwrgs = {'color': 'lightgrey', 'linewidth': 1.5, 'alpha': 1}
    for xcord in np.arange(0.5, ax_.get_xlim()[1], 2):
        ax_.plot([xcord, xcord], [-0.07, 1], linestyle='-', clip_on=False, **fakegrid_kwrgs)

    ax_.get_legend().remove()

    ax_.set_xlabel('')

    ax_.set_ylabel('Weight')

    return ax_


def plot_inverse_planning_kde_base_split(plotParam, inv_planning_df_, inv_planning_baseline_df_=None, feature_list=None, width=0.7, ax_=None):
    import numpy as np
    from matplotlib.collections import PolyCollection

    plt = plotParam['plt']
    sns = plotParam['sns']

    if not feature_list:
        feature_list = np.unique(inv_planning_df_['feature'].values)
    inv_planning_df = inv_planning_df_.astype({'weight': 'float'})

    df_ = inv_planning_df.loc[inv_planning_df['feature'].isin([feature for feature in feature_list if feature not in 'pi_a2']), :]

    vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=df_, ax=ax_,
                        order=feature_list,
                        hue_order=['C', 'D'],
                        split=True,
                        palette=dict(C='cornflowerblue', D='dimgrey'),
                        width=width,
                        bw='scott', cut=0, scale='area',
                        linewidth=0.0,
                        saturation=1,
                        inner=None)
    ax_.set_ylim([0, 1])

    for i_feature, feature in enumerate(feature_list):
        for a1 in ['C', 'D']:
            xcoord = i_feature + {'C': -width / 4, 'D': width / 4}[a1]
            yval = inv_planning_df.loc[(inv_planning_df['feature'] == feature) & (inv_planning_df['a_1'] == a1), 'weight'].mean()
            ax_.scatter([xcoord], [yval], s=350, marker='_', color=dict(C='cornflowerblue', D='dimgrey')[a1], linewidth=1.5)

    if 'pi_a2' in feature_list:
        from matplotlib.patches import Wedge
        kwargs = dict()
        kwargs.update(transform=ax_.transAxes, clip_on=False)
        xcoord_ = feature_list.index('pi_a2')

        for a1 in ['C', 'D']:
            xvals, ycounts = np.unique(inv_planning_df.loc[(inv_planning_df['a_1'] == a1) & (inv_planning_df['feature'] == 'pi_a2'), 'weight'], return_counts=True)
            ycounts_total = ycounts.sum()

            radii = (ycounts / ycounts_total)

            xcoord = xcoord_ + {'C': -width / 4, 'D': width / 4}[a1]
            for ixval, xval in enumerate(xvals):
                marker_style = dict(color='none', linestyle=':', marker='o', markersize=50 * radii[ixval], markerfacecoloralt={'C': 'cornflowerblue', 'D': 'dimgrey'}[a1], alpha=0.5)
                ax_.plot(xcoord_, xval, fillstyle={'C': 'right', 'D': 'left'}[a1], **marker_style)

    for art in ax_.get_children():
        if isinstance(art, PolyCollection):
            art.set_alpha(0.3)
            art.set_zorder(3)

    ax_.set_xlim([-0.5, len(feature_list) - 0.5])

    fakegrid_kwrgs = {'color': 'lightgrey', 'linewidth': 1.5, 'alpha': 1}

    for xcord in np.arange(0.5, ax_.get_xlim()[1], 1):
        ax_.plot([xcord, xcord], [-0.1, 1], linestyle='-', clip_on=False, **fakegrid_kwrgs)

    ax_.get_legend().remove()

    ax_.set_xlabel('')

    ax_.set_ylabel('Weight')

    return ax_


def composite_distal_fig(df_long_player_empir, df_long_player_model, stimid, paths, plot_param, **kwargs):

    import numpy as np
    import matplotlib.cbook as cbook

    #################################
    ### individual diffs summary
    #################################

    plt = plot_param['plt']

    plt.close('all')

    ### widths
    margin = 1
    graphic_width_photo = 2
    graphic_width_ip = 5
    graphic_width_scatter = 0.5
    graphic_width_emo = 0.5
    h_space = 1
    h_space_l = 0
    h_space_r = 0

    ### heights
    margin_top = 1
    header_height = 0.1
    legend_height = 0.25
    graphic_height = 1
    static_graphic = 2.5
    v_space = 0.5
    v_space_final = v_space + 0.15

    no_debug = True
    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    gs_col_widths = {
        'ml': margin,
        's0': h_space_l,
        'pic': graphic_width_photo,
        's1': h_space,
        'ip': graphic_width_ip,
        's2': h_space,
        'emoscatter': graphic_width_scatter,
        's3': h_space,
        'ia': graphic_width_emo,
        's4': h_space_r,
        'mr': margin,
    }

    gs_row_heights = {
        'mt': margin_top,
        'headers': header_height,
        'legends': legend_height,
        'g1': graphic_height,
        's1': v_space,
        'g2': graphic_height,
        's2': v_space,
        'mb': static_graphic,
    }

    def cm2inch(value):
        return value / 2.54
    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    gs_col = dict()
    for i_key, key in enumerate(gs_col_widths):
        gs_col[key] = i_key
    gs_row = dict()
    for i_key, key in enumerate(gs_row_heights):
        gs_row[key] = i_key

    width_temps = np.array(list(gs_col_widths.values()))
    widths = width_max * (width_temps / width_temps.sum())
    heights = np.array(list(gs_row_heights.values()))

    total_width = np.sum(widths) * 2.25
    total_height = np.sum(heights) * 2.25

    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    text_size_large = 12

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    ###################

    axd = dict()

    ########## Title

    axid = 'desc'
    axd[axid] = fig.add_subplot(gridspec[gs_row[f'headers'], gs_col['pic']:gs_col['ia'] + 1])
    axd[axid].text(0.5, 0.5, kwargs['title'])
    axd[axid].axis('off')

    ########## Photo

    axid = 'photo'
    axd[axid] = fig.add_subplot(gridspec[gs_row[f'g1'], gs_col['pic']])

    with cbook.get_sample_data(paths['stimuli'] / f'{stimid}.png') as image_file:
        image = plt.imread(image_file)

    im = axd[axid].imshow(image)

    axd[axid].axis('off')

    ######### Inv Planning

    axid = 'invp_empir'
    axd[axid] = fig.add_subplot(gridspec[gs_row[f'g1'], gs_col['ip']])
    axd[axid] = plot_inverse_planning_kde_base_split_HIST_(plot_param, df_long_player_empir, feature_list=['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA', 'pi_a2'], width=0.9, ax_=axd[axid])

    axd[axid].set_title(f"Empirical ({df_long_player_empir.shape[0]//7})")
    axd[axid].set_ylabel(axd[axid].get_ylabel(), fontdict={'fontsize': text_size_large})
    axd[axid].set_yticks([0, 0.5, 1])
    axd[axid].tick_params(axis="y", labelsize=text_size_large, pad=0)
    axd[axid].tick_params(axis="x", labelsize=text_size_large, pad=0)

    axd[axid].set_xticks([0, 0.5, 1, 2, 2.5, 3, 4, 4.5, 5, 5.99, 6])
    text_base = r'$\mathrm{base}$'
    text_repu = r'$\mathrm{repu}$'
    text_belief = r'$a_2=\mathrm{C}$'
    va_offset = -0.08
    axd[axid].set_xticklabels((text_base, '$Money$', text_repu, text_base, '$AIA$', text_repu, text_base, '$DIA$', text_repu, 'Belief', text_belief))

    va = [0, va_offset, 0, 0, va_offset, 0, 0, va_offset, 0, 0, va_offset]
    for t, y in zip(axd[axid].get_xticklabels(), va):
        t.set_y(y)

    ######### Inv Planning

    axid = 'invp_model'
    axd[axid] = fig.add_subplot(gridspec[gs_row[f'g2'], gs_col['ip']])
    axd[axid] = plot_inverse_planning_kde_base_split_HIST_(plot_param, df_long_player_model, feature_list=['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA', 'pi_a2'], width=0.9, ax_=axd[axid])

    axd[axid].set_title(f"Model ({df_long_player_model.shape[0]//7})")
    axd[axid].set_ylabel(axd[axid].get_ylabel(), fontdict={'fontsize': text_size_large})
    axd[axid].set_yticks([0, 0.5, 1])
    axd[axid].tick_params(axis="y", labelsize=text_size_large, pad=0)
    axd[axid].tick_params(axis="x", labelsize=text_size_large, pad=0)

    axd[axid].set_xticks([0, 0.5, 1, 2, 2.5, 3, 4, 4.5, 5, 5.99, 6])
    text_base = r'$base$'
    text_repu = r'$repu$'
    text_belief = r'$a_2=\mathrm{C}$'
    va_offset = -0.08
    axd[axid].set_xticklabels((text_base, 'Money', text_repu, text_base, 'AIA', text_repu, text_base, 'DIA', text_repu, 'Belief', text_belief))

    va = [0, va_offset, 0, 0, va_offset, 0, 0, va_offset, 0, 0, va_offset]
    for t, y in zip(axd[axid].get_xticklabels(), va):
        t.set_y(y)

    figsout_path = paths['figsOut'] / 'composite_inv_plan-specificPlayers'
    figsout_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figsout_path / f"invplan_{stimid}.pdf", bbox_inches='tight', pad_inches=0)


def expand_df_(df, nobsdf):
    import numpy as np
    import pandas as pd

    multilevel = isinstance(df.columns.to_list()[0], tuple)

    dfs_new_ = list()
    for pot in nobsdf.index:

        df_ = df.loc[df.index.get_level_values(0) == pot, :].copy().reset_index(drop=True)

        probs_in = ""
        if 'prob' in df_.columns.to_list() or ('prob', 'prob') in df_.columns.to_list():
            probs_ = df_['prob'].to_numpy()

            lcd_mult = 0
            lcd_found = False
            while not lcd_found and lcd_mult < 1000:
                lcd_mult += 1

                n_ = np.around(lcd_mult * probs_ / np.min(probs_)).astype(int)
                lcd = int(round(np.sum(n_)))
                lcd_found = np.allclose(lcd_mult * probs_ / np.min(probs_), n_)

            assert np.isclose(np.sum(probs_), 1.0)
            assert np.isclose(np.sum(n_), np.sum(lcd_mult * probs_ / np.min(probs_)))
        else:
            n_ = np.full([df_.shape[0], 1], 1, dtype=int)
            lcd = df_.shape[0]
            assert np.sum(n_) == df_.shape[0]

        expanded_dfs = list()
        if 'prob' in df_.columns.to_list():
            df_.drop(columns=['prob'], inplace=True)

        if lcd > 0:
            counter_ = 0

            for mult in np.arange(1, int(round(np.max(n_))) + 1):

                if np.sum(n_ == mult) > 0:
                    for _ in np.arange(1, mult + 1):
                        expanded_dfs.append(df_.loc[n_ == mult, :])
                        counter_ += df_.loc[n_ == mult, :].shape[0]

            df_expanded_ = pd.concat(expanded_dfs)
            if multilevel:
                df_expanded_[('prob', 'prob')] = lcd**-1
            else:
                df_expanded_['prob'] = lcd**-1
            df_expanded_['pots'] = pot
            df_expanded_.set_index('pots', inplace=True)

            assert df_expanded_.shape[0] == lcd, f"pot={pot}, df_.shape[0]: {df_.shape[0]}\nlcd: {lcd}; df_expanded_.shape[0]: {df_expanded_.shape[0]}\nprobs_in:: \n    {probs_in}"

            dfs_new_.append(df_expanded_)
    df_expanded = pd.concat(dfs_new_)

    return df_expanded


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

                a, b, _, _ = scipy.stats.beta.fit(rescaled_intensities, floc=0, fscale=1)
                distal_prior_param[stim][feature][marginal] = (a, b)

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


def get_empirical_inverse_planning_priors_(invPlanExtras_BaseGeneric, invPlanExtras_RepuSpecific):

    df_wide6, feature_list6, shorthand6, shorthand_list6 = invPlanExtras_BaseGeneric['df_wide'].copy(), invPlanExtras_BaseGeneric['feature_list'], invPlanExtras_BaseGeneric['shorthand'], invPlanExtras_BaseGeneric['shorthand_list']

    df_wide9, feature_list9, shorthand9, shorthand_list9 = invPlanExtras_RepuSpecific['df_wide'].copy(), invPlanExtras_RepuSpecific['feature_list'], invPlanExtras_RepuSpecific['shorthand'], invPlanExtras_RepuSpecific['shorthand_list']

    a1_labels = ['C', 'D']

    generic_prior_param, _, empratings6 = fit_priors_exp6_(a1_labels, df_wide6, feature_list6)

    distal_prior_param, rescale_intensities_, empratings9 = fit_priors_exp9_(a1_labels, df_wide9, feature_list9)

    return df_wide9, shorthand9, shorthand_list9, distal_prior_param, empratings9, df_wide6, shorthand6, shorthand_list6, generic_prior_param, empratings6, rescale_intensities_


def make_wide_df_for_IP(invplandf):
    import pandas as pd
    from cam_webppl_utils import unweightProbabilities

    nobsdf = invplandf['nobs']
    a1s = list(nobsdf.columns)
    df_array = list()
    for a1 in a1s:
        pots_by_a1_temp = nobsdf.index[nobsdf[a1] > 0]
        for i_pot, pot in enumerate(pots_by_a1_temp):
            data_slice = invplandf[a1].loc[pot, slice('feature', 'prob')]
            nobs = nobsdf[a1].loc[pot]

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


def composite_inverse_planning_split_violin(ppldata, paths, plotParam):
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    def remap_pia2(emp_):
        assert emp_ in [0, 1, 2, 3, 4, 5]
        remap = np.array([11, 9, 7, 5, 3, 1]) / 12
        return remap[emp_]

    """
    NB 
    For the generic players, we only collected inverse planning ratings for the anonymous game (exp6).
    For the specific players, we only collected inverse planning ratings for the public game (exp9).
    
    Thus, to show the emprical ratings of the inverse planning features for the public game, I'm using the exp9 data (specific players) and summing over the specific players.
    
    For this figure:
    Anon game, Human <- empirical ratings - anonymous game - generic players (exp6)
    
    Anon game, Model <- model inference, anonymous game (level1), priors used: empirical ratings - anonymous game - generic players (exp6) ((summed over: random faces, pot sizes, a_1))
    
    Public game, Human <- empirical ratings - public game - specific players (exp9) ((summed over the specific players))
    
    Public game, Model <- model inference, public game (level3); priors used: empirical ratings - public game - specific players (exp9) ((summed over: specific player descriptions, pot sizes, a_1))
    """

    ######

    """
    ppldata['empiricalInverseJudgmentsExtras_BaseGeneric']['df_wide'] - empirical ratings of generic players anonymous game
    df_wide6 - empirical ratings of generic players anonymous game
    
    ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide'] - empirical ratings of distal players public game
    df_wide9 - empirical ratings of distal players public game
    shorthand9 - prompts for feature ratings
    """

    df_wide9, shorthand9, shorthand_list9, distal_prior_param, empratings9, df_wide6, shorthand6, shorthand_list6, generic_prior_param, empratings6, rescale_intensities_ = get_empirical_inverse_planning_priors_(ppldata['empiricalInverseJudgmentsExtras_BaseGeneric'], ppldata['empiricalInverseJudgmentsExtras_RepuSpecific'])

    remap_ = np.vectorize(remap_pia2)

    '''Anon game, Human <- empirical ratings - anonymous game - generic players (exp6)'''

    """df_long6 - empirical ratings of generic players anonymous game"""
    df_wide6_copy = df_wide6.copy()
    df_wide6_copy.loc[:, 'pi_a2'] = remap_(df_wide6_copy.loc[:, 'pi_a2'].values)
    df_long6 = pd.melt(df_wide6_copy, id_vars=['a_1'], value_vars=['bMoney', 'bAIA', 'bDIA', 'pi_a2'], var_name='feature', value_name='weight').copy()

    '''Public game, Human <- empirical ratings - public game - specific players (exp9) ((summed over the specific players))'''

    """df_long9 - empirical ratings of distal players public game"""
    df_wide9_copy = df_wide9.copy()
    df_wide9_copy.loc[:, 'pi_a2'] = remap_(df_wide9_copy.loc[:, 'pi_a2'].values)
    df_long9 = pd.melt(df_wide9_copy, id_vars=['a_1'], value_vars=['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA', 'pi_a2'], var_name='feature', value_name='weight').copy()

    '''Anon game, Model <- model inference, anonymous game (level1), priors used: empirical ratings - anonymous game - generic players (exp6) ((summed over: random faces, pot sizes, a_1))'''

    """model ratings of generic players anonymous game ###"""
    model_level1_longdf = convert_model_level1_level3_widedf_to_longdf_(make_wide_df_for_IP(ppldata['level1']), list(shorthand6.keys()))

    '''Public game, Model <- model inference, public game (level3); priors used: empirical ratings - public game - specific players (exp9) ((summed over: specific player descriptions, pot sizes, a_1))'''

    """model ratings of generic players public game"""
    model_level3_longdf = convert_model_level1_level3_widedf_to_longdf_(make_wide_df_for_IP(ppldata['level3']), list(shorthand9.keys()))

    #################################
    ### inverse planning composite
    #################################

    plt = plotParam['plt']

    plt.close('all')

    no_debug = True

    ### heights 1
    margin_top = 1
    legend = 0.18
    text_title_h = 0.1
    x_labels_h = 0.2
    graphic_h = 0.7
    static_graphic = 2
    v_space = 0.05
    model_difs = 0.2

    if no_debug:
        static_graphic = 0.

    heights1 = [margin_top, legend + text_title_h, graphic_h, x_labels_h, static_graphic]

    heights2 = [model_difs, text_title_h, graphic_h, x_labels_h]

    heights3 = [v_space, text_title_h, graphic_h, x_labels_h]

    ### widths 1
    margin = 1
    y_label_w = 1

    horizontal_space_l = 0.0
    horizontal_space_r = 0.1
    graphic_base_w = 4
    h_space1 = 0.5

    wid_tem1 = np.array([0, horizontal_space_l, y_label_w, graphic_base_w, h_space1, y_label_w, graphic_base_w, horizontal_space_r, margin])

    ### widths 2
    graphic_repu_w = 3
    graphic_repu_w_diff = graphic_base_w - graphic_repu_w

    wid_tem2 = np.array([0, horizontal_space_l, y_label_w, graphic_repu_w, graphic_repu_w_diff, h_space1, y_label_w, graphic_repu_w, graphic_repu_w_diff, horizontal_space_r, margin])

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['single']

    np.isclose(wid_tem1.sum(), wid_tem2.sum())

    wid_tem_total = (np.sum(wid_tem1) + np.sum(wid_tem2)) * 0.5
    widths1 = width_max * (wid_tem1 / wid_tem_total.sum())
    widths2 = width_max * (wid_tem2 / wid_tem_total.sum())

    title_width = 0.1
    total_height = (np.sum(heights1) + np.sum(heights2) + np.sum(heights3)) * 2.25
    total_width = (margin + title_width + np.sum(widths1)) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    font_caption = 7  # helvet 7pt
    font_body = 9  # helvet 9pt

    text_size_small = 10
    text_size_medium = 11
    text_size_large = 12
    text_size_larger = 15
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'left'}
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    gs0 = fig.add_gridspec(ncols=2, nrows=3, width_ratios=[margin + title_width, np.sum(widths1)], height_ratios=[np.sum(heights1), np.sum(heights2), np.sum(heights3)], wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    gsbase = gs0[0, 1].subgridspec(ncols=len(widths1), nrows=len(heights1), width_ratios=widths1, height_ratios=heights1, wspace=0.0, hspace=0.0)
    gsrepu1 = gs0[1, 1].subgridspec(ncols=len(widths1), nrows=len(heights2), width_ratios=widths1, height_ratios=heights2, wspace=0.0, hspace=0.0)
    gsrepu2 = gs0[2, 1].subgridspec(ncols=len(widths2), nrows=len(heights2), width_ratios=widths2, height_ratios=heights3, wspace=0.0, hspace=0.0)

    gsbase_title = gs0[0, 0].subgridspec(ncols=2, nrows=len(heights1), width_ratios=[margin, title_width], height_ratios=heights1, wspace=0.0, hspace=0.0)
    gsrepu_title = gs0[1:, 0].subgridspec(ncols=2, nrows=1, width_ratios=[margin, title_width], wspace=0.0, hspace=0.0)

    gs_col1 = {
        'ml': 0,
        'sl': 1,
        'yl1': 2,
        'bh': 3,
        's1': 4,
        'yl2': 5,
        'bm': 6,
        'sr': 7,
        'mr': 8,
    }

    gs_row1 = {
        'mt': 0,
        'title': 1,
        'graphic': 2,
        'xl': 3,
        'static': 4,
    }

    gs_col2 = {
        'ml': 0,
        'sl': 1,
        'yl1': 2,
        'bh': 3,
        'diff1': 4,
        's1': 5,
        'yl2': 6,
        'bm': 7,
        'diff2': 8,
        'sr': 9,
        'mr': 10,
    }

    # ##################

    ax00 = fig.add_subplot(gsbase[gs_row1['title'], gs_col1['sl']:gs_col1['sr'] + 1])
    legend_elements = [
        Patch(label='Expectation:', facecolor='white', alpha=0.0, edgecolor='none', linewidth=0.),
        Line2D([0], [0], label='C', markerfacecolor='cornflowerblue', color='cornflowerblue', marker='o', alpha=1., markersize=0, linewidth=1.5),
        Line2D([0], [0], label='D', markerfacecolor='dimgrey', color='dimgrey', marker='o', alpha=1., markersize=0, linewidth=1.5),
        Patch(label='Density:', facecolor='white', alpha=0.0, edgecolor='none', linewidth=0.),
        Patch(label='C', facecolor='cornflowerblue', alpha=0.5, edgecolor='none', linewidth=0.),
        Patch(label='D', facecolor='dimgrey', alpha=0.5, edgecolor='none', linewidth=0.),
    ]

    ax00.axis('off')
    ax00.legend(handles=legend_elements, loc='center', columnspacing=1.0, bbox_to_anchor=(0.5, 0.8), frameon=False, handletextpad=0.31, ncol=len(legend_elements), prop={'size': text_size_medium})

    # ####################

    ax1 = fig.add_subplot(gsbase[gs_row1['graphic'], gs_col1['bh']])
    ax1 = plot_inverse_planning_kde_base_split(plotParam, df_long6, feature_list=['bMoney', 'bAIA', 'bDIA', 'pi_a2'], width=0.9, ax_=ax1)
    ax1.set_title('Human Judgment', fontdict={'fontsize': text_size_larger, 'fontweight': 'bold'}, y=1.03)

    ax2 = fig.add_subplot(gsbase[gs_row1['graphic'], gs_col1['bm']])
    ax2 = plot_inverse_planning_kde_base_split(plotParam, model_level1_longdf, feature_list=['bMoney', 'bAIA', 'bDIA', 'pi_a2'], width=0.9, ax_=ax2)
    ax2.set_title('Model Inverse Inference', fontdict={'fontsize': text_size_larger, 'fontweight': 'bold'}, y=1.03)

    ######

    ax3 = fig.add_subplot(gsrepu1[gs_row1['graphic'], gs_col1['bh']])
    ax3 = plot_inverse_planning_kde_base_split(plotParam, df_long9, feature_list=['bMoney', 'bAIA', 'bDIA', 'pi_a2'], width=0.9, ax_=ax3)
    ax3.set_title('Human Judgment', fontdict={'fontsize': text_size_larger, 'fontweight': 'bold'}, y=1.03)

    ax4 = fig.add_subplot(gsrepu1[gs_row1['graphic'], gs_col1['bm']])
    ax4 = plot_inverse_planning_kde_base_split(plotParam, model_level3_longdf, feature_list=['bMoney', 'bAIA', 'bDIA', 'pi_a2'], width=0.9, ax_=ax4)
    ax4.set_title('Model Inverse Inference', fontdict={'fontsize': text_size_larger, 'fontweight': 'bold'}, y=1.03)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': text_size_large})
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis="y", labelsize=text_size_large, pad=0)
        ax.tick_params(axis="x", labelsize=text_size_large, pad=0)

        ax.set_xticks([0, 1, 2, 3])
        text_base = r'$\mathrm{base}$'
        text_belief = r'$a_2=\mathrm{C}$'
        ax.set_xticklabels((text_base + '\n$Money$', text_base + '\n$AIA$', text_base + '\n$DIA$', 'Belief\n' + text_belief))

    ################

    ax0 = fig.add_subplot(gsrepu1[0, 0:8])
    ax0.plot([0, 1], [0.5, 0.5], 'k', linestyle='-', linewidth=1)
    ax0.axis('off')

    ax0 = fig.add_subplot(gsbase_title[gs_col1['yl1']:, 1])
    ax0.text(0.5, 0.5, f'Anonymous Game', horizontalalignment='center', verticalalignment='baseline', rotation_mode='anchor', fontdict={'fontsize': text_size_large + 3, 'fontweight': 'bold'}, rotation=90)
    ax0.axis('off')

    ax0 = fig.add_subplot(gsrepu_title[0, 1])
    ax0.text(0.5, 0.5, f'Public Game', horizontalalignment='center', verticalalignment='baseline', rotation_mode='anchor', fontdict={'fontsize': text_size_large + 3, 'fontweight': 'bold', 'horizontalalignment': 'center'}, rotation=90)
    ax0.axis('off')

    ax3 = fig.add_subplot(gsrepu2[gs_row1['graphic'], gs_col2['bh']])
    ax3 = plot_inverse_planning_kde_base_split(plotParam, df_long9, feature_list=['rMoney', 'rAIA', 'rDIA'], width=0.9, ax_=ax3)
    ax4 = fig.add_subplot(gsrepu2[gs_row1['graphic'], gs_col2['bm']])
    ax4 = plot_inverse_planning_kde_base_split(plotParam, model_level3_longdf, feature_list=['rMoney', 'rAIA', 'rDIA'], width=0.9, ax_=ax4)

    for ax in [ax3, ax4]:

        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': text_size_large})
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis="y", labelsize=text_size_large, pad=0)
        ax.tick_params(axis="x", labelsize=text_size_large, pad=0)

        ax.set_xticks([0, 1, 2])
        text_base = r'$\mathrm{repu}$'
        text_belief = r'$a_2=\mathrm{C}$'
        ax.set_xticklabels((text_base + '\n$Money$', text_base + '\n$AIA$', text_base + '\n$DIA$'))

    figsout = list()
    figsout.append((paths['figsOut'] / "composite_inv_plan_withprior-generic.pdf", fig, True))
    return figsout


def composite_inverse_planning_split_violin_specificplayer(ppldata, distal_prior_ppldata, paths, plot_param):
    import numpy as np
    import pandas as pd

    a1_labels = ppldata['labels']['decisions']

    for stimid in distal_prior_ppldata:

        dfs = dict()
        for a1_emp in a1_labels:
            a1_wppl = a1_emp

            nobsdf = distal_prior_ppldata[stimid][a1_emp]['level3']['nobs'][a1_emp]
            df_partial = distal_prior_ppldata[stimid][a1_emp]['level3'][a1_wppl]
            model_df_ = expand_df_(df_partial, nobsdf)

            dfs[a1_emp] = pd.melt(model_df_['feature'])

            dfs[a1_emp].columns = ['feature', 'weight']
            dfs[a1_emp]['a_1'] = a1_emp

        df_long_model = pd.concat(dfs.values())

        ##########

        assert np.min(ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']['pi_a2']) == 0
        assert np.max(ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']['pi_a2']) == 5

        dfs = dict()
        n_obs_counts_emp = dict()
        for a1_emp in a1_labels:

            nobsdf = ppldata['empiricalInverseJudgments_RepuSpecific']['nobs'][a1_emp].copy()
            nobsdf.loc[:] = 0
            df_partial0 = ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']
            cols = ['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA', 'pi_a2', 'pot']
            df_partial = df_partial0.loc[(df_partial0['face'] == stimid) & (df_partial0['a_1'] == a1_emp), cols].copy()

            n_obs_counts_emp[a1_emp] = df_partial.shape[0]

            df_partial.loc[:, 'pi_a2'] = ((np.abs(df_partial.loc[:, 'pi_a2'].to_numpy() - 5) * 2) + 1) / 12
            df_partial.set_index('pot', inplace=True)

            emp_df_ = expand_df_(df_partial, nobsdf)

            dfs[a1_emp] = pd.melt(emp_df_.drop(columns=['prob']))

            dfs[a1_emp].columns = ['feature', 'weight']
            dfs[a1_emp]['a_1'] = a1_emp

        df_long_emp = pd.concat(dfs.values())

        ##########

        composite_distal_fig(df_long_emp, df_long_model, stimid, paths, plot_param, title=f"{stimid} ($n_C$ = {n_obs_counts_emp['C']}, $n_D$ = {n_obs_counts_emp['D']})")

    # ####

    dfs = dict()
    for a1_emp in a1_labels:
        a1_wppl = a1_emp

        nobsdf = ppldata['level3']['nobs'][a1_emp]
        df_partial = ppldata['level3'][a1_wppl]
        model_df_ = expand_df_(df_partial, nobsdf)

        dfs[a1_emp] = pd.melt(model_df_['feature'])

        dfs[a1_emp].columns = ['feature', 'weight']
        dfs[a1_emp]['a_1'] = a1_emp

    df_long_model = pd.concat(dfs.values())

    ##########

    assert np.min(ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']['pi_a2']) == 0
    assert np.max(ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']['pi_a2']) == 5

    def remap_pia2(emp_):
        assert emp_ in [0, 1, 2, 3, 4, 5]
        remap = np.array([11, 9, 7, 5, 3, 1]) / 12
        return remap[emp_]

    remap_ = np.vectorize(remap_pia2)
    '''Public game, Human <- empirical ratings - public game - specific players (exp9) ((summed over the specific players))'''
    """df_long9 - empirical ratings of distal players public game"""
    df_wide9, shorthand9, shorthand_list9, distal_prior_param, empratings9, df_wide6, shorthand6, shorthand_list6, generic_prior_param, empratings6, rescale_intensities_ = get_empirical_inverse_planning_priors_(ppldata['empiricalInverseJudgmentsExtras_BaseGeneric'], ppldata['empiricalInverseJudgmentsExtras_RepuSpecific'])
    df_wide9_copy = df_wide9.copy()
    df_wide9_copy.loc[:, 'pi_a2'] = remap_(df_wide9_copy.loc[:, 'pi_a2'].values)
    df_long9 = pd.melt(df_wide9_copy, id_vars=['a_1'], value_vars=['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA', 'pi_a2'], var_name='feature', value_name='weight').copy()

    composite_distal_fig(df_long9, df_long_model, 'generic_avatar_male', paths, plot_param, title=f"generic")


def followup_analyses(**kwargs):

    import numpy as np
    import pandas as pd
    import dill
    import re
    from cam_plotfn_inverseplanning import run_inversePlanningAnalysis, inverse_planning_posterfigs_wrapper
    from cam_collect_torch_results import get_ppldata_from_cpar
    from cam_plot_utils import printFigList

    assert 'cpar' in kwargs or 'cpar_path' in kwargs
    if 'cpar' in kwargs:
        cpar = kwargs['cpar']
    else:
        with open(kwargs['cpar_path'], 'rb') as f:
            cpar = dill.load(f)
        cpar.cache['webppl'].update({'runModel': False, 'loadpickle': True})

    if 'ppldata' in kwargs:
        assert 'distal_prior_ppldata' in kwargs
        ppldata = kwargs['ppldata']
        distal_prior_ppldata = kwargs['distal_prior_ppldata']
    else:
        ppldata, distal_prior_ppldata = get_ppldata_from_cpar(cpar=cpar)

    if 'plotParam' in kwargs:
        plotParam = kwargs['plotParam']
    else:
        plotParam = cpar.plot_param
    plt = plotParam['plt']
    sns = plotParam['sns']

    paths = cpar.paths

    # %%

    #####
    ## make figs/composite_inv_plan_withprior.pdf
    #####

    figsout = printFigList(composite_inverse_planning_split_violin(ppldata, paths, plotParam), plotParam)

    # %%

    #####
    ## make figs/composite_inv_plan-specificPlayers/...
    #####

    composite_inverse_planning_split_violin_specificplayer(ppldata, distal_prior_ppldata, paths, plotParam)

    # %%

    #####
    ## make figs/feature_heatmaps/...
    #####

    features = ppldata['level4IAF']['CC'].columns.droplevel().drop('prob').to_list()
    outcomes = [key for key in list(ppldata['level4IAF'].keys()) if key != 'nobs']
    pots = np.unique(ppldata['level4IAF']['CC'].index.get_level_values(0).to_numpy())

    def ps_log1p(x): return np.sign(x) * np.log1p(np.abs(x))
    def ps_power04(x): return np.sign(x) * np.power(np.abs(x), 0.4)

    p = re.compile(r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2lnpotunval)')

    features_to_print = [s for s in features if p.match(s)]

    ### print features
    figsout = list()
    for ps_label, ps_fn in [('ps_power04', ps_power04)]:
        for renormalize_color_within_outcome in [True]:
            for i_feature, feature in enumerate(features_to_print):

                if 'PEa2' in feature:
                    ps_fn_ = None
                else:
                    ps_fn_ = ps_fn

                figsout.append(plotHeatmapDistOverPots_pub(**plotHeatmapDistOverPots_prep_emoDict_(ppldata['level4IAF'], i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=ps_fn_, transformation_label=ps_label), figoutpath=paths['figsOut'] / 'feature_heatmaps' / 'pub', renormalize_color_within_outcome=renormalize_color_within_outcome, plot_param=plotParam))

                figsout.append(plotHeatmapDistOverPots_pub(**plotHeatmapDistOverPots_prep_emoDict_(ppldata['level4IAF'], i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=ps_fn_, transformation_label=ps_label), figoutpath=paths['figsOut'] / 'feature_heatmaps' / 'pub-dense', linewidth=0.0, renormalize_color_within_outcome=renormalize_color_within_outcome, plot_param=plotParam))

    figsout = printFigList(figsout, plotParam)

    figsout = list()
    for ps_label, ps_fn in [('ps_power04', ps_power04), ('ps_log1p', ps_log1p)]:
        if ps_label == 'ps_power04':
            renormalize_color_within_outcome_opt = [True, False]
        else:
            renormalize_color_within_outcome_opt = [True]
        for renormalize_color_within_outcome in renormalize_color_within_outcome_opt:
            for i_feature, feature in enumerate(features):  # features_to_print

                if 'PEa2' in feature:
                    ps_fn_ = None
                else:
                    ps_fn_ = ps_fn

                figsout.append(plotHeatmapDistOverPots_pub(**plotHeatmapDistOverPots_prep_emoDict_(ppldata['level4IAF'], i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=ps_fn_, transformation_label=ps_label), figoutpath=paths['figsOut'] / 'feature_heatmaps' / 'objective', renormalize_color_within_outcome=renormalize_color_within_outcome, plot_param=plotParam))

    figsout = printFigList(figsout, plotParam)

    # %%

    #####
    ## Other inverse planning figures
    ## makes figs/inversePlanning/...
    #####

    _ = printFigList(run_inversePlanningAnalysis(ppldata, paths, plotParam['display_param'], {**plotParam, **cpar.plot_control['set1aba']}), plotParam)

    ######

    _ = printFigList(inverse_planning_posterfigs_wrapper(ppldata, paths, plotParam['display_param'], {**plotParam, **cpar.plot_control['set1aba']}), plotParam)

    # %%

    #########################
    ### Heat maps of model and empirical preferences
    #########################
    import scipy.stats
    from cam_webppl_utils import unweightProbabilities

    figsout = list()

    features = ['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA']

    plt.close('all')

    fig, axs = plt.subplots(figsize=(10, 10), nrows=2, ncols=2, sharex=False)

    for ia1, a1 in enumerate(['C', 'D']):

        df_temp = ppldata['level3'][a1].copy()
        df_temp_nobs = ppldata['level3']['nobs'][a1].sum()

        df_temp.columns = df_temp.columns.droplevel()
        df_temp.loc[:, 'prob'] = df_temp.loc[:, 'prob'].to_numpy() / len(np.unique(df_temp.index.get_level_values(0)))
        df_temp.reset_index(drop=True, inplace=True)

        unweighted_df = unweightProbabilities(df_temp, nobs=df_temp_nobs)

        corrmat = np.full((len(features), len(features)), np.nan)

        for i_feature, featurei in enumerate(features):
            for j_feature, featurej in enumerate(features):

                bb = unweighted_df.loc[:, featurei]
                aa = unweighted_df.loc[:, featurej]

                corrmat[i_feature, j_feature] = scipy.stats.pearsonr(aa, bb)[0]
                if i_feature == j_feature:
                    corrmat[i_feature, j_feature] = np.nan

        iu1 = np.triu_indices(len(features))
        mask = np.zeros(corrmat.shape)

        xlabels = features.copy()
        xlabels[-1] = None
        ylabels = features.copy()
        ylabels[0] = None

        ax = axs[1, ia1]

        ax = sns.heatmap(corrmat, mask=mask, square=True, cmap='coolwarm', ax=ax)
        ax.set_xticklabels(xlabels, rotation=-35, ha='left', rotation_mode='anchor')
        ax.set_yticklabels(ylabels, rotation=0, ha='right')
        ax.set_ylim((len(features), 0))

        ax.set_title(f"Model {a1}")

    ########

    df_wide9, _, _, _, _, _, _, _, _, _, _ = get_empirical_inverse_planning_priors_(ppldata['empiricalInverseJudgmentsExtras_BaseGeneric'], ppldata['empiricalInverseJudgmentsExtras_RepuSpecific'])

    for ia1, a1 in enumerate(['C', 'D']):

        df_temp = df_wide9.loc[df_wide9['a_1'] == a1, :].copy()

        df_temp.drop(columns=df_temp.columns[~df_temp.columns.isin(features)], inplace=True)

        corrmat = np.full((len(features), len(features)), np.nan)

        for i_feature, featurei in enumerate(features):
            for j_feature, featurej in enumerate(features):

                bb = df_temp.loc[:, featurei]
                aa = df_temp.loc[:, featurej]

                corrmat[i_feature, j_feature] = scipy.stats.pearsonr(aa, bb)[0]
                if i_feature == j_feature:
                    corrmat[i_feature, j_feature] = np.nan

        iu1 = np.triu_indices(len(features))
        mask = np.zeros(corrmat.shape)

        xlabels = features.copy()
        xlabels[-1] = None
        ylabels = features.copy()
        ylabels[0] = None

        ax = axs[0, ia1]

        ax = sns.heatmap(corrmat, mask=mask, square=True, cmap='coolwarm', ax=ax)
        ax.set_xticklabels(xlabels, rotation=-35, ha='left', rotation_mode='anchor')
        ax.set_yticklabels(ylabels, rotation=0, ha='right')
        ax.set_ylim((len(features), 0))

        ax.set_title(f"Empirical {a1}")

    plt.tight_layout()

    figsout.append((paths['figsOut'] / f"omega_corr_publicGame_modelandempirical.pdf", fig, True))
    plt.close(fig)
    figsout = printFigList(figsout, plotParam)

    plt.close('all')

    print('followup_analyses() fig printing complete')
