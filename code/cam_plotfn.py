#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_plotfn.py
"""


def plotDM(ax, sns, dm, mask=None, vmin=None, vmax=None, vcenter=None, cmap='YlGnBu', label_rename_dict=None, show_labels=True, linewidth=0.5):

    import numpy as np

    if vmin is None:
        vmin = np.floor(dm.min().min())
    if vmax is None:
        vmax = np.ceil(dm.max().max())
    if vcenter is None:
        vcenter = (vmax - vmin) / 2
    if mask is None:
        mask = np.full_like(dm, False)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(dm, ax=ax, mask=mask, vmin=vmin, vmax=vmax, center=vcenter, cmap=cmap, square=True, linewidths=linewidth, linecolor=(1, 1, 1, 1.0), cbar=False)

    ax.set_axisbelow(False)

    if label_rename_dict is None:
        xlabels_val = dm.columns.to_list()
        ylabels_val = dm.index.to_list()
    else:
        xlabels_val = [label_rename_dict[label] for label in dm.columns.to_list()]
        ylabels_val = [label_rename_dict[label] for label in dm.index.to_list()]

    if show_labels:

        ax.set_xticks(range(dm.shape[1]))
        ax.set_yticks(range(dm.shape[0]))
        ax.set_xticklabels(xlabels_val)
        ax.set_yticklabels(ylabels_val)

        xlabels = ax.get_xticklabels()
        ylabels = ax.get_yticklabels()

        for ilabel, xlabel in enumerate(xlabels):
            ax.text(ilabel + 0.25, ilabel, xlabel.get_text(), horizontalalignment='left', verticalalignment='bottom', rotation=45, fontsize=9)
            ax.text(-0.3, ilabel + 0.5, xlabel.get_text(), horizontalalignment='right', verticalalignment='center', rotation=0, fontsize=9)

    ax.set_xticklabels([''] * len(xlabels_val))
    ax.set_yticklabels([''] * len(ylabels_val))

    return ax


def plotDM_handler(dm, label_rename_dict=None, plotParam=None, outpath=None, show_labels=True, show_colorbar=True, invert_mask=False, grey_ones=False, linewidth=0.5):
    import numpy as np
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    import math

    assert len(np.unique(dm.shape)) == 1

    plt = plotParam['plt']
    sns = plotParam['sns']

    plt.close('all')

    cmap_diverging = sns.diverging_palette(200, 300, s=75, l=40, as_cmap=True)
    cmap_diverging.set_over('grey')
    cmap_diverging.set_under('grey')

    dm_ = dm.copy()

    if grey_ones:
        dm_ = dm_.applymap(lambda x: 1.01 if math.isclose(x, 1.0) else x)
        dm_ = dm_.applymap(lambda x: -1.01 if math.isclose(x, -1.0) else x)

    mask_allemotions_tril1 = np.zeros((dm.shape[0], dm.shape[0]), dtype=bool)
    mask_allemotions_tril1[np.triu_indices_from(mask_allemotions_tril1, k=1)] = True
    if invert_mask:
        mask_allemotions_tril1 = ~mask_allemotions_tril1

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = plotDM(ax, sns, dm_, mask=mask_allemotions_tril1, vmin=-1.0, vmax=1.0, vcenter=0, cmap=cmap_diverging, label_rename_dict=label_rename_dict, show_labels=show_labels, linewidth=linewidth)

    plt.grid(False)

    if show_colorbar:
        ### colorbar ###
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes('bottom', size='4%', pad='2%')
        fig.colorbar(ax.get_children()[0], ax=ax, cax=cax, orientation='horizontal', ticks=[-1, 0, 1], drawedges=False)
        cax.xaxis.set_ticks_position('bottom')

    figsout = list()
    figsout.append((outpath, fig, True))
    plt.close(fig)

    return figsout


def emo_marginals_plot(ppldataemodf, emoevdf=None, scale_factor=1.0, bandwidth=0.05, emotions_abbriv=None, fig_outpath=None, plotParam=None, verbose=False):

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

    """e.g.
    ppldataemodf = ppldata['empiricalEmotionJudgments']
    """
    # %%

    def gradient_image(ax, X, extent, direction=0.0, cmap_range=(0, 1), **kwargs):
        """
        Draw a gradient image based on a colormap.

        Parameters
        ----------
        ax : Axes
            The axes to draw on.
        extent
            The extent of the image as (xmin, xmax, ymin, ymax).
            By default, this is in Axes coordinates but may be
            changed using the *transform* keyword argument.
        direction : float
            The direction of the gradient. This is a number in
            range 0 (=vertical) to 1 (=horizontal).
        cmap_range : float, float
            The fraction (cmin, cmax) of the colormap that should be
            used for the gradient, where the complete colormap is (0, 1).
        **kwargs
            Other parameters are passed on to `.Axes.imshow()`.
            In particular useful is *cmap*.
        """

        im = ax.imshow(np.flipud(X), extent=extent, interpolation='bicubic',
                       vmin=0, vmax=1, **kwargs)
        return im

    plt = plotParam['plt']

    outcomes = plotParam['outcomes']

    emotions = ppldataemodf['CC'].loc[:, 'emotionIntensities'].columns.tolist()

    from mpl_toolkits.axes_grid1 import Divider, Size
    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(9.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(2.0), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    legend_str = dict(CC="CC (0.5, 0.5)", CD="CD (0, 1)", DC="DC (1, 0)", DD="DD (0, 0)")
    legend_plotted = list()

    bar_locs = np.array([-3, -1, 1, 3]) * 0.1
    bar_width = 0.1
    from scipy.stats import gaussian_kde
    kdes = dict()
    maxval = 0
    for i_emotion, emotion in enumerate(emotions):
        if verbose:
            print(f"making kde for {i_emotion + 1}")
        kdes[emotion] = dict()
        for i_outcome, outcome in enumerate(outcomes):
            observations = ppldataemodf[outcome].loc[:, ('emotionIntensities', emotion)]

            support_ = np.arange(0, 1.01, 0.01)
            kde = gaussian_kde(observations, bw_method=bandwidth)
            pd_ = kde.pdf(support_)
            kdes[emotion][outcome] = pd_ / np.sum(pd_)

    for i_emotion, emotion in enumerate(emotions):
        for i_outcome, outcome in enumerate(outcomes):
            i_x_major = i_emotion
            i_x_minor = i_outcome

            bar_loc_center = i_x_major + bar_locs[i_x_minor]
            bar_loc_l = bar_loc_center - (bar_width / 2)
            bar_loc_r = bar_loc_center + (bar_width / 2)

            kde_ = scale_factor * kdes[emotion][outcome]  # / np.max(kdes[emotion][outcome])

            if outcome == 'CC':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'CD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DC':
                cdict3 = {'red': [[0.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }

            X_ = np.tile(kde_, [2, 1]).T
            cutsom_cmap3 = LinearSegmentedColormap('testCmap', segmentdata=cdict3, N=256)
            gradient_image(ax, X_, extent=(bar_loc_l, bar_loc_r, 0.0, 1.0), direction=0, cmap=cutsom_cmap3, cmap_range=(0, 1.0), aspect='auto', zorder=50)

            # scatter_nudge = 0.02
            scatter_nudge = 0.0
            if outcome in legend_plotted:
                legend_label = None
            else:
                legend_label = legend_str[outcome]
                legend_plotted.append(outcome)
            color = dict(CC='green', CD='blue', DC='red', DD='black')[outcome]

            ax.scatter([bar_loc_center + scatter_nudge], [emoevdf.loc[outcome, emotion]], marker='o', s=45, color=color, facecolor='white', linewidth=1.5, zorder=51, label=legend_label)

    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(np.arange(0, 20))
    if emotions_abbriv is None:
        ax.set_xticklabels(emotions, rotation=-35, rotation_mode='anchor', ha='left')
    else:
        emotions_abbriv_labels = [emotions_abbriv[emotion] for emotion in emotions]
        ax.set_xticklabels(emotions_abbriv_labels)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='x', which='major', width=0, length=0, labelsize=11)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.tick_params(axis='y', which='major', width=0, length=0, labelsize=11, pad=3.0)

    # ax.grid(visible=None, which='major', axis='both', **kwargs)
    ax.xaxis.grid(visible=True, which='minor')
    ax.xaxis.grid(visible=False, which='major')

    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.96), ncol=4, frameon=False, fontsize=11, handlelength=0.9, handletextpad=0.3)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0.05)

    plt.close('all')

    # %%


def iaf_marginals_plot(ppldataemodf, emoevdf=None, scale_factor=1.0, bandwidth=0.05, emotions_abbriv=None, xrotate=True, yrange=None, fig_outpath=None, plotParam=None, verbose=False):

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MultipleLocator
    from mpl_toolkits.axes_grid1 import Divider, Size
    from scipy.stats import gaussian_kde

    # %%

    def gradient_image(ax, X, extent, direction=0.0, cmap_range=(0, 1), **kwargs):
        """
        Draw a gradient image based on a colormap.

        Parameters
        ----------
        ax : Axes
            The axes to draw on.
        extent
            The extent of the image as (xmin, xmax, ymin, ymax).
            By default, this is in Axes coordinates but may be
            changed using the *transform* keyword argument.
        direction : float
            The direction of the gradient. This is a number in
            range 0 (=vertical) to 1 (=horizontal).
        cmap_range : float, float
            The fraction (cmin, cmax) of the colormap that should be
            used for the gradient, where the complete colormap is (0, 1).
        **kwargs
            Other parameters are passed on to `.Axes.imshow()`.
            In particular useful is *cmap*.
        """

        im = ax.imshow(np.flipud(X), extent=extent, interpolation='bicubic',
                       vmin=0, vmax=1, **kwargs)
        return im

    plt = plotParam['plt']

    outcomes = plotParam['outcomes']

    emotions = ppldataemodf['CC'].loc[:, 'emotionIntensities'].columns.tolist()

    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the  axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(9.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(2.0), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    legend_str = dict(CC="CC (0.5, 0.5)", CD="CD (0, 1)", DC="DC (1, 0)", DD="DD (0, 0)")
    legend_plotted = list()

    bar_locs = np.array([-3, -1, 1, 3]) * 0.1
    bar_width = 0.1
    gradient_step_size = 0.01
    kdes = dict()
    for i_emotion, emotion in enumerate(emotions):
        if verbose:
            print(f"making kde for {i_emotion + 1}")
        kdes[emotion] = dict()
        for i_outcome, outcome in enumerate(outcomes):
            observations = ppldataemodf[outcome].loc[:, ('emotionIntensities', emotion)]

            support_ = np.arange(yrange[0], yrange[1] + gradient_step_size, gradient_step_size)
            try:
                kde = gaussian_kde(observations, bw_method=bandwidth)
                pd_ = kde.pdf(support_)
                pd_clipped = pd_
                pd_clipped[np.argwhere(support_ < np.min(observations))] = 0.0
                pd_clipped[np.argwhere(support_ > np.max(observations))] = 0.0
                if np.sum(pd_clipped) == 0:
                    kdes[emotion][outcome] = np.zeros(len(support_))
                else:
                    kdes[emotion][outcome] = pd_clipped / np.sum(pd_clipped)
            except:
                kdes[emotion][outcome] = np.zeros(len(support_))

    for i_emotion, emotion in enumerate(emotions):
        for i_outcome, outcome in enumerate(outcomes):
            i_x_major = i_emotion
            i_x_minor = i_outcome

            bar_loc_center = i_x_major + bar_locs[i_x_minor]
            bar_loc_l = bar_loc_center - (bar_width / 2)
            bar_loc_r = bar_loc_center + (bar_width / 2)

            kde_scaled = scale_factor * kdes[emotion][outcome]  # / np.max(kdes[emotion][outcome])

            if outcome == 'CC':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 1.0, 1.0],
                                    [1.0, 1.0, 1.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'CD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 1.0, 1.0],
                                   [1.0, 1.0, 1.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DC':
                cdict3 = {'red': [[0.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }
            elif outcome == 'DD':
                cdict3 = {'red': [[0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0]],
                          'green': [[0.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0]],
                          'blue': [[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0]],
                          'alpha': [[0.0, 0.0, 0.0],
                                    [1.0, 1.0, 1.0]], }

            X_ = np.tile(kde_scaled, [2, 1]).T
            cutsom_cmap3 = LinearSegmentedColormap('testCmap', segmentdata=cdict3, N=256)
            gradient_image(ax, X_, extent=(bar_loc_l, bar_loc_r, yrange[0], yrange[1]), direction=0, cmap=cutsom_cmap3, cmap_range=(0, 1.0), aspect='auto', zorder=50)

            # scatter_nudge = 0.02
            scatter_nudge = 0.0
            if outcome in legend_plotted:
                legend_label = None
            else:
                legend_label = legend_str[outcome]
                legend_plotted.append(outcome)
            color = dict(CC='green', CD='blue', DC='red', DD='black')[outcome]

            ax.scatter([bar_loc_center + scatter_nudge], [emoevdf.loc[outcome, emotion]], marker='o', s=45, color=color, facecolor='white', linewidth=1.5, zorder=51, label=legend_label)

    ax.set_xlim(-0.5, 19.5)
    ax.set_xticks(np.arange(0, len(emotions)))

    if xrotate:
        if emotions_abbriv is None:
            ax.set_xticklabels(emotions, rotation=-35, rotation_mode='anchor', ha='left')
        else:
            emotions_abbriv_labels = [emotions_abbriv[emotion] for emotion in emotions]
            ax.set_xticklabels(emotions_abbriv_labels, rotation=-35, rotation_mode='anchor', ha='left')
    else:
        emotions_abbriv_labels = [emotions_abbriv[emotion] for emotion in emotions]
        ax.set_xticklabels(emotions_abbriv_labels, fontdict={'fontsize': 10.5})

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='x', which='major', width=0, length=0, labelsize=11)

    ax.set_ylim(yrange)
    ax.set_yticks(np.arange(yrange[0], yrange[1] + 1, 1))
    ax.tick_params(axis='y', which='major', width=0, length=0, labelsize=11, pad=3.0)

    ax.xaxis.grid(visible=True, which='minor')
    ax.xaxis.grid(visible=False, which='major')

    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.96), ncol=4, frameon=False, fontsize=11, handlelength=0.9, handletextpad=0.3)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0.05)

    plt.close('all')

    # %%


def get_iaf_names(cfg):
    from cam_collect_torch_results import get_ppldata_from_cpar
    from cam_emotorch_utils import prep_generic_data_pair_, getEmpiricalModel_pair_

    ppldata, distal_prior_ppldata = get_ppldata_from_cpar(cpar_path=cfg['cpar_path_str'])

    feature_selector_label, feature_selector = cfg['pytorch_spec']['feature_selector']

    composite_emodict_, composite_iafdict_ = prep_generic_data_pair_(ppldata)
    Y_full, X_full, _ = getEmpiricalModel_pair_(composite_emodict_, composite_iafdict_, feature_selector=feature_selector, return_ev=False)

    return X_full.drop(['pot', 'outcome'], axis=1).columns.to_list()


def format_iaf_labels_multiline(iaf_labels):

    iaf_labels_formatted = list()
    for label in iaf_labels:
        label1 = label.replace('U[', 'AU[').replace('a1[', '{a1}[').replace('a2[', '{a2}[').replace('[base', '$\n${\mathrm{base}}[').replace('[repu', '$\n${\mathrm{repu}}[').replace('PEa2lnpotunval', '|PE\pi_{a_2}|').replace('[Money]', '$\n${\\mathit{Money}}').replace('[DIA]', '$\n${\\mathit{DIA}}').replace('[AIA]', '$\n${\\mathit{AIA}}').replace('[', '').replace(']', '')
        labelf = r"$" + label1 + r"$"
        iaf_labels_formatted.append(labelf)

    return iaf_labels_formatted


def format_iaf_labels_superphrase(iaf_labels):

    iaf_labels_formatted = list()
    for label in iaf_labels:
        label1 = label.replace('U[', 'AU[').replace('a1[', '{a1}[').replace('a2[', '{a2}[').replace('[base', '^{\mathrm{base}}[').replace('[repu', '^{\mathrm{repu}}[').replace('PEa2lnpotunval', '|PE\pi_{a_2}|').replace('[Money]', '_{\\mathit{Money}}').replace('[DIA]', '_{\\mathit{DIA}}').replace('[AIA]', '_{\\mathit{AIA}}').replace('[', '').replace(']', '')
        labelf = r"$" + label1 + r"$"
        iaf_labels_formatted.append(labelf)

    return iaf_labels_formatted


def composite_Aweights(learned_param_Amean=None, emo_labels=None, iaf_labels=None, lowercase_labels=False, fig_outpath=None, plotParam=None):

    import numpy as np
    from mpl_toolkits.axes_grid1 import Divider, Size
    # %%

    plt = plotParam['plt']
    sns = plotParam['sns']
    #######

    plt.close('all')

    text_size = 7

    iaf_labels_formatted = format_iaf_labels_superphrase(iaf_labels)

    A_ = learned_param_Amean.T
    maxval = np.max(np.absolute(A_.flatten()))

    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the  axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(4.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(3.75), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    cmap_blue_grey_red = sns.diverging_palette(260, 12, s=99, l=40, sep=2, as_cmap=True)
    cmap_green_white_purple = sns.diverging_palette(200, 300, s=75, l=40, sep=2, center='light', as_cmap=True)

    cmap_ = cmap_green_white_purple
    ax = sns.heatmap(A_, annot=True, linewidths=.5, center=0, vmin=-1 * maxval, vmax=maxval, cmap=cmap_, annot_kws={"size": 5}, fmt=".1f", ax=ax, cbar=False, )

    emo_labels_ = [x.lower() for x in emo_labels] if lowercase_labels else emo_labels
    ax.set_xticklabels(emo_labels_, rotation=-35, horizontalalignment='left', rotation_mode='anchor', fontdict={'fontsize': text_size})
    ax.set_yticklabels(iaf_labels_formatted, rotation=0, horizontalalignment='right', fontdict={'fontsize': text_size})

    ax.tick_params(axis="both", pad=-3, labelsize=text_size)

    ax.set_ylim((A_.shape[0], 0))

    # %%
    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    plt.close('all')


def single_Aweights(learned_param_A=None, emotion=None, emo_labels=None, iaf_labels=None, xrange=None, cialpha=None, cialpha_betaparam=None, fig_outpath=None, plotParam=None):

    import numpy as np
    import pandas as pd
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    i_emo = emo_labels.index(emotion)

    plt = plotParam['plt']
    emoparam = pd.DataFrame(learned_param_A[:, i_emo, :], columns=iaf_labels)

    from mpl_toolkits.axes_grid1 import Divider, Size
    fig = plt.figure(figsize=(7, 3))  # dims don't matter with fixed axis

    # The first & third items are for padding and the second items are for the  axes. Sizes are in inches.
    fa_width = [Size.Fixed(0.5), Size.Fixed(1.0), Size.Fixed(0.5)]
    fa_height = [Size.Fixed(0.5), Size.Fixed(3.75), Size.Fixed(0.5)]

    divider = Divider(fig, (0, 0, 1, 1), fa_width, fa_height, aspect=False)
    # The width and height of the rectangle are ignored.

    ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))

    for i_feature, feature in enumerate(iaf_labels):

        loadings = emoparam.loc[:, feature].to_list()

        alphatrim = len(loadings) * (cialpha_betaparam / 2)
        alphatrim_int = int(alphatrim)
        loadings_within_ci = sorted(loadings)[alphatrim_int:-alphatrim_int]
        loadings_ci = [np.min(loadings_within_ci), np.max(loadings_within_ci)]

        alphatrim_inner = len(loadings) * (cialpha / 2)
        alphatrim_inner_int = int(alphatrim_inner)
        loadings_within_inner_ci = sorted(loadings)[alphatrim_inner_int:-alphatrim_inner_int]
        loadings_inner_ci = [np.min(loadings_within_inner_ci), np.max(loadings_within_inner_ci)]

        if np.min(loadings_ci) <= 0 and np.max(loadings_ci) >= 0:
            color_ = '#929292'  # grey
        else:
            if np.min(loadings_ci) > 0:
                color_ = "#B31AB3"  # hsl(200, 75, 40) 'firebrick'
            elif np.max(loadings_ci) < 0:
                color_ = "#30666A"  # hsl(200, 75, 40) 'royalblue'
            else:
                color_ = 'green'

        ax.scatter([np.mean(loadings)], [i_feature], s=11, alpha=1.0, color=color_, facecolor='white', linewidth=1.0, zorder=99)
        ax.plot([np.min(loadings_ci), np.max(loadings_ci)], [i_feature, i_feature], color=color_, alpha=1.0, linewidth=1, solid_capstyle='butt')
        ax.plot([np.min(loadings_inner_ci), np.max(loadings_inner_ci)], [i_feature, i_feature], color=color_, alpha=1.0, linewidth=2, solid_capstyle='butt')

    ax.axvline(0, color='k', linestyle='-', linewidth=0.8)
    ax.set_yticks(range(len(iaf_labels)))
    iaf_labels_formatted = format_iaf_labels_superphrase(iaf_labels)
    ax.set_yticklabels(iaf_labels_formatted)
    ax.set_xscale('symlog', base=np.e)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    xlim = ax.get_xlim()
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 1))
    if xrange is not None:
        ax.set_xlim(xrange)
    ax.set_xticklabels([])

    ax.set_ylim((18.5, -0.5))

    text_size = 7
    ax.tick_params(axis="both", pad=-3, labelsize=text_size)

    ### For axis to appear in the same pixel location, regardless of labels,
    # plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    # plt.close('all')


def composite_emos_summarybars(bardatadf, x_tick_order=None, model_colors=None, model_order=None, fig_outpath=None, plotParam=None, x_tick_labels=None):

    import numpy as np
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    no_debug = True

    ### heights
    margin_top = 0.12
    allemotions_graphic = 1
    allemotions_xticks = 0.4
    allplayers_graphic = 1
    allplayers_xticks = 0.4
    static_graphic = 2.5
    v_space = 0.1

    ### widths
    margin = 1
    h_space_l = 0.9
    allemotions_graphic_w = 7
    h_space1 = 0.7
    scatter_w = 2
    h_space2 = h_space1
    fits_w = 1
    h_space_r = 0.4

    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    heights = [margin_top, allemotions_graphic, allemotions_xticks, v_space, allplayers_graphic, allplayers_xticks, static_graphic]
    wid_tem = np.array([margin, h_space_l, allemotions_graphic_w, h_space1, fits_w, h_space2, scatter_w, h_space_r, margin])
    widths = width_max * (wid_tem / wid_tem.sum())

    total_height = np.sum(heights) * 2.25
    total_width = np.sum(widths) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    text_size_small = 11
    text_size_large = 12
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'center'}

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    gs_col = {
        'ml': 0,
        's0': 1,
        'emo': 2,
        's1': 3,
        'fits': 4,
        's2': 5,
        'scatter': 6,
        's3': 7,
        'mr': 8,
    }

    gs_row = {
        'mt': 0,
        'emo': 1,
        'emoxt': 2,
        's1': 3,
        'players': 4,
        'playersxt': 5,
        'static': 6
    }

    ##################

    axs = list()

    ax = fig.add_subplot(gridspec[gs_row['players'], gs_col['fits']])
    axs.append(ax)

    ax = plotGroupedBars_new_withci_resized(bardatadf, ax=ax, colorin=model_colors, xlabels=x_tick_order, modellabels=model_order)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='y', which='major', width=0, length=0, labelsize=text_size_small, pad=3.0)

    ax.set_xlim([-0.5, 0.5])
    ax.xaxis.grid(False)
    ax.set_xticklabels(['overall'], fontdict=fontdict_ticks)
    ax.tick_params(axis='x', which='major', width=0, length=0, pad=5.0, labelsize=text_size_large)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0.05)

    # %%
    plt.close('all')


def composite_legend(model_labels=None, model_colors=None, fig_outpath=None, plotParam=None):

    import numpy as np
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    no_debug = True

    ### heights
    margin_top = 0.12
    allemotions_graphic = 1
    allemotions_xticks = 0.4
    allplayers_graphic = 1
    allplayers_xticks = 0.4
    static_graphic = 2.5
    v_space = 0.1

    ### widths
    margin = 1
    h_space_l = 0.9
    allemotions_graphic_w = 7
    h_space1 = 0.7
    scatter_w = 2
    h_space2 = h_space1
    fits_w = 1
    h_space_r = 0.4

    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    from matplotlib.patches import Patch

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    heights = [margin_top, allemotions_graphic, allemotions_xticks, v_space, allplayers_graphic, allplayers_xticks, static_graphic]
    wid_tem = np.array([margin, h_space_l, allemotions_graphic_w, h_space1, fits_w, h_space2, scatter_w, h_space_r, margin])
    widths = width_max * (wid_tem / wid_tem.sum())

    total_height = np.sum(heights) * 2.25
    total_width = np.sum(widths) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    text_size_small = 10
    text_size_large = 12

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    import matplotlib.cbook as cbook
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    gs_col = {
        'ml': 0,
        's0': 1,
        'emo': 2,
        's1': 3,
        'fits': 4,
        's2': 5,
        'scatter': 6,
        's3': 7,
        'mr': 8,
    }

    gs_row = {
        'mt': 0,
        'emo': 1,
        'emoxt': 2,
        's1': 3,
        'players': 4,
        'playersxt': 5,
        'static': 6
    }

    ##################

    axs = list()

    ax00 = fig.add_subplot(gridspec[gs_row['mt'], gs_col['emo']])

    legend_elements = [
        Patch(label=model_labels['caaFull'], facecolor=model_colors['caaFull'], edgecolor='none', linewidth=0.),
        Patch(label=model_labels['invplanLesion'], facecolor=model_colors['invplanLesion'], edgecolor='none', linewidth=0.),
        Patch(label=model_labels['socialLesion'], facecolor=model_colors['socialLesion'][0], edgecolor='none', linewidth=0.),
        Patch(label='Empirical CI', facecolor=(0., 0., 0., 0.2), edgecolor=(0., 0., 0., 0.6), linewidth=0.5, linestyle='-'),  # edgecolor=(1., 0., 0., 0.4), linestyle=(0, (3,1))
    ]

    ax00.set_axis_off()
    ax00.legend(handles=legend_elements, loc='center', columnspacing=1.0, bbox_to_anchor=(0.5, 0.8), frameon=False, handletextpad=0.3, ncol=len(legend_elements), prop={'size': text_size_small})

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0)

    # %%
    plt.close('all')


def composite_emos_concord(bardatadf, x_tick_order=None, model_colors=None, model_order=None, fig_outpath=None, plotParam=None, x_tick_labels=None, yrange=None, lowercase_labels=False):

    import numpy as np
    # %%

    plt = plotParam['plt']
    #######

    plt.close('all')

    no_debug = True

    ### heights
    margin_top = 0.12
    allemotions_graphic = 1
    allemotions_xticks = 0.4
    allplayers_graphic = 1
    allplayers_xticks = 0.4
    static_graphic = 2.5
    v_space = 0.1

    ### widths
    margin = 1
    h_space_l = 0.9
    allemotions_graphic_w = 7
    h_space1 = 0.7
    scatter_w = 2
    h_space2 = h_space1
    fits_w = 1
    h_space_r = 0.4

    if no_debug:
        allplayers_xticks, static_graphic = 0., 0.

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['double']

    heights = [margin_top, allemotions_graphic, allemotions_xticks, v_space, allplayers_graphic, allplayers_xticks, static_graphic]
    wid_tem = np.array([margin, h_space_l, allemotions_graphic_w, h_space1, fits_w, h_space2, scatter_w, h_space_r, margin])
    widths = width_max * (wid_tem / wid_tem.sum())

    total_height = np.sum(heights) * 2.25
    total_width = np.sum(widths) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    text_size_small = 11
    text_size_large = 12
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    gs_col = {
        'ml': 0,
        's0': 1,
        'emo': 2,
        's1': 3,
        'fits': 4,
        's2': 5,
        'scatter': 6,
        's3': 7,
        'mr': 8,
    }

    gs_row = {
        'mt': 0,
        'emo': 1,
        'emoxt': 2,
        's1': 3,
        'players': 4,
        'playersxt': 5,
        'static': 6
    }

    ##################

    axs = list()

    fakegrid_kwrgs = {'color': 'lightgrey', 'linewidth': 1.5, 'alpha': 1}

    ax = fig.add_subplot(gridspec[gs_row['players'], gs_col['emo']])
    axs.append(ax)

    ax = plotGroupedBars_new_withci_resized(bardatadf, ax=ax, colorin=model_colors, xlabels=x_tick_order, modellabels=model_order)
    ylim_original = ax.get_ylim()

    ax.set_ylabel(ax.get_ylabel(), fontdict=fontdict_axislab)

    ax.xaxis.grid(False)
    ax.set_xlim((-0.75, 19.75))

    if x_tick_labels is None:
        x_labels = [x.lower() for x in x_tick_order] if lowercase_labels else x_tick_order
    else:
        x_labels = [x_tick_labels[stimid] for stimid in x_tick_order]

    ax.tick_params(axis='y', which='major', width=0, length=0, labelsize=text_size_small, pad=3.0)
    ax.tick_params(axis='x', which='major', width=0, length=0, labelsize=text_size_large, pad=1.0)

    _ = ax.set_xticklabels(x_labels, rotation=-35, horizontalalignment='left', rotation_mode='anchor', fontdict={'fontsize': text_size_large})

    ylim_ = (np.sign(ylim_original[0]) * np.ceil(abs(ylim_original[0] * 10)) / 10, np.sign(ylim_original[1]) * np.ceil(abs(ylim_original[1] * 10)) / 10)
    if yrange is not None:
        ylim_ = yrange
    ax.set_ylim(ylim_)

    ycoord = ylim_[0] - ((ylim_[1] - ylim_[0]) * 0.05)

    exceed_ax = False
    for xcord in np.arange(19) + 0.5:
        ax.plot([xcord, xcord], [ycoord, ylim_[1]], linestyle='-', clip_on=exceed_ax, **fakegrid_kwrgs)

    ax.set_ylim(ylim_)

    i_ax = 1
    exceed_ax = False
    if i_ax == 1:
        from math import pi
        inv = ax.transData.inverted()
        magnitude = 40
        angle = -35. * (pi / 180.)
        dxx = magnitude * np.cos(angle)
        dyy = magnitude * np.sin(angle)
        p0d = ax.transData.transform([0, ycoord])
        for xcord in np.arange(19) + 0.5:

            p1d = ax.transData.transform([xcord, ycoord])
            p2d = [p1d[0] + dxx, p0d[1] + dyy]
            p2cord = inv.transform(p2d)

            ax.plot([xcord, p2cord[0]], [ycoord, p2cord[1]], linestyle='-', clip_on=exceed_ax, **fakegrid_kwrgs)

    ########################

    fig.align_ylabels(axs)

    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_outpath, bbox_inches='tight', pad_inches=0.05)

    # %%
    plt.close('all')


def plotGroupedBars_new_withci_resized(bardatadf, ax=None, colorin=None, xlabels=None, modellabels=None, width=0.35, major_space=0.35, minor_space=0.04):
    import numpy as np
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    n_major = len(xlabels)
    n_minor = len(modellabels)

    width = 0.23
    minor_space = 0.04
    ### 1 unit = (3 bars) + (2 minor space) + (1 major space)

    adjuster = [-1 * (width + minor_space), 0, (width + minor_space)]

    errorboxes = list()
    for i_x, xlabel in enumerate(xlabels):
        for i_y, model in enumerate(modellabels):
            ydata_ = bardatadf.loc[(bardatadf['model'] == model) & (bardatadf['xlabel'] == xlabel), :]
            y_ = ydata_['pe']
            ax.bar(i_x + adjuster[i_y], y_, width, color=colorin[model], edgecolor='none')

            cil_ = ydata_['cil'].item()
            ciu_ = ydata_['ciu'].item()
            ax.plot((i_x + adjuster[i_y], i_x + adjuster[i_y]), [cil_, ciu_], color='k', linewidth=1)

    for i_x, xlabel in enumerate(xlabels):
        ydata_ = bardatadf.loc[(bardatadf['model'] == 'empirical') & (bardatadf['xlabel'] == xlabel), :]
        cil_ = ydata_['cil'].item()
        ciu_ = ydata_['ciu'].item()
        # Loop over data points; create box from errors at each point
        empci_x_buffer = minor_space
        ci_emp_width = (width * len(adjuster)) + (minor_space * (len(adjuster) - 1)) + (empci_x_buffer * 2)
        x_emp_left = i_x - ci_emp_width / 2
        rect = Rectangle((x_emp_left, cil_), ci_emp_width, ciu_ - cil_)
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pce = PatchCollection(errorboxes, facecolor='none', alpha=0.6, edgecolor='k', linewidths=0.5, zorder=11)
    _ = ax.add_collection(pce)

    pc = PatchCollection(errorboxes, facecolor='k', alpha=0.2, edgecolor='none', zorder=10)
    _ = ax.add_collection(pc)

    _ = ax.set(xticks=range(n_major), xticklabels=xlabels)

    ax.set_ylim([bardatadf.loc[:, 'cil'].min(), bardatadf.loc[:, 'ciu'].max()])

    return ax
