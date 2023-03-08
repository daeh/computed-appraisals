
# %%


def plot_inverse_planning_kde_base_split_HIST_(plotParam, inv_planning_df_, inv_planning_baseline_df_=None, feature_list=None, width=0.7, ax_=None):
    import numpy as np
    from matplotlib.collections import PolyCollection

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    if not feature_list:
        feature_list = np.unique(inv_planning_df_['feature'].values)
    inv_planning_df = inv_planning_df_.astype({'weight': 'float'})

    # vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=inv_planning_df, ax=ax_,
    #             hue_order=['C','D'],
    #             split=True,
    #             palette=dict(C='cornflowerblue', D='dimgrey'),
    #             inner='quartile') #   scale='count',

    # print(inv_planning_df.dtypes)
    # inv_planning_df.astype({'weight': 'float'})
    # print(inv_planning_df.dtypes)
    # vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=inv_planning_df, ax=ax_,
    #             hue_order=['C','D'],
    #             split=True,
    #             palette=dict(C='cornflowerblue', D='dimgrey'),
    #             width=0.9,
    #             bw='scott', cut=2, scale='area',
    #             linewidth=1.0,
    #             saturation=1,
    #             inner='quartile')

    # if not inv_planning_baseline_df_ is None:
    #     inv_planning_baseline_df = inv_planning_baseline_df_.astype({'weight': 'float'})
    #     inv_planning_baseline_df.drop(['a_1'], axis=1, inplace=True)
    #     vp0 = sns.violinplot(x="feature", y="weight", data=inv_planning_baseline_df.loc[inv_planning_baseline_df['feature'].isin([feature for feature in feature_list if feature not in 'pi_a2']), :], ax=ax_,
    #                 split=True,
    #                 width=width,
    #                 # bw=0.1, cut=2, scale='area',
    #                 bw='scott', cut=2, scale='area',
    #                 linewidth=0.0,
    #                 saturation=1,
    #                 inner=None)

    df_ = inv_planning_df.loc[inv_planning_df['feature'].isin([feature for feature in feature_list if feature not in 'pi_a2']), :]
    vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=df_, ax=ax_,
                        order=feature_list,
                        hue_order=['C', 'D'],
                        split=True,
                        palette=dict(C='cornflowerblue', D='dimgrey'),
                        width=width,
                        # bw=0.1, cut=2, scale='area',
                        bw='scott', cut=0, scale='area',
                        # scale_hue=False,
                        linewidth=0.0,
                        saturation=1,
                        inner=None)
    ax_.set_ylim([-0.0, 1.0])

    for i_feature, feature in enumerate(feature_list):
        for a1 in ['C', 'D']:
            xcoord = i_feature + {'C': -width / 4, 'D': width / 4}[a1]
            yval = np.mean(inv_planning_df.loc[(inv_planning_df['feature'] == feature) & (inv_planning_df['a_1'] == a1), 'weight'])
            # ax_.scatter([xcoord], [yval], s=40, color=dict(C='cornflowerblue', D='dimgrey')[a1], edgecolor='none', linewidth=2)
            # ax_.scatter([xcoord], [yval], s=350, marker='_', color=dict(C='cornflowerblue', D='dimgrey')[a1], linewidth=1.5)
            # ax_.scatter([xcoord], [yval], s=40, color=dict(C='cornflowerblue', D='dimgrey')[a1], edgecolor='red', linewidth=2)
            # ax_.scatter([xcoord], [yval], s=40, color=color_, edgecolor=color_, linewidth=2)
            color_ = dict(C='cornflowerblue', D='dimgrey')[a1]
            ax_.plot([xcoord - width / 6, xcoord + width / 6], [yval, yval], color=color_, linewidth=2)

    ####

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

                # ax_.scatter([xcoord], [yval], s=40, color=dict(C='cornflowerblue', D='dimgrey')[a1], edgecolor='none', linewidth=2)
                # ax_.scatter([xcoord], [yval], s=350, marker='_', color=dict(C='cornflowerblue', D='dimgrey')[a1], linewidth=1.5)

                x_base = i_feature
                for i_binval, binval in enumerate(np.arange(0, 1 + 1 / 48, 1 / 48)):
                    xdelta = (width / 2) * hist[i_binval]
                    if a1 == 'C':
                        ax_.plot([i_feature, i_feature - xdelta], [binval, binval], linewidth=2, color='cornflowerblue', alpha=1)
                    else:
                        ax_.plot([i_feature, i_feature + xdelta], [binval, binval], linewidth=2, color='dimgrey', alpha=1)

    #####

    if 'pi_a2' in feature_list:
        # from matplotlib.patches import Wedge
        kwargs = dict()
        kwargs.update(transform=ax_.transAxes, clip_on=False)
        xcoord_ = feature_list.index('pi_a2')

        for a1 in ['C', 'D']:
            xvals, ycounts = np.unique(inv_planning_df.loc[(inv_planning_df['a_1'] == a1) & (inv_planning_df['feature'] == 'pi_a2'), 'weight'], return_counts=True)
            ycounts_total = ycounts.sum()
            # area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
            area = 1000
            sizes = area * ycounts / ycounts_total
            sizes = np.pi * (25 * ycounts / ycounts_total)**2  # 0 to 15 point radiuses

            radii = (ycounts / ycounts_total)
            angle = 90
            theta1, theta2 = angle, angle + 180

            # xcoord = 3 + {'C':-width/4, 'D':width/4}[a1]
            # for ixval,xval in enumerate(xvals):
            #     # ax_.scatter(np.repeat(xcoord, 6), xvals, s=sizes, alpha=0.05, edgecolor='none', linewidth=0, c={'C':'cornflowerblue', 'D':'dimgrey'}[a1], zorder=3)
            #     w1 = Wedge([xcoord,xval], radii[ixval], theta1, theta2, fc={'C':'cornflowerblue', 'D':'dimgrey'}[a1], **kwargs)
            #     ax_.add_artist(w1)

            sizes = np.pi * (ycounts / ycounts_total)**2  # 0 to 15 point radiuses
            xcoord = xcoord_ + {'C': -width / 4, 'D': width / 4}[a1]
            for ixval, xval in enumerate(xvals):
                marker_style = dict(color='none', linestyle=':', marker='o', markersize=50 * radii[ixval], markerfacecoloralt={'C': 'cornflowerblue', 'D': 'dimgrey'}[a1], alpha=0.5)
                ax_.plot(xcoord_, xval, fillstyle={'C': 'right', 'D': 'left'}[a1], **marker_style)

        # for a1 in ['C', 'D']:
        #     xvals,ycounts = np.unique(inv_planning_df.loc[ (inv_planning_df['a_1'] == a1) & (inv_planning_df['feature'] == 'pi_a2'), 'weight'], return_counts=True)
        #     ycounts_total =  ycounts.sum()
        #     # area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
        #     area = 800
        #     sizes = area*ycounts/ycounts_total
        #     sizes = np.pi * (25 * ycounts/ycounts_total)**2 # 0 to 15 point radiuses

        #     xcoord = 3 + {'C':-width/4, 'D':width/4}[a1]
        #     ax_.scatter(np.repeat(xcoord, 6), xvals, s=sizes, alpha=0.3, edgecolor='none', linewidth=0, c={'C':'cornflowerblue', 'D':'dimgrey'}[a1], zorder=3)

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
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    if not feature_list:
        feature_list = np.unique(inv_planning_df_['feature'].values)
    inv_planning_df = inv_planning_df_.astype({'weight': 'float'})

    # import seaborn as sns
    # vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=inv_planning_df, ax=ax_,
    #             hue_order=['C','D'],
    #             split=True,
    #             palette=dict(C='cornflowerblue', D='dimgrey'),
    #             inner='quartile') #   scale='count',

    # print(inv_planning_df.dtypes)
    # inv_planning_df.astype({'weight': 'float'})
    # print(inv_planning_df.dtypes)
    # vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=inv_planning_df, ax=ax_,
    #             hue_order=['C','D'],
    #             split=True,
    #             palette=dict(C='cornflowerblue', D='dimgrey'),
    #             width=0.9,
    #             bw='scott', cut=2, scale='area',
    #             linewidth=1.0,
    #             saturation=1,
    #             inner='quartile')

    # if not inv_planning_baseline_df_ is None:
    #     inv_planning_baseline_df = inv_planning_baseline_df_.astype({'weight': 'float'})
    #     inv_planning_baseline_df.drop(['a_1'], axis=1, inplace=True)
    #     vp0 = sns.violinplot(x="feature", y="weight", data=inv_planning_baseline_df.loc[inv_planning_baseline_df['feature'].isin([feature for feature in feature_list if feature not in 'pi_a2']), :], ax=ax_,
    #                 split=True,
    #                 width=width,
    #                 # bw=0.1, cut=2, scale='area',
    #                 bw='scott', cut=2, scale='area',
    #                 linewidth=0.0,
    #                 saturation=1,
    #                 inner=None)

    df_ = inv_planning_df.loc[inv_planning_df['feature'].isin([feature for feature in feature_list if feature not in 'pi_a2']), :]
    vp = sns.violinplot(x="feature", y="weight", hue="a_1", data=df_, ax=ax_,
                        order=feature_list,
                        hue_order=['C', 'D'],
                        split=True,
                        palette=dict(C='cornflowerblue', D='dimgrey'),
                        width=width,
                        # bw=0.1, cut=2, scale='area',
                        bw='scott', cut=0, scale='area',
                        linewidth=0.0,
                        saturation=1,
                        inner=None)
    ax_.set_ylim([0, 1])

    for i_feature, feature in enumerate(feature_list):
        for a1 in ['C', 'D']:
            xcoord = i_feature + {'C': -width / 4, 'D': width / 4}[a1]
            yval = inv_planning_df.loc[(inv_planning_df['feature'] == feature) & (inv_planning_df['a_1'] == a1), 'weight'].mean()
            # ax_.scatter([xcoord], [yval], s=40, color=dict(C='cornflowerblue', D='dimgrey')[a1], edgecolor='none', linewidth=2)
            ax_.scatter([xcoord], [yval], s=350, marker='_', color=dict(C='cornflowerblue', D='dimgrey')[a1], linewidth=1.5)

    if 'pi_a2' in feature_list:
        from matplotlib.patches import Wedge
        kwargs = dict()
        kwargs.update(transform=ax_.transAxes, clip_on=False)
        xcoord_ = feature_list.index('pi_a2')

        for a1 in ['C', 'D']:
            xvals, ycounts = np.unique(inv_planning_df.loc[(inv_planning_df['a_1'] == a1) & (inv_planning_df['feature'] == 'pi_a2'), 'weight'], return_counts=True)
            ycounts_total = ycounts.sum()
            # area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
            area = 1000
            sizes = area * ycounts / ycounts_total
            sizes = np.pi * (25 * ycounts / ycounts_total)**2  # 0 to 15 point radiuses

            radii = (ycounts / ycounts_total)
            angle = 90
            theta1, theta2 = angle, angle + 180

            # xcoord = 3 + {'C':-width/4, 'D':width/4}[a1]
            # for ixval,xval in enumerate(xvals):
            #     # ax_.scatter(np.repeat(xcoord, 6), xvals, s=sizes, alpha=0.05, edgecolor='none', linewidth=0, c={'C':'cornflowerblue', 'D':'dimgrey'}[a1], zorder=3)
            #     w1 = Wedge([xcoord,xval], radii[ixval], theta1, theta2, fc={'C':'cornflowerblue', 'D':'dimgrey'}[a1], **kwargs)
            #     ax_.add_artist(w1)

            sizes = np.pi * (ycounts / ycounts_total)**2  # 0 to 15 point radiuses
            xcoord = xcoord_ + {'C': -width / 4, 'D': width / 4}[a1]
            for ixval, xval in enumerate(xvals):
                marker_style = dict(color='none', linestyle=':', marker='o', markersize=50 * radii[ixval], markerfacecoloralt={'C': 'cornflowerblue', 'D': 'dimgrey'}[a1], alpha=0.5)
                ax_.plot(xcoord_, xval, fillstyle={'C': 'right', 'D': 'left'}[a1], **marker_style)

        # for a1 in ['C', 'D']:
        #     xvals,ycounts = np.unique(inv_planning_df.loc[ (inv_planning_df['a_1'] == a1) & (inv_planning_df['feature'] == 'pi_a2'), 'weight'], return_counts=True)
        #     ycounts_total =  ycounts.sum()
        #     # area = np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses
        #     area = 800
        #     sizes = area*ycounts/ycounts_total
        #     sizes = np.pi * (25 * ycounts/ycounts_total)**2 # 0 to 15 point radiuses

        #     xcoord = 3 + {'C':-width/4, 'D':width/4}[a1]
        #     ax_.scatter(np.repeat(xcoord, 6), xvals, s=sizes, alpha=0.3, edgecolor='none', linewidth=0, c={'C':'cornflowerblue', 'D':'dimgrey'}[a1], zorder=3)

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
    #################################
    ### individual diffs summary
    #################################
    import numpy as np

    # from iaa_model_comparison_wrapper import plot_inverse_planning_kde_base_split

    plt = plot_param['plt']
    sns = plot_param['sns']
    isInteractive = plot_param['isInteractive']
    showAllFigs = plot_param['showAllFigs']

    # stim_desc_dict9_shorthand = kwargs['stim_desc_dict9_shorthand']
    # iaf_model = kwargs['iaf_model']
    # df_wide9 = kwargs['df_wide9']
    # df_wide6 = kwargs['df_wide6']
    display_param = plot_param['display_param']

    # from webpypl_plot_inversePlanningRepu_specific_prior_induction import get_empirical_inverse_planning_priors_
    # df_wide9, shorthand9, shorthand_list9, distal_prior_param, empratings9, df_wide6, shorthand6, shorthand_list6, generic_prior_param, empratings6, rescale_intensities_ = get_empirical_inverse_planning_priors_(ppldata['empiricalInverseJudgmentsExtras_BaseGeneric'], ppldata['empiricalInverseJudgmentsExtras_RepuSpecific'])

    from webpypl_plotfun import plot_emo_comparison_scatter

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

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    font_caption = 7  # helvet 7pt
    font_body = 9  # helvet 9pt

    text_size_small = 10
    text_size_med = 11
    text_size_large = 12
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'left'}
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    import matplotlib.patches as patches
    import matplotlib.cbook as cbook

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # if not no_debug:
    #     ax_static = fig.add_subplot(gridspec[len(heights)-1, :])
    #     with cbook.get_sample_data(paths['code'] / f'sc.png') as image_file:
    #         image = plt.imread(image_file)
    #     im = ax_static.imshow(image)
    #     ax_static.axis('off')

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

    # axd[axid].text(axd[axid].get_xlim()[1]/2, axd[axid].get_ylim()[0]+25, stim_desc_dict9_shorthand[stimid], fontdict={'fontsize':text_size_large+2, 'fontweight':'normal', 'horizontalalignment':'center'})

    axd[axid].axis('off')

    ######### Inv Planning

    axid = 'invp_empir'
    axd[axid] = fig.add_subplot(gridspec[gs_row[f'g1'], gs_col['ip']])
    axd[axid] = plot_inverse_planning_kde_base_split_HIST_(plot_param, df_long_player_empir, feature_list=['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA', 'pi_a2'], width=0.9, ax_=axd[axid])

    axd[axid].set_title(f"Empirical ({df_long_player_empir.shape[0]//7})")
    axd[axid].set_ylabel(axd[axid].get_ylabel(), fontdict={'fontsize': text_size_large})  # , 'fontweight':'bold', 'horizontalalignment':'left'
    axd[axid].set_yticks([0, 0.5, 1])
    axd[axid].tick_params(axis="y", labelsize=text_size_large, pad=0)
    axd[axid].tick_params(axis="x", labelsize=text_size_large, pad=0)

    axd[axid].set_xticks([0, 0.5, 1, 2, 2.5, 3, 4, 4.5, 5, 5.99, 6])
    text_base = r'$\mathrm{base}$'
    text_repu = r'$\mathrm{repu}$'
    text_belief = r'$a_2=$C'
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
    axd[axid].set_ylabel(axd[axid].get_ylabel(), fontdict={'fontsize': text_size_large})  # , 'fontweight':'bold', 'horizontalalignment':'left'
    axd[axid].set_yticks([0, 0.5, 1])
    axd[axid].tick_params(axis="y", labelsize=text_size_large, pad=0)
    axd[axid].tick_params(axis="x", labelsize=text_size_large, pad=0)

    axd[axid].set_xticks([0, 0.5, 1, 2, 2.5, 3, 4, 4.5, 5, 5.99, 6])
    text_base = r'$base$'
    text_repu = r'$repu$'
    text_belief = r'$a_2=$C'
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

    # print(nobsdf)

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

            # probs_in = f"yes. np.sum(n_): {np.sum(n_)}.  np.min(probs_): {np.min(probs_)}, np.min(n_): {np.min(n_)}, np.max(n_): {np.max(n_)}; lcd: {lcd}"
            # print(f"np.sum(n_) :: {np.sum(n_)}")
            # print(f"lcd :: {lcd}")
            # print(f"lcd_mult :: {lcd_mult}")
            # print('')
            assert np.isclose(np.sum(probs_), 1.0)
            assert np.isclose(np.sum(n_), np.sum(lcd_mult * probs_ / np.min(probs_)))
        else:
            n_ = np.full([df_.shape[0], 1], 1, dtype=int)
            lcd = df_.shape[0]
            # probs_in = f"no. np.sum(n_): {np.sum(n_)}.  df_.shape[0]: {df_.shape[0]}, np.min(n_): {np.min(n_)}, np.max(n_): {np.max(n_)}; lcd: {lcd}"
            assert np.sum(n_) == df_.shape[0]

        expanded_dfs = list()
        if 'prob' in df_.columns.to_list():
            df_.drop(columns=['prob'], inplace=True)

        if lcd > 0:
            counter_ = 0
            # print(f'---- {lcd}')
            # print(f'np.sum(n_): {np.sum(n_)}')
            # print(f'np.max(n_): {np.max(n_)}')
            # print(f'round(np.max(n_)): {round(np.max(n_))}')
            # print(f"np.arange(1, int(round(np.max(n_)))+1): {np.arange(1, int(round(np.max(n_)))+1)}")
            # print(f"np.sum(n_) :: {np.sum(n_)}")

            for mult in np.arange(1, int(round(np.max(n_))) + 1):
                # print(f"  mult: {mult}")
                # print(f"    df_.shape :: {df_.loc[n_ == mult, :].shape[0]}" )
                # print(f"    np.sum(n_ == mult) :: {np.sum(n_ == mult)}")
                if np.sum(n_ == mult) > 0:
                    for _ in np.arange(1, mult + 1):
                        # print(f"        np.arange(1, mult+1): {np.arange(1, mult+1)}")
                        expanded_dfs.append(df_.loc[n_ == mult, :])
                        counter_ += df_.loc[n_ == mult, :].shape[0]
                    #     if mult < 4:
                    #         print('....')
                    #         print(expanded_dfs[-1])
                    # print(f"    counter : {counter_}")
                    # print(f"    expanded_dfs: {len(expanded_dfs)}")
                    # print('')

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


# %%

def composite_inverse_planning_split_violin(ppldata, paths, plotParam):
    import numpy as np
    import pandas as pd

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    from webpypl_pubplots_inverseplanning import convert_model_level1_level3_widedf_to_longdf_, make_wide_df_for_IP, aggregate_df6_df9
    from webpypl_plot_inversePlanningRepu_specific_prior_induction import get_empirical_inverse_planning_priors_

    def remap_pia2(emp_):
        assert emp_ in [0, 1, 2, 3, 4, 5]
        remap = np.array([11, 9, 7, 5, 3, 1]) / 12
        return remap[emp_]

    """
    NB 
    For the generic players, we only collected inverse planning ratings for the anonymous game (exp6).
    For the specific players, we only collected inverse planning ratings for the public game (exp9).
    
    Thus, to show the emprical ratings of the inverse planning features for the public game, I'm using the exp9 data (specific players) and summing over the specific players.
    
    Thus, for this figure:
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

    #################

    ### https://matplotlib.org/gallery/lines_bars_and_markers/scatter_piecharts.html#sphx-glr-gallery-lines-bars-and-markers-scatter-piecharts-py

    #################################
    ### inverse planning composite
    #################################

    plt = plotParam['plt']
    sns = plotParam['sns']
    isInteractive = plotParam['isInteractive']
    showAllFigs = plotParam['showAllFigs']

    # a1_labels = ppldata['labels']['decisions']

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
    horizontal_space_l2 = np.nan
    horizontal_space_r2 = np.nan
    graphic_repu_w = 3
    graphic_repu_w_diff = graphic_base_w - graphic_repu_w

    wid_tem2 = np.array([0, horizontal_space_l, y_label_w, graphic_repu_w, graphic_repu_w_diff, h_space1, y_label_w, graphic_repu_w, graphic_repu_w_diff, horizontal_space_r, margin])

    def cm2inch(value):
        return value / 2.54

    width_max = {'single': cm2inch(8.7), 'sc': cm2inch(11.4), 'double': cm2inch(17.8)}['single']

    # print(wid_tem2)
    # print("-------DEBUG:: np.isnan(wid_tem2)")  # TEMP
    # print(np.isnan(wid_tem2))  # TEMP
    # nans = np.isnan(wid_tem2)

    # print("-------DEBUG 1")
    # wid_tem2[nans] = (wid_tem1.sum() - wid_tem2[np.logical_not(nans)].sum()) / nans.sum()

    np.isclose(wid_tem1.sum(), wid_tem2.sum())

    wid_tem_total = (np.sum(wid_tem1) + np.sum(wid_tem2)) * 0.5
    widths1 = width_max * (wid_tem1 / wid_tem_total.sum())
    widths2 = width_max * (wid_tem2 / wid_tem_total.sum())

    # assert np.sum(widths1) == np.sum(widths2)

    title_width = 0.1
    total_height = (np.sum(heights1) + np.sum(heights2) + np.sum(heights3)) * 2.25
    total_width = (margin + title_width + np.sum(widths1)) * 2.25
    fig = plt.figure(figsize=(total_width, total_height), dpi=100)

    font_caption = 7  # helvet 7pt
    font_body = 9  # helvet 9pt

    # {'fontsize':11*scale_temp,'horizontalalignment':'left'}, rotation=-35, rotation_mode='anchor'
    text_size_small = 10
    text_size_medium = 11
    text_size_large = 12
    text_size_larger = 15
    fontdict_ticks = {'fontsize': text_size_small, 'horizontalalignment': 'left'}
    fontdict_axislab = {'fontsize': text_size_large, 'horizontalalignment': 'center'}

    # gridspec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0, top=1,bottom=0,right=1,left=0)

    gs0 = fig.add_gridspec(ncols=2, nrows=3, width_ratios=[margin + title_width, np.sum(widths1)], height_ratios=[np.sum(heights1), np.sum(heights2), np.sum(heights3)], wspace=0.0, hspace=0.0, top=1, bottom=0, right=1, left=0)

    gsbase = gs0[0, 1].subgridspec(ncols=len(widths1), nrows=len(heights1), width_ratios=widths1, height_ratios=heights1, wspace=0.0, hspace=0.0)
    gsrepu1 = gs0[1, 1].subgridspec(ncols=len(widths1), nrows=len(heights2), width_ratios=widths1, height_ratios=heights2, wspace=0.0, hspace=0.0)
    gsrepu2 = gs0[2, 1].subgridspec(ncols=len(widths2), nrows=len(heights2), width_ratios=widths2, height_ratios=heights3, wspace=0.0, hspace=0.0)

    gsbase_title = gs0[0, 0].subgridspec(ncols=2, nrows=len(heights1), width_ratios=[margin, title_width], height_ratios=heights1, wspace=0.0, hspace=0.0)
    gsrepu_title = gs0[1:, 0].subgridspec(ncols=2, nrows=1, width_ratios=[margin, title_width], wspace=0.0, hspace=0.0)

    # if not no_debug:
    #     ax_static = fig.add_subplot(gsbase[len(heights1) - 1, :])

    #     with cbook.get_sample_data(paths['code'] / f'single.png') as image_file:
    #         image = plt.imread(image_file)

    #     im = ax_static.imshow(image)
    #     ax_static.axis('off')

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
        # Line2D([0], [0], label='C', markerfacecolor='cornflowerblue', color='cornflowerblue', marker='o', alpha=1., markersize=7, linewidth=1.5),
        # Line2D([0], [0], label='D', markerfacecolor='dimgrey', color='dimgrey', marker='o', alpha=1., markersize=7, linewidth=1.5),
        Line2D([0], [0], label='C', markerfacecolor='cornflowerblue', color='cornflowerblue', marker='o', alpha=1., markersize=0, linewidth=1.5),
        Line2D([0], [0], label='D', markerfacecolor='dimgrey', color='dimgrey', marker='o', alpha=1., markersize=0, linewidth=1.5),
        Patch(label='Density:', facecolor='white', alpha=0.0, edgecolor='none', linewidth=0.),
        Patch(label='C', facecolor='cornflowerblue', alpha=0.5, edgecolor='none', linewidth=0.),
        Patch(label='D', facecolor='dimgrey', alpha=0.5, edgecolor='none', linewidth=0.),  # WIP fix color
    ]

    ax00.axis('off')
    ax00.legend(handles=legend_elements, loc='center', columnspacing=1.0, bbox_to_anchor=(0.5, 0.8), frameon=False, handletextpad=0.31, ncol=len(legend_elements), prop={'size': text_size_medium})  # bbox_to_anchor=(0.5, -0.3) ### , title="title"

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
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': text_size_large})  # , 'fontweight':'bold', 'horizontalalignment':'left'
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis="y", labelsize=text_size_large, pad=0)
        ax.tick_params(axis="x", labelsize=text_size_large, pad=0)

        ax.set_xticks([0, 1, 2, 3])
        text_base = r'$\mathrm{base}$'
        text_belief = r'$a_2=$C'
        # ax2.set_xticklabels((text_base, '$\omega_{Money}$', text_repu, text_base, '$\omega_{AIA}$', text_repu, text_base, '$\omega_{DIA}$', text_repu, 'Belief', text_belief))
        ax.set_xticklabels((text_base + '\n$Money$', text_base + '\n$AIA$', text_base + '\n$DIA$', 'Belief\n' + text_belief))

        # va_offset = -0.13 ### -0.13
        # va = [ 0, va_offset, 0, va_offset, 0, va_offset, 0, va_offset]
        # for t, y in zip( ax.get_xticklabels( ), va ):
        #     t.set_y( y )

    ################

    # gsrepu1
    ax0 = fig.add_subplot(gsrepu1[0, 0:8])
    # ax0.text(0.5, 0.1, f'Public Game', fontdict={'fontsize':text_size_large+3, 'fontweight':'bold', 'horizontalalignment':'center'})
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

        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': text_size_large})  # , 'fontweight':'bold', 'horizontalalignment':'left'
        ax.set_yticks([0, 0.5, 1])
        # ax.tick_params(axis="y", labelsize=text_size_large, pad=-5)
        # ax.tick_params(axis="x", labelsize=text_size_large, pad=-5)
        ax.tick_params(axis="y", labelsize=text_size_large, pad=0)
        ax.tick_params(axis="x", labelsize=text_size_large, pad=0)

        ax.set_xticks([0, 1, 2])
        text_base = r'$\mathrm{repu}$'
        text_belief = r'$a_2=$C'
        # ax2.set_xticklabels((text_base, '$\omega_{Money}$', text_repu, text_base, '$\omega_{AIA}$', text_repu, text_base, '$\omega_{DIA}$', text_repu, 'Belief', text_belief))
        ax.set_xticklabels((text_base + '\n$Money$', text_base + '\n$AIA$', text_base + '\n$DIA$'))

        # va_offset = -0.13 ### -0.13
        # va = [ 0, va_offset, 0, va_offset, 0, va_offset]
        # for t, y in zip( ax.get_xticklabels( ), va ):
        #     t.set_y( y )

    figsout = list()
    figsout.append((paths['figsOut'] / "composite_inv_plan_withprior-generic.pdf", fig, True))
    return figsout


def composite_inverse_planning_split_violin_specificplayer(ppldata, distal_prior_ppldata, paths, plot_param):
    import numpy as np
    import pandas as pd
    from webpypl_plot_inversePlanningRepu_specific_prior_induction import get_empirical_inverse_planning_priors_

    get_empirical_inverse_planning_priors_

    a1_labels = ppldata['labels']['decisions']

    # for stimid in list(distal_prior_ppldata.keys())[:1]:
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

        # feature_ = 'bMoney'
        # a1_ = 'D'
        # df_long_emp.loc[(df_long_emp['feature'] == feature_) & (df_long_emp['a_1'] == a1_), 'weight'] = 0.2
        # df_long_model.loc[(df_long_model['feature'] == feature_) & (df_long_model['a_1'] == a1_), 'weight'] = 0.2

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

    # '''Public game, Model <- model inference, public game (level3); priors used: empirical ratings - public game - specific players (exp9) ((summed over: specific player descriptions, pot sizes, a_1))'''
    # """model ratings of generic players public game"""
    # model_level3_longdf = convert_model_level1_level3_widedf_to_longdf_(make_wide_df_for_IP(ppldata['level3']), list(shorthand9.keys()))

    ##########

    assert np.min(ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']['pi_a2']) == 0
    assert np.max(ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']['pi_a2']) == 5

    # dfs = dict()
    # n_obs_counts_emp = dict()
    # for a1_emp in a1_labels:
    #     nobsdf = ppldata['empiricalInverseJudgments_RepuSpecific']['nobs'][a1_emp].copy()
    #     nobsdf.loc[:] = 0
    #     df_partial0 = ppldata['empiricalInverseJudgmentsExtras_RepuSpecific']['df_wide']
    #     cols = ['bMoney', 'rMoney', 'bAIA', 'rAIA', 'bDIA', 'rDIA', 'pi_a2', 'pot']
    #     df_partial = df_partial0.loc[(df_partial0['face'] == stimid) & (df_partial0['a_1'] == a1_emp), cols].copy()
    #     n_obs_counts_emp[a1_emp] = df_partial.shape[0]
    #     df_partial.loc[:, 'pi_a2'] = ((np.abs(df_partial.loc[:, 'pi_a2'].to_numpy() - 5) * 2) + 1) / 12
    #     df_partial.set_index('pot', inplace=True)
    #     emp_df_ = expand_df_(df_partial, nobsdf)
    #     dfs[a1_emp] = pd.melt(emp_df_.drop(columns=['prob']))
    #     dfs[a1_emp].columns = ['feature', 'weight']
    #     dfs[a1_emp]['a_1'] = a1_emp
    # df_long_emp = pd.concat(dfs.values())

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


# %%


def followup_analyses(cpar, ppldata, ppldata_exp3, distal_prior_ppldata, plotParam=None):

    import numpy as np
    import pandas as pd
    from webpypl_plot_inversePlanningRepu_specific_prior_induction import get_empirical_inverse_planning_priors_
    from webpypl_analysis_inverseplanning import run_inversePlanningAnalysis, inverse_planning_posterfigs_wrapper
    from webpypl_plotfun import printFigList

    paths = cpar.paths
    if plotParam is None:
        plot_param = cpar.plot_param
    else:
        plot_param = plotParam
    plt = plot_param['plt']
    sns = plot_param['sns']

    ###

    # %%

    #####
    ## make figs/composite_inv_plan_withprior.pdf
    #####

    figsout = printFigList(composite_inverse_planning_split_violin(ppldata, paths, plot_param), plot_param)

    # %%

    #####
    ## make figs/composite_inv_plan-specificPlayers/...
    #####

    composite_inverse_planning_split_violin_specificplayer(ppldata, distal_prior_ppldata, paths, plot_param)

    # %%

    #####
    ## make figs/feature_heatmaps/...
    #####

    """
    print the features in raw space (with to prospect transformation). e.g. if the webppl model uses a log1p prospect function, the subjective utility V will be transformed into U = exp(V) - 1, and U is printed here
    """

    features = ppldata['level4IAF']['CC'].columns.droplevel().drop('prob').to_list()
    outcomes = [key for key in list(ppldata['level4IAF'].keys()) if key != 'nobs']
    pots = np.unique(ppldata['level4IAF']['CC'].index.get_level_values(0).to_numpy())

    def ps_log1p(x): return np.sign(x) * np.log1p(np.abs(x))
    def ps_power05(x): return np.sign(x) * np.power(np.abs(x), 0.5)
    def ps_power025(x): return np.sign(x) * np.power(np.abs(x), 0.25)

    import re
    from webpypl_plotfun import plotHeatmapDistOverPots_prep_emoDict_, plotHeatmapDistOverPots_pub, plotHeatmapDistOverPots
    # p = re.compile(r'^(U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2lnpot|PEa2pot|PEa2raw|PEa2unval)')
    p = re.compile(r'^(EU\[.*\]|U\[.*\]|PE\[.*\]|CFa2\[.*\]|CFa1\[.*\]|PEa2lnpot|PEa2pot|PEa2raw|PEa2unval)')

    features_to_print = [s for s in features if p.match(s)]

    figsout = list()

    # for i_feature, feature in enumerate(features_to_print):
    #     for ps_label, ps_fn in [(None, None), ('ps_log1p', ps_log1p), ('ps_power05', ps_power05), ('ps_power025', ps_power025)]:
    #         for renormalize_color_within_outcome in [True, False]:

    #             if feature.startswith('PEa2'):
    #                 ps_fn_ = None
    #             else:
    #                 ps_fn_ = ps_fn

    #             figsout.append(plotHeatmapDistOverPots(**plotHeatmapDistOverPots_prep_emoDict_(ppldata['level4IAF'], i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=ps_fn_, transformation_label=ps_label), figoutpath=paths['figsOut'] / 'feature_heatmaps', renormalize_color_within_outcome=renormalize_color_within_outcome, plot_param=plot_param))

    # ### print objective features
    # for i_feature, feature in enumerate(features):
    #     for ps_label, ps_fn in [(None, None), ('ps_power025', ps_power025)]:
    #         for renormalize_color_within_outcome in [False]:

    #             if feature.startswith('PEa2'):
    #                 ps_fn_ = None
    #             else:
    #                 ps_fn_ = ps_fn

    #             figsout.append(plotHeatmapDistOverPots(**plotHeatmapDistOverPots_prep_emoDict_(ppldata['level4IAF'], i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=ps_fn_, transformation_label=ps_label), figoutpath=paths['figsOut'] / 'feature_heatmaps' / 'objective', renormalize_color_within_outcome=renormalize_color_within_outcome, plot_param=plot_param))

    ### print  features
    if True:
        for i_feature, feature in enumerate(features_to_print):  # features_to_print
            for ps_label, ps_fn in [('ps_power05', ps_power05)]:
                for renormalize_color_within_outcome in [True, False]:

                    figsout = list()
                    if feature.startswith('PEa2'):
                        ps_fn_ = None
                    else:
                        ps_fn_ = ps_fn

                    figsout.append(plotHeatmapDistOverPots_pub(**plotHeatmapDistOverPots_prep_emoDict_(ppldata['level4IAF'], i_feature=i_feature, feature=feature, outcomes=outcomes, pots=pots, transformation_fn=ps_fn_, transformation_label=ps_label), figoutpath=paths['figsOut'] / 'feature_heatmaps' / 'pub', renormalize_color_within_outcome=renormalize_color_within_outcome, plot_param=plot_param))

                    figsout = printFigList(figsout, plot_param)

    # %%

    #####
    ## Old inverse planning figures
    ## make figs/inversePlanning/...
    #####

    _ = printFigList(run_inversePlanningAnalysis(ppldata, paths, plot_param['display_param'], {**plot_param, **cpar.plot_control['set1aba']}), plot_param)

    ######

    _ = printFigList(inverse_planning_posterfigs_wrapper(ppldata, paths, plot_param['display_param'], {**plot_param, **cpar.plot_control['set1aba']}), plot_param)

    # %%

    #########################
    ### Heat maps of model and empirical preferences
    #########################
    import scipy.stats
    from webpypl import marginalizeContinuous, unweightProbabilities

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
        # for feature in ppldata['level4IAF']['CC']['compositeWeights']:
        for i_feature, featurei in enumerate(features):
            for j_feature, featurej in enumerate(features):

                bb = unweighted_df.loc[:, featurei]
                aa = unweighted_df.loc[:, featurej]

                corrmat[i_feature, j_feature] = scipy.stats.pearsonr(aa, bb)[0]
                if i_feature == j_feature:
                    corrmat[i_feature, j_feature] = np.nan

        iu1 = np.triu_indices(len(features))
        mask = np.zeros(corrmat.shape)
        # mask[iu1] = 1

        xlabels = features.copy()
        xlabels[-1] = None
        ylabels = features.copy()
        ylabels[0] = None

        ax = axs[0, ia1]

        # fig, ax = plt.subplots(figsize=(4,4))

        # ax = sns.heatmap(A_, annot=True, linewidths=.5, center=0, vmin=-1*A_.abs().max().max(), vmax=A_.abs().max().max(), cmap='coolwarm', annot_kws={"size": 10}, fmt=".2f", ax=ax)
        ax = sns.heatmap(corrmat, mask=mask, square=True, cmap='coolwarm', ax=ax)
        ax.set_xticklabels(xlabels, rotation=-35, ha='left', rotation_mode='anchor')
        ax.set_yticklabels(ylabels, rotation=0, ha='right')
        ax.set_ylim((len(features), 0))

        ax.set_title(f"Model {a1}")

        # figsout.append((paths['figsOut'] / f"omega_corr_modelLvl3_publicGame_{a1}.pdf", fig, True))

        plt.close(fig)

    ########

    df_wide9, _, _, _, _, _, _, _, _, _, _ = get_empirical_inverse_planning_priors_(ppldata['empiricalInverseJudgmentsExtras_BaseGeneric'], ppldata['empiricalInverseJudgmentsExtras_RepuSpecific'])

    # plt.close('all')
    for ia1, a1 in enumerate(['C', 'D']):

        df_temp = df_wide9.loc[df_wide9['a_1'] == a1, :].copy()

        df_temp.drop(columns=df_temp.columns[~df_temp.columns.isin(features)], inplace=True)

        corrmat = np.full((len(features), len(features)), np.nan)
        # for feature in ppldata['level4IAF']['CC']['compositeWeights']:
        for i_feature, featurei in enumerate(features):
            for j_feature, featurej in enumerate(features):

                bb = df_temp.loc[:, featurei]
                aa = df_temp.loc[:, featurej]

                corrmat[i_feature, j_feature] = scipy.stats.pearsonr(aa, bb)[0]
                if i_feature == j_feature:
                    corrmat[i_feature, j_feature] = np.nan

        iu1 = np.triu_indices(len(features))
        mask = np.zeros(corrmat.shape)
        # mask[iu1] = 1

        xlabels = features.copy()
        xlabels[-1] = None
        ylabels = features.copy()
        ylabels[0] = None

        ax = axs[1, ia1]
        # fig, ax = plt.subplots(figsize=(4,4))

        # ax = sns.heatmap(A_, annot=True, linewidths=.5, center=0, vmin=-1*A_.abs().max().max(), vmax=A_.abs().max().max(), cmap='coolwarm', annot_kws={"size": 10}, fmt=".2f", ax=ax)
        ax = sns.heatmap(corrmat, mask=mask, square=True, cmap='coolwarm', ax=ax)
        ax.set_xticklabels(xlabels, rotation=-35, ha='left', rotation_mode='anchor')
        ax.set_yticklabels(ylabels, rotation=0, ha='right')
        ax.set_ylim((len(features), 0))

        # figsout.append((paths['figsOut'] / f"omega_corr_empirical_publicGame_{a1}.pdf", fig, True))

        # plt.close(fig)

        ax.set_title(f"Empirical {a1}")

    plt.tight_layout()

    figsout.append((paths['figsOut'] / f"omega_corr_publicGame_modelandempirical.pdf", fig, True))
    plt.close(fig)
    figsout = printFigList(figsout, plot_param)

    plt.close('all')

    print('fig printing complete')
