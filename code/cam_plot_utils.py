#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_plot_utils.py
"""


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

    outcomes = ['CC', 'CD', 'DC', 'DD']

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
        'outcomes': outcomes,
    }

    return plot_param_
