#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_main.py
"""

import sys
import argparse


def main(projectdir=None):

    # %%

    isInteractive = False
    try:
        if __IPYTHON__:  # type: ignore
            get_ipython().run_line_magic('matplotlib', 'inline')  # type: ignore
            get_ipython().run_line_magic('load_ext', 'autoreload')  # type: ignore
            get_ipython().run_line_magic('autoreload', '2')  # type: ignore
            isInteractive = True
    except NameError:
        isInteractive = False

    from pathlib import Path
    import numpy as np
    from cam_utils import check_sbatch_job
    from cam_webppl_config import webppl_datagen_config
    from cam_assemble_torch_models import run_cv, relaunch_jobs_for_missing_cv_results, run_tt, relaunch_jobs_for_missing_tt_results, cleanup_torch_runs
    from cam_summary_analysis import compile_results, plot_summary

    # %%

    projectbase_path = None
    dataout_path = None
    if projectdir is not None:
        projectbase_path = Path(projectdir)
        assert projectbase_path.is_dir()
        dataout_path = Path(projectdir) / 'dataOut'
    print(f"\nprojectdir: {projectdir}\n")
    print(f"cwd: {Path().cwd().expanduser()}")
    print(f"home: {Path('~').expanduser()}")

    # %%

    ########
    # Run WebPPL models to generate computed appraisal samples
    # If webppl_cache.pkl files already exist, they will be loaded in lieu of running WebPPL.
    ########

    seed = 100

    spec_webppl = dict(
        exp_name='cam',
        inf_param='MCMC3k',
        prior_a2C_cmd=0.4,
        kde_width=0.01,
        generic_repu_values_from='internal',
        distal_repu_values_from='empiricalExpectation',
        refpoint_type='Norm',
        seed=seed,
    )
    hotpatch_cacheddata = True  # Use cached data if it exists

    dependency_wppl = webppl_datagen_config(spec_webppl, hotpatch_cacheddata=hotpatch_cacheddata, projectbase_path=projectbase_path, dataout_path=dataout_path)

    wppldatacache_full = dependency_wppl['full']['cpar'].paths['wpplDataCache']
    cpar_path_full = wppldatacache_full.parent / 'cpar.dill'
    wppldatacache_priors = dependency_wppl['priorsOnly']['cpar'].paths['wpplDataCache']
    cpar_path_priors = wppldatacache_priors.parent / 'cpar.dill'

    base_path = wppldatacache_full.parents[1]

    # %%

    specificplayers_desc_shorthand = {
        '239_1': 'Hospital nurse',
        '240_1': 'Aspiring rapper',
        '246_1': 'Corporate lawyer',
        '247_1': 'Eldercare worker',
        '249_2': 'Retired waitress',
        '254_2': 'Janitor',
        '255_2': 'Global health NGO',
        '256_2': 'Swimming coach',
        '263_1': 'Police officer',
        '263_2': 'Special edu. teacher',
        '266_2': 'City council member',
        '269_2': 'Boxing coach',
        '270_1': 'English teacher',
        '272_2': 'Software eng. \\male',
        '276_1': 'Dr. w/o Borders',
        '278_2': 'Software eng. \\female',
        '280_1': 'Stock broker',
        '281_1': 'Petrol exec.',
        '285_1': 'Investment analyst',
        '285_2': 'Design student'
    }
    specificplayers_stimids = sorted(list(specificplayers_desc_shorthand.keys()))

    ########
    # Specify the PyTorch models
    ########

    results_prefix = 'torchdata'
    niters = int(2000)
    logit_k = 0.4
    laplace_scales = np.arange(100., 820., 20.)
    n_tt_mixes = 40
    n_tt_folds = 4
    n_cv_random_reinits = 2
    n_tt_random_reinits = 5

    outpath_base = base_path / f'{results_prefix}_nfold{n_tt_folds}'

    output_paths = dict(
        datafolds=outpath_base / f'datafolds_nmix{n_tt_mixes}-nfold{n_tt_folds}.pkl',
        modelspecs_cv=outpath_base / f'modelspecs-cv_nmix{n_tt_mixes}-nfold{n_tt_folds}-nricv{n_cv_random_reinits}.pkl',
        modelspecs_tt=outpath_base / f'modelspecs-tt_nmix{n_tt_mixes}-nfold{n_tt_folds}-nritt{n_tt_random_reinits}-nricv{n_cv_random_reinits}.pkl',
        torch_cvres=outpath_base / f'torch_cvres_cache-nmix{n_tt_mixes}-nfold{n_tt_folds}-nritt{n_cv_random_reinits}.pkl',
        torch_ttres=outpath_base / f'torch_ttres_cache-nmix{n_tt_mixes}-nfold{n_tt_folds}-nritt{n_tt_random_reinits}.pkl',
    )

    # %%

    progress = dict(
        webppl=dict(status=None),
        torch_cv=dict(status=None),
        torch_tt=dict(status=None),
        torch=dict(status=None),
    )

    ########
    # Check if the WebPPL data exists
    ########

    satisfied = True
    if dependency_wppl['full']['depend'] is not None:
        if not check_sbatch_job(dependency_wppl['full']['depend']):
            satisfied = False
            progress['webppl']['status'] = 'running'
    if dependency_wppl['priorsOnly']['depend'] is not None:
        if not check_sbatch_job(dependency_wppl['priorsOnly']['depend']):
            satisfied = False
            progress['webppl']['status'] = 'running'
    for file_ in [base_path, wppldatacache_full, cpar_path_full, wppldatacache_priors, cpar_path_priors]:
        if not file_.exists():
            print(f"File {file_} does not exist")
            satisfied = False
            progress['webppl']['status'] = 'failed'
    if satisfied:
        progress['webppl']['status'] = 'complete'
    else:
        print(f"WebPPL data not present. Status: {progress['webppl']['status']}")

    if progress['webppl']['status'] == 'failed':
        raise RuntimeError("WebPPL data does not exist. Not continuing.")

    ########
    # Check if PyTorch models have been run
    ########

    if progress['webppl']['status'] == 'complete':
        if output_paths['torch_ttres'].is_file():
            print(f"model fit complete")
            progress['torch_cv']['status'] = 'complete'
            progress['torch_tt']['status'] = 'complete'
            progress['torch']['status'] = 'complete'
        else:
            progress['torch']['status'] = 'none'

    if not (progress['torch']['status'] == 'complete'):
        if output_paths['modelspecs_tt'].is_file():
            progress['torch_cv']['status'] = 'complete'
        elif (
            output_paths['datafolds'].is_file()
            and output_paths['modelspecs_cv'].is_file()
        ):
            progress['torch_cv']['status'] = 'started'
        else:
            print(f"torch cv not started")
            progress['torch_cv']['status'] = 'none'

        if progress['torch_cv']['status'] == 'started':
            print(f"missing cv results")
            nmissing_cv, _ = relaunch_jobs_for_missing_cv_results(modelspecs_cv_path=output_paths['modelspecs_cv'], run_missing=False)
            if nmissing_cv == 0:
                progress['torch_cv']['status'] = 'complete'

        if progress['torch_cv']['status'] == 'complete':
            if output_paths['modelspecs_tt'].is_file():
                progress['torch_tt']['status'] = 'started'
            else:
                print(f"torch tt not started")
                progress['torch_tt']['status'] = 'none'

        if progress['torch_tt']['status'] == 'started':
            print(f"missing tt results")
            nmissing_tt, _ = relaunch_jobs_for_missing_tt_results(modelspecs_tt_path=output_paths['modelspecs_tt'], run_missing=False)
            if nmissing_tt == 0:
                progress['torch_tt']['status'] = 'complete'

    # %%

    if progress['torch_cv']['status'] == 'none':
        ########
        # Run cross-validation
        # Train models on generic player data, cross-validate on subset of specific players, test on held-out specific players
        ########

        dependencies_cv_list = run_cv(outpath_base=outpath_base, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors, specificplayers_stimids=specificplayers_stimids, laplace_scales=laplace_scales, logit_k=logit_k, niters=niters, n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, n_cv_random_reinits=n_cv_random_reinits, seed=seed)

    elif progress['torch_cv']['status'] == 'started':
        ########
        # check if any results are missing, if so, launch jobs to generate them
        # if nmissing is 0, all cv jobs are finished, can move on to run_tt()
        ########

        nmissing, dependencies_cv_relaunch_list = relaunch_jobs_for_missing_cv_results(modelspecs_cv_path=output_paths['modelspecs_cv'], run_missing=True)

    elif progress['torch_tt']['status'] == 'none':
        ########
        # once all cv results exist, launch tt jobs
        # test on held-out specific players
        ########

        nmissing_cv, _ = relaunch_jobs_for_missing_cv_results(modelspecs_cv_path=output_paths['modelspecs_cv'], run_missing=False)
        assert nmissing_cv == 0, "Some cv results are missing. Run relaunch_jobs_for_missing_cv_results() to generate them."
        cleanup_torch_runs(output_paths['modelspecs_cv'])

        dependencies_tt_list = run_tt(n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, n_tt_random_reinits=n_tt_random_reinits, n_cv_random_reinits=n_cv_random_reinits, laplace_scales=laplace_scales, seed=seed, niters=niters, outpath_base=outpath_base, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors)

    elif progress['torch_tt']['status'] == 'started':
        ########
        # check if any results are missing, if so, launch jobs to generate them
        # if nmissing is 0, all tt jobs are finished
        ########

        nmissing, dependencies_tt_relaunch_list = relaunch_jobs_for_missing_tt_results(modelspecs_tt_path=output_paths['modelspecs_tt'], run_missing=True)

    # %%
    if progress['torch_tt']['status'] == 'complete':
        ########
        # Once all models have been run, aggregate data
        ########
        print("All pytorch models have been run. Moving on to summary analysis.")

        if progress['torch']['status'] == 'none':
            nmissing, _ = relaunch_jobs_for_missing_tt_results(modelspecs_tt_path=output_paths['modelspecs_tt'], run_missing=False)
            assert nmissing == 0, "Some test results are missing. Run relaunch_jobs_for_missing_tt_results() to generate them."
            cleanup_torch_runs(output_paths['modelspecs_tt'])

        cialpha = 0.05
        cialpha_betaparam = 0.01
        res = compile_results(n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, n_tt_random_reinits=n_tt_random_reinits, seed=seed, outpath_base=outpath_base, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors, specificplayers_desc_shorthand=specificplayers_desc_shorthand, cialpha=cialpha, cialpha_betaparam=cialpha_betaparam)

        # %%

        ########
        # Plot the results
        ########

        plot_summary(res, outpath_base=outpath_base, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors, specificplayers_desc_shorthand=specificplayers_desc_shorthand, cialpha=cialpha, cialpha_betaparam=cialpha_betaparam)

    # %%

    return 0


def _cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    parser.add_argument('-p', '--projectdir', type=str, help="Path to the project directory")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    print(f'\n-- Received {sys.argv} from shell --\n')

    exit_status = 1
    try:
        print('STARTING')
        exit_status = main(**_cli())
    except Exception as e:
        print(f'Got exception of type {type(e)}: {e}')
        print("Not sure what happened, so it's not safe to continue -- crashing the script!")
        sys.exit(1)
    finally:
        print(f"-- {main.__qualname__} from {__file__} ended with exit code {exit_status} --")

    if exit_status == 0:
        print("-- SCRIPT COMPLETED SUCCESSFULLY --")
    else:
        print(f"-- SOME ISSUE, EXITING:: {exit_status} --")

    sys.exit(exit_status)
