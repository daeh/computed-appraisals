#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""react_main.py
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
    from react_datagen_wrapper import wrapper_gen_webppl_data
    from react_collect_pytorch_cvresults import run_cv, relaunch_jobs_for_missing_results, run_tt, relaunch_jobs_for_missing_results_tt
    from react_summary_analysis import compile_results, plot_summary

    # %%

    projectbase_path = None
    dataout_path = None
    if projectdir is not None:
        projectbase_path = Path(projectdir)
        assert projectbase_path.is_dir()
        dataout_path = Path(projectdir) / 'dataOut'
    print(f"\nprojectdir: {projectdir}\n")

    # %%

    ########
    ### Run WebPPL models to generate inferred appraisals
    # If cacheddatain*.pkl files already exist, they will be loaded in lieu of running WebPPL.
    ########

    spec_webppl = dict(
        prefix='react',
        prospect_fn=['log', 'powerfn', 'nopsscale'][0],
        inf_param=['rejection', 'MCMC1', 'incrementalMH1', 'meddebug', 'rapidtest'][1],
        cfa2_form=['normedCFa1', 'unnoredCFa1'][0],
        prior_form=['kde', 'kdemixture', 'broadkdemixture', 'multivarkdemixture', 'multivarkdemixtureLogit', 'type1', 'agg69'][3],
        prior_a2C_cmd=['uniform', 'split', 'splitinv', 0.5, 0.4, 0.6, 0.1, 0.9, 0.45, 0.55, 0.35, 0.65, 0.01, 0.99, 1.0, 0.0][4],
        kde_width=[0, 0.01, 0.02, 0.04, 0.1][1],
        repu_values_from=['internal', 'empiricalKDE', 'empiricalExpectation', 'binary'][2],
        refpoint_type=['Power', 'Norm', 'Gamma', 'None', 'Norm350'][1],
        seed=100,
    )
    hotpatch_cacheddata = True  # Use cached data if it exists

    dependency_wppl = wrapper_gen_webppl_data(spec_webppl, hotpatch_cacheddata=hotpatch_cacheddata, projectbase_path=projectbase_path, dataout_path=dataout_path)

    wppldatacache_full = dependency_wppl['full']['cpar'].paths['wpplDataCache']
    cpar_path_full = wppldatacache_full.parent / 'cpar.dill'
    wppldatacache_priors = dependency_wppl['priorsOnly']['cpar'].paths['wpplDataCache']
    cpar_path_priors = wppldatacache_priors.parent / 'cpar.dill'

    base_path = wppldatacache_full.parents[1]

    ########
    ### Check if the WebPPL data exists
    ########

    satisfied = True
    if dependency_wppl['full']['depend'] is not None:
        satisfied = False
    if dependency_wppl['priorsOnly']['depend'] is not None:
        satisfied = False
    for file_ in [base_path, wppldatacache_full, cpar_path_full, wppldatacache_priors, cpar_path_priors]:
        if not file_.exists():
            satisfied = False
            print(f"File {file_} does not exist")
    if not satisfied:
        raise RuntimeError("WebPPL data does not exist. Not continuing.")

    # %%

    ########
    ### Specify the PyTorch models
    ########

    specificplayers_desc_shorthand = {
        '239_1': 'Hospital Nurse',
        '240_1': 'Aspiring rapper',
        '246_1': 'Corporate lawyer',
        '247_1': 'Eldercare worker',
        '249_2': 'Retired waitress',
        '254_2': 'Janitor',
        '255_2': 'Global health NGO',
        '256_2': 'Swimming coach',
        '263_1': 'Police officer',
        '263_2': 'Special edu teacher',
        '266_2': 'City council member',
        '269_2': 'Boxing coach',
        '270_1': 'English teacher',
        '272_2': 'Software eng. \\male',
        '276_1': 'Dr. w/o Borders',
        '278_2': 'Software eng. \\female',
        '280_1': 'Stock broker',
        '281_1': 'Petrol exec',
        '285_1': 'Investment analyst',
        '285_2': 'Design student'
    }
    specificplayers_stimids = sorted(list(specificplayers_desc_shorthand.keys()))
    seed = 1
    niters = int(2000)
    logit_k = 0.4
    laplace_scales = [100., 120., 140., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280., 290., 300.]
    n_tt_mixes = 40
    n_tt_folds = 4

    outpath_base = base_path / f'torch_results-{n_tt_folds}'

    # %%

    ########
    ### Check PyTorch models have been run
    ########

    if not (outpath_base / f'ttdatadump-{n_tt_mixes}-{n_tt_folds}.pkl').is_file():
        '''
        if ttdatadump_path exists, can go straight to summary_analysis(). Otherwise run_tt() either needs to be run, or it has been run and the results need to be collected.
        ttdatadump_path = outpath_base / f'ttdatadump-{n_tt_mixes}-{n_tt_folds}.pkl'
        '''

        ########
        ### If not, run cross-validation
        # Train models on generic player data, cross-validate on subset of specific players, test on held-out specific players
        ########

        dependency_cv = None
        if not (
            (outpath_base / f'datafolds_mixes-{n_tt_mixes}-{n_tt_folds}.pkl').is_file() and
                (outpath_base / f'sbatch_modelspecs_cv-{n_tt_mixes}-{n_tt_folds}.pkl').is_file()):

            dependency_cv = run_cv(base_path=base_path, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors, specificplayers_stimids=specificplayers_stimids, laplace_scales=laplace_scales, logit_k=logit_k, niters=niters, n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, seed=seed, dependency=None)

            ### wait for models to be trained ###
            raise RuntimeError("Cross-validation (train) has not been run. Jobs have been submitted. Wait until they complete before continuing.")

        else:
            print(f"Cross-validation (train) has been initiated. Checking for missing results.")

        ########
        ### check if any results are missing, if so, launch jobs to generate them
        # if relaunch_jobs_for_missing_results() returns 0, all cv jobs are finished, can move on to run_tt()
        ########

        run_missing = False
        nmissing, dependency_cv_relaunch = relaunch_jobs_for_missing_results(outpath_base=outpath_base, n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, dependency=dependency_cv, run_missing=run_missing)  # TODO dependency, delete extra

        if nmissing > 0:
            raise RuntimeError("Cross-validation (train) has not been run. Jobs have been submitted. Wait until they complete before continuing.")

        ########
        ### once all cv results exist, launch tt jobs
        # test on held-out specific players
        ########

        if not (outpath_base / f'datafolds_mixes-{n_tt_mixes}-{n_tt_folds}.pkl').is_file():

            dependency_tt = run_tt(n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, seed=seed, niters=niters, outpath_base=outpath_base, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors)

            ### wait for models to be trained ###
            raise RuntimeError("Cross-validation (test) has not been run. Jobs have been submitted. Wait until they complete before continuing.")

        else:
            print(f"Cross-validation (test) has been initiated. Checking for missing results.")

        ########
        ### check if any results are missing, if so, launch jobs to generate them
        # if relaunch_jobs_for_missing_results_tt() returns 0, all tt jobs are finished
        ########

        nmissing, dependency_tt_relaunch = relaunch_jobs_for_missing_results_tt(outpath_base=outpath_base, n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, dependency=None, run_missing=True, delete_extra=True)  # TODO dependency=dependency_cv, delete extra

        if nmissing > 0:
            raise RuntimeError("Cross-validation (test) has not been run. Jobs have been submitted. Wait until they complete before continuing.")

    else:
        print("Cross-validation (train) and (test) have been run. Moving on to summary analysis.")

    ########
    ### Once all models have been run, collect data
    ########

    cialpha = 0.05
    res = compile_results(n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, seed=seed, outpath_base=outpath_base, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors, specificplayers_desc_shorthand=specificplayers_desc_shorthand, cialpha=cialpha)

    # %%

    ########
    ### Plot the results
    ########

    plot_summary(res, n_tt_mixes=n_tt_mixes, n_tt_folds=n_tt_folds, seed=seed, outpath_base=outpath_base, cpar_path_full=cpar_path_full, cpar_path_priors=cpar_path_priors, specificplayers_desc_shorthand=specificplayers_desc_shorthand, cialpha=cialpha)

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
        print(f"-- main() from react_main.py ended with exit code {exit_status} --")

    if exit_status == 0:
        print("-- SCRIPT COMPLETED SUCCESSFULLY --")
    else:
        print(f"-- SOME ISSUE, EXITING:: {exit_status} --")

    sys.exit(exit_status)
