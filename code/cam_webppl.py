#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_webppl.py
"""


import argparse


def getdata_webppl(cpar, print_figs=True):

    from cam_webppl_gendata_wrapper import initialize_wrapper
    from cam_plotfn_webppl import followup_analyses

    """for testing
    cpar_full = dependency_wppl['full']['cpar']
    cpar = cpar_full
    cpar.cache['webppl']['runModel'] = False
    cpar.cache['webppl']['loadpickle'] = True
    cpar.cache['webppl']['removeOldData'] = False
    """

    print('---initialize_wrapper---')
    ppldata, distal_prior_ppldata = initialize_wrapper(cpar)

    if print_figs:
        print('---followup_analyses---')
        followup_analyses(cpar=cpar, ppldata=ppldata, distal_prior_ppldata=distal_prior_ppldata)

    print('---getdata_webppl done---')

    return 0


def load_cpar(data_pickle_path):
    import dill as pickle

    with open(data_pickle_path, 'rb') as f:
        cpar = pickle.load(f)

    return cpar


def _cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    parser.add_argument('data_pickle_path', type=str)
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    import sys
    print(f'\n---Received {sys.argv} from shell---\n')
    exit_status = 1
    try:
        exit_status = getdata_webppl(load_cpar(**_cli()))
    except Exception as e:
        print(f'Got exception of type {type(e)}: {e}')
        print("Not sure what happened, so it's not safe to continue -- crashing the script!")
        sys.exit(1)
    finally:
        print(f"-- {getdata_webppl.__qualname__} from {__file__} ended with exit code {exit_status} --")

    if exit_status == 0:
        print(f"--SCRIPT COMPLETED SUCCESSFULLY--")
    else:
        print(f"--SOME ISSUE, EXITING:: {exit_status}--")

    sys.exit(exit_status)
