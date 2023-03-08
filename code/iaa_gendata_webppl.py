#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".py
"""

import argparse
from iaa_control_utils import parse_param


def gendata_webppl(cpar):
    """."""

    from webpypl_initialize_wrapper import initialize_wrapper
    from iaa21_runwebppl_followup import followup_analyses

    """
    cpar = cpar_full
    cpar.cache['webppl']['runModel'] = False
    cpar.cache['webppl']['loadpickle'] = True
    cpar.cache['webppl']['removeOldData'] = False
    """

    print('---initialize_wrapper---')
    ppldata, ppldata_exp3, distal_prior_ppldata = initialize_wrapper(cpar)

    print('---followup_analyses---')
    followup_analyses(cpar, ppldata, ppldata_exp3, distal_prior_ppldata)

    print('gendata_webppl done')
    return 0


def load_data_cpar(data_pickle_path):
    import dill as pickle

    with open(data_pickle_path, 'rb') as f:
        cpar = pickle.load(f)

    return cpar


def load_data(data_pickle_path):
    import dill as pickle

    with open(data_pickle_path, 'rb') as f:
        update_dict = pickle.load(f)

    return update_dict


def _cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    parser.add_argument('data_pickle_path', action="store")
    args = parser.parse_args()
    print(args)
    return vars(args)


if __name__ == '__main__':
    import sys
    print(f'\n---Received {sys.argv} from shell---\n')
    exit_status = 1
    try:
        # gendata_webppl(parse_param(load_data(**_cli())))
        exit_status = gendata_webppl(load_data_cpar(**_cli()))
    except Exception as e:
        print(f'Got exception of type {type(e)}: {e}')
        print("Not sure what happened, so it's not safe to continue -- crashing the script!")
        sys.exit(1)
    finally:
        print(f"--gendata_webppl() from iaa_gendata_webppl.py ended with exit code {exit_status}--")

    if exit_status == 0:
        print("--WRAPPER gendata_webppl() COMPLETED SUCCESSFULLY--")
    else:
        print(f"--SOME ISSUE, EXITING:: {exit_status}--")

    sys.exit(exit_status)
