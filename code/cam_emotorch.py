#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_emotorch.py
"""

import argparse
import sys
import os


class EmoTorch():

    def __init__(self, verbose=True):

        self.cfg = None

        self.pickle_path = None

        self.dataout_base_path = None

        ###

        self.torchdata = dict()
        self.torchdata_cache_path = dict()

        self.data_transform = None
        self.data_transform_desc = None
        self.preprocdata = None
        self.preprocdata_cache_path = None

        self.optimized = None
        self.optimized_cache_path = None

        self.model_param = None
        self.model_name_brief = None

        self.verbose = verbose

        ####

    def dump(self, pickle_path, minimal=True):
        import dill
        self.pickle_path = pickle_path
        if minimal:
            saveattr = [
                'cfg',
                'pickle_path',
                'dataout_base_path',
                # 'torchdata',
                'torchdata_cache_path',
                # 'data_transform',
                'data_transform_desc',
                # 'preprocdata',
                'preprocdata_cache_path',
                # 'optimized',
                'optimized_cache_path',
                'model_param',
                'model_name_brief',
                # 'verbose',
            ]
        else:
            saveattr = [
                'cfg',
                'pickle_path',
                'dataout_base_path',
                'torchdata',
                # 'torchdata_cache_path',
                # 'data_transform',
                'data_transform_desc',
                'preprocdata',
                # 'preprocdata_cache_path',
                'optimized',
                'optimized_cache_path',
                'model_param',
                'model_name_brief',
                # 'verbose',
            ]
        objdict = dict()
        for key in saveattr:
            objdict[key] = getattr(self, key)

        objdict['pickle_path'] = pickle_path
        with open(self.pickle_path, 'wb') as f:
            dill.dump(objdict, f, protocol=-4)
        print(f'dumped to {self.pickle_path}')

    def load(self, pickle_path, minimal=True, use_relative_path=False):
        from pathlib import Path
        import dill

        if isinstance(pickle_path, str):
            pickle_path = Path(pickle_path)

        self.pickle_path = pickle_path

        if self.verbose:
            print(f'loading from {self.pickle_path}\n')

        with open(self.pickle_path, 'rb') as f:
            objdict = dill.load(f)

        for key, val in objdict.items():
            setattr(self, key, val)

        if self.optimized_cache_path is not None:
            if use_relative_path:
                optimized_cache_path = self.pickle_path.parent / self.optimized_cache_path.name
                assert optimized_cache_path.is_file()
                if optimized_cache_path != self.optimized_cache_path:
                    self._optimized_cache_path_original = str(self.optimized_cache_path)
                    self.optimized_cache_path = optimized_cache_path
            assert self.optimized_cache_path.is_file()
            with open(self.optimized_cache_path, 'rb') as f:
                optimized = dill.load(f)
            setattr(self, 'optimized', optimized)

        if not minimal:
            if self.preprocdata is not None or self.preprocdata_cache_path is not None:
                if self.verbose:
                    print(f">>fitting data_transform")
                self._preprocess_load_transform()
            else:
                if self.verbose:
                    print(f">>no preprocdata found, skipping data_transform fitting")

            _ = self.get_torchdata()

    def init_cfg(self, cfg, prefix=''):
        self.cfg = cfg
        self.model_name_brief = cfg['model_id_brief']
        self.model_param = cfg['model_param']

        ###########################

        model_param_str = "".join([f"_{k_}-{v_}" for k_, v_ in self.model_param.items()])

        prefix_ = f"{prefix}-" if prefix else ""
        results_fname_base = f"{prefix_}{self.model_name_brief}_{model_param_str}"

        self.dataout_base_path = cfg['dout_base_path'] / results_fname_base

        ###########################

        if self.verbose:
            print(f"\n\n=== model initalized at {self.dataout_base_path} ===\n")

    def get_torchdata(self):
        import pickle
        if len(self.torchdata) == 0 and len(self.torchdata_cache_path) > 0:
            for data_label, data_path in self.torchdata_cache_path.items():
                if self.verbose:
                    print(f">>loading torchdata {data_label} from {data_path}")
                assert data_path.is_file(), f"ERROR torchdata_cache_path for {data_label} {data_path} not found"
                with open(data_path, 'rb') as f:
                    self.torchdata[data_label] = pickle.load(f)
        else:
            assert len(self.torchdata) > 0, "ERROR torchdata not found"
        return self.torchdata

    def get_preprocdata(self):
        import pickle
        if self.preprocdata is None and self.preprocdata_cache_path is not None:
            if self.verbose:
                print(f">>loading preprocdata from {self.preprocdata_cache_path}")
            assert self.preprocdata_cache_path.is_file(), f"ERROR preprocdata_cache_path {self.preprocdata_cache_path} not found"
            with open(self.preprocdata_cache_path, 'rb') as f:
                self.preprocdata = pickle.load(f)
        else:
            assert len(self.preprocdata) > 0, "ERROR preprocdata not found"
        return self.preprocdata

    def _preprocess_fit_transform(self):
        """
        Y data stay in [0,1] space.
        X are transformed w/ standard scaler, prospect scale, etc. in data_transform
        """
        from cam_emotorch_utils import DataTransform, scale_appraisal_variables_func, prospect_transform_appraisal_variables_func

        preprocdata = self.get_preprocdata()
        pytorch_spec = self.cfg['pytorch_spec']

        whitening_label, scale_transform_param = pytorch_spec['whitening']

        prospect_transform_label, prospect_transform_param = pytorch_spec['prospect_param']

        thin = int(pytorch_spec.get('thin_samples_factor', 8))

        dataprep_kwargs = {
            'thin': thin,
            'pre_opt_y_affine': None,
            'scale_param': {
                'fn': scale_appraisal_variables_func,
                'param': {
                    'all': scale_transform_param, }, },
            'prospect_transform_param': {
                'fn': prospect_transform_appraisal_variables_func,
                'param': {
                    'base_kwargs': prospect_transform_param['base_kwargs'],
                    'repu_kwargs': prospect_transform_param['repu_kwargs'], }, }, }

        assert (isinstance(preprocdata, list) or isinstance(preprocdata, tuple)) and len(preprocdata) > 0

        ##################### fit x transform #####################

        self.data_transform = DataTransform(index_pad=0, _thin_=dataprep_kwargs['thin'], affine_intercept_=dataprep_kwargs['pre_opt_y_affine'], scale_param_=dataprep_kwargs['scale_param'], prospect_transform_param_=dataprep_kwargs['prospect_transform_param'], verbose=False)

        self.data_transform.fit_X_transform(preprocdata)

    def _preprocess_load_transform(self):
        from cam_utils import test_dicts_equal

        self._preprocess_fit_transform()

        data_transform_desc_reloaded = self.data_transform.get_params()
        test_dicts_equal(self.data_transform_desc['desc'], data_transform_desc_reloaded['desc'])

    def preprocess_fit_transform(self, preprocdata, cache=False):
        from cam_utils import cache_data

        if cache:
            shared_cache = self.cfg['dataincache_base_path']
            self.preprocdata_cache_path = cache_data(preprocdata, outdir=shared_cache, fname_base='data_preproc_cache', pickler='pickle', overwrite_existing=False)

        self.preprocdata = preprocdata

        self._preprocess_fit_transform()

        self.data_transform_desc = self.data_transform.get_params()

    def preprocess_apply_transform(self, ppldatasets, label=None, cache=True):
        from cam_emotorch_utils import format_data_for_torch
        from cam_utils import cache_data

        if label is None:
            label = 'data'

        ###
        ### apply preprocessing transform to data
        ###

        data = dict()
        for stimid, stimdata in ppldatasets.items():
            mats_, dfs_ = self.data_transform.gen_x_y_pair(stimdata['X'], stimdata['Y'])
            data[stimid] = format_data_for_torch(dfs_)

        ###
        ### format data for torch
        ###

        if cache:
            shared_cache = self.cfg['dataincache_base_path']
            assert label not in self.torchdata_cache_path
            self.torchdata_cache_path[label] = cache_data(data, outdir=shared_cache, fname_base=f'data_{label}_cache', overwrite_existing=False)
        else:
            assert label not in self.torchdata
            self.torchdata[label] = data

    def optimize(self, optimization_param_updates=None, inits=None, trackprogress=None):

        import dill
        import time
        from cam_utils import random_string_alphanumeric, gen_seed
        from cam_pytorch_lasso import run

        if optimization_param_updates is None:
            optimization_param_updates = dict()

        if trackprogress is None:
            trackprogress = self.cfg.get('trackprogress', True)

        torch_data = self.get_torchdata()

        ###

        trajectory_ = random_string_alphanumeric(4)
        optimization_param_defaults = dict(iter=10, seed=gen_seed())
        optimization_param = {**optimization_param_defaults, **self.cfg.get('optimization_param', dict()), **optimization_param_updates}
        niter = optimization_param['iter']

        ###

        tj_chain = list()

        res_type_str = 'optimize'
        res_specs_str = f"iter-{niter}_T-{trajectory_}"
        dout_path = self.dataout_base_path / f"{res_type_str}_{res_specs_str}"
        results_pickle_path = dout_path / 'res_op.dill'
        resultsnaive_pickle_path = results_pickle_path.parent / f"{results_pickle_path.with_suffix('').name}-naive.dill"

        tj_chain.append(dict(fn='optimize', tj=trajectory_, pklpath=results_pickle_path))
        res_str_dict = dict(res_type_str=res_type_str, res_specs_str=res_specs_str, trajectory=trajectory_)

        ######

        host_info = cpu_architecture()

        res = dict(
            kind='adam',
            desc='',
            ###
            model_name=self.model_name_brief,
            model_param=self.model_param,
            ###
            op_dict=None,
            op_stats=None,
            op_appliedfit=None,
            op_progressdf=None,
            fit_path=dout_path,
            ###
            torchdata_cache_path=self.torchdata_cache_path,
            data_transform_desc=self.data_transform_desc,
            ###
            optimization_param=optimization_param,
            inits=inits,
            res_str_dict=res_str_dict,
            ###
            trainset=self.cfg.get('trainset', None),
            testset=self.cfg.get('testset', None),
            ###
            et=0.0,
            host=host_info['host'],
            cpu=host_info['cpu'],
            trajectory=trajectory_,
            tj_chain=tj_chain,
            pickle_path=resultsnaive_pickle_path,
        )

        resultsnaive_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        if resultsnaive_pickle_path.is_file():
            resultsnaive_pickle_path.unlink()
        with open(resultsnaive_pickle_path, 'wb') as f:
            dill.dump(res, f, protocol=-5)
        self.optimized_cache_path = resultsnaive_pickle_path
        self.dump(results_pickle_path.parent / 'EmoTorchObj-naive.dill')

        ###

        ###
        print(f"---\nstarting optimization of {results_pickle_path}\n--fit param--vvv\n")
        print(optimization_param)
        print('\n--fit param--^^^')
        t0 = time.perf_counter()

        model_results = run(datatrain=torch_data['train'], datatest=torch_data['test'], model_param=self.model_param, optimization_param=optimization_param, trackprogress=trackprogress, outpath=dout_path / 'figs')

        elapsed_time = time.perf_counter() - t0
        print(f'Run finished, et: {elapsed_time}')

        save_appliedfit_data = True

        res['pickle_path'] = results_pickle_path
        res['op_dict'] = model_results.get('learned_param', None)
        res['op_stats'] = model_results.get('stats', None)
        if save_appliedfit_data:
            res['op_appliedfit'] = model_results.get('appliedfit', None)
        res['op_progressdf'] = model_results.get('progressdf', None)
        res['et'] = elapsed_time

        results_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        if results_pickle_path.is_file():
            results_pickle_path.unlink()
        with open(results_pickle_path, 'wb') as f:
            dill.dump(res, f, protocol=-5)
        self.optimized_cache_path = results_pickle_path
        self.optimized = res
        self.dump(results_pickle_path.parent / 'EmoTorchObj.dill')

        resultsnaive_pickle_path.unlink()
        (results_pickle_path.parent / 'EmoTorchObj-naive.dill').unlink()


def main_optimize(cfg):

    from cam_emotorch_utils import reformat_ppldata_allplayers

    print(f"generating data with >> {cfg['data_prep_label']} <<")
    assert cfg['data_prep_label'] == 'cv'
    assert cfg['pytorch_spec']['scale_transform_label'] == 'ScalePEa2raw'

    eto = EmoTorch()

    prefix = 'PTM'

    eto.init_cfg(cfg, prefix=prefix)

    trainstimid = cfg['trainset']
    teststimid = cfg['testset']

    ###
    ### reformat webppl data
    ###

    feature_selector_label, feature_selector = cfg['pytorch_spec']['feature_selector']
    ppldatasets = reformat_ppldata_allplayers(cpar_path=cfg['cpar_path_str'], feature_selector=feature_selector, feature_selector_label=feature_selector_label)

    generic_pots = sorted(ppldatasets['generic']['X']['pot'].unique().tolist())
    specific_pots = sorted(ppldatasets['239_1']['X']['pot'].unique().tolist())
    assert len(specific_pots) == 8

    ###
    ### collect train data for preprocessing transform
    ###

    X_generic = ppldatasets['generic']['X']
    Y_generic = ppldatasets['generic']['Y']

    train_pots = generic_pots[8:]

    fit_scale_transform_x_data = list()
    fit_scale_transform_x_data.append(X_generic.loc[X_generic['pot'].isin(train_pots), :])
    for stimid in trainstimid:
        fit_scale_transform_x_data.append(ppldatasets[stimid]['X'])

    ###
    ### fit preprocessing transform
    ###

    eto.preprocess_fit_transform(fit_scale_transform_x_data, cache=True)

    ##################### CV train data #####################

    ###
    ### collect train data
    ###

    ppldatasets_train = dict()
    ppldatasets_train['generic'] = dict(
        X=X_generic.loc[X_generic['pot'].isin(train_pots), :],
        Y=Y_generic.loc[Y_generic['pot'].isin(train_pots), :],
    )
    for stimid in trainstimid:
        ppldatasets_train[stimid] = ppldatasets[stimid]

    ###
    ### collect test data
    ###

    ppldatasets_test = dict()
    ppldatasets_test['generic'] = dict(
        X=X_generic.loc[X_generic['pot'].isin(specific_pots), :],
        Y=Y_generic.loc[Y_generic['pot'].isin(specific_pots), :],
    )
    for stimid in teststimid:
        ppldatasets_test[stimid] = ppldatasets[stimid]

    eto.preprocess_apply_transform(ppldatasets_train, label='train', cache=True)
    eto.preprocess_apply_transform(ppldatasets_test, label='test', cache=True)
    eto.optimize()


def cpu_architecture():
    import os
    import subprocess
    host_info = dict(
        cpu='unknown',
        host='unknown',
    )

    try:
        host_info['cpu'] = subprocess.check_output(["lscpu"], encoding='utf-8')
    except Exception as e:
        host_info['cpu'] = f"lscpu failed: {e}"

    try:
        host_info['host'] = os.environ['HOSTNAME']
    except Exception as e:
        host_info['host'] = f"HOSTNAME failed: {e}"

    return host_info


def load_jobdata(picklepath=None, jobnum=None):
    from pathlib import Path
    import dill
    from pprint import pprint

    job_num = jobnum - 1
    cfg_pickle_path = Path(picklepath)
    with open(cfg_pickle_path, 'rb') as f:
        cfg_pickle = dill.load(f)

    print(f"cfg_pickle length: {len(cfg_pickle)}")
    print(f"job_num: {job_num}")
    assert job_num >= 0 and job_num < len(cfg_pickle), f"job_num: {job_num}, len(cfg_pickle): {len(cfg_pickle)}"

    cfg_ = cfg_pickle[job_num]

    print(f"loaded #{job_num} from {cfg_pickle_path.name} at {cfg_pickle_path}")

    print('\n\nvvvvvvvv cfg -------------\n\n')
    pprint(cfg_)
    print('\n\n^^^^^^^^ cfg -------------\n\n')

    return cfg_


def check_multithread(cfgs=None):
    from joblib import Parallel, delayed, cpu_count

    if cfgs is None:
        print(f"\ncpu_count :: {cpu_count()}\n")
        return cpu_count()

    with Parallel(n_jobs=min(len(cfgs), cpu_count())) as pool:
        pool(delayed(main_optimize)(cfg) for cfg in cfgs)


def main(**kwargs):
    from pathlib import Path
    import pickle
    from cam_utils import random_string_alphanumeric

    behavior = kwargs.pop('behavior')
    picklepath = Path(kwargs['picklepath']) if isinstance(kwargs['picklepath'], str) else kwargs['picklepath']
    assert picklepath.is_file()
    jobnum = kwargs['jobnum']

    started_dir = picklepath.parent / 'started'
    finished_dir = picklepath.parent / 'finished'
    error_dir = picklepath.parent / 'error'

    started_dir.mkdir(parents=True, exist_ok=True)

    rndsuffix = ''
    sbatch_started_path = started_dir / f"run-{jobnum}{rndsuffix}.pkl"
    while sbatch_started_path.exists():
        rndsuffix = f"_{random_string_alphanumeric()}"
        sbatch_started_path = started_dir / f"run-{jobnum}{rndsuffix}.pkl"

    with open(sbatch_started_path, 'wb') as f:
        pickle.dump(kwargs, f, protocol=-5)

    run_basefilename = sbatch_started_path.with_suffix('').name

    exit_status = 0
    try:
        if behavior == 'optimize':

            cfg_ = load_jobdata(picklepath=picklepath, jobnum=jobnum)

            if isinstance(cfg_, list):
                if len(cfg_) == 1:
                    cfg = cfg_[0]
                    main_optimize(cfg)
                elif len(cfg_) > 1:
                    if check_multithread() > 1:
                        check_multithread(cfg_)
                    else:
                        for cfg in cfg_:
                            main_optimize(cfg)
            else:
                cfg = cfg_
                main_optimize(cfg)

    except Exception as e:
        error_dir.mkdir(parents=True, exist_ok=True)
        with open(error_dir / f"{run_basefilename}.pkl", 'wb') as f:
            pickle.dump(kwargs, f, protocol=-5)
        sbatch_started_path.unlink()
        error_text_file = error_dir / f"{run_basefilename}.txt"
        print('ERROR:\n')
        print(e)
        error_text_file.write_text("Exception Occured: \n" + str(e))
        exit_status = e

    else:
        finished_dir.mkdir(parents=True, exist_ok=True)
        with open(finished_dir / f"{run_basefilename}.pkl", 'wb') as f:
            pickle.dump(kwargs, f, protocol=-5)
        sbatch_started_path.unlink()
        if not list(started_dir.glob('*')):
            started_dir.rmdir()

    finally:
        return exit_status


def _cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS)
    parser.add_argument('-p', '--picklepath', default='', type=str, help='path to pickle of cfgs', required=True)
    parser.add_argument('-t', '--jobnum', default=0, type=int, help='which cfg to load from pickle')
    parser.add_argument('-b', '--behavior', default='optimize', type=str, help='pytorch behavior')
    qux_help = ("This argument will show its default in the help due to "
                "ArgumentDefaultsHelpFormatter")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    print(f'\n---Received {sys.argv} from shell---\n')
    exit_status = 1

    print(f"CC :: {os.getenv('CC')}")
    print(f"CXX :: {os.getenv('CXX')}")
    print(f"CXXFLAGS :: {os.getenv('CXXFLAGS')}")

    try:
        param_in = _cli()

        print(f'\nparam_in: {param_in}\n\n')

        exit_status = main(**param_in)

    except Exception as e:
        print(f'Got exception of type {type(e)}: {e}')
        print("Not sure what happened, so it's not safe to continue -- crashing the script!")
        sys.exit(1)

    finally:
        print(f"-- {main.__qualname__} from {__file__} ended with exit code {exit_status} --")

    if exit_status == 0:
        print("--SCRIPT COMPLETED SUCCESSFULLY--")
    else:
        print(f"--SOME ISSUE, EXITING:: {exit_status}--")

    sys.exit(exit_status)
