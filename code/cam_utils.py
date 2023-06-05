#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cam_utils.py
"""


def cache_data(data, outdir=None, fname_base='datacache', pickler='pickle', overwrite_existing=True, verbose=True, hashlen=16):
    from pathlib import Path
    import pickle
    import dill
    from hashlib import blake2b
    from cam_utils import random_string_alphanumeric

    if pickler == 'pickle':
        pickler_ = pickle
        suffix_ = '.pkl'
    elif pickler == 'dill':
        pickler_ = dill
        suffix_ = '.dill'

    assert isinstance(fname_base, str), type(fname_base)

    """
    demohash = hashlib.md5('somestring'.encode('ascii')).hexdigest()
    """

    ### save temporary file ###

    # fpath_temp0 = outdir.resolve() / fname_base
    fpath_temp0 = outdir / fname_base
    assert fpath_temp0.suffix in ['', suffix_]
    fpath_temp1 = fpath_temp0.with_suffix('')

    fpath_unique_ = False
    while not fpath_unique_:
        fpath_temp = fpath_temp1.parent / f"_tempcache_{fpath_temp1.name}_{random_string_alphanumeric()}{suffix_}"
        fpath_unique_ = not fpath_temp.exists()

    assert fpath_temp.parent == fpath_temp0.parent

    fpath_temp.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath_temp, 'wb') as f:
        pickler_.dump(data, f, protocol=-1)

    with open(fpath_temp, 'rb') as f:
        hasher = blake2b(digest_size=hashlen)
        while chunk := f.read(8192):  # 8 kiB
            hasher.update(chunk)
    file_hash_ = hasher.hexdigest()

    fpath_hashed = fpath_temp.parent / f"{fpath_temp1}_{file_hash_}{suffix_}"

    fpath_final = fpath_hashed
    if fpath_hashed.is_file():

        ### check of contents match ###
        try:
            with open(fpath_hashed, 'rb') as f:
                data_existing = pickler_.load(f)

            if not (test_dicts_equal(data, data_existing, result_only=True) == True):
                print(f"WARNING data mismatch at {fpath_hashed}")

                fpath_unique_ = False
                while not fpath_unique_:
                    fpath_final = fpath_temp.parent / f"{fpath_temp1}_{file_hash_}-{random_string_alphanumeric(stringLength=4)}{suffix_}"
                    fpath_unique_ = not fpath_temp.exists()

        except Exception as e:
            print(f"WARNING failed to load >>{e}<< at: {fpath_hashed}")

    if fpath_final.is_file():

        if overwrite_existing:
            try:
                fpath_final.unlink()
            except Exception as e:
                print(f"WARNING failed to delete >>{e}<< at: {fpath_final}")
            data_cache_path = fpath_temp.rename(fpath_final)
            if verbose:
                print(f"data cache replaced at {data_cache_path}")
                if fpath_temp.is_file():
                    print(f"ERROR temp file not deleted at {fpath_temp}")
        else:
            fpath_temp.unlink()
            data_cache_path = fpath_final
            if verbose:
                print(f"data cache already exists, leaving original at {data_cache_path}")
    else:
        data_cache_path = fpath_temp.rename(fpath_final)
        if verbose:
            print(f"data cache saved at {data_cache_path}")

    if fpath_temp.is_file():
        print(f"ERROR (2) temp file not deleted at {fpath_temp}")

    return data_cache_path


def random_string_alphanumeric(stringLength=8):
    """Generate a random string of letters and digits """
    import string
    import random
    lettersAndDigits = string.ascii_lowercase + string.digits
    return ''.join(random.choice(lettersAndDigits) for _ in range(stringLength))


def gen_seed():
    import random
    max_int = 2**31 - 1
    return random.randint(0, max_int - int(1e3))


def test_dicts_equal(x1, x2, result_only=True):
    import numpy as np
    import pandas as pd
    import math

    class TestEquiv():
        def __init__(self):
            self.key_path = list()
            self.key_current = None
            self.diffs = list()
            self.results_list = None
            self.result = None

        def __str__(self):
            obj_repr = [
                f"<{self.__class__.__name__}: "
                f"{self.result}. {len(self.diffs)} diffs."
            ]
            for i_diff, diff in enumerate(self.diffs):
                obj_repr.append(f" [{i_diff}] {diff}")
            obj_repr.append(">")
            obj_repr_str = "".join(obj_repr)
            if len(obj_repr_str) > 80:
                obj_repr_str = obj_repr_str[:76] + "...>"
            return obj_repr_str

        def _test_equiv(self, var1, var2):
            if type(var1) != type(var2):
                self.diffs.append(dict(err=f"types: {type(var1)} vs {type(var2)}", a=var1, b=var1))
                yield False

            if isinstance(var1, dict):

                keys1 = set(var1.keys())
                keys2 = set(var2.keys())

                if keys1 != keys2:
                    self.diffs.append(dict(err=f"keys", a=keys1, b=keys2))
                    yield False

                for key in keys1:
                    self.key_path.append(key)
                    self.key_current = key
                    yield from self._test_equiv(var1[key], var2[key])

            elif isinstance(var1, list) or isinstance(var1, tuple):
                if len(var1) != len(var2):
                    self.diffs.append(dict(err=f"length: {len(var1)} vs {len(var2)}", a=var1, b=var1))
                    yield False

                for item1, item2 in zip(var1, var2):
                    yield from self._test_equiv(item1, item2)

            elif isinstance(var1, np.ndarray):
                if np.array_equal(var1, var2):
                    yield True
                else:
                    self.diffs.append(dict(err=f"numpy", a=var1, b=var1))
                    yield False

            elif isinstance(var1, pd.DataFrame) or isinstance(var1, pd.Series):
                if var1.equals(var2):
                    yield True
                else:
                    self.diffs.append(dict(err=f"pandas", a=var1, b=var1))
                    yield False

            elif isinstance(var1, pd.Index):
                var1array, var2array = var1.to_numpy(), var2.to_numpy()
                yield from self._test_equiv(var1array, var2array)

            elif isinstance(var1, float):
                if math.isnan(var1) and math.isnan(var2):
                    yield True
                else:
                    if var1 == var2:
                        yield True
                    else:
                        self.diffs.append(dict(err=f"float", a=var1, b=var1))
                        yield False
            else:
                try:
                    if var1 == var2:
                        yield True
                    else:
                        self.diffs.append(dict(err=f"generic: {type(var1)}", a=var1, b=var1))
                        yield False
                except ValueError as e:
                    print(var1)
                    print(f"ERROR for data of type {type(var1)}")
                    raise Exception

        def test(self, v1, v2):
            self.results_list = [res for res in self._test_equiv(v1, v2)]
            self.result = all(self.results_list)

    testobj = TestEquiv()
    testobj.test(x1, x2)

    if result_only:
        return testobj.result
    else:
        return testobj


def check_sbatch_job(job_id):
    import subprocess
    import time
    timeout = 60 * 60 * 24  # 24 hours
    t_cycle = 10  # seconds
    n_cycles_max = timeout / t_cycle
    elapsed = -1
    complete = False
    while (not complete and elapsed < n_cycles_max):
        # Run the sacct command to get the job status
        result = subprocess.run(['sacct', '-j', str(job_id), '-o', 'State'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        error_output = result.stderr.decode('utf-8')

        # Parse the output to get the job status
        if 'PENDING' in output or 'RUNNING' in output:  # Job is still in the queue or running
            time.sleep(60)
        elif 'COMPLETED' in output:  # Job is complete
            complete = True
        else:  # Job has failed or been canceled
            print('Error: SBATCH job failed or was canceled')
            exit(1)
        elapsed += 1
    return complete


def _verify_data(x_in_, y_in_):
    import numpy as np

    if isinstance(x_in_, np.ndarray):
        x_ = x_in_
    elif isinstance(x_in_, list):
        x_ = np.array(x_in_)
    else:
        raise TypeError

    if isinstance(y_in_, np.ndarray):
        y_ = y_in_
    elif isinstance(y_in_, list):
        y_ = np.array(y_in_)
    else:
        raise TypeError

    for v_ in [x_, y_]:
        assert v_.dtype in [float, int, np.float32], v_.dtype

    assert x_.size == y_.size
    assert x_.size == x_.flatten().shape[0]
    assert y_.size == y_.flatten().shape[0]

    return x_, y_


def concordance_corr_(x_in_, y_in_, bias_corrected=False):
    import numpy as np
    ### https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    # cov := np.cov(x_, y_, ddof=1)[0][1] := pearsonr(x_, y_)[0] * np.sqrt( np.var(x_, ddof=1) * np.var(y_, ddof=1) )

    x_, y_ = _verify_data(x_in_, y_in_)

    # Whereas the ordinary correlation coefficient (Pearson's) is immune to whether the biased or unbiased versions for estimation of the variance is used, the concordance correlation coefficient is not.
    if bias_corrected:
        # Nickerson appears to have used the 1/(N-1) normalization
        # Carol A. E. Nickerson (December 1997). "A Note on "A Concordance Correlation Coefficient to Evaluate Reproducibility". Biometrics. 53 (4): 1503–1507. doi:10.2307/2533516
        ddof = 1
    else:
        # In the original article Lin suggested the 1/N normalization
        # Lawrence I-Kuei Lin (March 1989). "A concordance correlation coefficient to evaluate reproducibility". Biometrics. 45 (1): 255–268. doi:10.2307/2532051
        ddof = 0

    ccc = 2 * np.cov(x_, y_, ddof=ddof)[0][1] / (np.var(x_, ddof=ddof) + np.var(y_, ddof=ddof) + np.square(np.mean(x_) - np.mean(y_)))

    assert abs(ccc) <= 1.0, ccc

    x_mean = np.mean(x_)
    y_mean = np.mean(y_)
    x_var = np.sum(np.square(x_ - x_mean)) / (x_.size - ddof)
    y_var = np.sum(np.square(y_ - y_mean)) / (y_.size - ddof)
    xy_covar = np.dot((x_ - x_mean), (y_ - y_mean)) / (x_.size - ddof)
    ccc_ = (2 * xy_covar) / (x_var + y_var + np.square(x_mean - y_mean))

    assert np.isclose(ccc, ccc_)

    return ccc


def adjusted_corr_(x_in_, y_in_, Y):
    """
    This is the same as multiplying Pearson's r by the ratio of the standard deviation of y to the standard deviation of Y
    $\hat{r} = r * \sigma_y / \sigma_Y = cov(x,y) / (sigma_x * sigma_Y)
    """
    import numpy as np

    x, y = _verify_data(x_in_, y_in_)

    delt_x = x - np.mean(x)
    delt_y = y - np.mean(y)

    assert len(delt_x) == len(x)
    assert len(delt_y) == len(y)

    covariance = np.dot(delt_x, delt_y)

    var_x = np.sum(delt_x**2)
    var_y = np.sum(delt_y**2)
    var_Y = np.sum((Y - np.mean(Y))**2)

    pearson_r = covariance / np.sqrt(var_x * var_y)

    adjusted_r = covariance / np.sqrt(var_x * var_Y)

    return adjusted_r


def FTrz(r_array):
    import numpy as np

    fisher_transform = np.vectorize(lambda r: 0.5 * np.log((1 + r) / (1 - r)))  # = arctanh(r_i)

    return fisher_transform(r_array)


def FTzr(z_array):
    import numpy as np

    arctanh = np.vectorize(lambda z: (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1))  # = arctanh(z_i)

    return arctanh(z_array)


def get_expected_vector_from_multiplesets(df, col_labels, nobs, set_prior=None):
    import numpy as np
    import pandas as pd

    if set_prior is None:
        '''Use a uniform prior over sets (pots) if none is provided'''
        uniform_set_prior_prob = nobs.shape[0]**-1
        set_prior_templist = np.full((nobs.shape[0]), np.nan)
        for i_key, key in enumerate(nobs.index.to_list()):
            set_prior_templist[i_key] = uniform_set_prior_prob
        set_prior = pd.Series(set_prior_templist, index=nobs.index.to_list())

    e_vec = df.loc[:, col_labels].to_numpy()  # n_observations x emotions matrix
    obs_prob = df.loc[:, 'prob'].to_numpy()  # column vector of P(\vec{e}|outcome,pot)
    pots_col = df.index.get_level_values(0).to_numpy(dtype=float)  # column vector of set identity (which pot)

    set_prior_by_obs = np.full_like(pots_col, np.nan)
    for i_obspot, obspot in enumerate(pots_col):
        set_prior_by_obs[i_obspot] = set_prior.loc[obspot]  # column vector of P(set), i.e. P(pot)

    overall_vec_prob = set_prior_by_obs * obs_prob  # column vector of P(\vec{e}|outcome) = P(\vec{e}|outcome,pot) * P(pot)

    grand_prob = np.tile(overall_vec_prob.reshape(-1, 1), (1, e_vec.shape[1]))  # n_observations x n_emotions matrix

    assert np.allclose(grand_prob.sum(axis=0), 1)

    ### dot product of each column of intensity values e_i with P(e_i|outcome) yielding expected vector across pots, E[(\vec{e}|outcome)]
    expected_vector = pd.Series(np.sum(e_vec * grand_prob, axis=0), index=col_labels)
    variances = np.sum(np.square(e_vec) * grand_prob, axis=0) - np.square(np.sum(e_vec * grand_prob, axis=0))
    return expected_vector, variances


def get_ev_emodf(dfin0, col_labels, nobs, sets_prior=None):
    """
    We'll presume the nobs df has the relevant conditions (pots)
    the EV[ \vec{e} ] is, \sum \vec{e_{i,j}} \cdot P( \vec{e_{i,j}} | pot=j ) \cdot P( pot=j )

    df['prob'] gives P( \vec{e_{i,j}} | pot=j ) (for empirical data, simply 1/n_observations for that pot)
    set_prior gives P( pot=j )
    """

    import numpy as np
    import warnings

    warnings.warn('nobs currently only being used for index labels')

    dfin = dfin0.copy(deep=True)

    dfin.columns = dfin.columns.droplevel(level=0)

    ### make sure index and nobs are the same

    topkeys = np.unique(dfin.index.get_level_values(0).to_numpy(dtype=float))

    np.testing.assert_array_equal(np.unique(nobs.index.to_numpy(dtype=float)), topkeys)

    return get_expected_vector_from_multiplesets(dfin, col_labels, nobs, set_prior=sets_prior)


def get_expected_vector_by_outcome(emodict, emotions=None, outcomes=None, sets_prior=None):
    import numpy as np
    import pandas as pd

    assert emotions is not None

    if outcomes is None:
        outcomes = emodict['nobs'].columns.to_list()

    expected_vectors = list()
    for i_outcome, outcome in enumerate(outcomes):
        expected_vector, _ = get_ev_emodf(emodict[outcome], emotions, nobs=emodict['nobs'][outcome], sets_prior=sets_prior)
        expected_vector['outcome'] = outcome
        expected_vectors.append(expected_vector)

    return pd.concat(expected_vectors, axis=1).T
