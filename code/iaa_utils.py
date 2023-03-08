#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""iaa_utils.py
"""


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


def concordance_corr_adjusted_UNUSED(x_, y_, Y_):
    ### https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    # from scipy.stats import pearsonr
    # cov := np.cov(x_, y_, ddof=1)[0][1] := pearsonr(x_, y_)[0] * np.sqrt( np.var(x_, ddof=1) * np.var(y_, ddof=1) )
    import numpy as np

    assert np.issubdtype(x_.dtype, np.number)
    assert np.issubdtype(y_.dtype, np.number)
    assert np.issubdtype(Y_.dtype, np.number)

    # (np.var(y_, ddof=1) / np.var(Y_, ddof=1)) *
    # print(f"{(np.sqrt(np.var(y_, ddof=0)) / np.sqrt(np.var(Y_, ddof=0)))} // {(np.sqrt(np.var(y_, ddof=1)) / np.sqrt(np.var(Y_, ddof=1)))} * {2 * np.cov(x_, y_, ddof=1)[0][1] / ( np.var(x_, ddof=1) + np.var(y_, ddof=1) + np.square( np.mean(x_) - np.mean(y_) ) )}")
    # return (np.sqrt(np.var(y_, ddof=1)) / np.sqrt(np.var(Y_, ddof=1))) * 2 * np.cov(x_, y_, ddof=1)[0][1] / ( np.var(x_, ddof=1) + np.var(y_, ddof=1) + np.square( np.mean(x_) - np.mean(y_) ) )
    # print(f"{np.sqrt(np.var(y_, ddof=0))/np.sqrt(np.var(Y_, ddof=0))} /// {np.sqrt(np.var(y_, ddof=0))} vs {np.sqrt(np.var(Y_, ddof=0))}")
    # return 2 * np.cov(x_, y_, ddof=1)[0][1] / ( np.var(x_, ddof=1) + np.var(Y_, ddof=1) + np.square( np.mean(x_) - np.mean(y_) ) )

    delt_x = x_ - np.mean(x_)
    delt_y = y_ - np.mean(y_)
    var_x = np.sum(delt_x**2)
    var_y = np.sum(delt_y**2)
    covariance = np.dot(delt_x, delt_y)
    pearsonr = covariance / np.sqrt(var_x * var_y)

    pearsonr_adjusted = adjusted_corr_(x_, y_, Y_)[0]

    sd_x = np.sqrt(np.var(x_, ddof=1))
    sd_y = np.sqrt(np.var(y_, ddof=1))
    sd_Y = np.sqrt(np.var(Y_, ddof=1))

    Cb = 2 / ((sd_x / sd_y) + (sd_y / sd_x) + np.square(np.mean(x_) - np.mean(y_)) / (sd_x * sd_y))

    return pearsonr_adjusted * Cb
