#
# Copyright (c) 2016 Antonino Ingargiola and contributors.
#
"""
This module contains the 3 main pybroom's functions:

- :func:`glance`
- :func:`tidy`
- :func:`augment`

These functions take one or multiple fit results as input and return a
"tidy" (or long-form) DataFrame.
The `glance` function returns fit statistics, one for each
fit result (e.g. fit method, number of iterations, chi-square etc.).
The `tidy` function returns data for each fitted parameter
(e.g. fitted value, gradient, bounds, etc.).
The `augment` function returns data with the same size as the fitted
data points (evaluated best-fit model, residuals, etc.).

In the case of multiple fit results, pybroom functions accept a list, a
dict or a nested structure of dict and lists (for example a dict of lists
of fit results). The example below shows some use cases.

Note:
    pybroom functions are particularly convenient when tidying a
    collection of fit results. The following examples are valid for
    all the 3 pybroom functions. If `results` is a list
    of datasets (e.g. data replicates), the returned dataframe will
    have an additional "index" column containing the index of the
    dataset in the list. If `results` is a dict of fit results (e.g.
    results from different fit methods or models on the same dataset),
    then the "index" column contains the keys of the dict (each key
    identifies a fit result). In the previous two example, `var_names`
    should contains the name of the "index" column (a string).
    Nested structures are also possible. For example, when fitting
    a list of datasets with different methods, we can build a dict
    of lists of fit results where the dict keys are the method names
    and the items in the list are fit results for the different datasets.
    In this case the returned dataframe has two additional "index"
    columns: one with the dict keys and one with the list index.
    The tuple (key, list index) identifies each single fit result.
    In this case `var_names` should be a list of column names for
    the keys and index column respectively (list of strings)


Example:
    The following examples shows pybroom output when multiple fit results
    are used. The `glance` function is used as example but the same logic
    (and input arguments) can be also passsed to `tidy` and `augment`.

    Input is a list of fit results::

        >>> results = [fit_res1, fit_res2, fit_res3]
        >>> br.glance(results, var_names='dataset')

          num_params num_data_points      redchi      AIC  dataset
        0          6             101  0.00911793 -468.634        0
        1          6             101  0.00996431 -459.669        1
        2          6             101   0.0109456 -450.183        2

    Input is a dict of fit results::

        >>> results = {'A': fit_res1, 'B': fit_res2, 'C': fit_res3}
        >>> br.glance(results, var_names='function')

          num_params num_data_points      redchi      AIC function
        0          6             101  0.00911793 -468.634        A
        1          6             101  0.00996431 -459.669        B
        2          6             101   0.0109456 -450.183        C

    Input is a dict of lists of fit results::

        >>> results = {'A': [fit_res1, fit_res2], 'B': [fit_res3, fit_res4]}
        >>> br.glance(results, var_names=['function', 'dataset'])

          num_params num_data_points      redchi      AIC  dataset function
        0          6             101  0.00911793 -468.634        0        A
        1          6             101  0.00996431 -459.669        1        A
        2          6             101   0.0109456 -450.183        0        B
        3          6             101   0.0176529 -401.908        1        B


"""
from collections import OrderedDict
from functools import singledispatch
import pandas as pd

__version__ = '0.3.dev0'


@singledispatch
def tidy(result, var_names='key', **kwargs):
    """Tidy DataFrame containing fitted parameter data from `result`.

    A function to tidy any of the supported fit result
    (or a list of fit results). This function will identify input type
    and call the relative "specialized" tidying function. When the input
    is a list, the returned DataFrame contains data from all the fit
    results.
    Supported fit result objects are `lmfit.ModelResult`,
    `lmfit.MinimizeResult` and `scipy.optimize.OptimizeResult`.

    Arguments:
        result (fit result object or list): one of the supported fit result
            objects or a list of supported fit result objects. When a list,
            all the elements need to be of the same type.
        var_names (string or list): name(s) of the column(s) containing
            an "index" that is different for each element in the set of
            fit results.
        param_names (string or list of string): names of the fitted parameters
            for fit results which don't include parameter's names
            (such as scipy's OptimizeResult). It can either be a list of
            strings or a single string with space-separated names.
        **kwargs: additional arguments passed to the underlying specialized
            tidying function.

    Returns:
        A DataFrame with one row for each fitted parameter.
        Columns include parameter properties such as best-fit value,
        standard error, eventual bounds/constrains, etc.
        When a list of fit-result objects is passed, the column `var_name`
        (`'item'` by default) contains the index of the object
        in the list.

    See also:
        For more details on the returned DataFrame and on additional
        arguments refer to the specialized tidying functions:
        :func:`tidy_lmfit_result` and :func:`tidy_scipy_result`.
    """
    msg = 'Sorry, `tidy` does not support this object type (%s)'
    raise NotImplementedError(msg % type(result))


@singledispatch
def glance(results, var_names='key', **kwargs):
    """Tidy DataFrame containing fit summaries from`result`.

    A function to tidy any of the supported fit result
    (or a list of fit results). This function will identify input type
    and call the relative "specialized" tidying function. When the input
    is a list, the returned DataFrame contains data from all the fit
    results.
    Supported fit result objects are `lmfit.ModelResult`,
    `lmfit.MinimizeResult` and `scipy.optimize.OptimizeResult`.

    Arguments:
        result (fit result object or list): one of the supported fit result
            objects or a list of supported fit result objects. When a list,
            all the elements need to be of the same type.
        var_names (string or list): name(s) of the column(s) containing
            an "index" that is different for each element in the set of
            fit results.
        **kwargs: additional arguments passed to the underlying specialized
            tidying function.

    Returns:
        A DataFrame with one row for each passed fit result.
        Columns include fit summaries such as reduced chi-square,
        number of evaluation, successful convergence, AIC, BIC, etc.
        When a list of fit-result objects is passed, the column `var_name`
        (`'item'` by default) contains the index of the object
        in the list.

    See also:
        For more details on the returned DataFrame and on additional
        arguments refer to the specialized tidying functions:
        :func:`glance_lmfit_result` and :func:`glance_scipy_result`.
    """
    msg = 'Sorry, `glance` does not support this object type (%s)'
    raise NotImplementedError(msg % type(results))


@singledispatch
def augment(results, var_names='key', **kwargs):
    """Tidy DataFrame containing fit data from `result`.

    A function to tidy any of the supported fit result
    (or a list of fit results). This function will identify input type
    and call the relative "specialized" tidying function. When the input
    is a list or a dict of fit results, the returned DataFrame contains
    data from all the fit results. In this case data from different fit
    results is identified by the values in the additional "index"
    (or categorical) column(s) whose name(s) are specified in `var_names`.

    Arguments:
        results (fit result object or list): one of the supported fit result
            objects or a list of supported fit result objects. When a list,
            all the elements need to be of the same type.
        var_names (string or list): name(s) of the column(s) containing
            an "index" that is different for each element in the set of
            fit results. See the example section below.
        **kwargs: additional arguments passed to the underlying specialized
            tidying function.

    Returns:
        A DataFrame with one row for each data point used in the fit.
        It contains the input data, the model evaluated at the data points
        with best fitted parameters, error ranges, etc.
        When a list of fit-result objects is passed, the column `var_name`
        (`'item'` by default) contains the index of the object
        in the list.

    """
    msg = 'Sorry, `augment` does not support this object type (%s)'
    raise NotImplementedError(msg % type(results))


def _as_odict_copy(results):
    """Transform input into a OrderedDict, if needed. Returns a copy.
    """
    iterator = enumerate(results)
    if isinstance(results, dict):
        iterator = results.items()
    return OrderedDict((k, v) for k, v in iterator)


def _as_list_of_strings_copy(var_names):
    """Transform input into a list of strings, if needed. Returns a copy.
    """
    if isinstance(var_names, str):
        var_names = [var_names]
    return var_names.copy()


@tidy.register(list)
@tidy.register(dict)
def _tidy_multi_dataframe(results, var_names='key', **kwargs):
    return _multi_dataframe(tidy, results, var_names, **kwargs)


@glance.register(list)
@glance.register(dict)
def _glance_multi_dataframe(results, var_names='key', **kwargs):
    return _multi_dataframe(glance, results, var_names, **kwargs)


@augment.register(list)
@augment.register(dict)
def _augment_multi_dataframe(results, var_names='key', **kwargs):
    return _multi_dataframe(augment, results, var_names, **kwargs)


def _multi_dataframe(func, results, var_names, **kwargs):
    """Recursively call `func` on each item in `results` and concatenate output.

    Usually `func` is :func:`glance`, :func:`tidy` or :func:`augment`.
    The function `func` is also the calling function, therefore this implements
    a recursion which unpacks the nested `results` structure (a tree) and
    builds a global tidy DataFrame with "key" columns corresponding to
    the `results` structure.

    Arguments:
        func (function): function of the called on each element of `results`.
            Chose between `glance`, `tidy` or `augment`.
        results (dict or list): collection of fit results. It can be a list,
            a dict or a nested structure such as a dict of lists.
        var_names (list or string): names of DataFrame columns used to index
            the results. It can be a list of strings or single string in case
            only one categorical "index" is needed (i.e. a string is equivalent
            to a 1-element list of strings).

    Returns:
        "Tidy" DataFrame merging data from all the items in `results`.
        Necessary "key" columns are added to encode layout of fitting result
        objects in `results`.
    """
    if len(var_names) == 0:
        msg = ('The list `var_names` is too short. Its length should be equal '
               'to the nesting levels in `results`.')
        raise ValueError(msg)
    d = _as_odict_copy(results)
    var_names = _as_list_of_strings_copy(var_names)
    var_name = var_names.pop(0)
    for i, (key, res) in enumerate(d.items()):
        # Some result classes subclass dict, so isinstance fails
        if type(res) in {list, dict}:
            d[key] = func(res, var_names, **kwargs)
        else:
            d[key] = func(res, **kwargs)
        d[key][var_name] = key
    df = pd.concat(d, ignore_index=True)
    # Convert "key" column to categorical only if input was dict-type
    # not list/tuple.
    if isinstance(results, dict):
        kw = {var_name: lambda x: pd.Categorical(x[var_name], ordered=True)}
        df = df.assign(**kw)
    return df
