from collections import namedtuple
import numpy as np
import pandas as pd
import scipy.optimize as so
from .. import glance, tidy
from ..utils import dict_to_tidy


@tidy.register(so.OptimizeResult)
def tidy_optimize(result, param_names, **kwargs):
    """Tidy parameters data from scipy's `OptimizeResult`.

    Normally this function is not called directly but invoked by the
    general purpose function :func:`tidy`.
    Since `OptimizeResult` has a raw array of fitted parameters
    but no names, the parameters' names need to be passed in `param_names`.

    Arguments:
        result (`OptimizeResult`): the fit result object.
        param_names (string or list of string): names of the fitted parameters.
            It can either be a list of strings or a single string with
            space-separated names.

    Returns:
        A DataFrame in tidy format with one row for each parameter.

    Note:
        These two columns are always present in the returned DataFrame:

        - `name` (string): name of the parameter.
        - `value` (number): value of the parameter after the optimization.

        Optional columns (depending on the type of result) are:

        - `grad` (float): gradient for each parameter
        - `active_mask` (int)
    """
    if 'param_names' not in kwargs:
        msg = "The argument `param_names` is required for this input type."
        raise ValueError(msg)
    Params = namedtuple('Params', param_names)
    params = Params(*result.x)
    df = dict_to_tidy(params._asdict(), **kwargs)
    for var in ('grad', 'active_mask'):
        if hasattr(result, var):
            df[var] = result[var]
    return df


@glance.register(so.OptimizeResult)
def glance_optimize(result):
    """Tidy summary statistics from scipy's `OptimizeResult`.

    Normally this function is not called directly but invoked by the
    general purpose function :func:`glance`.

    Arguments:
        result (`OptimizeResult`): the fit result object.

    Returns:
        A DataFrame in tidy format with one row and several summary statistics
        as columns.

    Note:
        Possible columns of the returned DataFrame include:

        - `success` (bool): whether the fit succeed
        - `cost` (float): cost function
        - `optimality` (float): optimality parameter as returned by
          scipy.optimize.least_squares.
        - `nfev` (int): number of objective function evaluations
        - `njev` (int): number of jacobian function evaluations
        - `nit` (int): number of iterations
        - `status` (int): status returned by the fit routine
        - `message` (string): message returned by the fit routine
    """
    attr_names_all = ['success', 'cost', 'optimality', 'nfev', 'njev', 'nit'
                      'status', 'message']
    attr_names = [a for a in attr_names_all if hasattr(result, a)]
    if hasattr(result, 'fun') and np.size(result.fun) == 1:
        attr_names.append('fun')
    d = pd.DataFrame(index=range(1), columns=attr_names)
    for attr_name in attr_names:
        d.loc[0, attr_name] = getattr(result, attr_name)
    return d.apply(pd.to_numeric, errors='ignore')
