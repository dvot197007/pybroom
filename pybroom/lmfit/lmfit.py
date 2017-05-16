from collections import OrderedDict
import pandas as pd
import lmfit
from .. import glance, tidy, augment


@tidy.register(lmfit.model.ModelResult)
@tidy.register(lmfit.minimizer.MinimizerResult)
def tidy_lmfit(result):
    """Tidy parameters from lmfit's  `ModelResult` or `MinimizerResult`.

    Normally this function is not called directly but invoked by the
    general purpose function :func:`tidy`.

    Arguments:
        result (`ModelResult` or `MinimizerResult`): the fit result object.

    Returns:
        A DataFrame in tidy format with one row for each parameter.

    Note:
        The (possible) columns of the returned DataFrame are:

        - `name` (string): name of the parameter.
        - `value` (number): value of the parameter after the optimization.
        - `init_value` (number): initial value of the parameter before the
          optimization.
        - `min`, `max` (numbers): bounds of the parameter
        - `vary` (bool): whether the parameter has been varied during the
          optimization.
        - `expr` (string): constraint expression for the parameter.
        - `stderr` (float): standard error for the parameter.
    """
    params_attrs = ['name', 'value', 'min', 'max', 'vary', 'expr', 'stderr']
    columns = params_attrs + ['init_value']
    d = pd.DataFrame(index=range(result.nvarys), columns=columns)
    for i, (name, param) in enumerate(sorted(result.params.items())):
        for p in params_attrs:
            d.loc[i, p] = getattr(param, p)
        # Derived parameters may not have init value
        if name in result.init_values:
            d.loc[i, 'init_value'] = result.init_values[name]
    return d.apply(pd.to_numeric, errors='ignore')


@glance.register(lmfit.model.ModelResult)
@glance.register(lmfit.minimizer.MinimizerResult)
def glance_lmfit(result):
    """Tidy summary statistics from lmfit's `ModelResult` or `MinimizerResult`.

    Normally this function is not called directly but invoked by the
    general purpose function :func:`glance`.

    Arguments:
        result (`ModelResult` or `MinimizerResult`): the fit result object.

    Returns:
        A DataFrame in tidy format with one row and several summary statistics
        as columns.

    Note:
        The columns of the returned DataFrame are:

        - `model` (string): model name (only for `ModelResult`)
        - `method` (string): method used for the optimization (e.g. `leastsq`).
        - `num_params` (int): number of varied parameters
        - `ndata` (int):
        - `chisqr` (float): chi-square statistics.
        - `redchi` (float): reduced chi-square statistics.
        - `AIC` (float): Akaike Information Criterion statistics.
        - `BIC` (float): Bayes Information Criterion statistics.
        - `num_func_eval` (int): number of evaluations of the objective
          function during the fit.
        - `num_data_points` (int): number of data points (e.g. samples) used
          for the fit.

    """
    def _is_modelresult(res):
        return hasattr(res, 'model')

    result_attrs = ['name', 'method', 'nvarys', 'ndata', 'chisqr', 'redchi',
                    'aic', 'bic', 'nfev', 'success', 'message']
    attrs_map = OrderedDict((n, n) for n in result_attrs)
    attrs_map['name'] = 'model'
    attrs_map['aic'] = 'AIC'
    attrs_map['bic'] = 'BIC'
    attrs_map['nvarys'] = 'num_params'
    attrs_map['nfev'] = 'num_func_eval'
    attrs_map['ndata'] = 'num_data_points'

    # ModelResult has attribute `.model.name`, MinimizerResult does not
    if not _is_modelresult(result):
        attrs_map.pop('name')
    d = pd.DataFrame(index=range(1), columns=attrs_map.values())
    if _is_modelresult(result):
        d.loc[0, attrs_map.pop('name')] = result.model.name
    for attr_name, df_name in attrs_map.items():
        d.loc[0, df_name] = getattr(result, attr_name)
    # d.loc[0, 'num_components'] = len(result.components)
    if hasattr(result, 'kws') and result.kws is not None:
        for key, value in result.kws.items():
            d['_'.join((result.method, key))] = value
    return d.apply(pd.to_numeric, errors='ignore')


@augment.register(lmfit.model.ModelResult)
@augment.register(lmfit.minimizer.MinimizerResult)
def augment_lmfit(result):
    """Tidy data values and fitted model from `lmfit.model.ModelResult`.
    """
    columns = ['x', 'data', 'best_fit', 'residual']
    d = pd.DataFrame(index=range(result.ndata), columns=columns)
    for col in columns[1:]:
        d.loc[:, col] = getattr(result, col)

    independent_vars = result.model.independent_vars
    if len(independent_vars) == 1:
        independent_var = independent_vars[0]
    else:
        msg = ('Only 1 independent variable is currently supported.\n'
               'Found independent variables: %s' % str(independent_vars))
        raise NotImplementedError(msg)

    x_array = result.userkws[independent_var]
    d.loc[:, 'x'] = x_array

    if len(result.components) > 1:
        comp_names = [c.name for c in result.components]
        for cname, comp in zip(comp_names, result.components):
            d.loc[:, cname] = comp.eval(x=d.x, **result.values)
    return d.apply(pd.to_numeric, errors='ignore')
