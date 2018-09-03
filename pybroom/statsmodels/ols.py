from collections import OrderedDict
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from .. import glance, tidy, augment


@tidy.register(sm.regression.linear_model.RegressionResultsWrapper)
def tidy_statsmodels(result):
    """Tidy statsmodels `smf.ols` or `sm.OLS` fitted result.

    Arguments:
        result: the fit result object (`RegressionResultsWrapper`).

    Returns:
        A DataFrame in tidy format with one row for each parameter.
    """
    pass


@glance.register(sm.regression.linear_model.RegressionResultsWrapper)
def glance_statsmodels(result):
    """Glance statsmodels `sm.OLS` or `smf.ols` fitted result.

    Arguments:
        result: the fit result object (`RegressionResultsWrapper`).

    Returns:
        A DataFrame in tidy format with one row and several summary statistics
        as columns.

    Note:
        All attributes returned:
        https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html?highlight=regression%20linear_model%20regressionresults

        R `broom::glance` for `ols` equivalent `lm` also includes sigma,
        log_likelihood, and deviance.
    """
    df = pd.DataFrame({'r_squared': result.rsquared,
                       'adj_r_squared': result.rsquared_adj,
                       'statistic': result.fvalue,
                       'p_value': result.f_pvalue,
                       'df': result.df_model,
                       'df_residual': result.df_resid,
                       'aic': result.aic,
                       'bic': result.bic}, index=range(1))
    return df


@augment.register(sm.regression.linear_model.RegressionResultsWrapper)
def augment_statsmodels(result):
    """Augment statsmodels `sm.OLS` or `smf.ols` fitted result.

    Arguments:
        result: the fit result object (`RegressionResultsWrapper`).

    Returns:
        A DataFrame of the original data and additional columns such as
        predictions and residuals.

    Note:
        All attributes returned:
        https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html?highlight=regression%20linear_model%20regressionresults
    """
    design = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
    X = design.drop('Intercept', axis=1)
    y = pd.Series(result.model.endog, name=result.model.endog_names)

    estimated_values = pd.DataFrame({'_fitted': result.fittedvalues,
                                     '_se_fit': np.nan,
                                     '_resid': result.resid})

    estimated_values = (estimated_values
                        .reset_index()
                        .drop('index', axis=1))

    df = pd.concat([y, X, estimated_values], axis=1, sort=False)
    return df
