import pandas as pd


def tidy_to_dict(df, key='name', value='value', keys_exclude=None,
                 cast_value=float):
    """Convert a tidy DataFrame into a dictionary.

    This function converts two columns from an input tidy (or long-form)
    DataFrame into a dictionary. A typical use-case is passing
    parameters stored in tidy DataFrame to a python function. The arguments
    `key` and `value` contain the name of the DataFrame columns containing
    the keys and the values of the dictionary.

    Arguments:
        df (pandas.DataFrame): the "tidy" DataFrame containing the data.
            Two columns of this DataFrame should contain the keys and the
            values to construct the dictionary.
        key (string or scalar): name of the DataFrame column containing
            the keys of the dictionary.
        value (string or scalar ): name of the DataFrame column containing
            the values of the dictionary.
        keys_exclude (iterable or None): list of keys excluded when building
            the returned dictionary.
        cast_value (callable or None): callable used to cast
            the value of each item in the dictionary. If None, no casting
            is performed and the resulting values are 1-element
            `pandas.Series`. Default is the python built-in `float`.
            Other typical values may be `int` or `str`.

    Returns:
        A dictionary with keys and values extracted from the input (tidy)
        DataFrame.

    See also: :func:`dict_to_tidy`.
    """
    keys_list = set(df[key])
    if keys_exclude is not None:
        keys_list = keys_list - set(keys_exclude)
    if cast_value is None:
        def cast_value(x):
            return x
    return {var: cast_value(df.loc[df[key] == var, value])
            for var in keys_list}


def dict_to_tidy(dc, key='name', value='value', keys_exclude=None):
    """Convert a dictionary into a tidy DataFrame.

    This function converts a dictionary into a "tidy" (or long-form)
    DataFrame with two columns: one containing the keys and the other
    containing the values from the dictionary. Names of the columns
    can be specified with the `key` and `value` argument.

    Arguments:
        dc (dict): the input dictionary used to build the DataFrame.
        key (string or scalar): name of the DataFrame column containing
            the keys of the dictionary.
        value (string or scalar): name of the DataFrame column containing
            the values of the dictionary.
        keys_exclude (iterable or None): list of keys excluded when building
            the returned DataFrame.

    Returns:
        A two-columns tidy DataFrame containing the data in the dictionary.


    See also: :func:`tidy_to_dict`.
    """
    keys = dc.keys()  # this is a set
    if keys_exclude is not None:
        keys -= keys_exclude
    keys = sorted(keys)
    df = pd.DataFrame(columns=(key, value), index=range(len(keys)))
    df[key] = keys
    df[value] = [dc[k] for k in keys]
    return df


def _test_dict_to_tidy(dc, key='name', value='value', keys_exclude=None,
                       value_type=None):
    # Alternative implementation
    if keys_exclude is None:
        keys_exclude = []
    dc2 = {k: v for k, v in dc.items() if k not in keys_exclude}
    df = pd.DataFrame(columns=(key, value), index=range(len(dc2)))
    keys = sorted(dc2.keys())
    df[key] = keys
    df[value] = [dc2[k] for k in keys]
    # Test compliance
    assert all(df == dict_to_tidy(dc, key, value, keys_exclude, value_type))
    return df
