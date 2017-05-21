import numpy as np
from scipy.optimize import least_squares

from .conftest import BaseTest

N = 50
x = np.linspace(-10, 10, N)
random_state = np.random.RandomState(123)
y = x + random_state.randn(N)/3 + 3
x0 = [0, 0]


def residuals(parameters, x, y):
    a = parameters[0]
    b = parameters[1]
    y_fitted = a*x + b
    return y - y_fitted


ls_res1 = least_squares(residuals, x0, args=(x, y), loss='linear')
ls_res2 = least_squares(residuals, x0, args=(x, y), loss='soft_l1')


class TestOptimize(BaseTest):
    n = N
    result = ls_res1


class TestOptimizeList(BaseTest):
    n = [N, N]
    result = [ls_res1, ls_res2]


class TestOptimizeDict(BaseTest):
    n = {'m1': N, 'm2': N}
    result = {'m1': ls_res1,
              'm2': ls_res2}
