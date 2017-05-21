import numpy as np
import lmfit

from .conftest import BaseTest

N = 50
x = np.linspace(-10, 10, N)
random_state = np.random.RandomState(123)
y = x + random_state.randn(N)/3 + 3

model1 = lmfit.models.LinearModel()
model2 = lmfit.models.QuadraticModel()


class TestOneModel(BaseTest):
    n = N
    result = model1.fit(y, x=x)


class TestModelsList(BaseTest):
    n = [N, N]
    result = [model1.fit(y, x=x),
              model2.fit(y, x=x)]


class TestModelsDict(BaseTest):
    n = {'m1': N, 'm2': N}
    result = {'m1': model1.fit(y, x=x),
              'm2': model2.fit(y, x=x)}
