import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .conftest import BaseTest

N = 50
x = np.linspace(-10, 10, N)
random_state = np.random.RandomState(123)
y = x + random_state.randn(N)/3 + 3
df = pd.DataFrame({'x': x, 'y': y})

model = smf.ols('y ~ x', data=df)

class TestOneModel(BaseTest):
    n = N
    result = model.fit()
