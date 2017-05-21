import numpy as np
from pybroom import tidy, glance, augment


class BaseTest:
    """
    Base class for Model tests

    Does basic tests on the returned dataframe
    """
    # Subclasses must assign an instance of a model
    # result. The tests will be run against this
    # result
    result = None

    # No. of observations
    n = 0

    def test_tidy(self):
        """
        Test tidy
        """
        if isinstance(self.result, (list, dict)):
            return self._test_tidy_multi()

        try:
            df = tidy(self.result)
        except NotImplementedError:
            return

        assert len(df) > 0
        assert len(df.columns) > 0

    def test_glance(self):
        """
        Test glance
        """
        if isinstance(self.result, (list, dict)):
            return self._test_glance_multi()

        try:
            df = glance(self.result)
        except NotImplementedError:
            return

        assert len(df) == 1
        assert len(df.columns) > 0

    def test_augment(self):
        """
        Test augment
        """
        if isinstance(self.result, (list, dict)):
            return self._test_augment_multi()

        try:
            df = augment(self.result)
        except NotImplementedError:
            return

        assert len(df) == self.n
        assert len(df.columns) > 0

    @property
    def _result(self):
        """
        self.result as a list
        """
        if isinstance(self.result, dict):
            result = list(self.result.values())
        else:
            result = self.result

        return result

    @property
    def _n(self):
        """
        self.n as a list
        """
        if isinstance(self.n, dict):
            n = list(self.n.values())
        else:
            n = self.n

        return n

    def _test_tidy_multi(self):
        """
        Test tidy on a ``dict`` or ``list``
        """
        # In separate dataframes
        try:
            dfs = [tidy(res) for res in self._result]
        except NotImplementedError:
            return

        # Many model results
        df = tidy(self.result)

        # As many rows as there are observations
        # An extra column for the key
        # As many keys as there are models
        assert len(df) == np.sum([len(_df) for _df in dfs])
        assert len(df.columns) == len(dfs[0].columns) + 1
        assert df['key'].nunique() == len(dfs)

    def _test_glance_multi(self):
        """
        Test glance on a ``dict`` or ``list``
        """
        # In separate dataframes
        try:
            dfs = [glance(res) for res in self._result]
        except NotImplementedError:
            return

        # All in one dataframe
        df = glance(self.result)

        # As many rows as there are observations
        # An extra column for the key
        # As many keys as there are models
        assert len(df) == np.sum([len(_df) for _df in dfs])
        assert len(df.columns) == len(dfs[0].columns) + 1
        assert df['key'].nunique() == len(dfs)

    def _test_augment_multi(self):
        """
        Test augment on a ``dict`` or ``list``
        """
        # In separate dataframes
        try:
            dfs = [augment(res) for res in self._result]
        except NotImplementedError:
            return

        # All in one dataframe
        df = augment(self.result)

        # As many rows as there are observations
        # An extra column for the key
        # As many keys as there are models
        assert len(df) == np.sum(self._n)
        assert len(df.columns) == len(dfs[0].columns) + 1
        assert df['key'].nunique() == len(dfs)
