from .pybroom import tidy, glance, augment
from .utils import tidy_to_dict, dict_to_tidy  # noqa 401
from ._version import get_versions

__all__ = ['tidy', 'glance', 'augment']
__version__ = get_versions()['version']


def build_registries():
    """
    Build the registries used by the singledispatch

    The registries are automatically built when the modules
    in with the implementations functions are imported.
    This function automatically imports all implementations
    and handles cases for implementations where the Model
    package is not installed
    """
    import os
    from pkgutil import walk_packages
    from importlib import import_module
    from contextlib import suppress
    from glob import glob
    PATH = os.path.dirname(__file__)
    exclude = {'tests'}

    def subpackages():
        """
        Return the subpackages of pybroom

        These hold the implementations for all models of a
        given package, e.g lmfit, scipy, ...
        """
        def is_package(d):
            d = os.path.join(PATH, d)
            return (os.path.isdir(d) and
                    glob(os.path.join(d, '__init__.py*')))

        for d in os.listdir(PATH):
            if is_package(d) and d not in exclude:
                yield d

    def modules(package_string):
        """
        Return all the module paths in package

        e.g. Given 'pybroom.scipy' it would return
        `[pybroom.scipy.optimize]`.
        """
        package = import_module(package_string)
        for _, name, ispkg in walk_packages(package.__path__):
            if not ispkg:
                yield '{}.{}'.format(package.__name__, name)

    # Import all modules in the subpackages
    for package in subpackages():
        package_str = 'pybroom.{}'.format(package)
        for module in modules(package_str):
            with suppress(ImportError):
                import_module(module)


build_registries()
del build_registries
del get_versions
