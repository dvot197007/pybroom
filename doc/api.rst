Pybroom API Documentation
=========================

.. automodule :: pybroom

Main Functions
--------------

The 3 high-level functions :func:`glance`, :func:`tidy` and :func:`augment`
allows tidying one or more fit results.
These are pybroom's most generic functions, accepting all the
the supported fit result objects, as well as a list/dict of such objects.
See also the examples at the beginning of this page and the example notebooks.

.. currentmodule:: pybroom
.. autosummary::
   :toctree: generated/

   glance
   tidy
   augment

Specialized functions
---------------------

These are the specialized (i.e. low-level) functions, each converting one
specific object to a tidy DataFrame.

lmfit
*****

.. currentmodule:: pybroom.lmfit
.. autosummary::
   :toctree: generated/

   ~lmfit.glance_lmfit
   ~lmfit.tidy_lmfit
   ~lmfit.augment_lmfit

scipy
*****

.. currentmodule:: pybroom.scipy
.. autosummary::
   :toctree: generated/

   ~optimize.glance_optimize
   ~optimize.tidy_optimize


Utility Functions
-----------------

Dictionary conversions
**********************

The two functions :func:`tidy_to_dict` and :func:`dict_to_tidy` provide
the ability to convert a tidy DataFrame to and from a python dictionary.

.. currentmodule:: pybroom.utils
.. autosummary::
   :toctree: generated/

   tidy_to_dict
   dict_to_tidy
