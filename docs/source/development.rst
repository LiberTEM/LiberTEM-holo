Development
===========

Running notebooks in CI
-----------------------

Notebooks are tested as part of the CI run. When adding a notebook,
you can generate the notebook outputs in a clean environment using tox:

.. code-block:: shell

    $ tox -e notebooks_gen
    $ cp notebooks/generated/*.ipynb notebooks


