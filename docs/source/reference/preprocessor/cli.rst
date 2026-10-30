.. _cli:

**********************
Command line interface
**********************

*pyDeltaRCM* can be run from the command line without writing Python.
Two equivalent entry points are provided:

* the ``pyDeltaRCM`` console script installed with the package (the
  ``pyDeltaRCM.preprocessor:preprocessor_wrapper`` entry point), and
* ``python -m pyDeltaRCM``, which invokes the same wrapper via
  ``pyDeltaRCM/__main__.py``.

Both are implemented in the :mod:`pyDeltaRCM.preprocessor` module; see
:doc:`index` for the full API reference of the preprocessor classes.

.. code:: console

    $ pyDeltaRCM --config input_config.yaml --timesteps 500


Options
=======

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Meaning
   * - ``--config <path>``
     - Path to the YAML configuration file to use.
   * - ``--timesteps <N>``
     - Number of timesteps to run. Optional; if provided, overrides any
       value in the config file, and is overridden by the time options
       below.
   * - ``--time <seconds>``
     - Run the model until this model time, in seconds. Optional;
       overrides the config file and ``--timesteps``. Model time is scaled
       due to intermittency, which is set by ``--If`` (default 1).
   * - ``--time_years <years>``
     - Run the model until this model time, in years. Optional; overrides
       the config file and ``--timesteps``. Model time is scaled due to
       intermittency, which is set by ``--If`` (default 1).
   * - ``--If <fraction>``
     - Assumed intermittency factor for converting model time to
       real-world time: a value in ``(0, 1]`` representing the fraction of
       days per year that the delta is flooding. Optional, default 1.
   * - ``--dryrun``
     - Set up the run without executing timestepping.
   * - ``--parallel [Ncores]``
     - Run jobs in parallel, if possible. Given without a value, the number
       of parallel cores used depends on system specs as ``ncores - 1``;
       give an integer to choose the number of cores. Optional; the default
       is serial execution.
   * - ``--version``
     - Print the *pyDeltaRCM* version and exit.


Multi-run campaigns
===================

The command line runner is also a campaign runner. Configuration
specifications that define multiple runs — ``matrix``, ``set``, and
ensemble expansions — are expanded into a job list, one configuration per
job, each writing to its own output directory. By default the jobs run
sequentially; with ``--parallel``, the jobs are fanned out across processes
(all but one core, or the number of cores given).
