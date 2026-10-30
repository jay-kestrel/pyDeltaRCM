.. _benchmarking:

************
Benchmarking
************

Performance-oriented work needs a stable, reproducible way to measure the
model. This fork ships a two-regime per-timestep benchmark harness,
``benchmarks/benchmark_timestep.py``, which measures the wall time of
:meth:`~pyDeltaRCM.model.DeltaModel.update` in the two flow regimes that
matter for performance work.


What the harness measures
=========================

The harness runs two scenarios on the same grid and shared settings, so
their per-update times are directly comparable:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Scenario
     - Regime
     - Warm-up
   * - ``early_water``
     - water-dominated: flow routing on the initial, nearly flat bed
     - a fixed number of updates (default 5)
   * - ``evolved_sediment``
     - sediment-dominated: sediment routing is the dominant cost
     - runs until a deposition-coverage criterion is met, capped at
       ``--max-warmup-updates``

Both scenarios share the same full-size grid — 60×120 cells at ``dx: 50``
(``Length: 3000``, ``Width: 6000``) — and settings (``itermax: 3``,
``seed: 0``), with ``Np_water = Np_sed = 1000``. They differ in one knob,
deliberately: the ``evolved_sediment`` scenario runs ``Np_sed = 4000``.
Measured on this grid, the sediment phase stays a minority of update time
(21–27%) at ``Np_sed = 1000`` no matter how long the model evolves (probed
through 210 updates — flow channelizes and sediment parcel paths shorten),
while ``Np_sed = 4000`` makes sediment routing the dominant phase (53% of
phase time) with water forcing untouched. The scenario exists to stress the
sediment routers, so that is the configuration recorded in the benchmark
output.

The ``evolved_sediment`` warm-up ends once the bed has aggraded by at least
5 cm over enough cells to reach the coverage threshold — 5% of the domain
at full size — so the timed window starts with realistic morphology rather
than a flat bed. The harness records whether the threshold was actually met
before the warm-up cap.

Warm-up updates are discarded from the timing, so numba compilation and
initial transients never enter the numbers. Each timed update is measured
individually with :func:`time.perf_counter`, and each scenario reports the
per-update mean, standard deviation, median, minimum, and maximum wall time.
Alongside the timings, the output records the package versions (pyDeltaRCM,
numpy, numba, scipy), the git commit the package was imported from, the RNG
seed, the grid configuration, and the derived model timestep. All model
output (figures, grids, checkpoints) is disabled; the model writes only
into a throwaway temporary directory that is removed afterwards.


Running the harness
===================

.. code:: console

    # both scenarios, full size (~3-5 min; numba compiles once per process)
    $ python benchmarks/benchmark_timestep.py

    # single scenario
    $ python benchmarks/benchmark_timestep.py --scenarios evolved_sediment

    # CI smoke preset: 40x40 grid, 50 parcels, 3 timed updates
    $ python benchmarks/benchmark_timestep.py --smoke

Every run writes two timestamped files into ``--output-dir`` (default
``benchmarks/results/``): one JSON file with the full data (including every
per-update time) and one Markdown summary, which is also echoed to stdout.

All knobs can be overridden for targeted experiments: ``--timed-updates``,
``--warm-up-updates``, ``--morphology-threshold``, ``--max-warmup-updates``,
and ``--scenarios`` (``both``, ``early_water``, or ``evolved_sediment``).


The smoke preset
================

``--smoke`` shrinks the run to what continuous integration can afford: a
40×40 grid (``Length: 2000``, ``Width: 2000``, ``dx: 50``), 50 parcels, one
free-surface iteration (``itermax = 1``), 2 warm-up updates followed by 3
timed updates per scenario, and a deposition-coverage threshold of 2%
capped at 5 warm-up updates. It exercises the harness end-to-end (model
build, both regimes, timing, and output writing) at negligible wall time,
which is why CI runs it.


Benchmarking in CI
==================

The ``benchmark.yml`` workflow (a fork addition, see
:doc:`/meta/about_this_fork`) runs the smoke preset on every pull request
and on pushes to ``develop``, and uploads the resulting JSON and Markdown
files as a ``benchmark-smoke-results`` artifact. The other CI jobs are
described in :doc:`developer_guide`.


Reading the numbers
===================

Per-update wall times are specific to the machine, the Python and numba
versions, and the exact scenario configuration, so treat a single run as
**workload-specific** rather than absolute. Compare only numbers from the
same environment and scenario configuration, and rely on the recorded
metadata (versions, git commit, seed, grid) to confirm that a comparison is
valid. Port pull requests must report before/after numbers from this
harness against a recorded baseline — there is no speedup claim without
one (see ``benchmarks/README.md``).
