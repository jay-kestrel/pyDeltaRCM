**************
pyDeltaRCM
**************

.. image:: https://badge.fury.io/py/pyDeltaRCM.svg
    :target: https://badge.fury.io/py/pyDeltaRCM

.. image:: https://joss.theoj.org/papers/10.21105/joss.03398/status.svg
   :target: https://doi.org/10.21105/joss.03398

.. image:: https://github.com/DeltaRCM/pyDeltaRCM/actions/workflows/build.yml/badge.svg
    :target: https://github.com/DeltaRCM/pyDeltaRCM/actions

.. image:: https://codecov.io/gh/DeltaRCM/pyDeltaRCM/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/DeltaRCM/pyDeltaRCM

.. image:: https://app.codacy.com/project/badge/Grade/1c137d0227914741a9ba09f0b00a49a7
    :target: https://app.codacy.com/gh/DeltaRCM/pyDeltaRCM/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade




*pyDeltaRCM* is a computationally efficient, free and open source, and easy-to-customize numerical delta model based on the original DeltaRCM model design (`Matlab deltaRCM <https://csdms.colorado.edu/wiki/Model:DeltaRCM>`_ model by Man Liang; `Liang et al., 2015 <https://doi.org/10.5194/esurf-3-67-2015>`_).
*pyDeltaRCM* delivers improved model stability and capabilities, infrastructure to support exploration with minimal boilerplate code, and establishes an approach to extending model capabilities that ensures reproducible and comparable studies.


.. figure:: https://deltarcm.org/pyDeltaRCM/_images/cover.png

    Weighted random walks for 20 water parcels, in a *pyDeltaRCM* model run with default parameters.


Documentation
#############

`Find the complete documentation here <https://deltarcm.org/pyDeltaRCM/index.html>`_.

Documentation includes an `installation guide <https://deltarcm.org/pyDeltaRCM/meta/installing.html>`_, a thorough `guide for users <https://deltarcm.org/pyDeltaRCM/guides/user_guide.html>`_, detailed `API documentation for developers <https://deltarcm.org/pyDeltaRCM/reference/index.html>`_, a `plethora of examples <https://deltarcm.org/pyDeltaRCM/examples/index.html>`_ to use and develop pyDeltaRCM in novel scientific experiments, and more!


Installation
############

See our complete `installation guide <https://deltarcm.org/pyDeltaRCM/meta/installing.html>`_, especially if you are a developer planning to modify or contribute code (`developer installation guide <https://deltarcm.org/pyDeltaRCM/meta/installing.html#developer-installation>`_), or if you are new to managing Python `venv` or `conda` environments.

For a quick installation into an existing Python 3.x environment:

.. code:: console

    $ pip install pyDeltaRCM


Native water backend (this fork)
################################

Source installs on this branch require Python 3.11 or newer, a Rust toolchain
with Cargo, and a platform C linker. With those installed, run
``python -m pip install -e .`` from the checkout; maturin builds the PyO3
extension in release mode. No prebuilt wheel matrix is provided here.

Water-parcel routing, discharge accumulation, and free-surface reconstruction
use one native call per water iteration by default. Set the environment
variable ``PYDELTARCM_NATIVE=0`` before running the model to retain the original
Python/Numba implementation. The switch changes runtime dispatch, not build
requirements. Numba remains required for sediment routing and shared kernels.

The native RNG is seeded from the existing checkpointed random stream.
Identical native runs replay deterministically, but native and reference
trajectories deliberately differ; compare their distributions, not their
individual parcel paths. Keep the backend fixed when resuming a checkpoint
if exact continuation matters. Legacy golden-output tests explicitly select
the reference backend; separate native/reference ensemble tests exercise
bed relief, deposition volume, and channel-mouth flux.

Overridden water-phase internals or hooks, custom field precision, and
nonstandard walk geometry retain the reference call sequence rather than
silently bypassing subclass behavior. An unbuilt source checkout warns and
uses the reference backend; a broken installed extension raises its import
error. See ``benchmarks/README.md`` for reproducible backend comparisons.


Executing the model
###################

We recommend you check out our `pyDeltaRCM in 10 minutes tutorial <https://deltarcm.org/pyDeltaRCM/guides/10min.html>`_, which is part of our documentation.

Beyond that brief tutorial, we have a comprehensive `User Documentation <https://deltarcm.org/pyDeltaRCM/index.html#user-documentation>`_ and `Developer Documentation <https://deltarcm.org/pyDeltaRCM/index.html#developer-documentation>`_ to check out.


COBOL kernels (experimental)
############################

The weighted parcel direction pick -- :func:`pyDeltaRCM.shared_tools.random_pick`,
the arithmetic at the core of every per-parcel water and sediment routing
step -- has an optional GnuCOBOL backend (``cobol/random_pick.cob``),
compiled to a shared object and called in-process through a C shim and
ctypes::

	make -C cobol          # requires GnuCOBOL 3.x (e.g. apt gnucobol3)

Set ``PYDELTARCM_COBOL=1`` to route the ported pick through the COBOL
kernel; the flag takes precedence over the shared ``PYDELTARCM_NATIVE``
variable, which continues to govern the fused Rust water phase. With
routing off, behavior is byte-identical to the reference implementation.
There is no silent fallback: setting ``PYDELTARCM_COBOL=1`` without a
built shared object raises an error naming the build step. Random
numbers are always drawn in Python (the checkpointed MT19937 stream is
unchanged); the kernel receives pre-drawn uniforms and is a pure
function of its input buffers. The routine is also timed per backend by
``benchmarks/benchmark_timestep.py``, which asserts exact pick parity
between backends on its workload before reporting timings.


Citing pyDeltaRCM
#################

When citing *pyDeltaRCM*, please cite the `JOSS paper <https://doi.org/10.21105/joss.03398>`_:

	Moodie et al., (2021). pyDeltaRCM: a flexible numerical delta model. Journal of Open Source Software, 6(64), 3398, https://doi.org/10.21105/joss.03398
	
If you use BibTeX, you can add *pyDeltaRCM* to your `.bib` file using the following code:

.. code:: console

	@article{Moodie2021,
	doi = {10.21105/joss.03398},
	url = {https://doi.org/10.21105/joss.03398},
	year = {2021},
	publisher = {The Open Journal},
	volume = {6},
	number = {64},
	pages = {3398},
	author = {Andrew J. Moodie and Jayaram Hariharan and Eric Barefoot and Paola Passalacqua},
	title = {*pyDeltaRCM*: a flexible numerical delta model},
	journal = {Journal of Open Source Software}
	}


Additional notes
################

This repository no longer includes the `Basic Model Interface (BMI) <https://bmi.readthedocs.io/en/latest/?badge=latest>`_ wrapper to the DeltaRCM model.
*pyDeltaRCM* maintains BMI compatibility through another repository (`the BMI_pyDeltaRCM model <https://deltarcm.org/BMI_pyDeltaRCM/>`_).
