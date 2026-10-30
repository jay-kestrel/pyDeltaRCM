.. _about_this_fork:

***************
About this fork
***************

This repository — ``jay-kestrel/pyDeltaRCM`` — is a fork of the upstream
project `DeltaRCM/pyDeltaRCM <https://github.com/DeltaRCM/pyDeltaRCM>`_,
maintained by Kestrel for benchmarking and documentation work on top of the
upstream model.
The fork tracks upstream's ``develop`` branch, and this page describes what
is the same as upstream, what is different, and what to expect from the
project links and badges distributed through this site.


What is the same
================

The model code is the upstream *pyDeltaRCM* model: the same
reduced-complexity delta model documented throughout this site, described in
:doc:`the JOSS citation <citing>`, and released under the same
:doc:`MIT license <license>`.

Unless a page explicitly notes a fork-specific addition (as this page and
:doc:`the benchmarking guide </guides/benchmarking>` do), documentation on
this site describes the shared upstream/fork model.


What is different
=================

The fork's divergence from upstream is deliberately small and additive.
As of October 2026 the fork is exactly **one commit ahead** of upstream
``develop`` (and zero behind): the two-regime per-timestep benchmark
harness that measures :meth:`~pyDeltaRCM.model.DeltaModel.update` wall time
in water-dominated and sediment-dominated flow regimes
(:doc:`the benchmarking guide </guides/benchmarking>`).

Fork-specific documentation added on top of the upstream docs:

* this orientation page,
* a repository-level ``CONTRIBUTING.md`` pointing contributors at the
  :doc:`contributing documentation <contributing>`,
* :doc:`the benchmarking guide </guides/benchmarking>` for the benchmark
  harness,
* a :doc:`command line interface reference </reference/preprocessor/cli>`,
* :doc:`a technical specification </info/technical_specification>`, and
* a ``CHANGELOG.md`` seeded at the version current when the fork docs were
  written.

All of these are additive: no upstream file is rewritten on the fork, with
the small exception of the toctree entries that make the new pages
discoverable.


What still points upstream
==========================

Several user-facing identifiers continue to refer to the upstream project,
**on purpose**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - What
     - Points at
   * - Badges in ``README.rst`` and on this site (PyPI version, GitHub
       releases, GitHub Actions, Codecov, Codacy, JOSS status)
     - The upstream repository ``DeltaRCM/pyDeltaRCM``, so they report
       upstream CI, coverage, and release status — not this fork's.
   * - ``pip install pyDeltaRCM``
     - The upstream project's release on PyPI. It does **not** install this
       fork's code, including the benchmark harness.
   * - The ``deltarcm.org/pyDeltaRCM`` documentation links and the
       ``documentation``, ``homepage``, and ``repository`` URLs in
       ``pyproject.toml``
     - The upstream project and its rendered documentation.
   * - The issue-tracker links in the :doc:`contributing <contributing>`
       documentation
     - The upstream issue tracker.

This is a deliberate fork policy rather than an oversight: the fork tracks
upstream closely, and rewriting badges, URLs, and links that upstream also
edits would create merge conflicts on every upstream sync.
This page is the single, conflict-free place where the fork's identity is
documented.


Using this fork
===============

To run the fork's code rather than the upstream release, install from a
clone of this repository, following the
:ref:`developer installation <dev-install>` instructions, for example:

.. code:: console

    $ git clone https://github.com/jay-kestrel/pyDeltaRCM.git
    $ cd pyDeltaRCM
    $ pip install .

The benchmark harness can then be run as described in
:doc:`the benchmarking guide </guides/benchmarking>`.
