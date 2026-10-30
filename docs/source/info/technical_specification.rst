.. _technical_specification:

***********************
Technical specification
***********************

This page is the as-built technical specification of *pyDeltaRCM* as it
stands on this repository's ``develop`` branch (version 2.3.0, October
2026): the numerics, kernels, and invariants of the model as they are
actually coded. It complements the conceptual documentation in this
section — :doc:`hydrodynamics` and :doc:`morphodynamics` explain *what*
the model simulates; this page pins down *how* it is implemented.


Model structure and update sequence
===================================

The :obj:`~pyDeltaRCM.model.DeltaModel` class is composed from the
package's modules as mix-ins: :mod:`~pyDeltaRCM.init_tools` (domain,
derived quantities, and output-file initialization),
:mod:`~pyDeltaRCM.water_tools` (flow routing),
:mod:`~pyDeltaRCM.sed_tools` (sediment routing and bed evolution),
:mod:`~pyDeltaRCM.iteration_tools` (timestep orchestration, output, and
checkpoints), and :mod:`~pyDeltaRCM.shared_tools` (jitted utilities shared
by the other modules). Extension points are provided by
:mod:`~pyDeltaRCM.hook_tools`; command-line and campaign behavior by
:mod:`~pyDeltaRCM.preprocessor`.

At initialization, the model imports the configuration (a YAML file
and/or keyword arguments), determines the random seed, applies the
configuration to model properties, creates the derived variables and the
model domain, initializes the sediment router classes, and sets up the
subsidence fields.

Each call to :meth:`~pyDeltaRCM.model.DeltaModel.update` advances the
model one timestep through three stages defined in
:mod:`~pyDeltaRCM.iteration_tools`:

1. ``solve_water_and_sediment_timestep`` — route the water parcels, then
   route the sand and mud parcels and apply deposition, erosion, and
   topographic diffusion;
2. ``apply_subsidence`` — apply the subsidence field, if enabled;
3. ``finalize_timestep`` — advance the model time and counters, recompute
   the sand-fraction field, and write outputs and checkpoints on their
   configured cadences.


Water routing
=============

Water is routed as ``Np_water`` parcels per timestep performing weighted
random walks on the grid, across ``itermax`` flow iterations per update.

For each parcel step, a weight is assigned to each of the eight neighbor
cells from the free surface and the inertial continuation of the parcel's
direction (:func:`~pyDeltaRCM.shared_tools.get_weight_sfc_int` and
``water_tools._get_weight_at_cell_water``), with neighbor distances
weighted ``1`` for axis-aligned and :math:`\sqrt{2}` for diagonal moves
(D8). The relative weighting of free-surface and inertial contributions
is governed by ``gamma = g * S0 * dx / u0**2``. One neighbor is chosen
with a weighted random draw (:func:`~pyDeltaRCM.shared_tools.random_pick`),
and the parcel's discharge is accumulated into the directionally split
(``_update_dirQfield``) and absolute (``_update_absQfield``) discharge
fields. Parcels repeat until they leave the flow domain, are relocated by
the loop check (``_check_for_loops``), or exhaust ``stepmax`` jumps
(default ``2 * (L + W)``).

After the parcel walks, the accumulated discharge is converted to a flow
field: the free surface is accumulated along the walked paths
(``_accumulate_free_surface_walks``) and smoothed with an
under-relaxation scheme (``_smooth_free_surface``; ``Nsmooth`` iterations
per flow iteration, ``Csmooth`` smoothing coefficient), with the surface
and flow blended between flow iterations using the under-relaxation
factors ``omega_sfc`` and ``omega_flow`` (the per-iteration flow factor is
``omega_flow_iter = 2 / itermax``). See :doc:`hydrodynamics` for the
conceptual description of this scheme.


Sediment routing
================

Sediment is routed after water, as sand and mud parcels started in
proportion to ``f_bedload``: ``int(Np_sed * f_bedload)`` sand parcels and
``int(Np_sed * (1 - f_bedload))`` mud parcels. Both routers use the same
weighted random-walk machinery, with weights computed from the flow field
computed in the water phase and bed-slope and inertia terms
(``sed_tools._get_weight_at_cell_sediment``); the inertia term is scaled
per class by ``coeff_theta_sand`` (default 2.0) and ``coeff_theta_mud``
(default 1.0). Sand and mud routing are implemented by two numba
``jitclass`` es, ``SandRouter`` and ``MudRouter`` (see
`Jitted kernels`_).

Each parcel's deposition or erosion decision compares the local flow
velocity against critical thresholds — mud deposits below
``coeff_U_dep_mud`` times the characteristic velocity and erodes above
``coeff_U_ero_mud``; sand erodes above ``coeff_U_ero_sand`` — and the
eroded volume follows a power law in the local velocity with exponent
``beta`` (default 3):

.. math::

    V_\mathrm{ero} = m\, V_{p,\mathrm{sed}}\,
    \frac{U^\beta_\mathrm{loc} - U^\beta_\mathrm{ero}}{U^\beta_\mathrm{ero}}

where :math:`m` is a local erosion modifier and :math:`V_{p,\mathrm{sed}}`
is the parcel volume. Sand transport is additionally capacity-limited by a
transport capacity proportional to the local velocity to the ``beta``
power. See :doc:`morphodynamics` for the conceptual treatment.

After deposition, the bed is smoothed by topographic diffusion
(``topo_diffusion``): discrete convolutions (``scipy.ndimage.convolve``)
of the bed and sediment-flux fields, applied ``N_crossdiff`` substeps per
timestep — ``N_crossdiff = round(dVs / V0)`` — with a stability-scaled
multiplier ``diffusion_multiplier = dt / N_crossdiff * alpha * 0.5 / dx**2``.


Jitted kernels
==============

Performance-critical steps are compiled with
`numba <https://numba.pydata.org/>`_. The jitted surface consists of 19
``@njit`` functions and 2 ``jitclass`` es:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Kernels
   * - ``shared_tools`` (9)
     - ``set_random_seed``, ``get_random_uniform``, ``get_start_indices``,
       ``get_steps``, ``random_pick``, ``custom_unravel``,
       ``custom_ravel``, ``custom_pad``, ``get_weight_sfc_int``
   * - ``water_tools`` (9)
     - ``_get_weight_at_cell_water``, ``_get_water_weight_array``,
       ``_choose_next_directions``, ``_calculate_new_inds``,
       ``_check_for_loops``, ``_update_dirQfield``, ``_update_absQfield``,
       ``_accumulate_free_surface_walks``, ``_smooth_free_surface``
   * - ``sed_tools`` (1 + 2)
     - ``_get_weight_at_cell_sediment``; the ``SandRouter`` and
       ``MudRouter`` ``jitclass`` es

The ``jitclass`` es are compiled from a typed specification (``r_spec``):
model fields (``eta``, ``stage``, ``depth``, ``qx``, ``qy``, ``uw``, ...)
are ``float32`` arrays, cell-type and index arrays (``cell_type``,
``px``, ``py``, the walk buffers) are ``int64``, scalar configuration
(``_dt``, ``_dx``, ``stepmax``, ``dry_depth``) is ``float32``, and
``force_deposit`` is a boolean. ``jitclass`` is a numba experimental
feature (``numba.experimental.jitclass``). Note that numba compilation is
disabled for coverage runs (see :doc:`/guides/developer_guide`).


Field storage and indexing
==========================

The core state fields — bed elevation ``eta``, water surface ``stage``,
``depth``, discharge components ``qx``/``qy`` and their per-iteration
working counterparts ``qxn``/``qyn``/``qwn``, and velocity components —
are stored as ``float32`` arrays of shape ``(L, W)``.

Parcels are tracked by flattened (C-order) array indices.
:func:`~pyDeltaRCM.shared_tools.custom_ravel` and
:func:`~pyDeltaRCM.shared_tools.custom_unravel` convert between
``(i, j)`` tuples and flat indices (both jitted, with ``int64``
signatures).

Neighbor access uses
:func:`~pyDeltaRCM.shared_tools.custom_pad`, a jitted function equivalent
to ``np.pad(arr, 1, 'edge')``: arrays are padded with a one-cell ring
that replicates edge values, so every cell — corners included — has a
full 3×3 neighborhood to slice. This pattern is used pervasively in the
routing code and is the recommended neighbor-slicing approach for
subclass development (see :doc:`/guides/developer_guide`).


Randomness, determinism, and checkpoints
========================================

All model variability flows through the numba random number generator:

* The ``seed`` parameter (default ``null``) is resolved at initialization
  by ``determine_random_seed``: if unspecified, a random seed is drawn in
  ``[0, 2**32 - 1]`` and used; the seed value is always written to the
  log for the record.
* ``shared_tools.set_random_seed`` seeds the generator by calling
  ``np.random.seed`` *inside* a jitted function — crucially, this seeds
  the numba RNG (the one the parcel walks consume), not a Python-side
  generator. Valid seeds are integers in ``[0, 2**32]``.
* Every stochastic draw in the model goes through the jitted helpers
  ``get_random_uniform`` and ``random_pick``; drawing randoms outside
  jitted code breaks reproducibility (see
  :doc:`/guides/developer_guide`).
* Checkpoints are saved every ``checkpoint_dt`` (which defaults to
  ``save_dt`` when unset) when ``save_checkpoint`` is enabled. Loading a
  checkpoint restores the model state *and* the RNG state: instead of
  re-seeding, the saved generator state tuple is restored via
  ``shared_tools.set_random_state``, so a resumed run continues the exact
  random stream.
* If a checkpoint is resumed with a changed timestep, the subsidence
  field is rescaled by the ratio of the new timestep to the saved one.


The model timestep
==================

The timestep :math:`\Delta t` balances computation efficiency against
model stability, scaled by the reference volume of an inlet-channel cell
(:doc:`initialization`) and the sediment
inflow:

.. math::

    \Delta t = \frac{0.1\, {N_0}^2\, V_0}{Q_{s0}}

with the derived quantities (all computed in
``init_tools.create_other_variables``):

* ``V0 = h0 * dx**2`` — the reference volume (fill one cell to the
  characteristic depth ``h0``);
* ``Qw0 = u0 * h0 * N0 * dx`` — the constant water inflow discharge;
* ``dVs = 0.1 * N0**2 * V0`` — the total sediment volume added per
  timestep;
* ``Qs0 = Qw0 * C0`` — the total sediment input discharge, with
  ``C0 = C0_percent / 100``;
* ``Vp_sed = dVs / Np_sed`` — the volume of each sediment parcel.

The derivation follows the reduced-complexity formulation of `Liang,
Voller, and Paola (2015) <https://doi.org/10.5194/esurf-3-67-2015>`_.
Other stability-relevant derived quantities: ``u_max = 2 * u0`` (velocity
cap), ``dry_depth = min(0.1, 0.1 * h0)`` (dry-cell threshold), and
``stepmax = 2 * (L + W)`` (parcel jump cap).


Subsidence and the active layer
===============================

Subsidence is enabled with ``toggle_subsidence`` and proceeds at
``subsidence_rate`` (default :math:`2\times10^{-9}` m/s) beginning at
model age ``start_subsidence`` (default 216000 s). At initialization, the
subsidence field is computed as ``sigma = subsidence_mask * subsidence_rate * dt``
over the masked region, and it is applied to the bed each timestep by
``iteration_tools.apply_subsidence``.

The sand-fraction field is tracked through an active layer of thickness
``active_layer_thickness``, which defaults to ``h0 / 2`` and must be
positive. ``iteration_tools.compute_sand_frac`` updates the sand fraction
each timestep: where erosion does not exceed the active layer, the layer
simply exchanges sediment with the flow; where erosion digs into the
layer, ``sand_frac`` is reset to the underlying active layer value.


YAML parameters and validation
==============================

Every configurable parameter is declared in ``default.yml``, which ships
with the package and is loaded at model initialization. Each entry
declares a ``type`` — a single string (``'int'``, ``'float'``, ``'str'``,
``'bool'``) or a list of allowed alternatives, e.g.
``['float', 'int', 'None']`` — and a ``default`` value.

When a model is constructed, values from the YAML file and keyword
arguments (and, for command-line runs, CLI options) are merged into one
configuration, and each parameter is assigned through a typed
``DeltaModel`` property whose setter validates and normalizes the value:
``None`` defaults resolve to derived values (e.g.,
``active_layer_thickness`` defaults to ``h0 / 2``, ``checkpoint_dt`` to
``save_dt``, ``stepmax`` to ``2 * (L + W)``), and out-of-range or
wrong-typed values raise errors.

The complete schema — every parameter with its type and default — is
rendered in :doc:`/reference/model/yaml_defaults`, and the parameters'
physical meaning is described in :doc:`yamlparameters`.
