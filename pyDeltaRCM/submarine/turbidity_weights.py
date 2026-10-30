"""Bed-following water-weight kernel for the submarine-fan model.

This module provides the jitted replacement for the stock water-weight kernel
(:obj:`pyDeltaRCM.water_tools._get_water_weight_array`) used by
:obj:`pyDeltaRCM.submarine.SubmarineFanModel`.

The substitution is a single physics change: the slope (``weight_sfc``) term
is computed from the *bed* elevation ``eta`` instead of the ``stage`` field,
so the downslope pull follows bed topography -- the defining behavior of a
density current -- rather than a free surface relaxing toward a flat sea
level. The inertia term, the ``(depth * mod)**theta`` weighting, the
normalization, and every zero-weight fallback are the stock
:obj:`pyDeltaRCM.water_tools._get_weight_at_cell_water` code, reused
unchanged (feasibility spike ``spikes/SPIKE_DECISION.md``, spec ``V1``).
"""

import numpy as np
from numba import njit

from pyDeltaRCM.shared_tools import custom_pad, get_weight_sfc_int
from pyDeltaRCM.water_tools import _get_weight_at_cell_water


@njit
def _get_bed_following_water_weight_array(
    depth,
    eta,
    mod_water_weight,
    cell_type,
    qx,
    qy,
    ivec_flat,
    jvec_flat,
    distances_flat,
    dry_depth,
    gamma,
    theta_water,
):
    """Bed-following replacement for ``water_tools._get_water_weight_array``.

    Structurally identical to the stock kernel, with exactly one physics
    substitution: the slope (``weight_sfc``) term is computed from the bed
    elevation ``eta`` instead of the ``stage`` field, so ``weight_sfc``
    becomes a downdip bed-gravity term pulling parcels toward lower bed
    elevation. The inertia term, the ``(depth * mod)**theta`` weighting, and
    every zero-weight fallback are the stock
    :obj:`pyDeltaRCM.water_tools._get_weight_at_cell_water` code, reused
    unchanged.

    Parameters
    ----------
    depth : ndarray
        Flow thickness field (the density current's thickness), shape
        ``(L, W)``.
    eta : ndarray
        Bed elevation field, shape ``(L, W)``. Takes the role the ``stage``
        field plays in the stock kernel.
    mod_water_weight : ndarray
        Per-cell water-weight modifier array, shape ``(L, W)``.
    cell_type : ndarray
        Domain cell-type field, shape ``(L, W)``.
    qx, qy : ndarray
        Discharge-component fields, shape ``(L, W)`` (the inertia term).
    ivec_flat, jvec_flat, distances_flat : ndarray
        Flattened D8 neighbor unit-vector and distance arrays, from
        :obj:`pyDeltaRCM.init_tools.set_constants`.
    dry_depth : float
        Depth below which a neighbor is treated as dry.
    gamma : float
        Blend of the bed-gravity term (``gamma_density``) against the
        inertia term.
    theta_water : float
        Exponent of the ``(depth * mod)**theta`` weighting.

    Returns
    -------
    ndarray
        Step-direction weights, shape ``(L, W, 9)``. Neighborhood order is
        the stock D8 order: 0=NW 1=N 2=NE 3=W 4=self 5=E 6=SW 7=S 8=SE.
    """
    L, W = depth.shape
    pad_eta = custom_pad(eta)
    pad_depth = custom_pad(depth)
    mod_pad_water_weight = custom_pad(mod_water_weight)
    pad_cell_type = custom_pad(cell_type)

    water_weights = np.zeros((L, W, 9))

    for i in range(L):
        for j in range(W):
            eta_nbrs = pad_eta[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]
            depth_nbrs = pad_depth[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]
            mod_water_weight_nbrs = mod_pad_water_weight[
                i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1
            ]
            ct_nbrs = pad_cell_type[i - 1 + 1 : i + 2 + 1, j - 1 + 1 : j + 2 + 1]

            # the substitution: bed elevation takes the role of stage, so the
            #   slope term is a bed-gravity term toward lower bed elevation
            weight_sfc, weight_int = get_weight_sfc_int(
                eta[i, j],
                eta_nbrs.ravel(),
                qx[i, j],
                qy[i, j],
                ivec_flat,
                jvec_flat,
                distances_flat,
            )

            water_weights[i, j] = _get_weight_at_cell_water(
                (i, j),
                weight_sfc,
                weight_int,
                depth_nbrs.ravel(),
                mod_water_weight_nbrs.ravel(),
                ct_nbrs.ravel(),
                dry_depth,
                gamma,
                theta_water,
            )

    return water_weights
