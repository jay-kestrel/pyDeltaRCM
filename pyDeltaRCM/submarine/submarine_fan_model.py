"""Submarine-fan variant of the DeltaModel.

This module defines :obj:`SubmarineFanModel`, a subclass of
:obj:`~pyDeltaRCM.model.DeltaModel` that repurposes the stock parcel-routing
and two-fraction sediment machinery to route turbidity currents and build
submarine-fan-like deposits on a submerged slope-to-basin domain.
"""

import warnings

import numpy as np

from pyDeltaRCM.model import DeltaModel


def _submarine_bed_profile(
    n_rows: int, dx: float, slope_gradient: float, relief: float
) -> np.ndarray:
    """Initial submarine bed cross-section, laterally uniform.

    The bed starts at the canyon head (0 m at row 0), descends downdip
    (increasing row) at ``slope_gradient``, and stays flat at the basin
    floor (``-relief``) once the total relief is exhausted. A domain
    shorter than ``relief / (slope_gradient * dx)`` cells never reaches
    the floor and is a planar slope throughout.

    Parameters
    ----------
    n_rows : int
        Number of domain rows (domain length in cells).
    dx : float
        Cell size, in meters.
    slope_gradient : float
        Bed gradient of the canyon-to-slope section.
    relief : float
        Vertical relief from the canyon head to the basin floor, in meters.

    Returns
    -------
    np.ndarray
        Bed elevations along the flow direction, shape ``(n_rows,)``.
    """
    drop = np.arange(n_rows) * dx * slope_gradient
    return -np.minimum(drop, relief)


class SubmarineFanModel(DeltaModel):
    """Turbidity-current / submarine-fan variant of pyDeltaRCM.

    Reuses the stock parcel-routing and sand/mud sediment machinery,
    re-parameterized so the flow follows the bed instead of a flat sea level.
    The ambient water column is static and is not simulated; the flow fields
    describe the density current.

    The submarine parameters are registered as custom YAML parameters in
    :obj:`hook_import_files`, so they are type-checked, defaulted, and logged
    by the stock configuration machinery exactly like stock parameters.

    Notes
    -----
    The derived density-current constants mirror the stock riverine constants
    from :obj:`~pyDeltaRCM.init_tools.init_tools.create_boundary_conditions`.
    In particular, the bed-gravity weight coefficient mirrors the stock
    ``gamma = g * S0 * dx / u0**2`` scaling, with the slope-gradient parameter
    taking the role of the characteristic slope ``S0``::

        gamma_density = coeff_gamma_density * g * slope_gradient * dx
                        / u_current**2
    """

    def __init__(self, input_file=None, **kwargs):
        # inherit base DeltaModel methods
        super().__init__(input_file, **kwargs)

    def hook_import_files(self) -> None:
        """Register the submarine-model YAML parameters.

        Declared here so the stock ``process_input_to_model`` machinery
        validates, defaults, and logs them like any stock parameter.
        Defaults are starting points for calibration, not validated
        constants.
        """
        # volumetric sediment concentration of the inflow; literature
        #   plunging range is ~0.02-0.03 in marine salinity
        self.subclass_parameters["current_concentration"] = {
            "type": ["float", "int"],
            "default": 0.025,
        }
        # upper-slope/canyon gradient for the initial bathymetry; canyon
        #   range 1e-2-1e-1, fan range 1e-3-1e-2
        self.subclass_parameters["slope_gradient"] = {
            "type": ["float"],
            "default": 0.03,
        }
        # inlet current thickness; a None default falls back to h0
        #   in create_other_variables
        self.subclass_parameters["current_thickness"] = {
            "type": ["float", "int"],
            "default": None,
        }
        # scales the bed-gravity weight coefficient gamma_density, mirroring
        #   the stock gamma
        self.subclass_parameters["coeff_gamma_density"] = {
            "type": ["float"],
            "default": 1.0,
        }

    def create_other_variables(self) -> None:
        """Derive the density-current constants after the stock ones.

        Overridden rather than using :obj:`hook_create_other_variables`
        because that hook fires *before* the base method, which defines the
        ``g`` and ``dx`` attributes the derivations below need.
        """
        super().create_other_variables()

        # inlet current thickness, falling back to the reference depth
        if self.current_thickness is None:
            self.h_current = self.h0
        else:
            self.h_current = self.current_thickness

        # the current inherits the inlet velocity scale of the riverine model
        self.u_current = self.u0

        # bed-gravity weight coefficient, mirroring the stock
        #   gamma = g * S0 * dx / u0**2
        self.gamma_density = (
            self.coeff_gamma_density * self.g * self.slope_gradient * self.dx
            / self.u_current**2
        )

    def hook_after_create_domain(self) -> None:
        """Replace the riverine domain with a submerged slope-to-basin one.

        The stock ``create_domain`` builds a planar delta topset with a
        channel discharging into a flat, uniform-depth basin. This hook
        replaces that bathymetry with a fully submerged
        canyon-head-to-basin-floor profile, so a density current released
        at the inlet has a bed to run down:

        - the bed descends downdip from the canyon head (row 0, at the
          inlet) at ``slope_gradient`` and flattens at the basin floor,
          ``hb`` below the head (the stock basin-depth parameter, resolved
          by ``create_domain`` as ``hb or h0``);
        - every cell is wet: former land cells (stock ``cell_type == -2``)
          become ocean, so the domain has no shoreline; the stock elliptical
          edge mask (``cell_type == -1``) is kept as the outflow boundary
          and the inlet channel cells (``cell_type == 1``) are untouched;
        - the flow fields describe the density current, which starts at
          rest: ``stage`` trails the bed at the constant current thickness
          ``h_current``, ``depth`` is that thickness everywhere, and the
          discharge and velocity fields start at zero;
        - sea level is pinned to the basin floor, strictly below the
          current's surface everywhere, so the stock stage floor
          (``stage = max(stage, H_SL)``) never modifies the current's stage
          field; v1 does not simulate the ambient sea or its rise, so a
          nonzero ``SLR`` is zeroed with a warning.
        """
        # total relief from the canyon head to the basin floor; _hb is
        #   resolved by create_domain as "hb or h0"
        relief = float(self._hb)

        # laterally uniform canyon-head-to-basin-floor bed
        self.eta[:] = _submarine_bed_profile(
            self.L, self.dx, self.slope_gradient, relief
        )[:, np.newaxis]
        self.stage[:] = self.eta + self.h_current
        self.depth[:] = self.h_current

        # fully submerged: former land cells become ocean; the stock
        #   elliptical edge mask (-1) stays as the outflow boundary
        self.cell_type[self.cell_type == -2] = 0

        # the flow fields describe the current, which starts at rest
        self.qx[:] = 0
        self.qy[:] = 0
        self.qw[:] = 0
        self.ux[:] = 0
        self.uy[:] = 0
        self.uw[:] = 0

        # refresh the bed trackers to the submarine initial condition
        self.eta0[:] = self.eta
        self.eta_init[:] = self.eta

        # v1 sea-level bookkeeping: the ambient sea is not simulated, so the
        #   sea level is pinned at the basin floor -- strictly below the
        #   current's surface everywhere (the current's depth h_current > 0),
        #   which keeps the stock stage floor max(stage, H_SL) from ever
        #   modifying the current's stage field (spike-verified invariant).
        #   Note: "kept above the whole domain" is geometrically incompatible
        #   with that invariant -- above-sea-level flooring would flatten the
        #   current's stage field onto H_SL.
        self.H_SL = -relief

        # v1 does not simulate sea-level rise; a rising H_SL could eventually
        #   reach the current's surface and bind the stage floor, flattening
        #   the current, so it is zeroed.
        if self.SLR != 0.0:
            _msg = (
                "SLR is not simulated by the SubmarineFanModel in v1; the "
                f"configured rate ({self.SLR} m/s) is ignored and SLR is "
                "set to 0."
            )
            self.log_warning(_msg)
            warnings.warn(UserWarning(_msg))
            self.SLR = 0.0
