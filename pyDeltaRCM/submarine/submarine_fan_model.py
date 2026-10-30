"""Submarine-fan variant of the DeltaModel.

This module defines :obj:`SubmarineFanModel`, a subclass of
:obj:`~pyDeltaRCM.model.DeltaModel` that repurposes the stock parcel-routing
and two-fraction sediment machinery to route turbidity currents and build
submarine-fan-like deposits on a submerged slope-to-basin domain.
"""

from pyDeltaRCM.model import DeltaModel


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
