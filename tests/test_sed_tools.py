# unit tests for sed_tools.py

import numpy as np
from pathlib import Path

import pytest
import unittest.mock as mock

from pyDeltaRCM.model import DeltaModel
from pyDeltaRCM import sed_tools
from . import utilities


class TestSedimentRoute:

    def test_route_sediment(self, tmp_path: Path) -> None:
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        _delta = DeltaModel(input_file=p)

        # mock top-level methods
        _delta.log_info = mock.MagicMock()
        _delta.init_sediment_iteration = mock.MagicMock()
        _delta.route_all_sand_parcels = mock.MagicMock()
        _delta.topo_diffusion = mock.MagicMock()
        _delta.route_all_mud_parcels = mock.MagicMock()

        # run the method
        _delta.route_sediment()

        # methods called
        assert _delta.log_info.call_count == 4
        assert _delta.init_sediment_iteration.called is True
        assert _delta.route_all_sand_parcels.called is True
        assert _delta.topo_diffusion.called is True
        assert _delta.route_all_mud_parcels.called is True


class TestInitSedimentIteration:

    def test_fields_cleared(self, tmp_path: Path) -> None:
        # create a delta with default settings
        p = utilities.yaml_from_dict(tmp_path, "input.yaml")
        _delta = DeltaModel(input_file=p)

        # alter field for initial values going into function
        _delta.qs += np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.Vp_dep_sand += np.random.uniform(0, 1, size=_delta.eta.shape)
        _delta.Vp_dep_mud += np.random.uniform(0, 1, size=_delta.eta.shape)

        assert not np.all(_delta.qs == 0)  # field has info

        # call the method
        _delta.init_sediment_iteration()

        # assertions
        assert np.all(_delta.pad_depth[1:-1, 1:-1] == _delta.depth)
        assert np.all(_delta.qs == 0)  # field is cleared
        assert np.all(_delta.Vp_dep_sand == 0)
        assert np.all(_delta.Vp_dep_mud == 0)


class TestRouteAllSandParcels:

    def test_route_sand_parcels(self, tmp_path: Path) -> None:
        # create a delta with default settings
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"Np_sed": 1000, "f_bedload": 0.6}
        )
        _delta = DeltaModel(input_file=p)

        # mock top-level methods / objects
        _delta.log_info = mock.MagicMock()
        _delta._sr = mock.MagicMock()

        # mock the shared tools start indices
        def _patched_starts(inlet, inlet_weights, num_starts):
            return np.random.randint(0, 5, size=(num_starts,))

        patcher = mock.patch(
            "pyDeltaRCM.shared_tools.get_start_indices", new=_patched_starts
        )
        patcher.start()

        # run the method
        _delta.route_all_sand_parcels()

        # methods called
        assert _delta._sr.run.call_count == 1
        assert _delta.log_info.call_count == 3

        # stop the patch
        patcher.stop()


class TestRouteAllMudParcels:

    def test_route_mud_parcels(self, tmp_path: Path) -> None:
        # create a delta with default settings
        p = utilities.yaml_from_dict(
            tmp_path, "input.yaml", {"Np_sed": 1000, "f_bedload": 0.6}
        )
        _delta = DeltaModel(input_file=p)

        # mock top-level methods / objects
        _delta.log_info = mock.MagicMock()
        _delta._mr = mock.MagicMock()

        # mock the shared tools start indices
        def _patched_starts(inlet, inlet_weights, num_starts):
            return np.random.randint(0, 5, size=(num_starts,))

        patcher = mock.patch(
            "pyDeltaRCM.shared_tools.get_start_indices", new=_patched_starts
        )
        patcher.start()

        # run the method
        _delta.route_all_mud_parcels()

        # methods called
        assert _delta._mr.run.call_count == 1
        assert _delta.log_info.call_count == 3

        # stop the patch
        patcher.stop()


class TestForceDeposit:
    """Test the force_deposit flag"""

    def test_force_deposit(self, tmp_path: Path) -> None:
        # create a small domain with low stepmax
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "Width": 2000,
                "Length": 1000,
                "Np_sed": 100,
                "stepmax": 5,
                "force_deposit": True,
            },
        )
        _delta = DeltaModel(input_file=p)

        vol_lost = 0
        for _ in range(10):
            _delta.update()
            vol_lost += _delta._sr.Vp_lost + _delta._mr.Vp_lost

        # should be no lost volume since force_deposit = True
        assert vol_lost == pytest.approx(0)


class TestGetWeightAtCellSedimentFallback:
    """Regression tests for the zero-weight fallback at row-0 cells.

    The padded ghost cells above row 0 are edge-replicated copies of the
    row-0 cell (channel, not walls). The fallback must never weight
    them, because a parcel that steps there wraps Python negative
    indices onto the outflow row.
    """

    def _fallback_weights(self, ind):
        distances_flat = np.array(
            [np.sqrt(2), 1, np.sqrt(2), 1, 1, 1, np.sqrt(2), 1, np.sqrt(2)],
            dtype=np.float32,
        )
        # all neighbors dry => main-path weights all zero => fallback fires
        depth_nbrs = np.zeros(9, dtype=np.float32)
        # ghost window entries (0-2) replicate the row-0 channel cell;
        # cells beside the channel in the padded window are walls (-2)
        ct_nbrs = np.array([1, 1, 1, -2, 1, -2, 0, 1, 0], dtype=np.int64)
        weight_int = np.zeros(9, dtype=np.float32)
        mod_sed_weight = np.ones(9, dtype=np.float32)
        return sed_tools._get_weight_at_cell_sediment(
            ind,
            weight_int,
            depth_nbrs,
            ct_nbrs,
            0.1,  # dry_depth
            1.0,  # theta
            distances_flat,
            mod_sed_weight,
        )

    def test_row0_fallback_excludes_ghost_cells(self) -> None:
        weights = self._fallback_weights((0, 5))
        # the ghost cells above row 0 get no weight
        assert np.all(weights[0:3] == 0)
        # walls (3, 5) and the center (4) remain excluded
        assert np.all(weights[[3, 4, 5]] == 0)
        # in-domain options remain and the weights are normalized
        assert np.any(weights[6:] > 0)
        assert weights.sum() == pytest.approx(1.0)

    def test_non_row0_fallback_keeps_up_directions(self) -> None:
        # away from row 0, window entries 0-2 are in-domain cells
        weights = self._fallback_weights((1, 5))
        assert np.any(weights[0:3] > 0)
        assert weights.sum() == pytest.approx(1.0)



class TestGhostCellWeightFallback:
    """Regression tests: pad-ghost cells must never be movement targets.

    The zero-weight fallback in ``_get_weight_at_cell_sediment`` masked
    candidate targets by wall type only. For row-0 cells, the padded
    ghost cells above the domain replicate the row-0 cell's own channel
    type, so they are not walls and received the largest window weight;
    a parcel picking one stepped out of the domain (``px == -1``), and
    the negative row index wrapped flux and bed updates onto the outflow
    row (the last row of the domain).
    """

    def test_get_weight_at_cell_excludes_ghost_cells(self) -> None:
        # a row-0 cell of an aggraded-dry channel entrance: every
        # in-domain neighbor is dry channel (not wall), and the padded
        # ghost cells above the domain replicate the channel type
        ind = (0, 4)
        weight_int = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1], dtype=np.float64)
        depth_nbrs = np.zeros(9, dtype=np.float64)
        ct_nbrs = np.ones(9, dtype=np.int64)
        distances_flat = np.array(
            [np.sqrt(2), 1, np.sqrt(2), 1, 1, 1, np.sqrt(2), 1, np.sqrt(2)],
            dtype=np.float32,
        )
        mod_sed_weight = np.ones(9, dtype=np.float64)

        wts = sed_tools._get_weight_at_cell_sediment(
            ind,
            weight_int,
            depth_nbrs,
            ct_nbrs,
            0.1,  # dry_depth
            2.0,  # theta_sed
            distances_flat,
            mod_sed_weight,
        )

        # no weight on the padded ghost cells above the domain
        assert np.sum(wts[0:3]) == 0
        # remaining weights are a valid in-domain distribution
        assert wts.sum() == pytest.approx(1)
        assert np.sum(wts[3:]) == pytest.approx(1)

    def test_route_sand_parcels_flux_never_reaches_outflow_row(
        self, tmp_path: Path
    ) -> None:
        # 20x10 domain with stepmax 5: no parcel can legitimately reach
        # the outflow row (row 19) -- walking covers at most 5 rows from
        # the inlet and sediment routing has no relocation jumps -- so
        # any flux or bed update there is the negative-index wrap.
        p = utilities.yaml_from_dict(
            tmp_path,
            "input.yaml",
            {
                "Length": 20.0,
                "Width": 10.0,
                "dx": 1.0,
                "L0_meters": 1.0,
                "N0_meters": 2.0,
                "h0": 1.0,
                "Np_sed": 100,
                "f_bedload": 0.5,
                "itermax": 1,
                "seed": 1,  # verified to route a parcel into the ghost cell
                "stepmax": 5,
                "verbose": 0,
            },
        )
        delta = DeltaModel(input_file=p)
        delta.log_info = mock.MagicMock()

        delta.init_sediment_iteration()
        eta_outflow_before = delta.eta[-1, :].copy()

        # aggrade-dry the channel entrance
        cols = delta.inlet
        delta.depth[0:3, cols[0] - 1 : cols[-1] + 2] = 0

        delta.route_all_sand_parcels()

        # no sediment flux may be written on the outflow row
        assert np.all(delta.qs[-1, :] == 0)
        # and the outflow-row bed must be untouched by routing
        assert np.all(delta.eta[-1, :] == eta_outflow_before)
