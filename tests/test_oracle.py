import numpy as np
import pytest
from pymatgen.analysis.diffraction.neutron import NDCalculator
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator


def _assert_matches_pymatgen(structure, mode="xray", **kwargs):
    calculator = BraggCalculator(mode=mode, **kwargs).load(structure)
    actual_x, actual_y = calculator.line_pattern(scaled=True)
    oracle_type = XRDCalculator if mode == "xray" else NDCalculator
    oracle = oracle_type(
        wavelength=calculator.wavelength,
        debye_waller_factors=dict(calculator.debye_waller_factors),
    ).get_pattern(
        structure,
        two_theta_range=calculator.two_theta_range,
        scaled=True,
    )
    np.testing.assert_allclose(actual_x, oracle.x, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual_y, oracle.y, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("mode", ["xray", "neutron"])
def test_nacl_matches_oracle(nacl, mode):
    _assert_matches_pymatgen(nacl, mode=mode)


def test_equivalent_sites_are_not_discarded(strontium_titanate):
    calculator = BraggCalculator().load(strontium_titanate)
    assert len(calculator._symm["structure"]) == 5
    assert np.count_nonzero(calculator._symm["Z"] == 8) == 3
    _assert_matches_pymatgen(strontium_titanate)


def test_triclinic_matches_oracle(triclinic_structure):
    _assert_matches_pymatgen(triclinic_structure)


def test_disordered_occupancies_match_oracle():
    structure = Structure(
        Lattice.cubic(4.1),
        [{"Na": 0.7, "K": 0.3}, "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    calculator = BraggCalculator().load(structure)
    np.testing.assert_allclose(np.sort(calculator._symm["occ"]), [0.3, 0.7, 1.0])
    _assert_matches_pymatgen(structure)


def test_debye_waller_factors_match_oracle(strontium_titanate):
    _assert_matches_pymatgen(
        strontium_titanate,
        debye_waller_factors={"Sr": 0.4, "Ti": 0.25, "O": 0.75},
    )


def test_short_wavelength_range_is_not_truncated(nacl):
    _assert_matches_pymatgen(nacl, wavelength="AgKa1")


def test_conventional_and_primitive_cells_give_same_scaled_pattern():
    conventional = Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    primitive = conventional.get_primitive_structure()
    conv_x, conv_y = BraggCalculator().load(conventional).line_pattern(scaled=True)
    prim_x, prim_y = BraggCalculator().load(primitive).line_pattern(scaled=True)
    np.testing.assert_allclose(conv_x, prim_x, atol=1e-10)
    np.testing.assert_allclose(conv_y, prim_y, rtol=1e-10, atol=1e-10)
