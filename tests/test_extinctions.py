import numpy as np
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator


def _f2_for_hkl(calculator, hkl):
    matches = np.flatnonzero(np.all(calculator._hkl["hkl"] == hkl, axis=1))
    assert len(matches) == 1, hkl
    return float(calculator.fq(indices=matches)[0])


def test_body_centering_extinction_is_generated_by_structure_factor():
    structure = Structure(
        Lattice.cubic(3.0),
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    calculator = BraggCalculator(primitive=False).load(structure)
    assert _f2_for_hkl(calculator, [1, 0, 0]) < 1e-24
    assert _f2_for_hkl(calculator, [1, 1, 0]) > 1.0


def test_face_centering_extinctions_are_generated_by_structure_factor():
    structure = Structure(
        Lattice.cubic(4.0),
        ["Cu"] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    calculator = BraggCalculator(primitive=False).load(structure)
    assert _f2_for_hkl(calculator, [1, 0, 0]) < 1e-24
    assert _f2_for_hkl(calculator, [1, 1, 0]) < 1e-24
    assert _f2_for_hkl(calculator, [1, 1, 1]) > 1.0
    assert _f2_for_hkl(calculator, [2, 0, 0]) > 1.0


def test_diamond_basis_adds_222_extinction():
    structure = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.431), ["Si"], [[0, 0, 0]])
    calculator = BraggCalculator(primitive=False).load(structure)
    assert _f2_for_hkl(calculator, [2, 2, 2]) < 1e-24
    assert _f2_for_hkl(calculator, [1, 1, 1]) > 1.0
    assert _f2_for_hkl(calculator, [2, 2, 0]) > 1.0


def test_simple_cubic_first_powder_line_sums_six_reciprocal_points():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    calculator = BraggCalculator(two_theta_range=(1, 30)).load(structure)
    positions, individual = calculator.iq()
    first_position = float(np.min(positions))
    family = np.isclose(positions, first_position, atol=1e-10)
    assert np.count_nonzero(family) == 6
    lines, merged = calculator.line_pattern()
    assert merged[0] == np.sum(individual[family])
    assert lines[0] == first_position
