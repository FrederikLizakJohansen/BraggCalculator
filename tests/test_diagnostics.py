import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator
from braggcalculator.diagnostics import (
    apply_origin_shift,
    compare_calculators,
    match_reflections,
    mismatch_disk,
)


def test_reflection_matching_preserves_first_collection_order():
    hkl_a = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    hkl_b = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    match = match_reflections(hkl_a, hkl_b)
    np.testing.assert_array_equal(match.hkl, [[1, 0, 0], [1, 1, 0]])
    np.testing.assert_array_equal(match.indices_a, [0, 2])
    np.testing.assert_array_equal(match.indices_b, [1, 0])


def test_mismatch_disk_satisfies_identity_and_decomposition():
    hkl = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]])
    factor_a = np.array([2 + 1j, 1 - 2j, -3 + 0.5j, 0j])
    factor_b = np.array([1 + 2j, 2 - 1j, -2 - 1j, 1j])
    result = mismatch_disk(hkl, factor_a, factor_b, epsilon=1e-12)
    assert np.all(result.radius <= 1.0 + 1e-14)
    assert result.identity_error < 1e-14
    assert result.d_sf**2 == pytest.approx(
        result.d_amplitude**2 + result.d_phase**2, abs=1e-15
    )
    assert not result.phase_defined[-1]


def test_optimized_origin_removes_phase_ramp():
    hkl = np.array(
        [
            [h, k, ell]
            for h in range(-2, 3)
            for k in range(-2, 3)
            for ell in range(-2, 3)
        ]
    )
    hkl = hkl[np.any(hkl != 0, axis=1)]
    rng = np.random.default_rng(7)
    factor_a = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
    physical_shift = np.array([0.125, 0.25, 0.375])
    factor_b = apply_origin_shift(factor_a, hkl, physical_shift)
    unaligned = mismatch_disk(hkl, factor_a, factor_b)
    aligned = mismatch_disk(hkl, factor_a, factor_b, optimize_origin=True)
    assert unaligned.d_sf > 0.5
    assert aligned.d_sf < 1e-12
    np.testing.assert_allclose(
        (aligned.alignment.shift + physical_shift) % 1.0,
        np.zeros(3),
        atol=1e-12,
    )


def test_calculator_comparison_is_invariant_to_origin_and_atom_order():
    lattice = Lattice.from_parameters(4.1, 5.2, 6.3, 77, 83, 71)
    species = ["Si", "O", "N"]
    coordinates = np.array([[0.13, 0.21, 0.34], [0.31, 0.47, 0.11], [0.72, 0.08, 0.59]])
    shift = np.array([0.125, 0.25, 0.375])
    structure_a = Structure(lattice, species, coordinates)
    structure_b = Structure(lattice, species[::-1], (coordinates[::-1] + shift) % 1.0)
    calculator_a = BraggCalculator(q_range=(0.2, 6.0)).load(structure_a)
    calculator_b = BraggCalculator(q_range=(0.2, 6.0)).load(structure_b)
    result = compare_calculators(calculator_a, calculator_b, optimize_origin=True)
    assert len(result.match) > 10
    assert result.d_sf < 1e-12
    assert result.identity_error < 1e-14


def test_calculator_comparison_detects_structural_perturbation():
    lattice = Lattice.cubic(5.0)
    structure_a = Structure(lattice, ["Si", "O"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    structure_b = structure_a.copy()
    structure_b.translate_sites([1], [0.025, 0.0, 0.0], frac_coords=True)
    calculator_a = BraggCalculator(q_range=(0.2, 6.0)).load(structure_a)
    calculator_b = BraggCalculator(q_range=(0.2, 6.0)).load(structure_b)
    result = compare_calculators(calculator_a, calculator_b, optimize_origin=True)
    assert result.d_sf > 1e-3
    assert result.d_phase > 0
