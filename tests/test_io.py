from pathlib import Path

import numpy as np
import pytest
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

from braggcalculator import BraggCalculator
from braggcalculator.io import to_pmg_structure


def test_io_rejects_unknown():
    with pytest.raises(TypeError):
        to_pmg_structure(object())


def test_io_rejects_non_cif_path():
    with pytest.raises(TypeError):
        to_pmg_structure("structure.xyz")


def test_cif_path_is_loaded():
    structure = to_pmg_structure("demo/NaCl.cif")
    assert isinstance(structure, Structure)
    assert structure.composition.reduced_formula == "NaCl"


@pytest.mark.parametrize("path", ["demo/NaCl.cif", Path("demo/NaCl.cif")])
def test_calculator_accepts_cif_paths(path):
    calculator = BraggCalculator().load(path)
    actual_x, actual_y = calculator.line_pattern(scaled=True)
    expected = XRDCalculator(wavelength=calculator.wavelength).get_pattern(
        to_pmg_structure(path),
        two_theta_range=calculator.two_theta_range,
        scaled=True,
    )
    np.testing.assert_allclose(actual_x, expected.x, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual_y, expected.y, rtol=1e-10, atol=1e-10)


def test_pymatgen_structure_is_returned_without_copy():
    structure = Structure(Lattice.cubic(4.0), ["Cs", "Cl"], [[0, 0, 0], [0.5] * 3])
    assert to_pmg_structure(structure) is structure


def test_ase_input_is_supported():
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms(
        symbols="NaCl",
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    structure = to_pmg_structure(atoms)
    assert structure.composition.reduced_formula == "NaCl"
