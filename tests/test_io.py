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


def test_cif_isotropic_displacements_are_preserved(tmp_path):
    path = tmp_path / "isotropic.cif"
    path.write_text(
        """data_isotropic
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_U_iso_or_equiv
Na1 Na 0 0 0 0.0045
Cl1 Cl 0.5 0.5 0.5 0.0035
"""
    )
    calculator = BraggCalculator(primitive=False).load(path)
    np.testing.assert_allclose(
        np.sort(calculator._symm["B"]),
        8 * np.pi**2 * np.array([0.0035, 0.0045]),
    )


def test_cif_anisotropic_displacements_are_preserved_in_cartesian_form(tmp_path):
    path = tmp_path / "anisotropic.cif"
    path.write_text(
        """data_anisotropic
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.13 0.24 0.31
loop_
_atom_site_aniso_label
_atom_site_aniso_U_11
_atom_site_aniso_U_22
_atom_site_aniso_U_33
_atom_site_aniso_U_23
_atom_site_aniso_U_13
_atom_site_aniso_U_12
Si1 0.004 0.007 0.012 0.001 -0.0005 0.0008
"""
    )
    calculator = BraggCalculator(primitive=False).load(path)
    expected = np.array([[0.004, 0.0008, -0.0005], [0.0008, 0.007, 0.001], [-0.0005, 0.001, 0.012]])
    np.testing.assert_allclose(calculator._symm["U_cart"][0], expected)
    assert calculator._symm["has_anisotropic_displacement"]
    parameters = calculator.tensor_parameters()
    assert "u_cart" in parameters
    assert "b_iso" not in parameters
    np.testing.assert_allclose(parameters["u_cart"][0], expected)
    np.testing.assert_allclose(calculator.fq(parameters=parameters), calculator.fq())


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
