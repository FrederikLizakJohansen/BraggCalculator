import pytest
from braggcalculator.io import to_pmg_structure
from pymatgen.core import Lattice, Structure


def test_io_rejects_unknown():
    with pytest.raises(TypeError):
        to_pmg_structure(object())


def test_io_rejects_non_cif_path():
    with pytest.raises(TypeError):
        to_pmg_structure("structure.xyz")


def test_cif_path_is_loaded():
    structure = to_pmg_structure("examples/NaCl.cif")
    assert isinstance(structure, Structure)
    assert structure.composition.reduced_formula == "NaCl"


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
