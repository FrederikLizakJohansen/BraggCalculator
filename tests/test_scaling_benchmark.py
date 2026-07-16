import pytest

from benchmarks.scaling_cases import nacl_supercell, p1_structure, scaling_cases
from braggcalculator import BraggCalculator


@pytest.mark.parametrize("site_count", [4, 8, 16])
def test_p1_scaling_structure_has_requested_irreducible_sites(site_count):
    structure = p1_structure(site_count)
    calculator = BraggCalculator().load(structure)
    assert len(structure) == site_count
    assert len(calculator._symm["structure"]) == site_count
    assert calculator._symm["spacegroup_number"] == 1


@pytest.mark.parametrize(("factor", "input_sites"), [(1, 8), (2, 64)])
def test_nacl_scaling_structure_reduces_to_two_sites(factor, input_sites):
    structure = nacl_supercell(factor)
    calculator = BraggCalculator().load(structure)
    assert len(structure) == input_sites
    assert len(calculator._symm["structure"]) == 2


def test_scaling_cases_require_increasing_control_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        scaling_cases((8, 4), (1, 2))
