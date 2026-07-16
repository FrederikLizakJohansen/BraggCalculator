import pytest
from pymatgen.core import Lattice, Structure
from braggcalculator import BraggCalculator


@pytest.fixture
def calc():
    return BraggCalculator()


@pytest.fixture
def nacl():
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(5.6402), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ).get_primitive_structure()


@pytest.fixture
def strontium_titanate():
    return Structure(
        Lattice.cubic(3.905),
        ["Sr", "Ti", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


@pytest.fixture
def triclinic_structure():
    return Structure(
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73),
        ["Si", "O", "O"],
        [[0.13, 0.21, 0.34], [0.31, 0.47, 0.11], [0.72, 0.08, 0.59]],
    )
