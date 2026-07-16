"""Small, deterministic structures spanning important diffraction cases."""

from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure


def reference_structures() -> dict[str, Structure]:
    nacl = Structure.from_spacegroup(
        "Fm-3m",
        Lattice.cubic(5.6402),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    ).get_primitive_structure()
    silicon = Structure.from_spacegroup(
        "Fd-3m",
        Lattice.cubic(5.431),
        ["Si"],
        [[0, 0, 0]],
    ).get_primitive_structure()
    strontium_titanate = Structure(
        Lattice.cubic(3.905),
        ["Sr", "Ti", "O", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )
    triclinic = Structure(
        Lattice.from_parameters(4.2, 5.1, 6.3, 78, 82, 73),
        ["Si", "O", "O"],
        [[0.13, 0.21, 0.34], [0.31, 0.47, 0.11], [0.72, 0.08, 0.59]],
    )
    disordered = Structure(
        Lattice.cubic(4.1),
        [{"Na": 0.7, "K": 0.3}, "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    rng = np.random.default_rng(20240716)
    p1_40_atom = Structure(
        Lattice.from_parameters(11.8, 12.3, 13.1, 88, 93, 97),
        (["Si", "O", "Al", "Ca", "Na"] * 8),
        rng.random((40, 3)),
    )
    return {
        "NaCl": nacl,
        "Si": silicon,
        "SrTiO3": strontium_titanate,
        "triclinic-SiO2": triclinic,
        "NaKCl-disordered": disordered,
        "P1-40-atom": p1_40_atom,
    }
