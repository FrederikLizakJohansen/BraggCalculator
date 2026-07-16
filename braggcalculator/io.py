"""Convert supported structure inputs to pymatgen structures."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser


def to_pmg_structure(structure_like: Any) -> Structure:
    """Return a pymatgen structure from a CIF path, pymatgen, or ASE input."""
    if isinstance(structure_like, (str, PathLike)):
        path = Path(structure_like)
        if path.suffix.lower() != ".cif":
            raise TypeError("Structure paths must point to a .cif file")
        if not path.is_file():
            raise FileNotFoundError(path)
        structures = CifParser(path).parse_structures(primitive=True)
        if not structures:
            raise ValueError(f"No crystal structure found in {path}")
        return structures[0]
    if isinstance(structure_like, Structure):
        return structure_like

    try:
        from ase import Atoms
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        Atoms = None
    if Atoms is not None and isinstance(structure_like, Atoms):
        return AseAtomsAdaptor.get_structure(structure_like)
    raise TypeError("Unsupported structure type. Provide CIF path, pmg.Structure, or ASE Atoms.")
