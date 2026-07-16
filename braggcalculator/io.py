# braggcalculator/io.py
from os import PathLike
from pathlib import Path
from typing import Any
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser


def to_pmg_structure(struct_like: Any) -> Structure:
    # CIF path
    if isinstance(struct_like, (str, PathLike)):
        path = Path(struct_like)
        if path.suffix.lower() != ".cif":
            raise TypeError("Structure paths must point to a .cif file")
        if not path.is_file():
            raise FileNotFoundError(path)
        structures = CifParser(path).parse_structures(primitive=True)
        if not structures:
            raise ValueError(f"No crystal structure found in {path}")
        return structures[0]
    # pymatgen Structure
    if isinstance(struct_like, Structure):
        return struct_like
    # ASE Atoms
    try:
        from ase import Atoms  # optional
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError:
        Atoms = None
    if Atoms is not None and isinstance(struct_like, Atoms):
        return AseAtomsAdaptor.get_structure(struct_like)
    raise TypeError("Unsupported structure type. Provide CIF path, pmg.Structure, or ASE Atoms.")
