# braggcalculator/io.py
from typing import Any
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser

def to_pmg_structure(struct_like: Any) -> Structure:
    # CIF path
    if isinstance(struct_like, str) and struct_like.lower().endswith(".cif"):
        return CifParser(struct_like).parse_structures(primitive=True)[0]
    # pymatgen Structure
    if isinstance(struct_like, Structure):
        return struct_like
    # ASE Atoms
    try:
        from ase import Atoms  # optional
        from pymatgen.io.ase import AseAtomsAdaptor

        if isinstance(struct_like, Atoms):
            return AseAtomsAdaptor.get_structure(struct_like)
    except Exception:
        pass
    raise TypeError(
        "Unsupported structure type. Provide CIF path, pmg.Structure, or ASE Atoms."
    )
