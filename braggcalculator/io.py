"""Convert supported structure inputs to pymatgen structures."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifFile, CifParser, str2float


def _attach_cif_isotropic_displacements(path: Path, structure: Structure) -> Structure:
    """Preserve CIF Uiso/Biso values that pymatgen does not attach to sites."""
    cif = CifFile.from_file(path)
    for block in cif.data.values():
        data = block.data
        labels = data.get("_atom_site_label")
        if labels is None:
            continue
        field = None
        property_name = None
        for candidate, name in (
            ("_atom_site_B_iso_or_equiv", "B_iso"),
            ("_atom_site_U_iso_or_equiv", "U_iso"),
        ):
            if candidate in data:
                field = data[candidate]
                property_name = name
                break
        if field is None:
            continue
        values_by_label = {}
        for label, value in zip(labels, field):
            if str(value).strip() in {".", "?"}:
                continue
            parsed = float(str2float(str(value)))
            if not np.isfinite(parsed) or parsed < 0:
                raise ValueError(f"invalid isotropic displacement for CIF site {label}: {value}")
            values_by_label[str(label)] = parsed
        if values_by_label and all(site.label in values_by_label for site in structure):
            result = structure.copy()
            result.add_site_property(
                property_name,
                [values_by_label[site.label] for site in structure],
            )
            return result
    return structure


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
        return _attach_cif_isotropic_displacements(path, structures[0])
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
