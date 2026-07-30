"""Convert supported structure inputs to pymatgen structures."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifFile, CifParser, str2float


def _parse_cif_number(value):
    if str(value).strip() in {".", "?"}:
        return None
    parsed = float(str2float(str(value)))
    if not np.isfinite(parsed):
        raise ValueError(f"invalid CIF displacement value: {value}")
    return parsed


def _cif_u_to_cartesian(lattice, tensor):
    """Convert crystallographic-axis CIF Uij to Cartesian square angstrom."""
    matrix = np.asarray(lattice.matrix, dtype=np.float64)
    reciprocal_lengths = np.linalg.norm(np.linalg.inv(matrix.T), axis=1)
    orthogonalizer = matrix.T @ np.diag(reciprocal_lengths)
    return orthogonalizer @ tensor @ orthogonalizer.T


def _attach_cif_displacements(path: Path, structure: Structure) -> Structure:
    """Preserve CIF isotropic and anisotropic values omitted by pymatgen."""
    cif = CifFile.from_file(path)
    for block in cif.data.values():
        data = block.data
        labels = data.get("_atom_site_label")
        if labels is None:
            continue
        result = structure.copy()
        attached = False
        for candidate, name in (
            ("_atom_site_B_iso_or_equiv", "B_iso"),
            ("_atom_site_U_iso_or_equiv", "U_iso"),
        ):
            if candidate not in data:
                continue
            values_by_label = {}
            for label, value in zip(labels, data[candidate]):
                parsed = _parse_cif_number(value)
                if parsed is None:
                    continue
                if parsed < 0:
                    raise ValueError(
                        f"invalid isotropic displacement for CIF site {label}: {value}"
                    )
                values_by_label[str(label)] = parsed
            if values_by_label:
                result.add_site_property(
                    name,
                    [values_by_label.get(site.label) for site in structure],
                )
                attached = True
            break

        aniso_labels = data.get("_atom_site_aniso_label")
        components = ("11", "22", "33", "23", "13", "12")
        u_fields = tuple(f"_atom_site_aniso_U_{component}" for component in components)
        b_fields = tuple(f"_atom_site_aniso_B_{component}" for component in components)
        fields = u_fields if all(field in data for field in u_fields) else None
        scale = 1.0
        if fields is None and all(field in data for field in b_fields):
            fields = b_fields
            scale = 1.0 / (8.0 * np.pi**2)
        if aniso_labels is not None and fields is not None:
            tensors = {}
            for row, label in enumerate(aniso_labels):
                values = [_parse_cif_number(data[field][row]) for field in fields]
                if any(value is None for value in values):
                    continue
                u11, u22, u33, u23, u13, u12 = (scale * value for value in values)
                crystallographic = np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]])
                cartesian = _cif_u_to_cartesian(structure.lattice, crystallographic)
                if np.linalg.eigvalsh(cartesian).min() < -1e-10:
                    raise ValueError(f"CIF site {label} has a non-positive anisotropic tensor")
                tensors[str(label)] = cartesian
            if tensors:
                result.add_site_property(
                    "U_cart",
                    [tensors.get(site.label) for site in structure],
                )
                attached = True
        if attached:
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
        return _attach_cif_displacements(path, structures[0])
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
