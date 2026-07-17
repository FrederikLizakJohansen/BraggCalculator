"""Consistent primitive-cell and symmetry preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _site_b_iso(site) -> float:
    """Read a site-level isotropic displacement value when one is present."""
    properties = site.properties
    for key in ("B", "Biso", "B_iso", "b_iso"):
        if key in properties and properties[key] is not None:
            value = float(properties[key])
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"invalid isotropic B value {value!r}")
            return value
    for key in ("U", "Uiso", "U_iso", "u_iso"):
        if key in properties and properties[key] is not None:
            value = 8.0 * np.pi**2 * float(properties[key])
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"invalid isotropic U value {properties[key]!r}")
            return value
    return 0.0


class SymmetryEngine:
    """Prepare one internally consistent primitive cell.

    Symmetry operations and ``equivalent_atoms`` are always obtained from the
    exact structure returned here, never from a differently sized input cell.
    All sites are retained; equivalence information is metadata rather than a
    destructive reduction to asymmetric-unit representatives.
    """

    def __init__(
        self,
        symprec: float = 1e-3,
        angle_tolerance: float = 5.0,
        primitive: bool = True,
    ):
        self.symprec = float(symprec)
        self.angle_tolerance = float(angle_tolerance)
        self.primitive = bool(primitive)

    def reduce(self, pmg_structure) -> dict[str, Any]:
        input_analyzer = SpacegroupAnalyzer(
            pmg_structure,
            symprec=self.symprec,
            angle_tolerance=self.angle_tolerance,
        )
        structure = pmg_structure.copy()
        input_dataset = input_analyzer.get_symmetry_dataset()
        input_is_primitive = input_dataset is not None and len(
            np.unique(input_dataset.mapping_to_primitive)
        ) == len(pmg_structure)
        if self.primitive and not input_is_primitive:
            primitive = input_analyzer.find_primitive(keep_site_properties=True)
            if primitive is not None:
                structure = primitive
            analyzer = SpacegroupAnalyzer(
                structure,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance,
            )
            dataset = analyzer.get_symmetry_dataset()
        else:
            dataset = input_dataset
        if dataset is None:
            raise ValueError("spglib could not determine symmetry for the structure")

        equiv = np.asarray(dataset.equivalent_atoms, dtype=int)
        unique_ids = np.unique(equiv)
        orbits: list[np.ndarray] = [np.where(equiv == value)[0] for value in unique_ids]

        frac = []
        atomic_numbers = []
        symbols = []
        occupancies = []
        b_iso = []
        site_indices = []
        for site_index, site in enumerate(structure):
            site_b = _site_b_iso(site)
            for species, occupancy in site.species.items():
                frac.append(site.frac_coords)
                atomic_numbers.append(species.Z)
                symbols.append(species.symbol)
                occupancies.append(float(occupancy))
                b_iso.append(site_b)
                site_indices.append(site_index)

        return {
            "structure": structure,
            "lattice": np.asarray(structure.lattice.matrix, dtype=float),
            "spacegroup_symbol": dataset.international,
            "spacegroup_number": int(dataset.number),
            "crystal_system": SpacegroupAnalyzer(
                structure,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance,
            ).get_crystal_system(),
            "pointgroup_symbol": dataset.pointgroup,
            "frac_coords": np.asarray(frac, dtype=float),
            "Z": np.asarray(atomic_numbers, dtype=int),
            "symbols": tuple(symbols),
            "occ": np.asarray(occupancies, dtype=float),
            "B": np.asarray(b_iso, dtype=float),
            "site_indices": np.asarray(site_indices, dtype=int),
            "orbit_indices": orbits,
            "equiv_all": equiv,
            "symm_rot": np.asarray(dataset.rotations, dtype=int),
            "symm_trans": np.asarray(dataset.translations, dtype=float),
        }
