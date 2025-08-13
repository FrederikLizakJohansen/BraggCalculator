# braggcalculator/symmetry.py
from typing import Dict, Any, List
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class SymmetryEngine:
    """
    Wraps spglib (via pymatgen) to:
      - reduce to primitive
      - expose unique sites and a symmetry-expansion mapping
      - provide SG metadata
    """

    def reduce(self, pmg_structure) -> Dict[str, Any]:
        sga = SpacegroupAnalyzer(pmg_structure, symprec=1e-3, angle_tolerance=5.0)
        prim = sga.get_primitive_standard_structure()
        dataset = sga.get_symmetry_dataset()

        # All sites in primitive cell
        frac_all = np.array([site.frac_coords for site in prim.sites], dtype=float)
        Z_all = np.array([site.specie.Z for site in prim.sites], dtype=int)

        # spglib equivalent atoms: index of representative for each atom
        equiv = np.array(dataset["equivalent_atoms"], dtype=int)  # shape (Natoms,)
        # Build mapping: unique index -> indices in orbit
        unique_ids = np.unique(equiv)
        orbits: List[np.ndarray] = [np.where(equiv == u)[0] for u in unique_ids]

        # representative coords/Z for each unique site
        rep_idx = np.array([orb[0] for orb in orbits], dtype=int)
        frac_unique = frac_all[rep_idx]
        Z_unique = Z_all[rep_idx]

        # default occ & B (can be expanded per orbit later)
        occ_unique = np.ones(len(rep_idx), dtype=float)
        B_unique = np.zeros(len(rep_idx), dtype=float)

        # store symmetry operations in direct space (rot, trans)
        # dataset["rotations"]: (Nsym,3,3) integer; ["translations"]: (Nsym,3) float
        R = np.array(dataset["rotations"], dtype=int)
        t = np.array(dataset["translations"], dtype=float)

        return dict(
            lattice=np.array(prim.lattice.matrix, dtype=float),  # (3,3)
            spacegroup_symbol=dataset["international"],  # e.g. "Fm-3m"
            spacegroup_number=int(dataset["number"]),  # e.g. 225
            pointgroup_symbol=dataset["pointgroup"],  # e.g. "m-3m"
            frac_coords=frac_unique,  # (Nuniq,3)
            Z=Z_unique,  # (Nuniq,)
            occ=occ_unique,  # (Nuniq,)
            B=B_unique,  # (Nuniq,)
            # expansion mapping/orbits
            orbit_indices=orbits,  # list of arrays of site indices in prim
            equiv_all=equiv,  # (Natoms,)
            # symmetry ops
            symm_rot=R,  # (Nsym,3,3)
            symm_trans=t,  # (Nsym,3)
        )
