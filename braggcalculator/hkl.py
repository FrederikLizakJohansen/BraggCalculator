# braggcalculator/hkl.py
from typing import Dict, Any, Tuple
import numpy as np

_LAUE_TO_OPS = {
    # Minimal representative rotation matrices (reciprocal space) per Laue class.
    # For v0 we’ll approximate multiplicity by applying these ops to (h,k,l) and counting uniques.
    # TODO: expand set per point group properly.
    "m-3m": [np.eye(3, dtype=int)],  # placeholder; real set should include 48 ops
    "m-3": [np.eye(3, dtype=int)],
    "4/mmm": [np.eye(3, dtype=int)],
    "4/m": [np.eye(3, dtype=int)],
    "6/mmm": [np.eye(3, dtype=int)],
    "6/m": [np.eye(3, dtype=int)],
    "mmm": [np.eye(3, dtype=int)],
    "2/m": [np.eye(3, dtype=int)],
    "m": [np.eye(3, dtype=int)],
    "1̄": [np.eye(3, dtype=int)],
    "1": [np.eye(3, dtype=int)],
}


def _reciprocal_metric(lattice_matrix: np.ndarray) -> np.ndarray:
    # G is column-lattice or row? pymatgen returns rows as lattice vectors
    # Using a row-matrix L (3x3), metric G = L·L^T, reciprocal metric G* = (G)^-1
    L = np.array(lattice_matrix, dtype=float)  # (3,3)
    G = L @ L.T
    Gstar = np.linalg.inv(G)
    return Gstar


def _two_theta_from_hkl(
    hkl: np.ndarray, Gstar: np.ndarray, wavelength: float
) -> np.ndarray:
    # d*^2 = h^T G* h ; d = 1/sqrt(d*^2); sin(theta)=λ/(2d)
    dstar2 = (hkl @ Gstar * hkl).sum(axis=1)
    d = 1.0 / np.sqrt(dstar2)
    # guard domain
    arg = np.clip(0.5 * wavelength / d, 0.0, 1.0)
    return 2.0 * np.arcsin(arg)


def _laue_ops(pointgroup_symbol: str):
    # Map to a Laue class key known above (rough pass-through for now)
    key = pointgroup_symbol.replace(" ", "")
    return _LAUE_TO_OPS.get(key, [np.eye(3, dtype=int)])


def _multiplicity_for_hkl(hkl: np.ndarray, Rops: list) -> np.ndarray:
    # Apply each 3x3 integer rotation to hkl and count unique vectors per row
    # v0: treat (h,k,l) and (-h,-k,-l) as different; refinement later if desired.
    H = hkl.shape[0]
    m = np.empty(H, dtype=int)
    for i in range(H):
        images = []
        v = hkl[i]
        for R in Rops:
            images.append(tuple((R @ v).astype(int)))
        m[i] = len(set(images))
    return m


class HKLEnumerator:
    def __init__(self, wavelength: float, qmax: float, hkl_max: int = 16):
        self.wavelength = float(wavelength)
        self.qmax = float(qmax)
        self.hkl_max = int(hkl_max)

    def _candidate_hkls(self) -> np.ndarray:
        r = range(-self.hkl_max, self.hkl_max + 1)
        grid = np.array(
            [
                (h, k, l)
                for h in r
                for k in r
                for l in r
                if not (h == 0 and k == 0 and l == 0)
            ],
            dtype=int,
        )
        return grid

    def enumerate(self, lattice_matrix, pointgroup_symbol: str) -> Dict[str, Any]:
        Gstar = _reciprocal_metric(lattice_matrix)
        hkls = self._candidate_hkls()

        two_theta = _two_theta_from_hkl(hkls, Gstar, self.wavelength)
        q = 4.0 * np.pi * np.sin(two_theta / 2.0) / self.wavelength
        mask = np.isfinite(two_theta) & (two_theta > 0) & (q <= self.qmax)

        hkls = hkls[mask]
        two_theta = two_theta[mask]

        # multiplicity via (approx) Laue operations
        Rops = _laue_ops(pointgroup_symbol)
        multiplicity = _multiplicity_for_hkl(hkls, Rops)

        # v0 extinction mask: allow all (systematic absences to be added later)
        extinct_mask = np.ones_like(two_theta, dtype=bool)

        # sort by 2θ
        order = np.argsort(two_theta)
        return dict(
            hkl=hkls[order],
            two_theta=two_theta[order],
            multiplicity=multiplicity[order],
            extinct_mask=extinct_mask[order],
        )
