"""Complete reciprocal-lattice enumeration inside a metric ellipsoid."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def reciprocal_metric(lattice_matrix: np.ndarray) -> np.ndarray:
    """Return the crystallographic reciprocal metric (without ``2*pi``)."""
    lattice = np.asarray(lattice_matrix, dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError("lattice_matrix must be a finite 3 by 3 array")
    metric = lattice @ lattice.T
    if not np.isfinite(np.linalg.det(metric)) or np.linalg.det(metric) <= 0:
        raise ValueError("lattice_matrix must be non-singular")
    return np.linalg.inv(metric)


def _coordinate_bounds(reciprocal_metric_matrix: np.ndarray, gmax: float) -> np.ndarray:
    # The maximum coordinate of x in x.T A x <= r^2 is
    # r * sqrt(diag(A^-1)).  This bound is exact for the enclosing box.
    direct_metric = np.linalg.inv(reciprocal_metric_matrix)
    return np.ceil(gmax * np.sqrt(np.diag(direct_metric))).astype(int)


class HKLEnumerator:
    """Enumerate every integer reciprocal point satisfying the Bragg condition."""

    def __init__(
        self,
        wavelength: float,
        qmax: float,
        hkl_max: int | None = None,
        max_candidate_entries: int = 2_000_000,
    ):
        self.wavelength = float(wavelength)
        self.qmax = float(qmax)
        self.hkl_max = None if hkl_max is None else int(hkl_max)
        self.max_candidate_entries = int(max_candidate_entries)
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be positive and finite")
        if not np.isfinite(self.qmax) or self.qmax <= 0:
            raise ValueError("qmax must be positive and finite")
        if self.hkl_max is not None and self.hkl_max <= 0:
            raise ValueError("hkl_max must be positive when provided")
        if self.max_candidate_entries <= 0:
            raise ValueError("max_candidate_entries must be positive")

    def enumerate(
        self,
        lattice_matrix,
        pointgroup_symbol: str | None = None,
    ) -> Dict[str, Any]:
        """Return all HKLs within ``qmax`` and the Ewald limiting sphere.

        ``pointgroup_symbol`` is accepted for API compatibility.  No
        multiplicity is applied because all reciprocal points are explicitly
        represented.
        """
        del pointgroup_symbol
        gstar = reciprocal_metric(lattice_matrix)
        requested_gmax = self.qmax / (2.0 * np.pi)
        physical_gmax = np.nextafter(2.0 / self.wavelength, 0.0)
        gmax = min(requested_gmax, physical_gmax)
        bounds = _coordinate_bounds(gstar, gmax)
        if self.hkl_max is not None:
            bounds = np.minimum(bounds, self.hkl_max)

        k_values = np.arange(-bounds[1], bounds[1] + 1, dtype=int)
        l_values = np.arange(-bounds[2], bounds[2] + 1, dtype=int)
        plane_size = len(k_values) * len(l_values)
        h_per_chunk = max(1, self.max_candidate_entries // max(plane_size, 1))
        h_values = np.arange(-bounds[0], bounds[0] + 1, dtype=int)

        selected_hkl = []
        selected_g2 = []
        gmax2 = gmax * gmax
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, gmax2)

        for start in range(0, len(h_values), h_per_chunk):
            hs = h_values[start : start + h_per_chunk]
            h_grid, k_grid, l_grid = np.meshgrid(hs, k_values, l_values, indexing="ij")
            candidates = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
            g2 = np.einsum("hi,ij,hj->h", candidates, gstar, candidates)
            mask = (g2 > tolerance) & (g2 <= gmax2 + tolerance)
            if np.any(mask):
                selected_hkl.append(candidates[mask])
                selected_g2.append(g2[mask])

        if not selected_hkl:
            empty_hkl = np.empty((0, 3), dtype=int)
            empty = np.empty((0,), dtype=float)
            return {
                "hkl": empty_hkl,
                "g": empty,
                "q": empty,
                "two_theta": empty,
                "multiplicity": np.empty((0,), dtype=int),
            }

        hkls = np.concatenate(selected_hkl, axis=0)
        g = np.sqrt(np.concatenate(selected_g2, axis=0))
        order = np.lexsort((-hkls[:, 2], -hkls[:, 1], -hkls[:, 0], g))
        hkls = hkls[order]
        g = g[order]
        bragg_argument = 0.5 * self.wavelength * g
        two_theta = 2.0 * np.arcsin(bragg_argument)
        q = 2.0 * np.pi * g

        return {
            "hkl": hkls,
            "g": g,
            "q": q,
            "two_theta": two_theta,
            # Every row is one reciprocal point, so its multiplicity is one.
            "multiplicity": np.ones(len(hkls), dtype=int),
        }
