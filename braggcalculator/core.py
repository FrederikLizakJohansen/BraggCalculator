from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict, Literal

from .io import to_pmg_structure
from .symmetry import SymmetryEngine
from .hkl import HKLEnumerator
from .renderer import apply_lp_and_multiplicity, render_profile, render_profile_q
from .structure_factor import compute_F2
from .profiles import GaussianProfile, GaussianProfileQ
from .backends.numpy_backend import NumpyBackend


@dataclass
class BraggCalculator:
    mode: Literal["xray", "neutron"] = "xray"
    wavelength: float = 1.5406
    two_theta_range: Tuple[float, float] = (10.0, 80.0)
    two_theta_step: float = 0.01
    q_range: Tuple[float, float] = (0.0, 10.0)
    q_step: float = 0.005
    qmax: float = 12.0
    profile: Any = field(default_factory=GaussianProfile)
    profile_q: Any = field(default_factory=GaussianProfileQ)
    backend: Any = field(default_factory=NumpyBackend)

    # internal
    _pmg_structure: Any = field(default=None, init=False, repr=False)
    _symm: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _hkl: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def load(self, structure_like: Any) -> "BraggCalculator":
        self._pmg_structure = to_pmg_structure(structure_like)
        self._symm = SymmetryEngine().reduce(self._pmg_structure)
        self._hkl = HKLEnumerator(self.wavelength, self.qmax).enumerate(
            self._symm["lattice"], self._symm["pointgroup_symbol"]
        )
        return self

    def fq(self):
        """Return per-hkl |F|^2 (no LP/multiplicity)."""
        return compute_F2(
            mode=self.mode,
            backend=self.backend,
            hkl=self._hkl["hkl"],  # (H,3)
            two_theta=self._hkl["two_theta"],  # (H,)
            wavelength=self.wavelength,
            Z=self._symm["Z"],  # (N,)
            frac=self._symm["frac_coords"],  # (N,3)
            occ=self._symm["occ"],  # (N,)
            B=self._symm["B"],  # (N,)
        )

    def iq(self, domain: Literal["two_theta", "q"] = "two_theta"):
        """
        Return delta-line intensities after LP & multiplicity, in the requested domain.
        domain="two_theta" -> (two_theta_peaks, I_peaks)
        domain="q"         -> (q_peaks,         I_peaks)
        """
        F2 = self.fq()
        I_peaks = apply_lp_and_multiplicity(
            mode=self.mode,
            backend=self.backend,
            F2=F2,
            two_theta=self._hkl["two_theta"],
            multiplicity=self._hkl["multiplicity"],
        )

        if domain == "two_theta":
            x = self.backend.degrees(self._hkl["two_theta"])
        elif domain == "q":
            x = self.backend.q_from_two_theta(self._hkl["two_theta"], self.wavelength)
        else:
            raise ValueError("domain must be 'two_theta' or 'q'")
        return x, I_peaks

    def pattern(self, domain: Literal["two_theta", "q"] = "two_theta"):
        """
        Return gridded intensity in requested domain with appropriate profile broadening.
        domain="two_theta" -> (two_theta_grid, I_grid)
        domain="q"         -> (q_grid,         I_grid)
        """
        x_peaks, I_peaks = self.iq(domain=domain)

        if domain == "two_theta":
            tmin, tmax = self.two_theta_range
            grid = self.backend.linspace(
                tmin, tmax, int((tmax - tmin) / self.two_theta_step) + 1
            )
            I = render_profile(self.profile, self.backend, grid, x_peaks, I_peaks)
        elif domain == "q":
            qmin, qmax = self.q_range
            grid = self.backend.linspace(
                qmin, qmax, int((qmax - qmin) / self.q_step) + 1
            )
            I = render_profile_q(self.profile_q, self.backend, grid, x_peaks, I_peaks)
        else:
            raise ValueError("domain must be 'two_theta' or 'q'")

        return grid, I
