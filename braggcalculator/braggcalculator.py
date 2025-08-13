# braggcalculator/core.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple, Protocol, List, Literal, Union

# ---------------------------
# Backend protocol (NumPy / PyTorch / JAX interchangeable)
# ---------------------------
class Backend(Protocol):
    """Minimal tensor backend protocol."""

    def asarray(self, x: Any, dtype=None): ...
    def zeros(self, shape, dtype=None): ...
    def ones(self, shape, dtype=None): ...
    def exp(self, x): ...
    def pi(self): ...
    def sin(self, x): ...
    def cos(self, x): ...
    def sqrt(self, x): ...
    def abs(self, x): ...
    def real(self, x): ...
    def conj(self, x): ...
    def einsum(self, subscripts: str, *operands): ...
    def linspace(self, start, stop, num): ...
    def concat(self, xs, axis=0): ...

    complex64: Any
    float32: Any


# example numpy backend (fallback)
class NumpyBackend:
    import numpy as _np

    def __init__(self):
        self.np = self._np

    def asarray(self, x, dtype=None):
        return self.np.asarray(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return self.np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self.np.ones(shape, dtype=dtype)

    def exp(self, x):
        return self.np.exp(x)

    def pi(self):
        return self.np.pi

    def sin(self, x):
        return self.np.sin(x)

    def cos(self, x):
        return self.np.cos(x)

    def sqrt(self, x):
        return self.np.sqrt(x)

    def abs(self, x):
        return self.np.abs(x)

    def real(self, x):
        return self.np.real(x)

    def conj(self, x):
        return self.np.conj(x)

    def einsum(self, s, *ops):
        return self.np.einsum(s, *ops)

    def linspace(self, a, b, n):
        return self.np.linspace(a, b, n)

    def concat(self, xs, axis=0):
        return self.np.concatenate(xs, axis=axis)

    complex64 = _np.complex64
    float32 = _np.float32


# ---------------------------
# Modes: X-ray vs Neutron
# ---------------------------
class ScatteringMode(Protocol):
    name: Literal["xray", "neutron"]

    def form_factors(self, Z: Iterable[int], s: Any, backend: Backend) -> Any: ...
    def lp_factor(self, two_theta: Any, backend: Backend) -> Any: ...


@dataclass
class XRayMode:
    name: Literal["xray"] = "xray"
    polarization: float = 0.5  # example; refine later

    def form_factors(self, Z, s, backend: Backend):
        # TODO: tabulated Cromer–Mann or Waasmaier–Kirfel; placeholder = Z
        Z = backend.asarray(Z, dtype=backend.float32)
        return Z[None, :] * backend.ones((s.shape[0], len(Z)))

    def lp_factor(self, two_theta, backend: Backend):
        # Lorentz–polarization for Bragg-Brentano (placeholder)
        theta = two_theta / 2.0
        return 1.0 / (backend.sin(theta) * backend.sin(theta))


@dataclass
class NeutronMode:
    name: Literal["neutron"] = "neutron"

    def form_factors(self, Z, s, backend: Backend):
        # Use coherent scattering lengths b; placeholder = Z*0 + const
        import numpy as np

        b = np.array(
            [0.0 if z >= 200 else 5.0 for z in range(201)]
        )  # placeholder table
        return backend.asarray(b)[Z][None, :]

    def lp_factor(self, two_theta, backend: Backend):
        theta = two_theta / 2.0
        return 1.0 / backend.sin(theta)


# ---------------------------
# IO & structure representation (normalized to pymatgen.Structure)
# ---------------------------
StructureLike = Any  # Union[pymatgen.Structure, ase.Atoms, str(CIF path)]


class StructureAdaptor:
    @staticmethod
    def to_pmg(struct_like: StructureLike):
        # TODO: implement:
        # - if str & endswith .cif: parse via pymatgen
        # - if ASE Atoms: convert via AseAtomsAdaptor
        # - if already pymatgen.Structure: return as-is
        raise NotImplementedError


# ---------------------------
# Symmetry & hkl seams
# ---------------------------
@dataclass
class SymmetryEngine:
    """Wrap spglib/pymatgen ops & Wyckoff expansion."""

    def reduce(self, pmg_structure) -> Dict[str, Any]:
        # Return dict with: lattice, frac_coords (unique), species, wyckoff_info, sg, mults
        raise NotImplementedError

    def expand(self, unique_coords, wyckoff_info) -> Any:
        raise NotImplementedError


@dataclass
class HKLEnumerator:
    qmax: float
    wavelength: float

    def enumerate(self, lattice: Any, sg_symbol: str) -> Dict[str, Any]:
        """
        Returns dict with:
          hkl: (H,3)
          d:   (H,)
          two_theta: (H,)
          mult: (H,)
          extinct_mask: (H,)  # True if allowed
        """
        raise NotImplementedError


# ---------------------------
# Profiles (peak shapes)
# ---------------------------
class Profile(Protocol):
    def render(
        self, two_theta_grid, two_theta_peaks, I_peaks, backend: Backend
    ) -> Any: ...


@dataclass
class GaussianProfile:
    fwhm_deg: float = 0.1

    def render(self, grid, centers, amps, backend: Backend):
        sigma = self.fwhm_deg / (2.354820045)
        x = grid[:, None] - centers[None, :]
        return (amps[None, :] * backend.exp(-0.5 * (x / sigma) ** 2)).sum(axis=1)


# ---------------------------
# Main calculator
# ---------------------------
@dataclass
class BraggCalculator:
    mode: Literal["xray", "neutron"] = "xray"
    wavelength: float = 1.5406  # Å
    two_theta_range: Tuple[float, float] = (10.0, 80.0)
    two_theta_step: float = 0.01
    profile: Profile = field(default_factory=GaussianProfile)
    backend: Backend = field(default_factory=NumpyBackend)
    symmetry_engine: SymmetryEngine = field(default_factory=SymmetryEngine)
    # hkl cut set in q-space; ~ 4*pi*sin(theta)/lambda
    qmax: float = 12.0

    # internal state/cache
    _pmg_struct: Any = field(default=None, init=False, repr=False)
    _symm: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _hkl: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def set_mode(self, mode: Literal["xray", "neutron"]):
        self.mode = mode
        return self

    # -------- public API --------
    def load(self, structure: StructureLike) -> "BraggCalculator":
        self._pmg_struct = StructureAdaptor.to_pmg(structure)
        self._symm = self.symmetry_engine.reduce(self._pmg_struct)
        self._hkl = HKLEnumerator(qmax=self.qmax, wavelength=self.wavelength).enumerate(
            self._symm["lattice"], self._symm["sg"]
        )
        return self

    def fq(self) -> Any:
        """Return |F_hkl|^2 vs hkl (no LP/multiplicity); also returns mapping to q/2θ via self._hkl."""
        assert self._pmg_struct is not None, "Call .load(structure) first."
        bk = self.backend

        # gather needed arrays
        hkl = bk.asarray(self._hkl["hkl"])
        mask = self._hkl.get("extinct_mask", None)
        if mask is not None:
            hkl = hkl[mask]

        # atomic data
        frac = bk.asarray(
            self._symm["frac_coords"]
        )  # (N,3) full expanded or unique+ops
        occ = bk.asarray(self._symm.get("occ", None) or bk.ones((frac.shape[0],)))
        B = bk.asarray(self._symm.get("B", None) or bk.zeros((frac.shape[0],)))

        # scattering vector magnitude s = 2 sin(theta)/lambda; map from hkl via d-spacing
        two_theta = bk.asarray(self._hkl["two_theta"])
        if mask is not None:
            two_theta = two_theta[mask]
        s = 2.0 * bk.sin(two_theta / 2.0) / self.wavelength

        # choose scattering mode
        mode_impl: ScatteringMode = XRayMode() if self.mode == "xray" else NeutronMode()
        Z = self._symm["Z"]  # atomic numbers (N,)
        f_s = mode_impl.form_factors(Z, s, bk)  # shape (H, N)

        # phases exp(2π i h·r)
        phase = bk.einsum("hj,aj->ha", hkl, frac)  # (H,N)
        c = bk.exp(2j * bk.pi() * phase).astype(bk.complex64)
        dw = bk.exp(-(B[None, :] * (s[:, None] ** 2)) / 4.0)

        F = (f_s * occ[None, :] * c * dw).sum(axis=1)  # (H,)
        return bk.real(F * bk.conj(F))  # |F|^2

    def iq(self) -> Tuple[Any, Any]:
        """Return intensity vs q (applies multiplicity & LP) and q-grid for peaks (delta lines)."""
        I_hkl = self.fq()
        bk = self.backend
        mult = bk.asarray(self._hkl["mult"])
        two_theta = bk.asarray(self._hkl["two_theta"])
        # LP factor
        mode_impl: ScatteringMode = XRayMode() if self.mode == "xray" else NeutronMode()
        LP = mode_impl.lp_factor(two_theta, bk)

        I_peaks = I_hkl * mult * LP
        # convert to q-line positions
        q = 4.0 * bk.pi() * bk.sin(two_theta / 2.0) / self.wavelength
        return q, I_peaks

    def pattern(self) -> Tuple[Any, Any]:
        """Return (two_theta_grid, intensity_grid) with peak profiles & binning."""
        q, I_peaks = self.iq()
        bk = self.backend
        two_theta_peaks = self._hkl["two_theta"]
        t_min, t_max = self.two_theta_range
        grid = bk.linspace(t_min, t_max, int((t_max - t_min) / self.two_theta_step) + 1)
        intensity = self.profile.render(grid, two_theta_peaks, I_peaks, bk)
        return grid, intensity

    # convenience
    def as_dict(self) -> Dict[str, Any]:
        return dict(
            mode=self.mode,
            wavelength=self.wavelength,
            two_theta_range=self.two_theta_range,
            two_theta_step=self.two_theta_step,
            qmax=self.qmax,
            sg=self._symm.get("sg") if self._symm else None,
        )
