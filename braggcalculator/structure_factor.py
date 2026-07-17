"""Vectorized crystallographic structure-factor kernels."""

from __future__ import annotations

from typing import Any, Literal, Mapping

from .factors import neutron_b_coherent, xray_form_factors


def reflection_geometry(backend: Any, hkl, lattice, wavelength: float):
    """Return ``(g, two_theta)`` for a fixed HKL topology.

    ``g`` is ``1 / d`` in inverse angstroms and ``two_theta`` is in radians.
    The lattice contains direct-space row vectors in angstroms.
    """
    bk = backend
    hkl = bk.asarray(hkl, dtype=bk.dtype)
    lattice = bk.asarray(lattice, dtype=bk.dtype)
    metric = bk.einsum("ij,kj->ik", lattice, lattice)
    reciprocal_metric = bk.inverse(metric)
    g2 = bk.einsum("hi,ij,hj->h", hkl, reciprocal_metric, hkl)
    g = bk.sqrt(g2)
    bragg_argument = 0.5 * wavelength * g
    if int(bragg_argument.shape[0]):
        if getattr(bk, "is_torch", False):
            largest_argument = float(bragg_argument.detach().max().cpu())
        else:
            largest_argument = float(bk.max(bragg_argument))
        if largest_argument >= 1.0:
            raise ValueError(
                "The supplied lattice moved a prepared reflection outside the Bragg "
                "limiting sphere; rebuild the calculator for this lattice"
            )
    two_theta = 2.0 * bk.arcsin(bragg_argument)
    return g, two_theta


def compute_F(
    mode: Literal["xray", "neutron"],
    backend: Any,
    hkl,
    two_theta,
    wavelength,
    Z,
    frac,
    occ,
    B,
    *,
    neutron_scattering_lengths: Mapping[str | int, float | str] | None = None,
    phase_chunk_entries: int = 4_194_304,
):
    """Compute complex ``F(hkl)`` using integrated site occupancies.

    ``B`` is the isotropic Debye-Waller ``B`` value in square angstroms, so
    the amplitude correction is ``exp(-B * s**2)`` with
    ``s = sin(theta) / wavelength``.
    """
    if mode not in {"xray", "neutron"}:
        raise ValueError("mode must be 'xray' or 'neutron'")
    if phase_chunk_entries <= 0:
        raise ValueError("phase_chunk_entries must be positive")

    bk = backend
    hkl = bk.asarray(hkl, dtype=bk.dtype)
    frac = bk.asarray(frac, dtype=bk.dtype)
    occ = bk.asarray(occ, dtype=bk.dtype)
    B = bk.asarray(B, dtype=bk.dtype)
    two_theta = bk.asarray(two_theta, dtype=bk.dtype)

    atom_count = int(frac.shape[0])
    if atom_count == 0:
        raise ValueError("a structure must contain at least one species contribution")
    chunk_size = max(1, phase_chunk_entries // atom_count)
    outputs = []

    if getattr(B, "requires_grad", False):
        zero_b = False
    elif getattr(bk, "is_torch", False):
        zero_b = not bool(B.detach().count_nonzero().cpu())
    else:
        zero_b = not bool((B != 0).any())

    neutron_b = None
    if mode == "neutron":
        neutron_b = neutron_b_coherent(Z, bk, overrides=neutron_scattering_lengths)

    for start in range(0, int(hkl.shape[0]), chunk_size):
        stop = min(start + chunk_size, int(hkl.shape[0]))
        hkl_part = hkl[start:stop]
        angle_part = two_theta[start:stop]
        s = bk.sin(angle_part / 2.0) / wavelength
        if mode == "xray":
            scattering = xray_form_factors(Z, s, bk)
        else:
            scattering = neutron_b[None, :]

        phase = 2.0 * bk.pi() * bk.matmul(hkl_part, frac.T)
        phase_factor = bk.cos(phase) + 1j * bk.sin(phase)
        dw = 1.0 if zero_b else bk.exp(-B[None, :] * (s[:, None] ** 2))
        amplitude = bk.sum(
            scattering * occ[None, :] * phase_factor * dw,
            axis=1,
        )
        outputs.append(amplitude)

    if not outputs:
        complex_dtype = bk.complex128 if bk.dtype == bk.float64 else bk.complex64
        return bk.zeros((0,), dtype=complex_dtype)
    return outputs[0] if len(outputs) == 1 else bk.concat(outputs, axis=0)


def compute_F2(
    mode: Literal["xray", "neutron"],
    backend: Any,
    hkl,
    two_theta,
    wavelength,
    Z,
    frac,
    occ,
    B,
    *,
    neutron_scattering_lengths: Mapping[str | int, float | str] | None = None,
    phase_chunk_entries: int = 4_194_304,
):
    """Compute ``|F(hkl)|^2`` from the shared complex-amplitude kernel."""
    amplitude = compute_F(
        mode=mode,
        backend=backend,
        hkl=hkl,
        two_theta=two_theta,
        wavelength=wavelength,
        Z=Z,
        frac=frac,
        occ=occ,
        B=B,
        neutron_scattering_lengths=neutron_scattering_lengths,
        phase_chunk_entries=phase_chunk_entries,
    )
    return backend.real(amplitude * backend.conj(amplitude))
