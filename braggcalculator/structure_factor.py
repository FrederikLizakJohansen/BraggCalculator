from typing import Any, Literal
from .factors import xray_form_factors, neutron_b_coherent


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
):
    bk = backend
    hkl = bk.asarray(hkl)
    frac = bk.asarray(frac)
    occ = bk.asarray(occ)
    B = bk.asarray(B)
    two_theta = bk.asarray(two_theta)

    s = 2.0 * bk.sin(two_theta / 2.0) / wavelength  # (H,)
    if mode == "xray":
        f = xray_form_factors(Z, s, bk)  # (H,N)
    else:
        b = neutron_b_coherent(Z, bk)  # (N,)
        f = bk.asarray(b)[None, :].repeat(s.shape[0], axis=0)

    phase = bk.einsum("hj,aj->ha", hkl, frac)  # (H,N)
    c = bk.exp(2j * bk.pi() * phase).astype(bk.complex64)
    dw = bk.exp(-(B[None, :] * (s[:, None] ** 2)) / 4.0)
    F = (f * occ[None, :] * c * dw).sum(axis=1)  # (H,)
    return bk.real(F * bk.conj(F))  # |F|^2
