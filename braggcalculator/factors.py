from typing import Any
import numpy as np


def xray_form_factors(Z, s, backend) -> Any:
    # TODO: Waasmaier–Kirfel tables with smooth interpolation (autograd-safe)
    Z = backend.asarray(Z)
    return backend.ones((s.shape[0], Z.shape[0])) * Z  # placeholder


def neutron_b_coherent(Z, backend) -> Any:
    # TODO: load b_coh from table
    b = np.full(119, 5.0)  # placeholder up to Z=118
    return backend.asarray(b)[Z]  # (N,)
