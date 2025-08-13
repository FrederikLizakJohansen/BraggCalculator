from typing import Any, Literal


def lp_factor(two_theta, backend, mode: Literal["xray", "neutron"]):
    theta = two_theta / 2.0
    if mode == "xray":
        return 1.0 / (backend.sin(theta) ** 2)  # placeholder
    else:
        return 1.0 / backend.sin(theta)  # placeholder


def apply_lp_and_multiplicity(mode, backend, F2, two_theta, multiplicity):
    LP = lp_factor(two_theta, backend, mode)
    return F2 * multiplicity * LP


def render_profile(profile, backend, grid, centers, amplitudes):
    centers = backend.asarray(centers)
    amplitudes = backend.asarray(amplitudes)
    return profile.render(grid, centers, amplitudes, backend)


# Q-space version (identical call signature, different profile object)
def render_profile_q(profile_q, backend, grid_q, centers_q, amplitudes):
    centers_q = backend.asarray(centers_q)
    amplitudes = backend.asarray(amplitudes)
    return profile_q.render(grid_q, centers_q, amplitudes, backend)
