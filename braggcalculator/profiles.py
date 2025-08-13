from dataclasses import dataclass


@dataclass
class GaussianProfile:
    fwhm_deg: float = 0.1

    def render(self, grid, centers, amplitudes, backend):
        sigma = self.fwhm_deg / 2.354820045
        x = grid[:, None] - centers[None, :]
        return (amplitudes[None, :] * backend.exp(-0.5 * (x / sigma) ** 2)).sum(axis=1)

# Possibly useful later
@dataclass
class GaussianProfileQ:
    fwhm_q: float = 0.02  # in Å^-1

    def render(self, grid_q, centers_q, amplitudes, backend):
        sigma = self.fwhm_q / 2.354820045
        x = grid_q[:, None] - centers_q[None, :]
        return (amplitudes[None, :] * backend.exp(-0.5 * (x / sigma) ** 2)).sum(axis=1)
