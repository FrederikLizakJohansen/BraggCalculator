"""Scaled Jacobian and local identifiability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .results import JacobianDiagnostics


@dataclass(frozen=True)
class ParameterPath:
    """One scalar tensor entry and its characteristic physical step."""

    name: str
    index: tuple[int, ...]
    scale: float
    label: str | None = None

    @property
    def display_name(self) -> str:
        return self.label or f"{self.name}{self.index}"


def analyze_jacobian(
    jacobian,
    *,
    residual=None,
    weights=None,
    covariance=None,
    parameter_scales=None,
    parameter_names=None,
    prior_precision=None,
    prior_jacobian=None,
    rcond: float | None = None,
) -> JacobianDiagnostics:
    """Analyze data and posterior information in characteristic parameter units.

    ``prior_precision`` and ``prior_jacobian`` are expressed in the unscaled
    input-parameter coordinates. Priors may make the posterior full rank, but
    ``rank`` and ``covariance_is_identifiable`` continue to describe the data
    Jacobian alone.
    """
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian must be a finite two-dimensional matrix")
    observation_count, parameter_count = matrix.shape
    if observation_count == 0 or parameter_count == 0:
        raise ValueError("jacobian must have at least one row and one column")
    if weights is not None and covariance is not None:
        raise ValueError("provide weights or covariance, not both")

    if parameter_scales is None:
        scales = np.ones(parameter_count, dtype=np.float64)
    else:
        scales = np.asarray(parameter_scales, dtype=np.float64)
        if (
            scales.shape != (parameter_count,)
            or np.any(scales <= 0)
            or not np.all(np.isfinite(scales))
        ):
            raise ValueError("parameter_scales must be one positive value per parameter")
    if parameter_names is None:
        names = tuple(f"p{index}" for index in range(parameter_count))
    else:
        names = tuple(parameter_names)
        if len(names) != parameter_count or any(not name for name in names):
            raise ValueError("parameter_names must contain one non-empty name per parameter")

    scaled = matrix * scales[None, :]
    local_information = None
    if covariance is not None:
        covariance_array = np.asarray(covariance, dtype=np.float64)
        expected_shape = (observation_count, observation_count)
        if covariance_array.shape != expected_shape or not np.all(np.isfinite(covariance_array)):
            raise ValueError(f"covariance must be a finite matrix with shape {expected_shape}")
        if not np.allclose(covariance_array, covariance_array.T, rtol=1e-12, atol=1e-14):
            raise ValueError("covariance must be symmetric")
        try:
            cholesky = np.linalg.cholesky(covariance_array)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error
        whitened_jacobian = np.linalg.solve(cholesky, scaled)
        whitened_residual = (
            None
            if residual is None
            else np.linalg.solve(cholesky, _validated_residual(residual, observation_count))
        )
    else:
        if weights is None:
            weight_array = np.ones(observation_count, dtype=np.float64)
        else:
            weight_array = np.asarray(weights, dtype=np.float64)
            if (
                weight_array.shape != (observation_count,)
                or np.any(weight_array < 0)
                or not np.all(np.isfinite(weight_array))
                or not np.any(weight_array > 0)
            ):
                raise ValueError("weights must be a finite non-negative observation vector")
        square_root_weight = np.sqrt(weight_array)
        whitened_jacobian = square_root_weight[:, None] * scaled
        local_information = weight_array[:, None] * scaled**2
        whitened_residual = (
            None
            if residual is None
            else square_root_weight * _validated_residual(residual, observation_count)
        )

    normal = whitened_jacobian.T @ whitened_jacobian
    prior_scaled = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    if prior_precision is not None:
        precision = np.asarray(prior_precision, dtype=np.float64)
        if precision.shape != (parameter_count, parameter_count) or not np.all(
            np.isfinite(precision)
        ):
            raise ValueError("prior_precision must be a finite square parameter matrix")
        if not np.allclose(precision, precision.T, rtol=1e-12, atol=1e-14):
            raise ValueError("prior_precision must be symmetric")
        if np.linalg.eigvalsh(precision).min() < -1e-12 * max(
            1.0, np.linalg.norm(precision, ord=2)
        ):
            raise ValueError("prior_precision must be positive semidefinite")
        prior_scaled += scales[:, None] * precision * scales[None, :]
    if prior_jacobian is not None:
        prior_matrix = np.asarray(prior_jacobian, dtype=np.float64)
        if (
            prior_matrix.ndim != 2
            or prior_matrix.shape[1] != parameter_count
            or not np.all(np.isfinite(prior_matrix))
        ):
            raise ValueError("prior_jacobian must be finite with one column per parameter")
        scaled_prior_jacobian = prior_matrix * scales[None, :]
        prior_scaled += scaled_prior_jacobian.T @ scaled_prior_jacobian
    posterior_normal = normal + prior_scaled
    sensitivity = np.sqrt(np.maximum(np.diag(normal), 0.0))
    denominator = sensitivity[:, None] * sensitivity[None, :]
    column_cosine = np.divide(
        normal,
        denominator,
        out=np.zeros_like(normal),
        where=denominator > 0,
    )
    residual_support = None
    if whitened_residual is not None:
        projection = whitened_jacobian.T @ whitened_residual
        residual_support = np.divide(
            projection,
            sensitivity,
            out=np.zeros_like(projection),
            where=sensitivity > 0,
        )

    singular_values, right_vectors = _svd_diagnostics(whitened_jacobian)
    default_rcond = np.finfo(np.float64).eps * max(whitened_jacobian.shape)
    cutoff_ratio = default_rcond if rcond is None else float(rcond)
    if not np.isfinite(cutoff_ratio) or cutoff_ratio < 0:
        raise ValueError("rcond must be finite and non-negative")
    cutoff = cutoff_ratio * singular_values[0] if len(singular_values) else 0.0
    rank = int(np.count_nonzero(singular_values > cutoff))
    condition_number = _condition_number(singular_values, rank, parameter_count)
    prior_singular_values = np.sqrt(np.maximum(np.linalg.eigvalsh(prior_scaled)[::-1], 0.0))
    prior_cutoff = cutoff_ratio * prior_singular_values[0] if len(prior_singular_values) else 0.0
    prior_rank = int(np.count_nonzero(prior_singular_values > prior_cutoff))
    posterior_singular_values = np.sqrt(np.maximum(np.linalg.eigvalsh(posterior_normal)[::-1], 0.0))
    posterior_cutoff = (
        cutoff_ratio * posterior_singular_values[0] if len(posterior_singular_values) else 0.0
    )
    posterior_rank = int(np.count_nonzero(posterior_singular_values > posterior_cutoff))
    posterior_condition_number = _condition_number(
        posterior_singular_values, posterior_rank, parameter_count
    )
    generalized_scaled = np.linalg.pinv(normal, rcond=cutoff_ratio, hermitian=True)
    generalized_physical = scales[:, None] * generalized_scaled * scales[None, :]
    diagonal = np.diag(generalized_scaled)
    covariance_denominator = np.sqrt(np.maximum(diagonal[:, None] * diagonal[None, :], 0.0))
    correlation = np.divide(
        generalized_scaled,
        covariance_denominator,
        out=np.full_like(generalized_scaled, np.nan),
        where=covariance_denominator > 0,
    )
    posterior_covariance_scaled = np.linalg.pinv(
        posterior_normal, rcond=cutoff_ratio, hermitian=True
    )
    posterior_covariance_physical = (
        scales[:, None] * posterior_covariance_scaled * scales[None, :]
    )
    posterior_diagonal = np.diag(posterior_covariance_scaled)
    posterior_denominator = np.sqrt(
        np.maximum(posterior_diagonal[:, None] * posterior_diagonal[None, :], 0.0)
    )
    posterior_correlation = np.divide(
        posterior_covariance_scaled,
        posterior_denominator,
        out=np.full_like(posterior_covariance_scaled, np.nan),
        where=posterior_denominator > 0,
    )
    if posterior_rank == parameter_count:
        standard_errors_scaled = np.sqrt(
            np.maximum(np.diag(posterior_covariance_scaled), 0.0)
        )
        standard_errors_physical = scales * standard_errors_scaled
    else:
        standard_errors_scaled = np.full(parameter_count, np.nan)
        standard_errors_physical = np.full(parameter_count, np.nan)

    return JacobianDiagnostics(
        parameter_names=names,
        parameter_scales=scales,
        jacobian=matrix,
        scaled_jacobian=scaled,
        normal_matrix=normal,
        sensitivity=sensitivity,
        residual_support=residual_support,
        column_cosine=column_cosine,
        generalized_covariance_scaled=generalized_scaled,
        generalized_covariance_physical=generalized_physical,
        correlation=correlation,
        local_information=local_information,
        singular_values=singular_values,
        right_singular_vectors=right_vectors,
        rank=rank,
        condition_number=condition_number,
        covariance_is_identifiable=rank == parameter_count,
        prior_precision_scaled=prior_scaled,
        posterior_normal_matrix=posterior_normal,
        prior_rank=prior_rank,
        posterior_rank=posterior_rank,
        posterior_condition_number=posterior_condition_number,
        posterior_covariance_is_identifiable=posterior_rank == parameter_count,
        posterior_covariance_scaled=posterior_covariance_scaled,
        posterior_covariance_physical=posterior_covariance_physical,
        posterior_correlation=posterior_correlation,
        null_space_vectors=right_vectors[rank:].copy(),
        standard_errors_scaled=standard_errors_scaled,
        standard_errors_physical=standard_errors_physical,
    )


def _validated_residual(residual, observation_count: int) -> np.ndarray:
    result = np.asarray(residual, dtype=np.float64)
    if result.shape != (observation_count,) or not np.all(np.isfinite(result)):
        raise ValueError("residual must be one finite value per observation")
    return result


def _svd_diagnostics(whitened_jacobian):
    _, singular_values, right_vectors = np.linalg.svd(
        whitened_jacobian, full_matrices=True
    )
    return singular_values, right_vectors


def _condition_number(singular_values, rank, parameter_count):
    return (
        float(singular_values[0] / singular_values[-1])
        if rank == parameter_count and len(singular_values) and singular_values[-1] > 0
        else float("inf")
    )


def torch_profile_jacobian(calculator, parameters, paths, *, domain: str = "q"):
    """Return ``(grid, profile, J)`` for selected scalar Torch parameter paths."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without the optional extra
        raise ImportError("torch_profile_jacobian requires the 'torch' extra") from error
    if not getattr(calculator.backend, "is_torch", False):
        raise TypeError("calculator must use TorchBackend")
    selected = tuple(paths)
    if not selected:
        raise ValueError("at least one ParameterPath is required")
    base = {name: value.detach().clone() for name, value in parameters.items()}
    values = []
    for path in selected:
        if path.name not in base:
            raise ValueError(f"unknown parameter tensor {path.name!r}")
        if not np.isfinite(path.scale) or path.scale <= 0:
            raise ValueError(f"invalid scale for {path.display_name}")
        try:
            values.append(base[path.name][path.index])
        except IndexError as error:
            raise ValueError(f"invalid index for {path.display_name}") from error
    vector = torch.stack(values).clone().detach().requires_grad_(True)

    def render(selected_values):
        updated = {name: value.clone() for name, value in base.items()}
        for index, path in enumerate(selected):
            updated[path.name][path.index] = selected_values[index]
        return calculator.pattern(domain=domain, parameters=updated)[1]

    grid, profile = calculator.pattern(domain=domain, parameters=base)
    jacobian = torch.autograd.functional.jacobian(render, vector, vectorize=True)
    return (
        grid.detach().cpu().numpy(),
        profile.detach().cpu().numpy(),
        jacobian.detach().cpu().numpy(),
    )
