"""Local, measurement-driven input-output sensitivity estimation (Eq. 15-17
of the paper). S_hat is never treated as a plant model; it is only used to
assess readiness (responsiveness/coupling) from the current data window.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SensitivityResult:
    S_hat: np.ndarray            # (n_y, n_u)
    singular_values: np.ndarray
    rho: float                   # sensitivity energy, (1/n_u)*||S_hat||_F^2
    epsilon_s: float             # normalized residual energy
    condition_number: float
    excitation_energy: float     # trace(Delta~U W Delta~U^T)


def estimate_sensitivity(delta_u_norm: np.ndarray, delta_y_norm: np.ndarray,
                          weights: np.ndarray, lambda_s: float,
                          eps0: float = 1e-9) -> SensitivityResult:
    """Weighted, regularized least-squares sensitivity estimate:
        S_hat = DY W DU^T (DU W DU^T + lambda_s I)^-1

    delta_u_norm : (n_u, N) = Delta~U_k
    delta_y_norm : (n_y, N) = Delta~Y_k
    weights      : (N,) diagonal of W_k (forgetting/confidence weights)
    """
    n_u = delta_u_norm.shape[0]
    w = np.asarray(weights, dtype=float)

    Uw = delta_u_norm * w[None, :]
    Yw = delta_y_norm * w[None, :]

    M = Uw @ delta_u_norm.T + lambda_s * np.eye(n_u)
    cross = Yw @ delta_u_norm.T

    S_hat = cross @ np.linalg.inv(M)

    excitation_energy = float(np.trace(Uw @ delta_u_norm.T))

    if excitation_energy < 1e-8:
        # Degenerate window (e.g. the plant is fully saturated and even
        # probing is being clipped to zero at the actuator bound): Delta U
        # carries no information, so S_hat is trivially the zero matrix and
        # would otherwise "perfectly" explain a likewise-trivial Delta Y --
        # a residual of ~0 that reads as ideal confidence despite there
        # being no data to be confident about. Force the worst-case
        # residual instead, so lack of excitation never looks like
        # reliability.
        epsilon_s = 1.0
    else:
        E_k = delta_y_norm - S_hat @ delta_u_norm
        epsilon_s = float(np.linalg.norm(E_k, "fro") / (np.linalg.norm(delta_y_norm, "fro") + eps0))

    svals = np.linalg.svd(S_hat, compute_uv=False)
    rho = float(np.sum(svals ** 2) / n_u)
    cond = float(svals.max() / max(svals.min(), 1e-12)) if svals.size else float("inf")

    return SensitivityResult(
        S_hat=S_hat, singular_values=svals, rho=rho,
        epsilon_s=epsilon_s, condition_number=cond,
        excitation_energy=excitation_energy,
    )


def coupling_raw(S_hat: np.ndarray, eps0: float = 1e-9) -> float:
    """c_k, Eq. (20): column-normalized off-diagonal-vs-peak energy ratio.

    A near-zero S_hat (no informative excitation, e.g. a fully saturated
    actuator) carries no evidence either way about coupling; conservatively
    treat it as maximally coupled (c_k=1, i.e. CI=0) rather than letting an
    empty 0/eps0 ratio read as "perfectly decoupled."
    """
    abs_S = np.abs(S_hat)
    total = np.sum(abs_S)
    if total < 1e-8:
        return 1.0
    col_sum = abs_S.sum(axis=0)
    col_max = abs_S.max(axis=0)
    num = np.sum(col_sum - col_max)
    return float(num / (total + eps0))
