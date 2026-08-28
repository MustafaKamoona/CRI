"""Safe micro-probing (Eq. 11-14): a deterministic, repeatable bounded
excitation sequence, scaled down automatically by the current safety margin
so that probing intensity is self-limiting near operational constraints.
"""
from typing import Optional, Tuple
import numpy as np


def _hadamard4() -> np.ndarray:
    H2 = np.array([[1.0, 1.0], [1.0, -1.0]])
    return np.kron(H2, H2)   # 4x4, orthogonal rows/columns, entries +-1


_H4 = _hadamard4()


def need_probe(delta_u_window: np.ndarray, min_natural_excitation: float) -> bool:
    """True when recent commanded-input variation is too small to inform a
    reliable sensitivity estimate on its own."""
    if delta_u_window.size == 0:
        return True
    level = float(np.mean(np.linalg.norm(delta_u_window, axis=0)))
    return level < min_natural_excitation


def generate_probe(k: int, n_u: int, period_steps: int, channel_scale: np.ndarray) -> np.ndarray:
    """Deterministic bounded orthogonal excitation xi_k in [-1,1]^n_u.

    Cycles through one Hadamard COLUMN PER SAMPLE (not held over several
    samples): this is what makes Delta u nonzero on nearly every probed
    step, giving the sensitivity estimator persistent, well-conditioned
    excitation instead of brief spikes at infrequent block boundaries. The
    overall sign flips every `period_steps` samples purely to avoid an
    exactly-4-sample-periodic pattern aliasing with the plant's own
    dynamics; it does not gate how often Delta u is nonzero.
    """
    H = _H4 if n_u == 4 else np.sign(np.cos(2 * np.pi * np.outer(np.arange(n_u), np.arange(n_u)) / n_u) + 1e-9)
    col = k % n_u
    block = (k // max(period_steps, 1)) % 2
    sign = 1.0 if block == 0 else -1.0
    xi = sign * H[:, col]
    return np.asarray(channel_scale, dtype=float) * xi


def scale_probe_with_smi(delta_max: float, smi_k: float) -> float:
    """Eq. (13): delta_bar_k = delta_max * SMI_k -- probing is automatically
    attenuated as the operating point approaches a constraint."""
    return float(delta_max) * float(np.clip(smi_k, 0.0, 1.0))


def check_actuator_bounds(u_k: np.ndarray, delta_u: np.ndarray,
                           u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    """Clip delta_u so that u_k + delta_u stays within [u_min, u_max]
    (Eq. 12's feasible probing set P_k)."""
    lo = u_min - u_k
    hi = u_max - u_k
    return np.clip(delta_u, lo, hi)


def probing_perturbation(k: int, u_k: np.ndarray, smi_k: float, delta_max: float,
                          period_steps: int, channel_scale: np.ndarray,
                          u_min: np.ndarray, u_max: np.ndarray) -> Tuple[np.ndarray, float]:
    """Full safe-probing pipeline for one step: returns (delta_u_k, delta_bar_k)."""
    delta_bar_k = scale_probe_with_smi(delta_max, smi_k)
    xi = generate_probe(k, u_k.size, period_steps, channel_scale)
    delta_u = delta_bar_k * xi
    delta_u = check_actuator_bounds(u_k, delta_u, u_min, u_max)
    return delta_u, delta_bar_k
