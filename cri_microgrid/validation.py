"""Simulation-only ground truth, used EXCLUSIVELY for post-hoc evaluation.

Every array consumed here (`*_true`, `g1_true`, noise level, load-drift
rate) is present in a run's log dict only because sim.py chose to record it
for offline analysis -- cri.py and controllers.py never see any of it. This
module is what turns "the CRI is measurement-driven" from a design
intention into something checkable: z_k^unsafe is built from privileged,
noise-free information the readiness layer never had access to.
"""
from typing import Dict, Optional, Tuple
import numpy as np

from .config import ConstraintParams


def compute_unsafe_truth(log: Dict[str, np.ndarray], cons: ConstraintParams,
                          noise_level_pct: float = 0.0,
                          authority_threshold: float = 0.5,
                          noise_threshold_pct: float = 1.5,
                          drift_threshold: float = 0.15,
                          noise_window: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """z_k^unsafe = hard constraint violation OR weak true actuator
    authority OR excessive true measurement noise OR excessive true
    operating-point drift rate -- all evaluated on the *_true fields.

    noise_window: pass the same (t0, t1) as the scenario's Scenario.noise_window
    if it has one (e.g. S6), so the "noisy" ground-truth label only applies
    while noise was actually injected, not for the whole run."""
    V1, V2 = log["V1_true"], log["V2_true"]
    I1, I2 = log["I1_true"], log["I2_true"]
    Vdc = log["Vdc_true"]
    g1, g2 = log["g1_true"], log["g2_true"]
    t = log["t"]

    hard_violation = (
        (V1 < cons.Vac_min) | (V1 > cons.Vac_max) |
        (V2 < cons.Vac_min) | (V2 > cons.Vac_max) |
        (I1 > cons.I_max) | (I2 > cons.I_max) |
        (Vdc < cons.Vdc_min) | (Vdc > cons.Vdc_max)
    )

    weak_authority = (g1 < authority_threshold) | (g2 < authority_threshold)
    noise_active = np.full_like(t, fill_value=True, dtype=bool)
    if noise_window is not None:
        noise_active = (t >= noise_window[0]) & (t < noise_window[1])
    noisy = noise_active & (noise_level_pct > noise_threshold_pct)

    Ts = float(t[1] - t[0]) if len(t) > 1 else 1.0
    Ptot = log["Pload1_true"] + log["Pload2_true"]
    dP = np.gradient(Ptot, Ts)
    scale = np.maximum(np.abs(Ptot), 1e-3)
    drift_rate = np.abs(dP) / scale
    excessive_drift = drift_rate > drift_threshold

    return (hard_violation | weak_authority | noisy | excessive_drift).astype(int)


def confusion_matrix_table(s: np.ndarray, z_unsafe: np.ndarray) -> Dict[str, int]:
    from .metrics import confusion_counts
    return confusion_counts(s, z_unsafe)
