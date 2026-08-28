"""Isolated, high-resolution timing of the CRI update, used to connect the
theoretical complexity O(n_y n_u N + n_u^2 N + n_u^3) to measured wall-clock
cost as a function of the window length N and the input dimension n_u.

Benchmarks `sensitivity.estimate_sensitivity` directly (not the full
CRISupervisor.step) because the paper is explicit that this is the
dominant operation -- "the remaining indicator and decision calculations
require only vector operations and threshold comparisons" (Sec. III-D) --
and because the microgrid-specific margin formulas in cri.py are fixed to
n_u=4, n_y=7 and cannot be swept over arbitrary dimensions.
"""
import time
import platform
from typing import List
import numpy as np

from .sensitivity import estimate_sensitivity


def env_info() -> dict:
    return dict(platform=platform.platform(), python_version=platform.python_version(),
                numpy_version=np.__version__)


def _stats_us(arr_ns) -> dict:
    arr_us = np.asarray(arr_ns, dtype=float) / 1e3
    return dict(mean=float(np.mean(arr_us)), std=float(np.std(arr_us, ddof=1)) if len(arr_us) > 1 else 0.0,
                p95=float(np.percentile(arr_us, 95)), max=float(np.max(arr_us)))


def benchmark_cri_step(N: int, n_u: int = 4, n_y: int = 7, n_warmup: int = 200,
                        n_measure: int = 2000, seed: int = 0, lambda_s: float = 1e-3) -> dict:
    """Isolated timing of one regularized sensitivity estimate over an
    (n_u, N)/(n_y, N) window, repeated n_measure times on freshly drawn
    synthetic data (so no cross-call state or caching affects the timing)."""
    rng = np.random.default_rng(seed)
    n_total = n_warmup + n_measure
    U = rng.normal(0.0, 0.05, size=(n_total, n_u, N))
    Y = rng.normal(0.0, 0.05, size=(n_total, n_y, N))
    w = np.ones(N)

    for k in range(n_warmup):
        estimate_sensitivity(U[k], Y[k], w, lambda_s)

    t_ns = np.zeros(n_measure)
    for j in range(n_measure):
        k = n_warmup + j
        t0 = time.perf_counter_ns()
        estimate_sensitivity(U[k], Y[k], w, lambda_s)
        t1 = time.perf_counter_ns()
        t_ns[j] = t1 - t0

    return dict(N=N, n_u=n_u, n_y=n_y, n_measure=n_measure, env=env_info(), timing_us=_stats_us(t_ns))


def sweep_N(N_values: List[int], n_u: int = 4, n_y: int = 7, n_measure: int = 2000, seed: int = 0) -> List[dict]:
    return [benchmark_cri_step(N, n_u=n_u, n_y=n_y, n_measure=n_measure, seed=seed) for N in N_values]


def sweep_n_u(n_u_values: List[int], N: int = 40, n_y: int = 7, n_measure: int = 2000, seed: int = 0) -> List[dict]:
    return [benchmark_cri_step(N, n_u=nu, n_y=n_y, n_measure=n_measure, seed=seed) for nu in n_u_values]
