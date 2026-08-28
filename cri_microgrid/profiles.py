"""Generic exogenous-input building blocks (loads, PV infeed, POI voltage)
shared by every scenario. scenarios.py composes these into the specific
time-varying schedules for S1-S10; nothing here is scenario-specific."""
from typing import Dict, List, Sequence, Tuple
import numpy as np

from .config import ProfileParams


def ramp(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    """Linear ramp: y0 for t<=t0, y1 for t>=t1, linear in between."""
    if t1 <= t0:
        return y1 if t >= t0 else y0
    frac = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return float(y0 + frac * (y1 - y0))


def piecewise_linear(t: float, knots: Sequence[Tuple[float, float]]) -> float:
    """knots: sorted list of (time, value); flat-extrapolated outside range."""
    if t <= knots[0][0]:
        return float(knots[0][1])
    if t >= knots[-1][0]:
        return float(knots[-1][1])
    for (t0, y0), (t1, y1) in zip(knots[:-1], knots[1:]):
        if t0 <= t <= t1:
            return ramp(t, t0, t1, y0, y1)
    return float(knots[-1][1])


def sinusoidal(t: float, base: float, amplitude: float, period: float, phase: float = 0.0) -> float:
    return float(base + amplitude * np.sin(2 * np.pi * t / period + phase))


def baseline_exogenous(pp: ProfileParams) -> Dict[str, float]:
    """Flat, nominal exogenous inputs (used as the S1 baseline and as the
    starting point every other scenario perturbs)."""
    P1, P2 = pp.P_load_base
    Q1, Q2 = pp.Q_load_base
    return dict(
        Pload1=P1 * pp.load_scale, Qload1=Q1 * pp.load_scale,
        Pload2=P2 * pp.load_scale, Qload2=Q2 * pp.load_scale,
        Ppv=pp.P_pv_base * pp.pv_scale, Pdcload=pp.Pdcload_base * pp.load_scale,
        Vpoi=1.0,
    )


def drift_multiplier(t: float, ramp_kind: str, t_start: float = 5.0, period: float = 30.0,
                      amplitude: float = 0.25) -> float:
    """A dimensionless multiplicative drift signal in roughly [1-amplitude,
    1+amplitude], whose *rate* of change is set by `ramp_kind` -- used by S5
    (and reused inside S8) to drive the operating point at normal, fast, or
    very-fast speed while keeping the same total swing amplitude."""
    speed = {"flat": 0.0, "normal": 1.0, "fast": 3.0, "very_fast": 8.0}.get(ramp_kind, 0.0)
    if speed == 0.0:
        return 1.0
    return float(1.0 + amplitude * np.sin(2 * np.pi * speed * max(t - t_start, 0.0) / period))


def exogenous_with_drift(pp: ProfileParams, t: float, ramp_kind: str = "flat",
                          amplitude: float = 0.25, t_start: float = 5.0, period: float = 30.0) -> Dict[str, float]:
    base = baseline_exogenous(pp)
    m = drift_multiplier(t, ramp_kind, t_start=t_start, period=period, amplitude=amplitude)
    base["Pload1"] *= m
    base["Pload2"] *= m
    base["Ppv"] *= (2.0 - m)   # PV and load drift in opposite directions (worst-case net-power swing)
    return base
