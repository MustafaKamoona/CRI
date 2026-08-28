"""Readiness/activation metrics and control/disturbance metrics (Sec. V of
the paper), plus a few simulation-level metrics (unsafe active fraction,
probe energy, min/max signal envelopes) that make the notebook result
tables self-explanatory to a reviewer.
"""
import numpy as np


def summary_stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, std=np.nan, median=np.nan, min=np.nan, max=np.nan)
    return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                median=float(np.median(x)), min=float(np.min(x)), max=float(np.max(x)))


# --- Readiness and activation metrics --------------------------------------
def first_activation_time(t, CRI, eta_A) -> float:
    """T_A: first time CRI_k >= eta_A. NaN if it never happens."""
    t, CRI = np.asarray(t), np.asarray(CRI)
    idx = np.where(CRI >= eta_A)[0]
    return float(t[idx[0]]) if idx.size else float("nan")


def active_duration(s, Ts) -> float:
    """T_on = sum_k s_k * Ts."""
    return float(np.sum(np.asarray(s)) * Ts)


def false_activation_count(s, z_unsafe) -> int:
    """N_FA = # samples with s_k=1 while z_k^unsafe=1."""
    s, z = np.asarray(s), np.asarray(z_unsafe)
    return int(np.sum((s == 1) & (z == 1)))


def unsafe_active_fraction(s, z_unsafe, eps=1e-9) -> float:
    """sum(s_k*z_k^unsafe) / (sum(z_k^unsafe)+eps): near 0 for an effective
    supervisor -- the controller is almost never active while the plant is
    truly in an unsafe condition."""
    s, z = np.asarray(s, dtype=float), np.asarray(z_unsafe, dtype=float)
    return float(np.sum(s * z) / (np.sum(z) + eps))


def safe_availability_pct(s, z_unsafe) -> float:
    """Percent of SAFE samples (z_unsafe=0) during which the controller was
    active -- how much useful availability the supervisor preserves."""
    s, z = np.asarray(s, dtype=float), np.asarray(z_unsafe, dtype=float)
    safe_mask = z == 0
    if not np.any(safe_mask):
        return float("nan")
    return float(100.0 * np.mean(s[safe_mask]))


def confusion_counts(s, z_unsafe) -> dict:
    s, z = np.asarray(s), np.asarray(z_unsafe)
    return dict(
        correct_activation=int(np.sum((s == 1) & (z == 0))),
        unnecessary_delay=int(np.sum((s == 0) & (z == 0))),
        false_activation=int(np.sum((s == 1) & (z == 1))),
        correct_rejection=int(np.sum((s == 0) & (z == 1))),
    )


def cri_degradation(cri_nominal_mean, cri_dist_mean, eps=1e-9) -> float:
    """D_CRI = (CRI_nom - CRI_dist) / (CRI_nom + eps)."""
    return float((cri_nominal_mean - cri_dist_mean) / (cri_nominal_mean + eps))


# --- Control / disturbance metrics -----------------------------------------
def voltage_deviation(V1, V2, Vref=1.0) -> float:
    """E_V = mean_k || [V1_k,V2_k] - Vref ||_2."""
    V1, V2 = np.asarray(V1), np.asarray(V2)
    dev = np.stack([V1 - Vref, V2 - Vref], axis=0)
    return float(np.mean(np.linalg.norm(dev, axis=0)))


def current_violation(I1, I2, I_max) -> float:
    """E_I = mean_k ( max(0,I1-Imax) + max(0,I2-Imax) )."""
    I1, I2 = np.asarray(I1), np.asarray(I2)
    return float(np.mean(np.maximum(0.0, I1 - I_max) + np.maximum(0.0, I2 - I_max)))


def dc_voltage_deviation(Vdc, Vdc_ref=1.0) -> float:
    return float(np.mean(np.abs(np.asarray(Vdc) - Vdc_ref)))


def poi_power_deviation(P_POI, P_POI_ref) -> float:
    return float(np.mean(np.abs(np.asarray(P_POI) - P_POI_ref)))


def violation_duration_pct(V1, V2, I1, I2, Vdc, cons) -> float:
    n = len(V1)
    viol = ((V1 < cons.Vac_min) | (V1 > cons.Vac_max) | (V2 < cons.Vac_min) | (V2 > cons.Vac_max) |
            (I1 > cons.I_max) | (I2 > cons.I_max) | (Vdc < cons.Vdc_min) | (Vdc > cons.Vdc_max))
    return float(100.0 * np.mean(viol))


# --- Probing / adaptation / timing metrics ---------------------------------
def probe_energy(probe_norm, Ts) -> float:
    return float(np.sum(np.asarray(probe_norm) ** 2) * Ts)


def detection_delay(t, series, event_time, threshold, direction="below") -> float:
    """First time after event_time that `series` crosses `threshold` in the
    given direction, minus event_time. NaN if it never crosses."""
    t, series = np.asarray(t), np.asarray(series)
    mask_t = t >= event_time
    if not np.any(mask_t):
        return float("nan")
    idx0 = np.argmax(mask_t)
    seg_t, seg_s = t[idx0:], series[idx0:]
    cond = seg_s <= threshold if direction == "below" else seg_s >= threshold
    idx = np.where(cond)[0]
    return float(seg_t[idx[0]] - event_time) if idx.size else float("nan")


def recovery_time(t, cls_code, event_end_time, activate_code=3) -> float:
    """First time after event_end_time that the classification returns to
    Activate, minus event_end_time."""
    return detection_delay(t, cls_code, event_end_time, activate_code - 0.5, direction="above")
