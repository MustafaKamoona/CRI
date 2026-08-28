"""Master closed-loop simulation: wires the plant, the fallback and OFO
candidate controllers, and (for the 'cri' strategy) the CRI supervisor and
safe micro-probing together into a single reproducible run.

Three strategies share every other setting (same plant, same scenario, same
OFO parameters, same random seed):
  'direct'    : s_k=1, kappa_k=1 always -- the candidate always drives the plant.
  'threshold' : a conventional hard-limit rule-based supervisor -- active iff
                every instantaneous constraint margin is currently satisfied.
  'cri'       : the full CRI-supervised activation of Algorithm 1.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .config import (PlantParams, ControllerParams, ConstraintParams, CRIParams,
                      ProbeParams, ProfileParams, SimulationParams, DEFAULT_D_u, DEFAULT_D_y)
from .plant import HybridACDCPlant
from .controllers import FallbackDroopController, OFOVSCController, blend_command
from .cri import CRISupervisor, microgrid_margins, compute_SMI
from . import probing as pb
from .scenarios import Scenario

LOG_FIELDS = [
    "t", "s", "kappa", "cls_code", "dt_flag",
    "CRI", "CRI_hat", "RI", "CI", "SMI", "AI", "NCI", "beta",
    "epsilon_s", "rho", "condition_number", "excitation_energy", "sigma_hat_sq", "a_raw",
    "probe_bar", "probe_norm",
    "V1", "V2", "Vdc", "I1", "I2", "P_POI", "Q_POI",
    "V1_true", "V2_true", "Vdc_true", "I1_true", "I2_true",
    "P1_true", "Q1_true", "P2_true", "Q2_true",
    "P1_cmd", "Q1_cmd", "P2_cmd", "Q2_cmd",
    "P1_candidate", "Q1_candidate", "P2_candidate", "Q2_candidate",
    "g1_true", "g2_true", "execution_time",
    "Pload1_true", "Pload2_true", "Ppv_true",
    "R1_self_true", "R2_self_true", "Rmutual_true",
]

_CLASS_CODE = {"Activate": 3, "Monitor": 2, "Delay": 1, "Reject": 0}


_NO_NOISE = dict(enabled=False, level_pct=0.0)


def _active_noise_cfg(scenario: Scenario, t: float) -> Dict:
    """scenario.noise_cfg, gated by scenario.noise_window if one is set
    (None means always active, preserving every scenario's original
    behavior except S6, which uses this to start clean)."""
    if scenario.noise_window is None:
        return scenario.noise_cfg
    t0, t1 = scenario.noise_window
    return scenario.noise_cfg if (t0 <= t < t1) else _NO_NOISE


def _u_bounds(cons: ConstraintParams):
    lo = np.array([cons.P_min, cons.Q_min, cons.P_min, cons.Q_min])
    hi = np.array([cons.P_max, cons.Q_max, cons.P_max, cons.Q_max])
    return lo, hi


def run_closed_loop(scenario: Scenario, plant_params: PlantParams, ctrl_params: ControllerParams,
                     cons: ConstraintParams, cri_params: CRIParams, probe_params: ProbeParams,
                     sim_params: SimulationParams, strategy: Optional[str] = None,
                     probing_enabled: bool = True, measure_time: bool = False,
                     capture_shat_times: Optional[list] = None,
                     plant=None, fallback=None, candidate=None) -> Dict[str, np.ndarray]:
    """capture_shat_times: optional list of times (s); the CRI supervisor's
    S_hat at the nearest sample to each requested time is stored in the
    returned log under log["shat_snapshots"][t_requested] (strategy='cri'
    only) -- used only by plotting/analysis notebooks, never by the
    control loop itself.

    plant, fallback, candidate: optional pre-built instances. Pass your own
    to test a different candidate controller (most common: keep the
    reference microgrid plant/fallback, swap in your own `candidate`) or
    fallback law against the exact same CRI supervisor, probing, and
    metrics used throughout this study -- see cri_microgrid/interfaces.py
    and notebooks/11_bring_your_own.ipynb for the required interface and a
    worked example. Each defaults to the built-in microgrid
    plant/controllers when omitted, so every existing call site is
    unaffected. Note this function's logging (V1,V2,Vdc,I1,I2,P_POI,Q_POI /
    P1,Q1,P2,Q2) is specific to the 4-input/7-output microgrid layout; for
    an entirely different plant (different dimensions), use
    `run_closed_loop_generic` instead.
    """
    strategy = strategy or sim_params.strategy
    assert strategy in ("direct", "threshold", "cri")

    Ts = plant_params.Ts
    n = int(np.floor(scenario.T_end / Ts)) + 1
    t_axis = np.arange(n) * Ts

    rng = np.random.default_rng(sim_params.seed)
    plant = plant if plant is not None else HybridACDCPlant(plant_params, noise_rng=rng)
    fallback = fallback if fallback is not None else FallbackDroopController(ctrl_params, cons, plant_params)
    candidate = candidate if candidate is not None else OFOVSCController(ctrl_params, cons, plant_params)
    cri_sup = CRISupervisor(cri_params, D_u=DEFAULT_D_u, D_y=DEFAULT_D_y) if strategy == "cri" else None
    u_lo, u_hi = _u_bounds(cons)

    log = {k: np.zeros(n) for k in LOG_FIELDS}
    shat_snapshots = {}
    capture_steps = {int(round(tc / Ts)): tc for tc in capture_shat_times} if capture_shat_times else {}

    exo0 = scenario.exogenous_fn(0.0)
    y_meas, hidden0 = plant.measure(exo0, _active_noise_cfg(scenario, 0.0))
    u_applied = candidate.candidate_command().copy()

    s_prev = 1 if strategy == "direct" else 0
    kappa_prev = 1.0
    # Tracks the CONTROLLER-INTENDED command movement only (before any probe
    # is added on top), so probing's own effect on u never feeds back into
    # the decision of whether more probing is needed (that self-reference
    # would make probing flicker on/off every step).
    natural_du_hist = deque(maxlen=cri_params.N)

    import time as _time

    for k in range(n):
        t = t_axis[k]
        exo = scenario.exogenous_fn(t)
        gain_true = scenario.actuator_gain_fn(t)
        topo = scenario.topology_fn(t)

        t0 = _time.perf_counter_ns() if measure_time else None

        u_safe = fallback.command(y_meas)
        u_candidate = candidate.candidate_command()
        delta_nom = candidate.propose_nominal_step(y_meas)

        probe_bar = 0.0
        probe_vec = np.zeros(4)

        if strategy == "direct":
            s_k, kappa_k, cls = 1, 1.0, "Activate"
            cri_out = None
        elif strategy == "threshold":
            margins = microgrid_margins(y_meas, u_applied, cons)
            ok = min(margins) >= sim_params.threshold_margin
            s_k = 1 if ok else 0
            kappa_k = 1.0
            cls = "Activate" if ok else "Reject"
            cri_out = None
        else:  # 'cri'
            cri_out = cri_sup.step(u_applied, y_meas, cons)
            s_k, kappa_k, cls = cri_out.s, cri_out.kappa, cri_out.cls

        u_intended = blend_command(u_safe, u_candidate, s_k, kappa_k)
        natural_du_hist.append((u_intended - u_applied) / DEFAULT_D_u)

        if strategy == "cri" and probing_enabled:
            smi_now = cri_out.SMI
            du_hist = np.stack(natural_du_hist, axis=1) if len(natural_du_hist) else np.zeros((4, 0))
            if pb.need_probe(du_hist, probe_params.min_natural_excitation):
                probe_vec, probe_bar = pb.probing_perturbation(
                    k, u_intended, smi_now, cri_params.delta_max,
                    probe_params.period_steps, np.array(probe_params.channel_scale),
                    u_lo, u_hi)

        u_next = np.clip(u_intended + probe_vec, u_lo, u_hi)

        candidate.shadow_update(delta_nom, kappa_k)

        y_next, hidden = plant.step(u_next, t, exo, gain_true, topo, _active_noise_cfg(scenario, t))

        if measure_time:
            log["execution_time"][k] = (_time.perf_counter_ns() - t0) / 1e9

        log["t"][k] = t
        log["s"][k] = s_k
        log["kappa"][k] = kappa_k
        log["cls_code"][k] = _CLASS_CODE[cls]
        log["probe_bar"][k] = probe_bar
        # Infinity norm, matching the paper's actual probing bound
        # ||delta u||_inf <= delta_bar_k (Eq. 12) -- an L2 norm here would
        # let a 4-channel probe exceed delta_bar_k by up to a factor of 2.
        log["probe_norm"][k] = float(np.max(np.abs(probe_vec)))

        if k in capture_steps and cri_out is not None:
            shat_snapshots[capture_steps[k]] = cri_out.S_hat.copy()

        if cri_out is not None:
            log["dt_flag"][k] = cri_out.dt_flag
            log["CRI"][k] = cri_out.CRI
            log["CRI_hat"][k] = cri_out.CRI_hat
            log["RI"][k] = cri_out.RI
            log["CI"][k] = cri_out.CI
            log["SMI"][k] = cri_out.SMI
            log["AI"][k] = cri_out.AI
            log["NCI"][k] = cri_out.NCI
            log["beta"][k] = cri_out.beta
            log["epsilon_s"][k] = cri_out.epsilon_s
            log["rho"][k] = cri_out.rho
            log["condition_number"][k] = cri_out.condition_number
            log["excitation_energy"][k] = cri_out.excitation_energy
            log["sigma_hat_sq"][k] = cri_out.sigma_hat_sq
            log["a_raw"][k] = cri_out.a_raw
        else:
            margins = microgrid_margins(y_meas, u_applied, cons)
            log["SMI"][k] = compute_SMI(margins)
            log["CRI"][k] = float(s_k)
            log["RI"][k] = log["CI"][k] = log["AI"][k] = log["NCI"][k] = np.nan

        log["V1"][k], log["V2"][k], log["Vdc"][k] = y_meas[0], y_meas[1], y_meas[2]
        log["I1"][k], log["I2"][k] = y_meas[3], y_meas[4]
        log["P_POI"][k], log["Q_POI"][k] = y_meas[5], y_meas[6]

        log["V1_true"][k], log["V2_true"][k], log["Vdc_true"][k] = hidden["V1"], hidden["V2"], hidden["Vdc"]
        log["I1_true"][k], log["I2_true"][k] = hidden["I1"], hidden["I2"]
        log["P1_true"][k], log["Q1_true"][k] = hidden["P1"], hidden["Q1"]
        log["P2_true"][k], log["Q2_true"][k] = hidden["P2"], hidden["Q2"]

        log["P1_cmd"][k], log["Q1_cmd"][k], log["P2_cmd"][k], log["Q2_cmd"][k] = u_next
        log["P1_candidate"][k], log["Q1_candidate"][k], log["P2_candidate"][k], log["Q2_candidate"][k] = u_candidate
        log["g1_true"][k], log["g2_true"][k] = gain_true
        log["Pload1_true"][k], log["Pload2_true"][k], log["Ppv_true"][k] = exo["Pload1"], exo["Pload2"], exo["Ppv"]
        log["R1_self_true"][k], log["R2_self_true"][k] = topo.R_self[0], topo.R_self[1]
        log["Rmutual_true"][k] = topo.R_mutual

        u_applied = u_next
        y_meas = y_next
        s_prev, kappa_prev = s_k, kappa_k

    log["strategy"] = strategy
    log["scenario_name"] = scenario.name
    log["shat_snapshots"] = shat_snapshots
    return log


def run_closed_loop_generic(scenario: Scenario, cri_params: CRIParams, probe_params: ProbeParams,
                             sim_params: SimulationParams, plant, fallback, candidate,
                             n_u: int, n_y: int, cons, margin_fn,
                             D_u: Optional[np.ndarray] = None, D_y: Optional[np.ndarray] = None,
                             u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                             strategy: Optional[str] = None, probing_enabled: bool = True) -> Dict:
    """Dimension-agnostic version of `run_closed_loop`, for an entirely
    custom plant (not the built-in 4-input/7-output microgrid). Everything
    the CRI framework needs is supplied explicitly rather than assumed:

    plant, fallback, candidate : your objects (see cri_microgrid/interfaces.py
                                  for the exact required methods).
    n_u, n_y                   : dimensions of your command and measurement vectors.
    cons                       : your constraints object, opaque to this
                                  function -- only `margin_fn` and your own
                                  controllers interpret it.
    margin_fn(y_meas, u_applied, cons) -> List[float] : your normalized
                                  constraint margins (Eq. 22-23); required
                                  for the 'cri' and 'threshold' strategies.
    D_u, D_y                   : representative per-channel scales for
                                  normalization (Eq. 10); default to all-ones.
    u_bounds                   : (u_min, u_max) arrays, length n_u; default
                                  to +/- inf (no probing clipping).

    Returns a log dict with generic fields: t, s, kappa, cls_code, dt_flag,
    CRI, CRI_hat, RI, CI, SMI, AI, NCI, beta, epsilon_s, rho,
    condition_number, excitation_energy, sigma_hat_sq, a_raw, probe_bar,
    probe_norm (all length-n 1D arrays), u_applied and y_meas (n x n_u /
    n x n_y arrays), and hidden_truth_history (a length-n list of whatever
    your plant's step()/measure() returned as hidden_truth -- schema is
    entirely yours, never touched by cri.py or controllers.py).
    """
    strategy = strategy or sim_params.strategy
    assert strategy in ("direct", "threshold", "cri")

    D_u = np.ones(n_u) if D_u is None else np.asarray(D_u, dtype=float)
    D_y = np.ones(n_y) if D_y is None else np.asarray(D_y, dtype=float)
    u_lo, u_hi = ((-np.inf * np.ones(n_u), np.inf * np.ones(n_u)) if u_bounds is None
                  else (np.asarray(u_bounds[0], dtype=float), np.asarray(u_bounds[1], dtype=float)))

    Ts = sim_params.Ts
    n = int(np.floor(scenario.T_end / Ts)) + 1
    t_axis = np.arange(n) * Ts

    cri_sup = CRISupervisor(cri_params, D_u=D_u, D_y=D_y, n_u=n_u, n_y=n_y, margin_fn=margin_fn) \
        if strategy == "cri" else None

    scalar_fields = ["t", "s", "kappa", "cls_code", "dt_flag", "CRI", "CRI_hat", "RI", "CI", "SMI",
                      "AI", "NCI", "beta", "epsilon_s", "rho", "condition_number", "excitation_energy",
                      "sigma_hat_sq", "a_raw", "probe_bar", "probe_norm"]
    log = {k: np.zeros(n) for k in scalar_fields}
    log["u_applied"] = np.zeros((n, n_u))
    log["y_meas"] = np.zeros((n, n_y))
    log["hidden_truth_history"] = []

    exo0 = scenario.exogenous_fn(0.0)
    y_meas, _ = plant.measure(exo0, _active_noise_cfg(scenario, 0.0))
    u_applied = np.asarray(candidate.candidate_command(), dtype=float).copy()
    natural_du_hist = deque(maxlen=cri_params.N)

    for k in range(n):
        t = t_axis[k]
        exo = scenario.exogenous_fn(t)
        gain_true = scenario.actuator_gain_fn(t)
        topo = scenario.topology_fn(t)

        u_safe = np.asarray(fallback.command(y_meas), dtype=float)
        u_candidate = np.asarray(candidate.candidate_command(), dtype=float)
        delta_nom = np.asarray(candidate.propose_nominal_step(y_meas), dtype=float)

        probe_bar, probe_vec = 0.0, np.zeros(n_u)

        if strategy == "direct":
            s_k, kappa_k, cls = 1, 1.0, "Activate"
            cri_out = None
        elif strategy == "threshold":
            margins = margin_fn(y_meas, u_applied, cons)
            ok = min(margins) >= sim_params.threshold_margin
            s_k, kappa_k, cls = (1 if ok else 0), 1.0, ("Activate" if ok else "Reject")
            cri_out = None
        else:
            cri_out = cri_sup.step(u_applied, y_meas, cons)
            s_k, kappa_k, cls = cri_out.s, cri_out.kappa, cri_out.cls

        u_intended = blend_command(u_safe, u_candidate, s_k, kappa_k)
        natural_du_hist.append((u_intended - u_applied) / D_u)

        if strategy == "cri" and probing_enabled:
            du_hist = np.stack(natural_du_hist, axis=1) if len(natural_du_hist) else np.zeros((n_u, 0))
            if pb.need_probe(du_hist, probe_params.min_natural_excitation):
                probe_vec, probe_bar = pb.probing_perturbation(
                    k, u_intended, cri_out.SMI, cri_params.delta_max,
                    probe_params.period_steps, np.ones(n_u), u_lo, u_hi)

        u_next = np.clip(u_intended + probe_vec, u_lo, u_hi)
        candidate.shadow_update(delta_nom, kappa_k)
        y_next, hidden = plant.step(u_next, t, exo, gain_true, topo, _active_noise_cfg(scenario, t))

        log["t"][k], log["s"][k], log["kappa"][k] = t, s_k, kappa_k
        log["cls_code"][k] = _CLASS_CODE[cls]
        log["probe_bar"][k] = probe_bar
        log["probe_norm"][k] = float(np.max(np.abs(probe_vec))) if n_u else 0.0
        log["u_applied"][k] = u_next
        log["y_meas"][k] = y_meas
        log["hidden_truth_history"].append(hidden)

        if cri_out is not None:
            for f in ["dt_flag", "CRI", "CRI_hat", "RI", "CI", "SMI", "AI", "NCI", "beta",
                      "epsilon_s", "rho", "condition_number", "excitation_energy", "sigma_hat_sq", "a_raw"]:
                log[f][k] = getattr(cri_out, f)
        else:
            margins = margin_fn(y_meas, u_applied, cons)
            log["SMI"][k] = compute_SMI(margins)
            log["CRI"][k] = float(s_k)
            log["RI"][k] = log["CI"][k] = log["AI"][k] = log["NCI"][k] = np.nan

        u_applied, y_meas = u_next, y_next

    log["strategy"] = strategy
    log["scenario_name"] = scenario.name
    return log
