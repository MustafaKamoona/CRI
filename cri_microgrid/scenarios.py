"""Scenario builders S1-S8 (Table II of the paper) plus the S9 parameter-
sweep and S10 statistical-validation helpers. Each scenario is a bundle of
time-varying callables (exogenous inputs, true actuator gain, true
topology, noise config) that sim.py's master loop consumes generically --
sim.py has no scenario-specific logic anywhere in it.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple
import copy
import numpy as np

from .config import PlantParams, ProfileParams
from .plant import TopologyState, nominal_topology
from . import profiles as pf


@dataclass
class Scenario:
    name: str
    T_end: float
    exogenous_fn: Callable[[float], Dict[str, float]]
    actuator_gain_fn: Callable[[float], Tuple[float, float]]
    topology_fn: Callable[[float], TopologyState]
    noise_cfg: Dict
    description: str = ""
    event_time: Optional[float] = None
    event_end: Optional[float] = None
    # If set, noise_cfg is only active for t in [noise_window[0], noise_window[1]);
    # outside it the plant is noise-free. None (default) means noise_cfg applies
    # for the whole run, unchanged from every scenario's original behavior.
    noise_window: Optional[Tuple[float, float]] = None


def _const_gain(g1: float = 1.0, g2: float = 1.0) -> Callable[[float], Tuple[float, float]]:
    return lambda t: (g1, g2)


def _const_topology(pp: PlantParams, coupling_scale: float = 1.0) -> Callable[[float], TopologyState]:
    topo = nominal_topology(pp, coupling_scale=coupling_scale)
    return lambda t: topo


# -----------------------------------------------------------------------
# S1 - Nominal / healthy
# -----------------------------------------------------------------------
def build_S1(pp: PlantParams, prof: ProfileParams, T_end: float = 40.0) -> Scenario:
    return Scenario(
        name="S1", T_end=T_end,
        exogenous_fn=lambda t: pf.baseline_exogenous(prof),
        actuator_gain_fn=_const_gain(1.0, 1.0),
        topology_fn=_const_topology(pp, 1.0),
        noise_cfg=dict(enabled=False, level_pct=0.0),
        description="Nominal loading, clean measurements, full VSC headroom.",
    )


# -----------------------------------------------------------------------
# S2 - Constraint-limited operation (loading ramps toward V/I/Vdc limits)
# -----------------------------------------------------------------------
def build_S2(pp: PlantParams, prof: ProfileParams, T_end: float = 40.0,
             t0: float = 6.0, t1: float = 34.0, load_factor: float = 3.0) -> Scenario:
    def exo(t):
        base = pf.baseline_exogenous(prof)
        m = pf.ramp(t, t0, t1, 1.0, load_factor)
        # Stresses the AC voltage/current margins directly (heavier local
        # P+Q demand plus a mild upstream POI sag), rather than starving the
        # DC-link power balance -- that is what S2 ("near voltage/current
        # limits") is meant to isolate, as distinct from S3's actuator
        # authority loss or S8's combined stress.
        base["Pload1"] *= m
        base["Pload2"] *= m
        base["Qload1"] *= (1.0 + 1.5 * (m - 1.0))
        base["Qload2"] *= (1.0 + 1.5 * (m - 1.0))
        base["Vpoi"] = pf.ramp(t, t0, t1, 1.0, 0.975)
        return base

    return Scenario(
        name="S2", T_end=T_end, exogenous_fn=exo,
        actuator_gain_fn=_const_gain(1.0, 1.0),
        topology_fn=_const_topology(pp, 1.0),
        noise_cfg=dict(enabled=False, level_pct=0.0),
        description="High loading pushing toward voltage/current/DC-link limits.",
        event_time=t0, event_end=t1,
    )


# -----------------------------------------------------------------------
# S3 - Weak actuator authority (VSC1 loses effectiveness for a window)
# -----------------------------------------------------------------------
def build_S3(pp: PlantParams, prof: ProfileParams, T_end: float = 40.0,
             t0: float = 10.0, t1: float = 25.0, weak_gain: float = 0.20,
             weak_gain2: Optional[float] = None) -> Scenario:
    """weak_gain2: if given, VSC2 also loses effectiveness (to weak_gain2)
    over the same window, instead of the default single-VSC1 fault. A
    single-actuator fault lets the healthy VSC compensate, which the
    aggregate (sum-of-squares) RI energy metric rho_k = (1/n_u)||S_hat||_F^2
    cannot see through -- the unaffected VSC's rising column energy masks
    the faulted VSC's falling one in the sum. Faulting both symmetrically
    removes that masking, exercising RI as actually defined."""
    def gain_fn(t):
        if t0 <= t <= t1:
            return (weak_gain, weak_gain2 if weak_gain2 is not None else 1.0)
        return (1.0, 1.0)

    both = weak_gain2 is not None
    return Scenario(
        name="S3", T_end=T_end,
        exogenous_fn=lambda t: pf.baseline_exogenous(prof),
        actuator_gain_fn=gain_fn,
        topology_fn=_const_topology(pp, 1.0),
        noise_cfg=dict(enabled=False, level_pct=0.0),
        description=("Both VSCs lose actuator authority together (no compensating actuator) "
                      "while limits stay close to nominal." if both else
                      "VSC1 actuator authority collapses (saturation/lost headroom) while limits still look nominal."),
        event_time=t0, event_end=t1,
    )


# -----------------------------------------------------------------------
# S4 - Strong AC/DC / feeder coupling
# -----------------------------------------------------------------------
def build_S4(pp: PlantParams, prof: ProfileParams, T_end: float = 40.0,
             coupling_scale: float = 6.0) -> Scenario:
    return Scenario(
        name="S4", T_end=T_end,
        exogenous_fn=lambda t: pf.baseline_exogenous(prof),
        actuator_gain_fn=_const_gain(1.0, 1.0),
        topology_fn=_const_topology(pp, coupling_scale),
        noise_cfg=dict(enabled=False, level_pct=0.0),
        description="Strong mutual feeder coupling between the two VSC buses.",
    )


# -----------------------------------------------------------------------
# S5 - Fast operating-point drift
# -----------------------------------------------------------------------
def build_S5(pp: PlantParams, prof: ProfileParams, ramp_kind: str = "fast",
             T_end: float = 40.0, amplitude: float = 0.30) -> Scenario:
    def exo(t):
        return pf.exogenous_with_drift(prof, t, ramp_kind=ramp_kind, amplitude=amplitude, t_start=5.0, period=20.0)

    return Scenario(
        name=f"S5_{ramp_kind}", T_end=T_end, exogenous_fn=exo,
        actuator_gain_fn=_const_gain(1.0, 1.0),
        topology_fn=_const_topology(pp, 1.0),
        noise_cfg=dict(enabled=False, level_pct=0.0),
        description=f"Operating-point drift at '{ramp_kind}' speed.",
        event_time=5.0,
    )


# -----------------------------------------------------------------------
# S6 - Measurement noise / sensor degradation
# -----------------------------------------------------------------------
def build_S6(pp: PlantParams, prof: ProfileParams, noise_level_pct: float = 1.0,
             T_end: float = 40.0, t_onset: float = 10.0) -> Scenario:
    """Starts clean (like every other scenario) so the CRI has a chance to
    build initial confidence and activate BEFORE noise is introduced at
    t_onset -- otherwise, with noise present from t=0, the window never
    contains a clean baseline and the CRI can never reach Activate in the
    first place, which tests something different (bootstrapping trust
    under permanent noise) from what S6 is meant to isolate (does NCI
    selectively degrade an already-trusted state as noise appears)."""
    return Scenario(
        name=f"S6_{noise_level_pct:g}pct", T_end=T_end,
        exogenous_fn=lambda t: pf.baseline_exogenous(prof),
        actuator_gain_fn=_const_gain(1.0, 1.0),
        topology_fn=_const_topology(pp, 1.0),
        noise_cfg=dict(enabled=(noise_level_pct > 0.0), level_pct=noise_level_pct),
        noise_window=(t_onset, T_end),
        description=f"Measurement noise at {noise_level_pct}% of full scale, starting at t={t_onset:g}s.",
        event_time=t_onset,
    )


# -----------------------------------------------------------------------
# S7 - Topology change / converter outage
# -----------------------------------------------------------------------
def build_S7(pp: PlantParams, prof: ProfileParams, T_end: float = 40.0,
             event_time: float = 18.0, kind: str = "impedance_increase") -> Scenario:
    topo_pre = nominal_topology(pp, 1.0)
    if kind == "vsc2_outage":
        topo_post = TopologyState(
            R_self=(pp.R_self[0], pp.R_self[1] * 8.0),
            X_self=(pp.X_self[0], pp.X_self[1] * 8.0),
            R_mutual=pp.R_mutual * 0.2, X_mutual=pp.X_mutual * 0.2,
            vsc_online=(True, False),
        )
        gain_fn = lambda t: (1.0, 1.0) if t < event_time else (1.0, 0.0)
    elif kind == "impedance_increase":
        topo_post = TopologyState(
            R_self=(pp.R_self[0] * 3.0, pp.R_self[1] * 3.0),
            X_self=(pp.X_self[0] * 3.0, pp.X_self[1] * 3.0),
            R_mutual=pp.R_mutual * 3.0, X_mutual=pp.X_mutual * 3.0,
            vsc_online=(True, True),
        )
        gain_fn = _const_gain(1.0, 1.0)
    else:
        raise ValueError(f"unknown topology_event_kind {kind!r}")

    def topo_fn(t):
        return topo_pre if t < event_time else topo_post

    return Scenario(
        name="S7", T_end=T_end,
        exogenous_fn=lambda t: pf.baseline_exogenous(prof),
        actuator_gain_fn=gain_fn,
        topology_fn=topo_fn,
        noise_cfg=dict(enabled=False, level_pct=0.0),
        description=f"Topology event ('{kind}') at t={event_time:g}s.",
        event_time=event_time,
    )


# -----------------------------------------------------------------------
# S8 - Combined severe case
# -----------------------------------------------------------------------
def build_S8(pp: PlantParams, prof: ProfileParams, T_end: float = 50.0,
             t0: float = 15.0, t1: float = 25.0, event_time: float = 30.0) -> Scenario:
    def exo(t):
        base = pf.exogenous_with_drift(prof, t, ramp_kind="very_fast", amplitude=0.25, t_start=5.0, period=15.0)
        m = pf.ramp(t, t0, t1, 1.0, 2.0)
        base["Pload1"] *= m
        base["Pload2"] *= m
        base["Ppv"] *= max(0.0, 2.0 - m)
        return base

    def gain_fn(t):
        if 10.0 <= t <= 35.0:
            return (0.35, 1.0)
        return (1.0, 1.0)

    topo_pre = nominal_topology(pp, 3.0)
    topo_post = TopologyState(
        R_self=(pp.R_self[0], pp.R_self[1] * 8.0), X_self=(pp.X_self[0], pp.X_self[1] * 8.0),
        R_mutual=pp.R_mutual * 3.0 * 0.2, X_mutual=pp.X_mutual * 3.0 * 0.2,
        vsc_online=(True, False),
    )

    def topo_fn(t):
        return topo_pre if t < event_time else topo_post

    def gain_fn_combined(t):
        g1, g2 = gain_fn(t)
        if t >= event_time:
            g2 = 0.0
        return (g1, g2)

    return Scenario(
        name="S8", T_end=T_end, exogenous_fn=exo,
        actuator_gain_fn=gain_fn_combined, topology_fn=topo_fn,
        noise_cfg=dict(enabled=True, level_pct=2.0),
        description="Combined: high loading + weak authority + noise + fast drift + topology outage.",
        event_time=event_time,
    )


# -----------------------------------------------------------------------
# S10 - Randomized trial (statistical validation)
# -----------------------------------------------------------------------
def build_randomized_trial(rng: np.random.Generator, pp: PlantParams, prof: ProfileParams,
                            T_end: float = 40.0) -> Scenario:
    load_scale = rng.uniform(0.8, 1.3)
    noise_pct = rng.uniform(0.0, 1.5)
    weak_gain = rng.uniform(0.3, 1.0)
    weak_t0 = rng.uniform(8.0, 15.0)
    weak_dur = rng.uniform(5.0, 12.0)
    coupling_scale = rng.uniform(1.0, 4.0)
    ramp_kind = rng.choice(["flat", "normal", "fast"])
    topo_event_time = rng.uniform(20.0, 32.0)
    do_topology_event = rng.uniform() < 0.5

    prof_local = copy.deepcopy(prof)
    prof_local.load_scale = load_scale

    def exo(t):
        return pf.exogenous_with_drift(prof_local, t, ramp_kind=str(ramp_kind), amplitude=0.2, t_start=5.0, period=20.0)

    def gain_fn(t):
        if weak_t0 <= t <= weak_t0 + weak_dur:
            return (weak_gain, 1.0)
        return (1.0, 1.0)

    topo_pre = nominal_topology(pp, coupling_scale)
    if do_topology_event:
        topo_post = TopologyState(
            R_self=(pp.R_self[0], pp.R_self[1] * 5.0), X_self=(pp.X_self[0], pp.X_self[1] * 5.0),
            R_mutual=pp.R_mutual * coupling_scale * 0.3, X_mutual=pp.X_mutual * coupling_scale * 0.3,
            vsc_online=(True, True),
        )
        topo_fn = lambda t: topo_pre if t < topo_event_time else topo_post
    else:
        topo_fn = lambda t: topo_pre

    return Scenario(
        name="S10_trial", T_end=T_end, exogenous_fn=exo, actuator_gain_fn=gain_fn, topology_fn=topo_fn,
        noise_cfg=dict(enabled=(noise_pct > 0.0), level_pct=noise_pct),
        description="Randomized trial for statistical validation.",
        event_time=(topo_event_time if do_topology_event else None),
    )


ALL_S1_S8_BUILDERS = {
    "S1": build_S1, "S2": build_S2, "S3": build_S3, "S4": build_S4,
    "S6": build_S6, "S7": build_S7, "S8": build_S8,
}
