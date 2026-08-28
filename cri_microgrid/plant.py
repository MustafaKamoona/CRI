"""Two-VSC hybrid AC/DC microgrid plant.

Physics summary (per-unit, linearized-DistFlow-style averaged model):
  - Each VSC tracks its commanded (P*, Q*) setpoint through first-order
    dynamics, after an "actuator effectiveness" gain and an apparent-power
    saturation circle (models weak authority / saturated converters, S3).
  - Each AC bus voltage follows a linear voltage-drop law driven by the net
    injected active/reactive power at both buses (a mutual term lets one
    VSC's action move the other bus's voltage -- the coupling mechanism the
    Coupling Index is designed to detect, S4).
  - The shared DC link follows an energy-balance ODE from renewable infeed,
    a DC load, and the AC-side power drawn by both VSCs.
  - Feeder current is the apparent power over voltage, I_i = S_i / V_i.

Critical design rule for this paper's claims: the plant may know and expose
everything (topology, true actuator gain, true noise level, true drift) via
`hidden_truth`, but that dictionary is consumed ONLY by validation.py to
build simulation-only ground-truth labels. controllers.py and cri.py never
receive it -- they only see whatever sim.py explicitly passes them: applied
command, (possibly noisy) measured output, operational limits, and
measurement-quality metadata.
"""
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple
import numpy as np

from .config import PlantParams


@dataclass
class PlantState:
    P1: float
    Q1: float
    P2: float
    Q2: float
    V1: float
    V2: float
    Vdc: float


@dataclass
class TopologyState:
    """Time-varying feeder/coupling parameters, supplied externally by
    scenarios.py/sim.py at each step (topology is an input to the plant, not
    something the plant decides on its own)."""
    R_self: Tuple[float, float]
    X_self: Tuple[float, float]
    R_mutual: float
    X_mutual: float
    vsc_online: Tuple[bool, bool] = (True, True)


def nominal_topology(p: PlantParams, coupling_scale: float = 1.0) -> TopologyState:
    return TopologyState(
        R_self=tuple(p.R_self),
        X_self=tuple(p.X_self),
        R_mutual=p.R_mutual * coupling_scale,
        X_mutual=p.X_mutual * coupling_scale,
        vsc_online=(True, True),
    )


def _apparent_power_saturate(P: float, Q: float, S_max: float) -> Tuple[float, float]:
    S = np.hypot(P, Q)
    if S <= S_max or S < 1e-12:
        return P, Q
    scale = S_max / S
    return P * scale, Q * scale


class HybridACDCPlant:
    """Discrete-time simulator. `step()` is the ONLY entry point used by
    sim.py; it returns (y_measured, hidden_truth)."""

    def __init__(self, params: PlantParams, noise_rng: Optional[np.random.Generator] = None):
        self.p = params
        self.rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
        self.state = PlantState(
            P1=params.S_rated_per_vsc * 0.30, Q1=0.0,
            P2=params.S_rated_per_vsc * 0.30, Q2=0.0,
            V1=params.Vac_nom, V2=params.Vac_nom, Vdc=params.Vdc_nom,
        )

    def reset(self, state: Optional[PlantState] = None):
        if state is not None:
            self.state = state
        else:
            p = self.p
            self.state = PlantState(
                P1=p.S_rated_per_vsc * 0.30, Q1=0.0,
                P2=p.S_rated_per_vsc * 0.30, Q2=0.0,
                V1=p.Vac_nom, V2=p.Vac_nom, Vdc=p.Vdc_nom,
            )

    def _voltage_targets(self, x: PlantState, exo: Dict[str, float], topo: TopologyState) -> Tuple[float, float]:
        Pnet1 = x.P1 - exo["Pload1"]
        Qnet1 = x.Q1 - exo["Qload1"]
        Pnet2 = x.P2 - exo["Pload2"]
        Qnet2 = x.Q2 - exo["Qload2"]

        R11, R22 = topo.R_self
        X11, X22 = topo.X_self
        R12 = X12 = 0.0  # mutual coupling enters symmetrically via R_mutual/X_mutual below
        Rm, Xm = topo.R_mutual, topo.X_mutual

        V1_target = exo["Vpoi"] + (R11 * Pnet1 + X11 * Qnet1) + (Rm * Pnet2 + Xm * Qnet2)
        V2_target = exo["Vpoi"] + (R22 * Pnet2 + X22 * Qnet2) + (Rm * Pnet1 + Xm * Qnet1)
        return V1_target, V2_target

    def measure(self, exo: Dict[str, float], noise_cfg: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Compute the output vector from the CURRENT state without
        integrating dynamics -- used only to obtain the very first
        measurement at t=0 so the closed loop has something to react to."""
        x = self.state
        p = self.p
        I1 = np.hypot(x.P1, x.Q1) / max(x.V1, 0.2)
        I2 = np.hypot(x.P2, x.Q2) / max(x.V2, 0.2)
        P_POI = (x.P1 - exo["Pload1"]) + (x.P2 - exo["Pload2"])
        Q_POI = (x.Q1 - exo["Qload1"]) + (x.Q2 - exo["Qload2"])
        y_true = np.array([x.V1, x.V2, x.Vdc, I1, I2, P_POI, Q_POI], dtype=float)

        y_meas = y_true.copy()
        noise_std = np.zeros(7)
        if noise_cfg is not None and noise_cfg.get("enabled", False):
            lvl = noise_cfg["level_pct"] / 100.0
            rated = np.array([p.Vac_nom, p.Vac_nom, p.Vdc_nom, p.S_rated_per_vsc, p.S_rated_per_vsc, 1.0, 1.0])
            noise_std = lvl * rated
            y_meas = y_true + self.rng.normal(0.0, noise_std)

        hidden_truth = dict(
            y_true=y_true, P1=x.P1, Q1=x.Q1, P2=x.P2, Q2=x.Q2,
            V1=x.V1, V2=x.V2, Vdc=x.Vdc, I1=I1, I2=I2,
            actuator_gain=(1.0, 1.0), topology=None, noise_std=noise_std,
        )
        return y_meas, hidden_truth

    def step(self, u: np.ndarray, t: float, exo: Dict[str, float],
              actuator_gain: Tuple[float, float], topo: TopologyState,
              noise_cfg: Optional[Dict] = None) -> Tuple[Dict[str, float], Dict]:
        """Advance the plant by one sample.

        u             : commanded [P1*, Q1*, P2*, Q2*] (pu)
        exo           : dict with Pload1,Qload1,Pload2,Qload2,Ppv,Pdcload,Vpoi
        actuator_gain : (g1, g2) true effectiveness multipliers, in [0, 1]
        topo          : current TopologyState (may reflect an outage/coupling change)
        noise_cfg     : dict(enabled, level_pct) for the returned measurement only

        Returns (y_measured, hidden_truth).
        """
        p = self.p
        Ts = p.Ts
        x = self.state
        g1, g2 = actuator_gain
        online1, online2 = topo.vsc_online

        P1_cmd = g1 * u[0] * float(online1)
        Q1_cmd = g1 * u[1] * float(online1)
        P2_cmd = g2 * u[2] * float(online2)
        Q2_cmd = g2 * u[3] * float(online2)

        P1_cmd, Q1_cmd = _apparent_power_saturate(P1_cmd, Q1_cmd, p.S_rated_per_vsc)
        P2_cmd, Q2_cmd = _apparent_power_saturate(P2_cmd, Q2_cmd, p.S_rated_per_vsc)

        # Exact (zero-order-hold) discretization of each first-order lag,
        # x_{k+1} = x_k + (1-exp(-Ts/tau))*(target-x_k): unlike forward
        # Euler, this is unconditionally stable for any Ts/tau ratio, which
        # matters here because tau_p/tau_q/tau_v are deliberately set fast
        # relative to Ts (see PlantParams) to keep the supervisory sample
        # close to the paper's quasi-static regime -- forward Euler would
        # diverge outright once Ts/tau > 1.
        ap = 1.0 - np.exp(-Ts / p.tau_p)
        aq = 1.0 - np.exp(-Ts / p.tau_q)
        av = 1.0 - np.exp(-Ts / p.tau_v)

        P1n = x.P1 + ap * (P1_cmd - x.P1)
        Q1n = x.Q1 + aq * (Q1_cmd - x.Q1)
        P2n = x.P2 + ap * (P2_cmd - x.P2)
        Q2n = x.Q2 + aq * (Q2_cmd - x.Q2)

        V1_target, V2_target = self._voltage_targets(x, exo, topo)
        V1n = x.V1 + av * (V1_target - x.V1)
        V2n = x.V2 + av * (V2_target - x.V2)
        V1n = float(np.clip(V1n, 0.5, 1.5))
        V2n = float(np.clip(V2n, 0.5, 1.5))

        P_dc_out = (P1n + P2n) / p.eta_conv
        P_balance = exo["Ppv"] - exo["Pdcload"] - P_dc_out
        E = 0.5 * p.Cdc * x.Vdc * x.Vdc
        E_next = max(1e-9, E + P_balance * Ts)
        Vdc_n = float(np.sqrt(2.0 * E_next / p.Cdc))
        Vdc_n = float(np.clip(Vdc_n, 0.5, 1.5))

        self.state = PlantState(P1=P1n, Q1=Q1n, P2=P2n, Q2=Q2n, V1=V1n, V2=V2n, Vdc=Vdc_n)

        I1 = np.hypot(P1n, Q1n) / max(V1n, 0.2)
        I2 = np.hypot(P2n, Q2n) / max(V2n, 0.2)
        P_POI = (P1n - exo["Pload1"]) + (P2n - exo["Pload2"])
        Q_POI = (Q1n - exo["Qload1"]) + (Q2n - exo["Qload2"])

        y_true = np.array([V1n, V2n, Vdc_n, I1, I2, P_POI, Q_POI], dtype=float)

        noise_std = np.zeros(7)
        y_meas = y_true.copy()
        if noise_cfg is not None and noise_cfg.get("enabled", False):
            lvl = noise_cfg["level_pct"] / 100.0
            rated = np.array([p.Vac_nom, p.Vac_nom, p.Vdc_nom, p.S_rated_per_vsc, p.S_rated_per_vsc, 1.0, 1.0])
            noise_std = lvl * rated
            y_meas = y_true + self.rng.normal(0.0, noise_std)

        hidden_truth = dict(
            y_true=y_true,
            P1=P1n, Q1=Q1n, P2=P2n, Q2=Q2n,
            V1=V1n, V2=V2n, Vdc=Vdc_n, I1=I1, I2=I2,
            actuator_gain=actuator_gain,
            topology=topo,
            noise_std=noise_std,
        )
        return y_meas, hidden_truth
