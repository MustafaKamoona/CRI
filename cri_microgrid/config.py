"""Central configuration for the CRI hybrid AC/DC microgrid study.

Every tunable number used anywhere in the simulation lives in one of the
dataclasses below. Nothing in plant.py, controllers.py, cri.py, sensitivity.py,
probing.py, sim.py, or the notebooks hardcodes a parameter value directly --
they all read it from one of these dataclasses, so a single edit here is
enough to re-run the whole study under a different setting.

All electrical quantities are in per-unit (pu): Vac and Vdc are normalized to
their nominal magnitude (1.0 pu), P/Q setpoints are normalized to a VSC's
rated apparent power, and feeder currents are normalized to the feeder's
rated current. This keeps every figure and every CRI computation
dimensionless, matching the D_u^-1, D_y^-1 normalization used throughout the
paper's readiness formulas.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List, Dict, Any
import numpy as np


# =============================================================================
# Plant (two coordinated VSCs on a shared DC link, each feeding an AC bus)
# =============================================================================
@dataclass
class PlantParams:
    n_vsc: int = 2

    Ts: float = 0.02                 # supervisory sample time (s) -> 50 Hz

    Vac_nom: float = 1.0             # pu
    Vdc_nom: float = 1.0             # pu

    # Self feeder impedance (VSC i terminal -> POI), pu, one entry per VSC.
    R_self: Tuple[float, float] = (0.04, 0.05)
    X_self: Tuple[float, float] = (0.12, 0.14)

    # Mutual coupling impedance (shared line/transformer segment before the
    # feeders split): nonzero values let VSC j's power injection move VSC
    # i's bus voltage, which is exactly the mechanism the Coupling Index is
    # designed to detect. Nominal value is intentionally small (weak
    # coupling); S4 scales this up via ScenarioParams.coupling_scale.
    R_mutual: float = 0.01
    X_mutual: float = 0.03

    # Time constants are set fast relative to Ts (supervisory sample time)
    # so that, from the readiness layer's perspective, one sample is close
    # to the paper's quasi-static assumption (Sec III-C): the plant has
    # essentially settled to a Delta u before the next Delta u is applied,
    # so the local Delta y ~= S*Delta u regression is a physically sound
    # local linearization rather than fighting first-order lag dynamics.
    tau_v: float = 0.05              # AC bus voltage transient time constant (s)
    tau_p: float = 0.012             # VSC active-power tracking time constant (s)
    tau_q: float = 0.012             # VSC reactive-power tracking time constant (s)

    Cdc: float = 0.5                 # DC-link capacitance (pu energy storage constant)
    eta_conv: float = 0.98           # converter efficiency (DC -> AC side power draw)

    # POI (point of interconnection) voltage is treated as a stiff reference
    # that can itself sag slightly under heavy net loading.
    Vpoi_nom: float = 1.0

    S_rated_per_vsc: float = 1.0     # pu apparent-power rating used for saturation


@dataclass
class ControllerParams:
    # --- Fallback conventional droop/PI controller (u_safe) ---
    Kp_droop_P: float = 2.5          # P_i* correction per pu Vdc error
    Kq_droop_Q: float = 3.0          # Q_i* correction per pu bus-voltage error
    # Nominal per-VSC active-power dispatch: sized so P_nom[0]+P_nom[1] sits
    # close to the DC-side power available under ProfileParams' defaults
    # (P_pv_base - Pdcload_base)*eta_conv, so the DC link starts near
    # balance rather than perpetually charging/discharging.
    P_nom: Tuple[float, float] = (0.29, 0.29)
    Q_nom: Tuple[float, float] = (0.0, 0.0)

    # --- Candidate OFO (online feedback optimization) VSC setpoint coordinator ---
    # OFO uses its own fixed, approximate internal sensitivity model (a
    # standard OFO ingredient -- see e.g. Bolognani et al.). This is
    # deliberately distinct from, and never shared with, the CRI module's
    # measurement-driven S_hat: the candidate controller is allowed to know
    # an approximate model, the CRI supervisor is not.
    eta_ofo: float = 0.35            # OFO projected-gradient step size
    w_v: float = 1.0                 # AC voltage tracking weight
    w_dc: float = 1.0                # DC-link voltage tracking weight
    w_share: float = 0.3             # active-power-sharing (mismatch) weight
    w_du: float = 0.05               # step-size regularization weight
    ofo_du_max: float = 0.08         # max per-step |Delta u| (pu), independent of kappa

    # Direct-activation and CRI-supervised cases must use identical candidate
    # controller parameters -- enforced by both strategies constructing the
    # OFO controller from this same dataclass.


@dataclass
class ConstraintParams:
    Vac_min: float = 0.95
    Vac_max: float = 1.05
    I_max: float = 1.0                        # per-feeder rated current, pu
    Vdc_min: float = 0.92
    Vdc_max: float = 1.08
    # Asymmetric P box (converters export far more often than they import):
    # centered close to the nominal per-VSC dispatch point so the actuator
    # headroom margin (which, by the paper's own min(headroom-fraction)
    # formula, structurally tops out at 0.5 exactly at the box center)
    # actually reaches close to that ceiling under healthy nominal
    # operation, rather than being permanently starved by an off-center box.
    P_min: float = -0.4
    P_max: float = 1.0
    Q_min: float = -0.6
    Q_max: float = 0.6


@dataclass
class ProbeParams:
    """Safe micro-probing signal generator settings."""
    period_steps: int = 8              # samples between probe sign flips (avoids a period-4 alias)
    channel_scale: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # D_p diagonal
    # The droop+OFO commands move both VSCs almost in lockstep under normal
    # operation (a near-rank-deficient natural excitation that cannot
    # separate individual-channel effects), so this threshold is set high
    # enough that probing stays on through ordinary operation and only
    # backs off once a genuinely large natural transient is already
    # exciting the system on its own (e.g. during S2/S7/S8 events).
    min_natural_excitation: float = 0.05


@dataclass
class ProfileParams:
    """Load / PV / wind exogenous-input profile shaping."""
    load_scale: float = 1.0
    pv_scale: float = 1.0
    P_load_base: Tuple[float, float] = (0.30, 0.30)   # local AC-bus load per bus, pu (met via P_POI exchange)
    Q_load_base: Tuple[float, float] = (0.10, 0.10)
    # DC-side renewable infeed and DC-side load: sized so that, combined
    # with eta_conv, (P_pv_base - Pdcload_base)*eta_conv is close to
    # ControllerParams.P_nom[0]+P_nom[1] -- the DC link's only two power
    # paths are this infeed/load pair and the two VSCs' AC-side draw, so a
    # mismatch here means Vdc drifts to its saturation floor/ceiling even
    # under "nominal" conditions, regardless of the controller.
    P_pv_base: float = 0.65
    Pdcload_base: float = 0.05
    ramp_kind: str = "flat"           # flat | normal | fast | very_fast
    ramp_amplitude: float = 0.25      # pu swing used by ramp profiles


@dataclass
class NoiseParams:
    """Zero-mean Gaussian measurement noise, as a fraction of full scale per
    channel. Only the measured signal fed to the controller/CRI is
    corrupted; plant state integration always uses the true state."""
    enabled: bool = False
    level_pct: float = 0.0            # percent of rated value, e.g. 1.0 = 1%


@dataclass
class ScenarioParams:
    """Generic, flexible scenario descriptor. scenarios.py builds S1-S10 by
    filling in the relevant subset of these fields; unused fields keep their
    nominal (no-effect) default."""
    name: str = "S1"
    T_end: float = 40.0
    seed: int = 0

    load_scale: float = 1.0
    ramp_kind: str = "flat"
    ramp_amplitude: float = 0.25

    noise_level_pct: float = 0.0

    # Weak actuator authority: static per-VSC effectiveness multiplier
    # applied to the commanded P/Q before it reaches the plant dynamics
    # (models saturation / lost gain, invisible directly in the voltage
    # constraints).
    actuator_gain: Tuple[float, float] = (1.0, 1.0)
    actuator_gain_window: Optional[Tuple[float, float]] = None   # (t_start, t_end)

    coupling_scale: float = 1.0

    high_loading: bool = False        # pushes toward the voltage/current limits

    topology_event_time: Optional[float] = None
    topology_event_kind: Optional[str] = None   # 'vsc2_outage' | 'impedance_increase'

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CRIParams:
    # --- Data window / sensitivity estimation ---
    N: int = 40                       # moving-window length (samples)
    Na: int = 20                      # adaptability moving-average window (samples)
    lambda_s: float = 1.0e-3          # sensitivity regularization

    # --- Indicator normalization constants ---
    r0: float = 0.06                  # responsiveness normalization (rho_0)
    a0: float = 0.12                  # adaptability normalization (max acceptable drift)
    # Calibrated so that the healthy nominal baseline -- which, with
    # continuous safe probing running, always carries some irreducible
    # high-frequency content from the probe itself -- reads as high
    # confidence, while still responding clearly once real measurement
    # noise (S6) is added on top.
    sigma_y0_sq: float = 3.0e-3       # max acceptable measurement-noise variance (normalized)
    # Sample-to-sample differencing (Delta u, Delta y) amplifies noise
    # relative to the (naturally smooth) true signal, so eps_s responds
    # steeply once measurement noise is present. Calibrated jointly against
    # S3 (weak actuator authority, where a degraded eps_s is the main
    # rejection mechanism) and S6 (measurement noise): eps_0s=0.5 is the
    # largest value that still gives S3 a clean, near-complete rejection of
    # the unsafe window -- above it, NCI stops being sensitive enough to
    # eps_s and S3's false-activation rate jumps sharply.
    epsilon_0s: float = 0.50          # acceptable sensitivity-residual level

    # --- CRI combination weights (safety-oriented, converter-dominated microgrid) ---
    omega_R: float = 0.20
    omega_C: float = 0.10
    omega_S: float = 0.30
    omega_A: float = 0.15
    omega_N: float = 0.25

    # --- Decision thresholds ---
    # Calibrated against this plant's own achievable nominal ceiling: the
    # paper's own actuator-margin formula (Sec. IV-B) structurally caps SMI
    # at 0.5 even at the best possible operating point (it is a min of two
    # headroom fractions that always sum to 1), and the shared-POI terms in
    # the microgrid output vector give the Coupling Index an honest
    # baseline around 0.55-0.6 even under weak physical coupling -- so a
    # healthy nominal CRI here settles near 0.70, not 1.0. Thresholds are
    # set with headroom below that ceiling rather than against an
    # unreachable value.
    eta_A: float = 0.64               # activate
    eta_M: float = 0.565              # monitor
    eta_D: float = 0.40               # delay / reject boundary
    eta_S: float = 0.30               # DT escalation: safety-margin threshold
    eta_N: float = 0.35               # DT escalation: noise-confidence threshold

    # --- Dynamic (recursive) CRI filter ---
    beta_min: float = 0.05
    beta_max: float = 0.50
    kappa_beta: float = 2.0

    # --- Adaptation scaling ---
    kappa_min: float = 0.05

    # --- Safe micro-probing ---
    delta_max: float = 0.05           # max probing amplitude (pu)

    # --- Numerical safety ---
    epsilon_regularization: float = 1.0e-6

    def weights(self) -> np.ndarray:
        return np.array([self.omega_R, self.omega_C, self.omega_S, self.omega_A, self.omega_N])

    def __post_init__(self):
        assert 0.0 <= self.eta_D < self.eta_M < self.eta_A <= 1.0, \
            "CRI thresholds must satisfy 0 <= eta_D < eta_M < eta_A <= 1"
        w = self.weights()
        assert np.all(w >= 0.0), "CRI weights must be nonnegative"
        assert abs(w.sum() - 1.0) < 1e-9, "CRI weights must sum to 1"


@dataclass
class SimulationParams:
    Ts: float = 0.02
    T_end: float = 40.0
    seed: int = 0
    strategy: str = "cri"             # 'direct' | 'threshold' | 'cri'

    # Simple hard-limit threshold-based supervisor (the "conventional rule-
    # based supervisor" baseline compared against CRI): activates whenever
    # every instantaneous constraint is currently satisfied, with no
    # awareness of coupling, drift, or noise.
    threshold_margin: float = 0.0


DEFAULT_D_u = np.array([0.5, 0.3, 0.5, 0.3])   # representative scale of [P1*,Q1*,P2*,Q2*] increments
DEFAULT_D_y = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.6, 0.3])
# representative scale of [Vac1,Vac2,Vdc,I1,I2,P_POI,Q_POI] increments

Y_LABELS = ["Vac1", "Vac2", "Vdc", "I1", "I2", "P_POI", "Q_POI"]
U_LABELS = ["P1*", "Q1*", "P2*", "Q2*"]
