"""Controllers: the always-safe fallback droop/PI law, and the candidate
OFO (online feedback optimization) VSC setpoint coordinator supervised by
the CRI layer.

Both the direct-activation and CRI-supervised strategies construct the OFO
controller from the SAME ControllerParams instance, so any performance
difference between strategies in the notebooks is attributable only to the
supervisory layer, never to a differently tuned candidate controller.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .config import ControllerParams, ConstraintParams, PlantParams


def _clip_box(u: np.ndarray, cons: ConstraintParams) -> np.ndarray:
    lo = np.array([cons.P_min, cons.Q_min, cons.P_min, cons.Q_min])
    hi = np.array([cons.P_max, cons.Q_max, cons.P_max, cons.Q_max])
    return np.clip(u, lo, hi)


class FallbackDroopController:
    """Conventional decentralized droop/PI fallback: Vdc-P and V-Q droop per
    VSC, purely proportional (no coordination, no shared state), used as
    u_safe by every strategy and as the sole controller when s_k=0."""

    def __init__(self, cp: ControllerParams, cons: ConstraintParams, plant_nom: PlantParams):
        self.cp = cp
        self.cons = cons
        self.Vdc_ref = plant_nom.Vdc_nom
        self.Vac_ref = plant_nom.Vac_nom

    def command(self, y_meas: np.ndarray) -> np.ndarray:
        V1, V2, Vdc = y_meas[0], y_meas[1], y_meas[2]
        # P1,P2 are DC-bus DISCHARGE power (DC -> AC): if Vdc sags below
        # its reference the bus is short on energy, so exported power must
        # be reduced (not increased) to let it recover -- dP < 0 when
        # Vdc < Vdc_ref.
        dP = self.cp.Kp_droop_P * (Vdc - self.Vdc_ref)
        P1 = self.cp.P_nom[0] + dP
        P2 = self.cp.P_nom[1] + dP
        Q1 = self.cp.Q_nom[0] + self.cp.Kq_droop_Q * (self.Vac_ref - V1)
        Q2 = self.cp.Q_nom[1] + self.cp.Kq_droop_Q * (self.Vac_ref - V2)
        u = np.array([P1, Q1, P2, Q2], dtype=float)
        return _clip_box(u, self.cons)


class OFOVSCController:
    """Candidate controller: a projected-gradient online feedback
    optimizer over VSC setpoints u=[P1*,Q1*,P2*,Q2*], minimizing AC/DC
    voltage deviation plus an active-power-sharing regularizer.

    Uses a FIXED, approximate internal sensitivity model (built once from
    the nominal plant parameters) -- a standard OFO ingredient. This model
    is intentionally never updated online and is completely independent of
    the CRI module's own measurement-driven S_hat; a topology change makes
    this internal model stale, which is precisely the situation the CRI
    layer is meant to catch from measurements alone.
    """

    def __init__(self, cp: ControllerParams, cons: ConstraintParams, plant_nom: PlantParams):
        self.cp = cp
        self.cons = cons
        self.Vdc_ref = plant_nom.Vdc_nom
        self.Vac_ref = plant_nom.Vac_nom
        self.y_target = np.array([self.Vac_ref, self.Vac_ref, self.Vdc_ref, 0.0, 0.0, 0.0, 0.0])
        self.w_y = np.array([cp.w_v, cp.w_v, cp.w_dc, 0.0, 0.0, 0.0, 0.0])

        R11, R22 = plant_nom.R_self
        X11, X22 = plant_nom.X_self
        Rm, Xm = plant_nom.R_mutual, plant_nom.X_mutual
        dvdc_dP = -plant_nom.Ts / (plant_nom.eta_conv * plant_nom.Cdc * plant_nom.Vdc_nom)
        # Rows: Vac1, Vac2, Vdc, I1, I2, P_POI, Q_POI ; Cols: P1,Q1,P2,Q2
        self.J_model = np.array([
            [R11, X11, Rm, Xm],
            [Rm, Xm, R22, X22],
            [dvdc_dP, 0.0, dvdc_dP, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        self.u = np.array([cp.P_nom[0], cp.Q_nom[0], cp.P_nom[1], cp.Q_nom[1]], dtype=float)

    def reset(self, u0: Optional[np.ndarray] = None):
        if u0 is not None:
            self.u = np.array(u0, dtype=float)
        else:
            self.u = np.array([self.cp.P_nom[0], self.cp.Q_nom[0], self.cp.P_nom[1], self.cp.Q_nom[1]])

    def candidate_command(self) -> np.ndarray:
        """pi_phi_k(y_k, r_k, D_k): the candidate command implied by the
        controller's CURRENT internal state (before this step's update)."""
        return self.u.copy()

    def propose_nominal_step(self, y_meas: np.ndarray) -> np.ndarray:
        """The full (kappa=1) update the OFO would apply, Delta_phi_k^nom
        in the paper's notation -- a projected-gradient step on the
        tracking + sharing objective."""
        e_y = y_meas - self.y_target
        grad_u_track = self.J_model.T @ (self.w_y * e_y)

        share_err = self.u[0] - self.u[2]   # P1* - P2*
        grad_u_share = self.cp.w_share * np.array([share_err, 0.0, -share_err, 0.0])

        du = -self.cp.eta_ofo * (grad_u_track + grad_u_share)
        du = np.clip(du, -self.cp.ofo_du_max, self.cp.ofo_du_max)

        u_next = _clip_box(self.u + du, self.cons)
        return u_next - self.u

    def shadow_update(self, delta_u_nominal: np.ndarray, kappa_k: float):
        """Apply the CRI-scaled internal parameter update, eq. (32):
        phi_{k+1} = phi_k + kappa_k * Delta_phi_k_nom. Runs every step,
        independent of s_k, so the candidate stays warmed up while
        de-activated."""
        self.u = _clip_box(self.u + kappa_k * delta_u_nominal, self.cons)


def blend_command(u_safe: np.ndarray, u_candidate: np.ndarray, s_k: int, kappa_k: float) -> np.ndarray:
    """Eq. (38): u_k = u_safe + s_k * kappa_k * (pi_phi_k - u_safe)."""
    return u_safe + float(s_k) * float(kappa_k) * (u_candidate - u_safe)
