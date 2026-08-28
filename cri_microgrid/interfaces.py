"""Interface contracts for plugging your own controller and/or your own
plant into the CRI framework.

Nothing in this file is imported by the rest of `cri_microgrid` -- it is
pure documentation, expressed as `typing.Protocol`s so it is also
machine-checkable (a static type checker, or `isinstance(obj, Protocol)`
at runtime with `@runtime_checkable`, can confirm your object satisfies
it). Everything here is duck-typed: your class does not need to inherit
from anything, it just needs to provide these methods with these
signatures.

Two entry points, two levels of reuse:

1. Bring your own CANDIDATE CONTROLLER (most common case): keep the
   reference microgrid plant and fallback controller, swap in your own
   optimizer/learning policy/tuner as the `candidate` argument to
   `sim.run_closed_loop(..., candidate=your_controller)`. You get CRI
   supervision, safe probing, and every metric in `metrics.py` for free,
   on the exact same benchmark plant used throughout this study.

2. Bring your own PLANT (a different system entirely): implement
   `PlantProtocol`, `CandidateControllerProtocol`, `FallbackControllerProtocol`,
   and a `MarginFn`, then call `sim.run_closed_loop_generic(...)`. See
   notebooks/11_bring_your_own.ipynb for a complete worked example of both.
"""
from typing import Dict, List, Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class CandidateControllerProtocol(Protocol):
    """The adaptive/learning/optimization-driven controller the CRI
    supervises (pi_phi in the paper's notation)."""

    def candidate_command(self) -> np.ndarray:
        """Return pi_phi_k(y_k, r_k, D_k): the command implied by the
        controller's CURRENT internal state, before this step's update.
        Called once per sample, before propose_nominal_step()."""
        ...

    def propose_nominal_step(self, y_meas: np.ndarray) -> np.ndarray:
        """Return Delta_phi_k^nom (Eq. 37): the FULL (kappa=1) internal
        update your controller would apply given the just-measured output,
        expressed in the same units/space as candidate_command()'s return
        value (e.g. if candidate_command() returns a setpoint vector, this
        returns the proposed change to that setpoint vector). Does not
        mutate internal state -- shadow_update() does that."""
        ...

    def shadow_update(self, delta_nominal: np.ndarray, kappa_k: float) -> None:
        """Apply state += kappa_k * delta_nominal (Eq. 37). Called every
        step regardless of the activation status s_k -- this is the
        "shadow learning" that keeps your controller warmed up while
        de-activated. If your controller does not want CRI-scaled shadow
        learning, implement this as `state += delta_nominal` (ignore
        kappa_k) or as a no-op; both are valid, just document which."""
        ...

    def reset(self, u0: np.ndarray = None) -> None:
        """Reset internal state, optionally to a given initial command."""
        ...


@runtime_checkable
class FallbackControllerProtocol(Protocol):
    """The conservative, always-trusted controller used as u_safe and as
    the sole controller whenever s_k=0. Should require no tuning
    coordination with the candidate and should be simple enough that you
    would trust it unsupervised."""

    def command(self, y_meas: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class PlantProtocol(Protocol):
    """Your simulated system. May know everything about itself internally
    -- the CRI supervisor never receives this object, only what sim.py
    explicitly extracts from its return values."""

    def measure(self, exo: Dict, noise_cfg: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Return (y_measured, hidden_truth) from the CURRENT state
        WITHOUT integrating dynamics -- used once, to seed the closed loop
        at t=0. hidden_truth may contain anything (true topology, true
        actuator effectiveness, true noise level, ...): it is recorded for
        offline validation.py use only and must never be read by cri.py or
        controllers.py."""
        ...

    def step(self, u: np.ndarray, t: float, exo: Dict, actuator_gain, topo,
             noise_cfg: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Advance one sample given the applied command `u`, current time
        `t`, and the scenario's exogenous_fn/actuator_gain_fn/topology_fn
        outputs (exo, actuator_gain, topo -- opaque to sim.py, interpreted
        however your plant likes). Returns (y_measured, hidden_truth), same
        contract as measure()."""
        ...


# MarginFn: Callable[[np.ndarray, np.ndarray, Any], List[float]]
#   margin_fn(y_meas, u_applied, cons) -> [psi_1, ..., psi_ng]
#   One normalized margin per operational constraint (Eq. 22-23): each
#   psi_ell should be near 1 when comfortably inside the limit and near or
#   below 0 at/beyond it. SMI = min_ell sat_[0,1](psi_ell) is taken by
#   cri.compute_SMI() -- you only need to supply the list of margins.
#   `cons` is opaque: pass whatever your margin function needs (a dataclass,
#   a dict, a YAML-loaded config -- cri.py and sim.py never inspect it).

REQUIRED_METHODS = dict(
    candidate=["candidate_command", "propose_nominal_step", "shadow_update", "reset"],
    fallback=["command"],
    plant=["measure", "step"],
)


def check_interfaces(candidate=None, fallback=None, plant=None) -> None:
    """Quick pre-flight check: raises AttributeError with a clear message
    naming exactly what is missing, instead of failing deep inside a
    50-step simulation loop."""
    objs = dict(candidate=candidate, fallback=fallback, plant=plant)
    for name, obj in objs.items():
        if obj is None:
            continue
        for method in REQUIRED_METHODS[name]:
            if not hasattr(obj, method):
                raise AttributeError(
                    f"{name} object of type {type(obj).__name__} is missing required "
                    f"method '{method}()' -- see cri_microgrid/interfaces.py "
                    f"({name.capitalize()}{'ControllerProtocol' if name != 'plant' else 'Protocol'})."
                )
    print("Interface check passed for:", ", ".join(k for k, v in objs.items() if v is not None))
