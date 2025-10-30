import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Optional


# ===============================
# Dynamics: Kinematic Bicycle
# ===============================
@dataclass
class BicycleParams:
    wheelbase: float = 2.7  # meters
    dt: float = 0.05        # seconds
    max_steer: float = math.radians(35.0)


def bicycle_dynamics(x: np.ndarray, u: np.ndarray, params: BicycleParams) -> np.ndarray:
    """
    Discrete-time kinematic bicycle dynamics (Euler forward).
    State x = [px, py, yaw, v, delta]
    Control u = [a, ddelta]
    """
    px, py, yaw, v, delta = x
    a, ddelta = u
    L = params.wheelbase
    dt = params.dt

    # Update
    px_next = px + v * math.cos(yaw) * dt
    py_next = py + v * math.sin(yaw) * dt
    yaw_next = yaw + (v / L) * math.tan(delta) * dt
    v_next = v + a * dt
    delta_next = delta + ddelta * dt

    # Optional smooth saturation of steering for forward rollout (keeps demo realistic)
    if params.max_steer is not None:
        delta_next = np.clip(delta_next, -params.max_steer, params.max_steer)

    return np.array([px_next, py_next, yaw_next, v_next, delta_next])


def bicycle_linearize(x: np.ndarray, u: np.ndarray, params: BicycleParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytic Jacobians A = d f / d x, B = d f / d u for the kinematic bicycle.
    """
    px, py, yaw, v, delta = x
    a, ddelta = u
    L = params.wheelbase
    dt = params.dt

    # Precompute trig
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    tdelta = math.tan(delta)
    sec2 = 1.0 / (math.cos(delta) ** 2)

    # A matrix (5x5)
    A = np.eye(5)
    A[0, 2] = -v * sy * dt        # d(px_next)/d(yaw)
    A[0, 3] = cy * dt             # d(px_next)/d(v)

    A[1, 2] = v * cy * dt         # d(py_next)/d(yaw)
    A[1, 3] = sy * dt             # d(py_next)/d(v)

    A[2, 3] = (tdelta / L) * dt   # d(yaw_next)/d(v)
    A[2, 4] = (v / L) * sec2 * dt # d(yaw_next)/d(delta)

    A[3, 3] = 1.0                 # v_next depends linearly on v via +0 and a via B
    A[4, 4] = 1.0

    # B matrix (5x2)
    B = np.zeros((5, 2))
    # d(px_next)/d a, ddelta = 0, 0 (since px_next uses current v)
    # d(py_next)/d a, ddelta = 0, 0
    B[2, 0] = 0.0                 # yaw_next w.r.t a is 0
    B[2, 1] = 0.0                 # yaw_next w.r.t ddelta is 0 (delta rate only influences delta_next)
    B[3, 0] = dt                  # v_next w.r.t a
    B[4, 1] = dt                  # delta_next w.r.t ddelta

    return A, B


# ===============================
# Quadratic Tracking Cost
# ===============================
@dataclass
class CostWeights:
    Q: np.ndarray      # state tracking weight (5x5)
    R: np.ndarray      # control effort weight (2x2)
    Rd: np.ndarray     # control rate weight (2x2)
    Qf: np.ndarray     # terminal state weight (5x5)


def stage_cost_quadratic(x: np.ndarray, u: np.ndarray, x_ref: np.ndarray, u_prev: np.ndarray, w: CostWeights) -> float:
    dx = x - x_ref
    du = u
    du_rate = u - u_prev
    return float(dx.T @ w.Q @ dx + du.T @ w.R @ du + du_rate.T @ w.Rd @ du_rate)


def terminal_cost_quadratic(x: np.ndarray, x_ref: np.ndarray, w: CostWeights) -> float:
    dx = x - x_ref
    return float(dx.T @ w.Qf @ dx)


# ===============================
# iLQR Optimizer
# ===============================
@dataclass
class ILQRConfig:
    max_iter: int = 50
    alpha_list: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05)
    reg_min: float = 1e-6
    reg_max: float = 1e6
    reg_init: float = 1e-3
    reg_scale_up: float = 10.0
    reg_scale_down: float = 0.3
    tol_cost: float = 1e-6


@dataclass
class ILQRResult:
    u_seq: np.ndarray
    K_seq: np.ndarray
    x_seq: np.ndarray
    cost: float
    iters: int
    converged: bool


def rollout(x0: np.ndarray,
            u_seq: np.ndarray,
            f: Callable[[np.ndarray, np.ndarray], np.ndarray],
            cost_stage: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
            cost_terminal: Callable[[np.ndarray, np.ndarray], float],
            x_refs: np.ndarray,
            w: CostWeights) -> Tuple[np.ndarray, float]:
    """
    Rollout dynamics and compute total cost. x_refs has shape (N+1, n).
    """
    N = u_seq.shape[0]
    n = x0.shape[0]

    x_seq = np.zeros((N + 1, n))
    x_seq[0] = x0.copy()
    total_cost = 0.0

    u_prev = np.zeros_like(u_seq[0])
    for t in range(N):
        x_t = x_seq[t]
        u_t = u_seq[t]
        total_cost += cost_stage(x_t, u_t, x_refs[t], u_prev)
        x_seq[t + 1] = f(x_t, u_t)
        u_prev = u_t

    total_cost += cost_terminal(x_seq[N], x_refs[N])
    return x_seq, total_cost


def ilqr(x0: np.ndarray,
         u_init: np.ndarray,
         f: Callable[[np.ndarray, np.ndarray], np.ndarray],
         linearize: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
         w: CostWeights,
         x_refs: np.ndarray,
         cfg: ILQRConfig = ILQRConfig()) -> ILQRResult:
    """
    iLQR core solving a finite-horizon problem around nominal (x,u) with quadratic tracking cost.
    x_refs: shape (N+1, n) terminal ref at index N.
    """
    N, m = u_init.shape
    n = x0.shape[0]

    # Cost derivatives for quadratic cost (constant):
    Q = 2.0 * w.Q
    R = 2.0 * w.R
    Rd = 2.0 * w.Rd
    Qf = 2.0 * w.Qf

    def l_x(x, t):
        return Q @ (x - x_refs[t])

    def l_u(u, u_prev):
        return (R + Rd) @ u - Rd @ u_prev

    l_xx = Q
    l_uu = R + Rd
    l_ux = np.zeros((m, n))

    # Initialize nominal trajectories
    u_bar = u_init.copy()
    x_bar = np.zeros((N + 1, n))

    # Utility closures for dynamics and costs for rollout
    u_prev_dummy = np.zeros(m)

    def stage_cost_for_rollout(x, u, x_ref, u_prev):
        # reuse stage cost with given weights
        dx = x - x_ref
        du = u
        du_rate = u - u_prev
        return float(dx.T @ w.Q @ dx + du.T @ w.R @ du + du_rate.T @ w.Rd @ du_rate)

    def terminal_cost_for_rollout(x, x_ref):
        dx = x - x_ref
        return float(dx.T @ w.Qf @ dx)

    # initial rollout
    def f_wrap(x, u):
        return f(x, u)

    x_bar[0] = x0.copy()
    for t in range(N):
        x_bar[t + 1] = f_wrap(x_bar[t], u_bar[t])

    best_cost = (
        sum(stage_cost_for_rollout(x_bar[t], u_bar[t], x_refs[t], u_prev_dummy if t == 0 else u_bar[t-1]) for t in range(N))
        + terminal_cost_for_rollout(x_bar[N], x_refs[N])
    )

    reg = cfg.reg_init
    converged = False

    for it in range(cfg.max_iter):
        # Linearize dynamics along nominal
        A = np.zeros((N, n, n))
        B = np.zeros((N, n, m))
        for t in range(N):
            A[t], B[t] = linearize(x_bar[t], u_bar[t])

        # Backward pass
        V_x = Qf @ (x_bar[N] - x_refs[N])
        V_xx = Qf.copy()

        K_seq = np.zeros((N, m, n))
        k_seq = np.zeros((N, m))

        diverged = False
        for t in reversed(range(N)):
            u_prev_t = u_bar[t-1] if t > 0 else np.zeros(m)

            Q_x = l_x(x_bar[t], t) + A[t].T @ V_x
            Q_u = l_u(u_bar[t], u_prev_t) + B[t].T @ V_x
            Q_xx = l_xx + A[t].T @ V_xx @ A[t]
            Q_ux = l_ux + B[t].T @ V_xx @ A[t]
            Q_uu = l_uu + B[t].T @ V_xx @ B[t]

            # Regularize Q_uu
            Q_uu_reg = Q_uu + reg * np.eye(m)

            # Solve for gains
            try:
                K_t = -np.linalg.solve(Q_uu_reg, Q_ux)
                k_t = -np.linalg.solve(Q_uu_reg, Q_u)
            except np.linalg.LinAlgError:
                diverged = True
                break

            K_seq[t] = K_t
            k_seq[t] = k_t

            # Value function update (Gauss-Newton/iLQR form)
            V_x = Q_x - Q_ux.T @ np.linalg.solve(Q_uu_reg, Q_u)
            V_xx = Q_xx - Q_ux.T @ np.linalg.solve(Q_uu_reg, Q_ux)
            # Symmetrize to avoid numerical drift
            V_xx = 0.5 * (V_xx + V_xx.T)

        if diverged:
            reg = min(cfg.reg_max, reg * cfg.reg_scale_up)
            continue

        # Forward line search
        accepted = False
        for alpha in cfg.alpha_list:
            x_new = np.zeros_like(x_bar)
            u_new = np.zeros_like(u_bar)
            x_new[0] = x0.copy()
            u_prev = np.zeros(m)
            cost_new = 0.0
            for t in range(N):
                dx = x_new[t] - x_bar[t]
                du = alpha * k_seq[t] + K_seq[t] @ dx
                u_new[t] = u_bar[t] + du
                cost_new += stage_cost_for_rollout(x_new[t], u_new[t], x_refs[t], u_prev)
                x_new[t + 1] = f_wrap(x_new[t], u_new[t])
                u_prev = u_new[t]
            cost_new += terminal_cost_for_rollout(x_new[N], x_refs[N])

            if cost_new < best_cost - cfg.tol_cost:
                best_cost = cost_new
                x_bar = x_new
                u_bar = u_new
                accepted = True
                reg = max(cfg.reg_min, reg * cfg.reg_scale_down)
                break

        if not accepted:
            reg = min(cfg.reg_max, reg * cfg.reg_scale_up)
        else:
            # Check convergence with small improvement
            if (best_cost - cost_new) < cfg.tol_cost:
                converged = True
                return ILQRResult(u_seq=u_bar, K_seq=K_seq, x_seq=x_bar, cost=best_cost, iters=it+1, converged=True)

    return ILQRResult(u_seq=u_bar, K_seq=K_seq, x_seq=x_bar, cost=best_cost, iters=cfg.max_iter, converged=converged)


# ===============================
# MPC Demo: Lane-change tracking
# ===============================
@dataclass
class MPCConfig:
    horizon: int = 40
    iters: int = 10


def build_reference(total_steps: int, dt: float, lane_width: float = 3.5, v_ref: float = 12.0) -> np.ndarray:
    """
    Build a smooth lane-change reference (x increases, y shifts by lane_width).
    Returns x_ref array shape (total_steps+1, 5)
    """
    x_ref = np.zeros((total_steps + 1, 5))
    s = np.linspace(0.0, 1.0, total_steps + 1)
    # Smooth S-curve for y
    y_shift = lane_width * (3*s**2 - 2*s**3)

    for t in range(total_steps + 1):
        if t == 0:
            x_ref[t, 0] = 0.0
        else:
            x_ref[t, 0] = x_ref[t - 1, 0] + v_ref * dt
        x_ref[t, 1] = y_shift[t]
        x_ref[t, 2] = 0.0   # small yaw reference
        x_ref[t, 3] = v_ref
        x_ref[t, 4] = 0.0
    return x_ref


def run_mpc_demo(seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Params and weights
    params = BicycleParams(wheelbase=2.7, dt=0.05, max_steer=math.radians(35.0))
    n, m = 5, 2

    Q = np.diag([4.0, 6.0, 2.0, 0.2, 0.1])
    R = np.diag([0.05, 0.05])
    Rd = np.diag([0.5, 0.5])
    Qf = np.diag([10.0, 12.0, 4.0, 1.0, 0.5])
    w = CostWeights(Q=Q, R=R, Rd=Rd, Qf=Qf)

    # Total simulation steps and horizon
    total_time = 8.0
    total_steps = int(total_time / params.dt)
    horizon = 40

    x_ref_full = build_reference(total_steps, params.dt, lane_width=3.5, v_ref=12.0)

    # Initial state
    x = np.array([0.0, 0.0, 0.0, 12.0, 0.0])

    # Buffers
    X_log = [x.copy()]
    U_log = []

    # Warm-start
    u_warm = np.zeros((horizon, m))

    ilqr_cfg = ILQRConfig(max_iter=10, tol_cost=1e-5)

    def f_dyn(x_: np.ndarray, u_: np.ndarray) -> np.ndarray:
        return bicycle_dynamics(x_, u_, params)

    def lin_dyn(x_: np.ndarray, u_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return bicycle_linearize(x_, u_, params)

    t = 0
    while t < total_steps:
        # Build refs for the current horizon window
        t_end = min(total_steps, t + horizon)
        x_refs = x_ref_full[t: t_end + 1]
        H = t_end - t

        if H <= 0:
            break

        # Adjust warm start to current horizon length
        if u_warm.shape[0] != H:
            u_warm = np.zeros((H, m))

        # Solve iLQR on the horizon
        result = ilqr(x0=x, u_init=u_warm, f=f_dyn, linearize=lin_dyn, w=w, x_refs=x_refs, cfg=ilqr_cfg)

        # Apply first control and step system
        u0 = result.u_seq[0]
        x = f_dyn(x, u0)

        # Log
        U_log.append(u0)
        X_log.append(x.copy())

        # Warm start for next MPC cycle: shift controls
        if result.u_seq.shape[0] > 1:
            u_warm = np.vstack([result.u_seq[1:], result.u_seq[-1]])
        else:
            u_warm = result.u_seq

        t += 1

    X_log = np.array(X_log)
    U_log = np.array(U_log)

    return {
        "X": X_log,
        "U": U_log,
        "X_ref": x_ref_full,
        "dt": np.array([params.dt]),
    }


def _maybe_plot(sim: Dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available; skipping plots. Error:", e)
        return

    X = sim["X"]
    U = sim["U"]
    X_ref = sim["X_ref"]
    dt = float(sim["dt"][0])

    T = X.shape[0] - 1
    tgrid = np.arange(T + 1) * dt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(X_ref[:T+1, 0], X_ref[:T+1, 1], 'k--', label='ref path')
    plt.plot(X[:T+1, 0], X[:T+1, 1], 'b-', label='vehicle')
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.title('Trajectory')

    plt.subplot(2, 2, 2)
    plt.plot(tgrid[:-1], U[:, 0], label='a [m/s^2]')
    plt.plot(tgrid[:-1], U[:, 1], label='ddelta [rad/s]')
    plt.xlabel('time [s]')
    plt.legend()
    plt.title('Controls')

    plt.subplot(2, 2, 4)
    plt.plot(tgrid, X[:T+1, 3], label='v [m/s]')
    plt.plot(tgrid, X_ref[:T+1, 3], 'k--', label='v_ref')
    plt.xlabel('time [s]')
    plt.legend()
    plt.title('Speed')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sim = run_mpc_demo()
    _maybe_plot(sim)
    # Print final tracking error summary
    X = sim["X"]; X_ref = sim["X_ref"]
    pos_err = np.linalg.norm(X[-1, :2] - X_ref[min(len(X_ref)-1, len(X)-1), :2])
    print(f"Final position error: {pos_err:.3f} m")
