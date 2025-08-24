# PINN03.py — time-dependent 1D–2D coupled oxygen transport with PINNs (DeepXDE)
# Minimal changes per request:
#   1) Removed RAR & any use of pde_residual (bug source + unnecessary cost).
#   2) Added a small "save artifacts" block at the end:
#        - DeepXDE model checkpoint via model.save(...)
#        - run_meta.json with learned params and iteration
#        - final-time predictions as .npy
#   3) Training kept simple: Adam (with resampler/early stop) -> L-BFGS.
#      No extra polish phases, no best-ckpt callback.

from __future__ import annotations

import os
os.environ["DDE_BACKEND"] = "pytorch"  # ensure PyTorch backend

import json, time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import deepxde as dde
from deepxde.backend import backend as F

# If PyTorch backend is active, import torch for safe trig ops in drivers
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ----------------------------
# Problem setup and constants
# ----------------------------
@dataclass
class TrueParams:
    # Geometry (mm)
    L: float = 1.0      # capillary length
    H: float = 0.2      # tissue height
    R: float = 0.005    # capillary radius

    # Capillary transport
    u: float = 1.0      # mm/s axial speed
    Dc: float = 1e-3    # mm^2/s axial diffusion

    # Tissue diffusion
    Dt: float = 1e-3    # mm^2/s

    # Interface permeability
    P: float = 5e-3     # mm/s

    # Metabolism (Michaelis–Menten)
    Vmax: float = 2e-3  # 1/s * [conc]
    Km: float = 0.03    # [conc]
    beta: float = 0.3   # motion scaling coefficient (quadratic)

    # Inflow + motion drivers
    Cin0: float = 0.10  # baseline inlet
    Cin1: float = 0.02  # pulsation amplitude
    f_c: float = 1.0    # Hz inflow pulsation
    I0: float = 1.0     # baseline intensity
    I1: float = 0.5     # intensity oscillation amplitude
    f_m: float = 0.5    # Hz motion frequency

    # Time horizon (s)
    T: float = 2.0


# FD discretization for forward solver
FD_NX = 161
FD_NY = 65
FD_NT = 201
INNER_ITERS = 3
np.random.seed(7)


# Utility: time signals (numpy for forward solver)
def Cin_t(params: TrueParams, t: np.ndarray) -> np.ndarray:
    return params.Cin0 + params.Cin1 * np.sin(2 * np.pi * params.f_c * t)


def I_t(params: TrueParams, t: np.ndarray) -> np.ndarray:
    return params.I0 + params.I1 * np.sin(2 * np.pi * params.f_m * t)


# --------------------------------------------------------
# Forward transient solver (finite difference, semi-implicit)
# --------------------------------------------------------

def solve_forward_fd(params: TrueParams):
    L, H, R = params.L, params.H, params.R
    u, Dc, Dt = params.u, params.Dc, params.Dt
    P, Vmax, Km, beta = params.P, params.Vmax, params.Km, params.beta

    Nx, Ny, Nt = FD_NX, FD_NY, FD_NT
    x = np.linspace(0.0, L, Nx)
    y = np.linspace(0.0, H, Ny)
    t = np.linspace(0.0, params.T, Nt)
    dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]

    # State arrays
    Oc = np.zeros((Nt, Nx))          # capillary: (t, x)
    Ot = np.zeros((Nt, Ny, Nx))      # tissue: (t, y, x)

    # Initial conditions
    Oc[0, :] = params.Cin0
    Ot[0, :, :] = 0.8 * params.Cin0

    def update_Ot_interface(Oc_line, Ot_slice):
        # Robin: -Dt (Ot[1]-Ot[0])/dy = P (Oc - Ot[0])
        Ot0 = (Dt * Ot_slice[1, :] / dy + P * Oc_line) / (Dt / dy + P)
        Ot_slice[0, :] = Ot0
        return Ot_slice

    def solve_Oc_tridiag(Oc_prev, Ot_wall, Cin_next):
        # (Oc^{n+1} - Oc^n)/dt + u dOc/dx - Dc d2Oc/dx2 + 2P/R (Oc - Ot_wall) = 0
        Nx = Oc_prev.size
        a = np.zeros(Nx); b = np.zeros(Nx); c = np.zeros(Nx); rhs = np.zeros(Nx)
        sink = 2.0 * P / R
        # left Dirichlet
        b[0] = 1.0
        rhs[0] = Cin_next
        # interior upwind/central blend
        for i in range(1, Nx - 1):
            a[i] = - (u / (2*dx)) - (Dc / dx**2)
            b[i] = 1.0/dt + (2 * Dc / dx**2) + (u / (2*dx)) + sink
            c[i] = - (Dc / dx**2)
            rhs[i] = Oc_prev[i]/dt + sink * Ot_wall[i]
        # right Neumann: dOc/dx=0
        i = Nx - 1
        a[i] = - (u / (2*dx)) - (Dc / dx**2)
        b[i] = 1.0/dt + (Dc / dx**2) + (u / (2*dx)) + sink
        c[i] = 0.0
        rhs[i] = Oc_prev[i]/dt + sink * Ot_wall[i]
        # Thomas algorithm
        for i in range(1, Nx):
            m = a[i] / b[i-1]
            b[i] -= m * c[i-1]
            rhs[i] -= m * rhs[i-1]
        sol = np.zeros(Nx)
        sol[-1] = rhs[-1] / b[-1]
        for i in range(Nx-2, -1, -1):
            sol[i] = (rhs[i] - c[i] * sol[i+1]) / b[i]
        return sol

    # Time marching
    for n in range(Nt - 1):
        Cin_next = Cin_t(params, np.array([t[n+1]]))[0]
        Veff = Vmax * (1.0 + beta * I_t(params, np.array([t[n+1]]))[0]**2)

        Oc_next = Oc[n, :].copy()
        Ot_next = Ot[n, :, :].copy()

        for _ in range(INNER_ITERS):
            # Tissue BCs (Neumann top/left/right)
            Ot_next[-1, :] = Ot_next[-2, :]
            Ot_next[:, 0] = Ot_next[:, 1]
            Ot_next[:, -1] = Ot_next[:, -2]
            # Interface Robin using current Oc_next
            Ot_next = update_Ot_interface(Oc_next, Ot_next)

            # Semi-implicit tissue interior
            a = Veff / (Km + Ot_next + 1e-12)
            denom = 1.0 + dt * (a + 2*Dt*(1/dx**2 + 1/dy**2))
            for j in range(1, Ny-1):
                for i in range(1, Nx-1):
                    Ot_next[j, i] = (
                        Ot[n, j, i]
                        + dt * Dt * ((Ot_next[j, i+1] + Ot_next[j, i-1]) / dx**2
                                     + (Ot_next[j+1, i] + Ot_next[j-1, i]) / dy**2)
                    ) / denom[j, i]

            # Capillary update using current Ot_wall
            Oc_next = solve_Oc_tridiag(Oc[n, :], Ot_next[0, :], Cin_next)

        Oc[n+1, :] = Oc_next
        Ot[n+1, :, :] = Ot_next

    # Build observation sets
    Xcap = []  # (x, 0, t)
    Ycap = []  # Oc(x,t)
    Xtis = []  # (x, y, t)
    Ytis = []  # Ot(x,y,t)
    for n in range(Nt):
        for i in range(Nx):
            Xcap.append([x[i], 0.0, t[n]])
            Ycap.append([Oc[n, i]])
        # subsample tissue per time slice
        idx_x = np.linspace(0, Nx-1, 64, dtype=int)
        idx_y = np.linspace(0, Ny-1, 32, dtype=int)
        for j in idx_y:
            for i in idx_x:
                Xtis.append([x[i], y[j], t[n]])
                Ytis.append([Ot[n, j, i]])

    return (x, y, t), Oc, Ot, np.array(Xcap), np.array(Ycap), np.array(Xtis), np.array(Ytis)


# ---------------------------------------
# PINN inverse model (DeepXDE, time-dependent)
# ---------------------------------------

def build_inverse_pinn(params_true: TrueParams,
                       obs_cap_pts: np.ndarray,
                       obs_cap_vals: np.ndarray,
                       obs_tis_pts: np.ndarray,
                       obs_tis_vals: np.ndarray,
                       y_eps: float = 2e-3,
                       num_domain: int = 12000,
                       num_boundary: int = 1800,
                       num_initial: int = 1200):
    L, H, T = params_true.L, params_true.H, params_true.T
    geom = dde.geometry.Rectangle([0.0, 0.0], [L, H])
    timedomain = dde.geometry.TimeDomain(0.0, T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Trainable parameters (inverse targets)
    P_var   = dde.Variable(params_true.P * 0.8)
    Dt_var  = dde.Variable(params_true.Dt * 1.2)
    Vmax_var= dde.Variable(params_true.Vmax * 1.1)
    Km_var  = dde.Variable(params_true.Km * 0.9)

    # Fixed constants
    u, Dc, R, beta = params_true.u, params_true.Dc, params_true.R, params_true.beta

    # Time drivers
    def I_of_t(t):
        if torch is not None:
            return params_true.I0 + params_true.I1 * torch.sin(2 * torch.pi * params_true.f_m * t)
        return params_true.I0 + params_true.I1 * F.sin(2 * np.pi * params_true.f_m * t)

    # PDE residuals
    def pde(X, Y):
        x = X[:, 0:1]
        y = X[:, 1:2]
        t = X[:, 2:3]
        Oc = Y[:, 0:1]
        Ot = Y[:, 1:2]

        # Time derivatives
        Oc_t = dde.grad.jacobian(Oc, X, i=0, j=2)
        Ot_t = dde.grad.jacobian(Ot, X, i=0, j=2)
        # Spatial derivatives
        Oc_x  = dde.grad.jacobian(Oc, X, i=0, j=0)
        Oc_xx = dde.grad.hessian(Oc, X, i=0, j=0)
        Ot_xx = dde.grad.hessian(Ot, X, i=0, j=0)
        Ot_yy = dde.grad.hessian(Ot, X, i=0, j=1)

        # Tissue reaction
        Veff = Vmax_var * (1.0 + beta * I_of_t(t) ** 2)
        reac = Veff * Ot / (Km_var + Ot + 1e-12)

        # Residuals
        res_t = Ot_t - Dt_var * (Ot_xx + Ot_yy) + reac
        sink  = (2.0 * P_var / R) * (Oc - Ot)
        res_c_raw = Oc_t + u * Oc_x - Dc * Oc_xx + sink

        # Enforce capillary PDE only near y=0 using a smooth mask
        if torch is not None:
            y_eps_t = torch.tensor(y_eps, dtype=y.dtype, device=y.device)
            z = y / y_eps_t
            mask = torch.exp(-(z * z))
        else:
            z = y / y_eps
            mask = F.exp(-(z * z))
        res_c = mask * res_c_raw

        return [res_c, res_t]

    # Boundary selectors (space–time)
    def on_left_cap(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], 0.0) and (X[1] <= y_eps)

    def on_right_cap(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], L) and (X[1] <= y_eps)

    def on_top_tis(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[1], H)

    def on_left_tis(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], 0.0) and (X[1] > y_eps)

    def on_right_tis(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[0], L) and (X[1] > y_eps)

    def on_interface(X, on_boundary):
        return on_boundary and dde.utils.isclose(X[1], 0.0)

    # Initial conditions at t=0
    ic_cap = dde.icbc.IC(
        geomtime,
        lambda X: params_true.Cin0 * np.ones((len(X), 1)),
        lambda _, on_initial: on_initial,
        component=0,
    )
    ic_tis = dde.icbc.IC(
        geomtime,
        lambda X: 0.8 * params_true.Cin0 * np.ones((len(X), 1)),
        lambda _, on_initial: on_initial,
        component=1,
    )

    # Capillary inflow Dirichlet at x=0
    bc_in = dde.icbc.DirichletBC(
        geomtime,
        lambda X: params_true.Cin0 + params_true.Cin1 * np.sin(2 * np.pi * params_true.f_c * X[:, 2:3]),
        on_left_cap,
        component=0,
    )

    # Capillary outflow Neumann at x=L (dOc/dx = 0)
    def dOc_dx(inputs, outputs, X):
        Oc = outputs[:, 0:1]
        return dde.grad.jacobian(Oc, inputs, i=0, j=0)

    bc_out = dde.icbc.OperatorBC(geomtime, dOc_dx, on_right_cap)

    # Tissue zero-flux at top and side boundaries
    def dOt_dy_top(inputs, outputs, X):
        Ot = outputs[:, 1:2]
        return dde.grad.jacobian(Ot, inputs, i=0, j=1)

    def dOt_dx_lr(inputs, outputs, X):
        Ot = outputs[:, 1:2]
        return dde.grad.jacobian(Ot, inputs, i=0, j=0)

    bc_top   = dde.icbc.OperatorBC(geomtime, dOt_dy_top, on_top_tis)
    bc_left  = dde.icbc.OperatorBC(geomtime, dOt_dx_lr, on_left_tis)
    bc_right = dde.icbc.OperatorBC(geomtime, dOt_dx_lr, on_right_tis)

    # Interface Robin flux: P (Oc − Ot) + Dt * dOt/dy = 0 at y=0
    def interface_flux(inputs, outputs, X):
        Oc = outputs[:, 0:1]
        Ot = outputs[:, 1:2]
        dOt_dy = dde.grad.jacobian(Ot, inputs, i=0, j=1)
        return P_var * (Oc - Ot) + Dt_var * dOt_dy

    bc_interface = dde.icbc.OperatorBC(geomtime, interface_flux, on_interface)

    # Observational constraints
    obs_cap = dde.icbc.PointSetBC(obs_cap_pts, obs_cap_vals, component=0)
    obs_tis = dde.icbc.PointSetBC(obs_tis_pts, obs_tis_vals, component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [ic_cap, ic_tis, bc_in, bc_out, bc_top, bc_left, bc_right, bc_interface, obs_cap, obs_tis],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        train_distribution="uniform",
    )

    net = dde.nn.FNN([3] + [64] * 4 + [2], "tanh", "Glorot uniform")  # (x,y,t)->(Oc,Ot)
    model = dde.Model(data, net)

    model.compile(
        optimizer="adam",
        lr=1e-3,
        external_trainable_variables=[P_var, Dt_var, Vmax_var, Km_var],
    )
    variables = {"P": P_var, "Dt": Dt_var, "Vmax": Vmax_var, "Km": Km_var}
    return model, variables


# ---------------------------
# Visualization utilities
# ---------------------------

def plot_results(space, Oc_true, Ot_true, Oc_pred_tend, Ot_pred_tend,
                 learned_params: dict, true_params, out_prefix="oxygen_pinn_td"):
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- unpack & coerce ----
    x, y, t = space
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    Oc_true = np.asarray(Oc_true)
    Ot_true = np.asarray(Ot_true)

    # Oc_pred_tend: (len(x),) expected
    Oc_pred_tend = np.asarray(Oc_pred_tend)
    if Oc_pred_tend.ndim > 1:
        Oc_pred_tend = Oc_pred_tend.reshape(-1)
    if Oc_pred_tend.size != x.size:
        raise ValueError(f"Oc_pred_tend has {Oc_pred_tend.size} elements; expected {x.size} (len(x)).")

    # Ot_pred_tend: reshape to (len(y), len(x)) to match meshgrid/contourf
    Ot_pred_tend = np.asarray(Ot_pred_tend)
    if Ot_pred_tend.ndim == 1:
        if Ot_pred_tend.size != x.size * y.size:
            raise ValueError(
                f"Ot_pred_tend has {Ot_pred_tend.size} elements; expected {x.size * y.size} (=len(x)*len(y))."
            )
        Z_pred = Ot_pred_tend.reshape(len(y), len(x))
    elif Ot_pred_tend.ndim == 2:
        if Ot_pred_tend.shape == (len(y), len(x)):
            Z_pred = Ot_pred_tend
        elif Ot_pred_tend.shape == (len(x), len(y)):
            Z_pred = Ot_pred_tend.T  # auto-fix swapped dims
        else:
            raise ValueError(f"Ot_pred_tend has shape {Ot_pred_tend.shape}; expected {(len(y), len(x))}.")
    else:
        raise ValueError("Ot_pred_tend must be 1D or 2D.")

    # Build rectilinear grid for contourf (X,Y same shape as Z) 
    Xg, Yg = np.meshgrid(x, y, indexing="xy")

    # ---- 1) Capillary line plot at t = T ----
    Oc_true_T = Oc_true[-1, :].reshape(-1)
    plt.figure()
    plt.plot(x, Oc_true_T, label="Oc true (t=T)")
    plt.plot(x, Oc_pred_tend, "--", label="Oc PINN (t=T)")
    plt.xlabel("x [mm]")
    plt.ylabel("Capillary O$_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_capillary_tend.png", dpi=160)

    # ---- 2) Tissue heatmaps at t = T ----
    Z_true = Ot_true[-1, :, :]
    plt.figure()
    cs = plt.contourf(Xg, Yg, Z_true, levels=30)
    plt.colorbar(cs)
    plt.xlabel("x [mm]"); plt.ylabel("y [mm]")
    plt.title("Tissue O$_2$ true (t=T)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_tissue_true_tend.png", dpi=160)

    plt.figure()
    cs = plt.contourf(Xg, Yg, Z_pred, levels=30)
    plt.colorbar(cs)
    plt.xlabel("x [mm]"); plt.ylabel("y [mm]")
    plt.title("Tissue O$_2$ PINN (t=T)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_tissue_pinn_tend.png", dpi=160)

    # ---- 3) Learned vs true params ----
    names = ["P", "Dt", "Vmax", "Km"]
    truth = [true_params.P, true_params.Dt, true_params.Vmax, true_params.Km]
    learned = [float(learned_params[n]) for n in names]
    pos = np.arange(len(names))

    plt.figure()
    plt.bar(pos - 0.2, truth, 0.4, label="True")
    plt.bar(pos + 0.2, learned, 0.4, label="Learned")
    plt.xticks(pos, names)
    plt.ylabel("Value")
    plt.title("Parameter comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_params.png", dpi=160)

    plt.close("all")

def _to_scalar_any(v):
    """Convert DeepXDE trainable variable to a Python float, robust across backends."""
    # 1) PyTorch tensor
    if torch is not None and isinstance(v, torch.Tensor):
        return float(v.detach().cpu().numpy().reshape(-1)[0])
    # 2) DeepXDE Variable wrappers sometimes expose .value (tensor/array)
    val = getattr(v, "value", None)
    if val is not None:
        # value may itself be a torch tensor or a numpy-like
        if torch is not None and isinstance(val, torch.Tensor):
            return float(val.detach().cpu().numpy().reshape(-1)[0])
        return float(np.array(val).reshape(-1)[0])
    # 3) TF-style objects exposing .numpy()
    if hasattr(v, "numpy"):
        return float(np.array(v.numpy()).reshape(-1)[0])
    # 4) Plain numbers
    return float(v)


# ---------------------------
# End-to-end demo
# ---------------------------

def main():
    params_true = TrueParams()

    # 1) Forward synthetic data
    space, Oc, Ot, Xcap, Ycap, Xtis, Ytis = solve_forward_fd(params_true)

    # Subsample observations
    rng = np.random.default_rng(0)
    keep_cap = min(4000, len(Xcap))
    keep_tis = min(12000, len(Xtis))
    if keep_cap < len(Xcap):
        idx_cap = rng.choice(len(Xcap), keep_cap, replace=False)
        Xcap, Ycap = Xcap[idx_cap], Ycap[idx_cap]
    if keep_tis < len(Xtis):
        idx_tis = rng.choice(len(Xtis), keep_tis, replace=False)
        Xtis, Ytis = Xtis[idx_tis], Ytis[idx_tis]

    # Add light noise
    noise = 0.01 * params_true.Cin0
    Ycap_noisy = Ycap + noise * np.random.randn(*Ycap.shape)
    Ytis_noisy = Ytis + noise * np.random.randn(*Ytis.shape)

    # 2) Build inverse PINN
    model, vars_dict = build_inverse_pinn(
        params_true,
        obs_cap_pts=Xcap,
        obs_cap_vals=Ycap_noisy,
        obs_tis_pts=Xtis,
        obs_tis_vals=Ytis_noisy,
        y_eps=2e-3,
        num_domain=12000,
        num_boundary=1800,
        num_initial=1200,
    )

    # -----------------------------
    # 3) Training schedule (simple)
    # -----------------------------
    outdir = "checkpoints"
    os.makedirs(outdir, exist_ok=True)

    # Phase A: Adam + periodic resampling (standard, light)
    resampler = dde.callbacks.PDEPointResampler(period=200)
    earlyA = dde.callbacks.EarlyStopping(min_delta=1e-5, patience=3000)
    model.train(iterations=4000, display_every=200, callbacks=[resampler, earlyA])

    # Phase B: L-BFGS (standard DeepXDE defaults)
    try:
        from deepxde.optimizers import config as lbfgs_cfg
        # Use documented defaults; adjust if needed.
        lbfgs_cfg.set_LBFGS_options(maxiter=15000, maxcor=100, ftol=0.0, gtol=1e-8, maxls=50)
    except Exception:
        pass
    model.compile(optimizer="L-BFGS", external_trainable_variables=list(vars_dict.values()))
    model.train()

    # Extract learned parameters
    learned = {k: _to_scalar_any(v) for k, v in vars_dict.items()}

    print("\nLearned parameters:")
    for k, v in learned.items():
        print(f"  {k}: {v:.6g}")

    # 4) Predict fields at final time t=T
    x, y, t = space
    T = t[-1]
    cap_pts = np.column_stack([x, np.zeros_like(x), np.full_like(x, T)])
    pred_cap = model.predict(cap_pts)[:, 0]

    Xg, Yg = np.meshgrid(x, y)
    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, T)])
    pred_tis = model.predict(grid_pts)[:, 1]

    # 5) Visualize
    plot_results(space, Oc, Ot, pred_cap, pred_tis, learned, params_true)

    # 6) ---- Save artifacts (model + metadata + arrays) ----
    # DeepXDE uses backend-appropriate serialization under the hood; on PyTorch this
    # produces a .pt checkpoint containing model + optimizer states. Use restore() to load.
    save_path = model.save(os.path.join(outdir, "oxygen_pinn"), verbose=1)  # -> "...oxygen_pinn-<N>.pt"

    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "pytorch",
        "final_iteration": int(model.train_state.iteration),
        "learned_params": {k: float(v) for k, v in learned.items()},
        "true_params": vars(params_true),
        "checkpoint_path": save_path,
    }
    with open(os.path.join(outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    np.save(os.path.join(outdir, "pred_cap_T.npy"), pred_cap)
    np.save(os.path.join(outdir, "pred_tis_T.npy"), pred_tis)

    print(f"Artifacts saved under: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()

