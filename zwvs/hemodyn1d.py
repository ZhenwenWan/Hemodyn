import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from CoDoSol import CoDoSol
from hemodyn1d_seq import SequentialHemodynamics1D
import plots


# --- 1. Setup parameters ---
try:
    print("Setting up parameters")
    nx, nt = 201, 101
    L, T = 1.0, 1.0
    dx, dt = L/(nx-1), T/(nt-1)
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)

    A0 = 1e-4
    Q0 = 1e-4
    Q1 = 1e-4
    P0 = 1e5

    # Dimensionless boundary conditions
    Q_in  = lambda tt: (Q0 * np.sin(2*np.pi*tt/T) * np.sin(2*np.pi*tt/T)) / Q0
    P_in  = lambda tt: (P0 * (1 + 0.05 * np.sin(2*np.pi*tt/T))) / P0
    P_out = lambda tt: (P0 * (1 + 0.025 * np.sin(2*np.pi*tt/T))) / P0
    Q_out = lambda tt: (Q1 * np.sin(2*np.pi*tt/T) * np.sin(2*np.pi*tt/T)) / Q0

    # Dimensionless parameters
    params = {
        'alpha': 1.1,
        'rho_scaled': 1060 * Q0**2 / A0,
        'beta_scaled': 1e11 * A0 / P0,
        'K_R_scaled': 0.1 * Q0 / A0,
        'R': 1e3
    }
    print("Parameters defined successfully:", params)
    print(f"dx={dx:.2e}, dt={dt:.2e}")
except Exception as e:
    print("Error in parameter setup:", e)
    sys.exit(1)


# --- 2. Run and Verify Sequential Solver ---
try:
    print("Initializing U0 for sequential solver")
    U0_seq = np.zeros(3 * nx).copy()  # Ensure writable array
    margin = 1e-5
    for i in range(nx):
        U0_seq[i]          = 1 
        U0_seq[nx + i]     = 0
        U0_seq[2 * nx + i] = 1

    lb_seq = np.full_like(U0_seq, -np.inf)
    ub_seq = np.full_like(U0_seq, np.inf)
    lb_seq[:nx] = 0.5
    ub_seq[:nx] = 2
    lb_seq[nx:2*nx] = -2 
    ub_seq[nx:2*nx] = 10
    lb_seq[2*nx:] = 0.5
    ub_seq[2*nx:] = 2

    seq_solver = SequentialHemodynamics1D(nx, dx, dt, params, Q_in, P_in, P_out, Q_out, (lb_seq.copy(), ub_seq.copy()))
    A_seq, Q_seq, P_seq = seq_solver.solve(U0_seq.copy(), x, nt, t)  # Ensure writable input
    plots.display_step_result(nt -1, x, t, A_seq, Q_seq, P_seq, A0, Q0, P0)
    plots.visualize_all_steps(x, t, A_seq, Q_seq, P_seq, A0, Q0, P0)

    A_seq_physical = A_seq * A0
    Q_seq_physical = Q_seq * Q0
    P_seq_physical = P_seq * P0

    if A_seq is not None and Q_seq is not None and P_seq is not None and A_seq.shape[0] == nt:
        print("Sequential solver completed successfully")
        # (Plotting remains unchanged)
    else:
        print("Sequential solver failed: A, Q, or P is None or incomplete")
        # (Partial plotting remains unchanged)
        sys.exit(1)
except Exception as e:
    print(f"Error in sequential solver execution: {e}")
    raise
