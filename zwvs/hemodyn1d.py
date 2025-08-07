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
    nx, nt = 101, 101
    L, T = 10.0, 100.0
    dx, dt = L/(nx-1), T/(nt-1)
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)

    A0 = 1 
    Q0 = 100
    Q1 = 101
    P0 = 1.3e5

    # Dimensionless boundary conditions
    Q_in  = lambda tt: Q0 * 0.05  * np.sin(2*np.pi*tt/T) * np.sin(2*np.pi*tt/T)
    P_in  = lambda tt: P0 * (1 + 0.05  * np.cos(2*np.pi*tt/T)* np.cos(2*np.pi*tt/T))
    P_out = lambda tt: P0 * (1 + 0.025 * np.sin(2*np.pi*tt/T))
    Q_out = lambda tt: Q1 * np.sin(2*np.pi*tt/T) * np.sin(2*np.pi*tt/T)

    # Dimensionless parameters
    params = {
        'alpha': 1.333,
        'rho': 1.060e3,
        'beta': 2e7,
        'K_R': 25.0,
        'Miu': 1.0e-5,
        'R_out': 10 ,
        'P_ref': 1.3e5,
        'A0': A0,
        'Q0': Q0,
        'P0': P0
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
        U0_seq[i]          = A0 
        U0_seq[nx + i]     = 0
        U0_seq[2 * nx + i] = P0

    lb_seq = np.full_like(U0_seq, -np.inf)
    ub_seq = np.full_like(U0_seq, np.inf)
    lb_seq[:nx] = 0.5*A0
    ub_seq[:nx] = 2*A0
    lb_seq[nx:2*nx] = -10*Q0 
    ub_seq[nx:2*nx] = 10*Q0
    lb_seq[2*nx:] = 0.5*P0
    ub_seq[2*nx:] = 2*P0

    seq_solver = SequentialHemodynamics1D(nx, dx, dt, params, Q_in, P_in, P_out, Q_out, (lb_seq.copy(), ub_seq.copy()))
    A_seq, Q_seq, P_seq = seq_solver.solve(U0_seq.copy(), x, 150, t)  # Ensure writable input
    plots.display_step_result(149, x, t, A_seq, Q_seq, P_seq, A0, Q0, P0)
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
