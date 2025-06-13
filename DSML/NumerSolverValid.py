import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Parameters
nx = 100
nt = 100
x = np.linspace(0, 1, nx)
t = np.linspace(0, 1, nt)
dx = x[1] - x[0]
dt = 0.01

# Parameter combinations
param_combos = [(0.1, 0.1), (0.1, 0.5), (0.5, 0.1), (0.5, 0.5)]

def analytical_solution(x, t, alpha, beta):
    X, T = np.meshgrid(x, t)
    return np.exp(-np.pi**2 * alpha * T) * np.sin(np.pi * X) + beta / (np.pi**2 * alpha) * (1 - np.exp(-np.pi**2 * alpha * T)) * np.sin(np.pi * X)

def compute_f(u, x, alpha, beta, dx):
    Nx = len(x)
    f = np.zeros(Nx)
    for i in range(1, Nx - 1):
        f[i] = alpha * (u[i + 1] - 2 * u[i] + u[i - 1]) / (dx ** 2) + beta * np.sin(np.pi * x[i])
    return f

def numerical_solution_implicit(x, t_output, alpha, beta, dx=0.01, dt=0.01):
    Nx = len(x)
    Nt = len(t_output)
    u = np.zeros((Nt, Nx))
    u[0, :] = np.sin(np.pi * x)
    
    # Define matrix A for Crank–Nicolson
    r = alpha * dt / (dx ** 2)
    main_diag = (1 + r) * np.ones(Nx - 2)
    off_diag = (-r / 2) * np.ones(Nx - 3)
    A = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')

    B = sp.diags([r / 2 * np.ones(Nx - 3), (1 - r) * np.ones(Nx - 2), r / 2 * np.ones(Nx - 3)],
                 offsets=[-1, 0, 1], format='csc')

    t_full = np.linspace(0, t_output[-1], int(t_output[-1]/dt) + 1)
    u_full = np.zeros((len(t_full), Nx))
    u_full[0, :] = u[0, :]

    for n in range(len(t_full) - 1):
        rhs = B.dot(u_full[n, 1:-1]) + dt * beta * np.sin(np.pi * x[1:-1])
        u_full[n + 1, 1:-1] = spla.spsolve(A, rhs)
        u_full[n + 1, 0] = 0
        u_full[n + 1, -1] = 0

    # Resample u to match t_output
    step_interval = int(0.01 / dt)
    u_output = u_full[::step_interval][:Nt]
    return u_output

def numerical_solution(x, t_output, alpha, beta, dx=0.01, dt=0.01):
    Nx = len(x)
    t_full = np.arange(0, 1.0001, dt)
    Nt_full = len(t_full)
    u = np.zeros((Nt_full, Nx))
    u[0, :] = np.sin(np.pi * x)
    u[:, 0] = 0
    u[:, -1] = 0

    for n in range(Nt_full - 1):
        u_n = u[n, :].copy()
        k1 = compute_f(u_n, x, alpha, beta, dx)
        u_temp = u_n + (dt / 2) * k1
        u_temp[0] = u_temp[-1] = 0
        k2 = compute_f(u_temp, x, alpha, beta, dx)
        u_temp = u_n + (dt / 2) * k2
        u_temp[0] = u_temp[-1] = 0
        k3 = compute_f(u_temp, x, alpha, beta, dx)
        u_temp = u_n + dt * k3
        u_temp[0] = u_temp[-1] = 0
        k4 = compute_f(u_temp, x, alpha, beta, dx)
        u[n + 1, :] = u_n + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        u[n + 1, 0] = u[n + 1, -1] = 0

    step_interval = int(0.01 / dt)
    u_output = u[::step_interval, :]
    return u_output

def compute_stats_aligned(U_analytical, U_numerical):
    min_time_steps = min(U_analytical.shape[0], U_numerical.shape[0])
    U_analytical_aligned = U_analytical[:min_time_steps]
    U_numerical_aligned = U_numerical[:min_time_steps]
    error = U_numerical_aligned - U_analytical_aligned
    std_dev = np.std(error)
    mean_ana = np.mean(np.abs(U_analytical_aligned))
    mean_num = np.mean(np.abs(U_numerical_aligned))
    return std_dev, mean_ana, mean_num

# Prepare plots
fig, axes = plt.subplots(len(param_combos), 2, figsize=(12, 10))
common_cmap = 'viridis'

for idx, (alpha, beta) in enumerate(param_combos):
    U_analytical = analytical_solution(x, t, alpha, beta)
    U_numeric_stable = numerical_solution_implicit(x, t, alpha, beta, dx=dx, dt=dt)
    vmin_shared = min(U_analytical.min(), U_numeric_stable.min())
    vmax_shared = max(U_analytical.max(), U_numeric_stable.max())
    std_dev, mean_ana, mean_num = compute_stats_aligned(U_analytical, U_numeric_stable)
    print(f"α={alpha}, β={beta} --> Std Dev: {std_dev:.5f}, Mean Ana: {mean_ana:.5f}, Mean Num: {mean_num:.5f}")

    # Analytical plot
    im1 = axes[idx, 0].imshow(U_analytical, extent=[0, 1, 0, 1], origin='lower',
                              aspect='auto', cmap=common_cmap, vmin=vmin_shared, vmax=vmax_shared)
    axes[idx, 0].set_title(f'Analytical α={alpha}, β={beta}')
    axes[idx, 0].set_xlabel('x')
    axes[idx, 0].set_ylabel('t')
    fig.colorbar(im1, ax=axes[idx, 0])

    # Numerical plot
    im2 = axes[idx, 1].imshow(U_numeric_stable, extent=[0, 1, 0, 1], origin='lower',
                              aspect='auto', cmap=common_cmap, vmin=vmin_shared, vmax=vmax_shared)
    axes[idx, 1].set_title(f'Numerical α={alpha}, β={beta}')
    axes[idx, 1].set_xlabel('x')
    axes[idx, 1].set_ylabel('t')
    fig.colorbar(im2, ax=axes[idx, 1])

plt.tight_layout()
plt.show()

